"""Inferencia para comparar agentes evaluados en las MISMAS series de test.

Todas las funciones reciben matrices (semillas, series): cada fila es un entrenamiento y
cada columna una serie de test común a todos los agentes. Una referencia determinista
(B&H, una regla) es una matriz de una fila.

- `contrast`: diferencia media, Wilcoxon pareado (sobre la media por serie), IC95%
  bootstrap jerárquico (semillas × series), tamaños del efecto y test de equivalencia.
- `friedman_nemenyi`: test global de k agentes y diferencia crítica (Demšar, 2006).
- `iqm_ci`, `prob_improvement`: resúmenes robustos entre ejecuciones (Agarwal et al., 2021).
- `variance_components`, `mde`: potencia y tamaño muestral del diseño.
- `holm`, `bh`: corrección por comparaciones múltiples (error de familia y FDR).
"""
import numpy as np
from scipy import stats

from .evaluate import holm

# Valores críticos q_alpha de Nemenyi (alfa = 0.05): rango studentizado con infinitos grados de
# libertad dividido por sqrt(2). k = 2..10 de Demšar (2006), tabla 5a; k = 11, 12 con la misma fórmula.
Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
             11: 3.219, 12: 3.268}


def bh(pvals):
    """Benjamini-Hochberg: p-valores ajustados que controlan la tasa de falsos descubrimientos."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    ranked = p[ok][idx] * m / np.arange(1, m + 1)
    adj = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def hier_boot(X, Y, n_boot=4000, seed=0):
    """Distribución bootstrap de mean(X) - mean(Y) remuestreando semillas de cada agente y las series
    (las mismas para ambos, porque están emparejadas). Vectorizado con pesos multinomiales."""
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    rng = np.random.default_rng(seed)
    sx, sy, n = len(X), len(Y), X.shape[1]
    wj = rng.multinomial(n, np.full(n, 1 / n), size=n_boot) / n
    wx = rng.multinomial(sx, np.full(sx, 1 / sx), size=n_boot) / sx
    wy = rng.multinomial(sy, np.full(sy, 1 / sy), size=n_boot) / sy
    mx = np.einsum("bs,sn,bn->b", wx, X, wj)
    my = np.einsum("bs,sn,bn->b", wy, Y, wj)
    return mx - my


def rank_biserial(d):
    """Correlación rango-biserial emparejada: (W+ - W-)/(W+ + W-), en [-1, 1]; tamaño del efecto de Wilcoxon."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if not np.any(d != 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def contrast(X, Y, delta=None, n_boot=4000, seed=0, LX=None, LY=None):
    """Contraste emparejado de X frente a Y (matrices semillas × series de la métrica principal, Σr).

    delta: margen de equivalencia (TOST con el IC90% jerárquico, que equivale a dos tests
    unilaterales al 5%). LX, LY: log-riqueza por serie para el cociente geométrico de capital."""
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    d = X.mean(0) - Y.mean(0)
    boots = hier_boot(X, Y, n_boot, seed)
    out = dict(diff=float(d.mean()), ci_low=float(np.quantile(boots, 0.025)),
               ci_high=float(np.quantile(boots, 0.975)), p=wilcoxon_p(d), r_rb=rank_biserial(d),
               win=float(np.mean(d > 0) + 0.5 * np.mean(d == 0)) * 100, n_x=len(X), n_y=len(Y),
               se=float(boots.std()))
    if delta is not None:
        lo90, hi90 = np.quantile(boots, [0.05, 0.95])
        out.update(delta=float(delta), ci90_low=float(lo90), ci90_high=float(hi90),
                   equivalent=bool(-delta < lo90 and hi90 < delta),
                   p_tost=float(max(np.mean(boots <= -delta), np.mean(boots >= delta))))
    if LX is not None and LY is not None:
        lb = hier_boot(LX, LY, n_boot, seed + 1)
        dl = np.atleast_2d(LX).mean(0) - np.atleast_2d(LY).mean(0)
        out.update(cap_ratio=float(np.exp(dl.mean())), cap_low=float(np.exp(np.quantile(lb, 0.025))),
                   cap_high=float(np.exp(np.quantile(lb, 0.975))))
    return out


def verdict(row, alpha=0.05, p_col="p_adj"):
    """Regla única de veredicto de toda la memoria.

    gana / pierde: p ajustada < alfa Y el IC95% jerárquico excluye el 0 (en el mismo sentido);
    equivalente: el IC90% jerárquico cae dentro de ±delta (TOST); si no, no concluyente."""
    p = row.get(p_col, np.nan)
    if p < alpha and row["ci_low"] > 0:
        return "gana"
    if p < alpha and row["ci_high"] < 0:
        return "pierde"
    if row.get("equivalent", False):
        return "equivalente"
    return "no concluyente"


def friedman_nemenyi(M, alpha=0.05):
    """M: (series, k agentes), mayor es mejor. Devuelve estadístico y p de Friedman, rangos medios
    (1 = mejor) y diferencia crítica de Nemenyi para alfa = 0.05."""
    M = np.asarray(M, float)
    n, k = M.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)
    chi, p = stats.friedmanchisquare(*M.T)
    cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * n))
    return dict(chi2=float(chi), p=float(p), avg_rank=ranks.mean(0), cd=float(cd), n=n, k=k)


def iqm(x):
    """Media intercuartílica: media del 50% central (robusta a colas, eficiente frente a la mediana)."""
    x = np.sort(np.ravel(x))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def iqm_ci(X, n_boot=2000, seed=0):
    """IQM de todas las puntuaciones (semillas × series) con IC95% por bootstrap estratificado."""
    X = np.atleast_2d(np.asarray(X, float))
    rng = np.random.default_rng(seed)
    s, n = X.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = iqm(X[rng.integers(0, s, s)][:, rng.integers(0, n, n)])
    return iqm(X), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def prob_improvement(X, Y):
    """P(X > Y): probabilidad de que un entrenamiento de X supere a uno de Y en una serie de test al azar
    (empates cuentan 1/2)."""
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    gt = (X[:, None, :] > Y[None, :, :]).mean()
    eq = (X[:, None, :] == Y[None, :, :]).mean()
    return float(gt + 0.5 * eq)


def variance_components(X, Y):
    """Fuentes de incertidumbre de mean(X) - mean(Y): varianza entre semillas de cada agente y
    varianza entre series de la diferencia. Devuelve (sd_semilla_x, sd_semilla_y, sd_series)."""
    X, Y = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Y, float))
    sdx = float(X.mean(1).std(ddof=1)) if len(X) > 1 else 0.0
    sdy = float(Y.mean(1).std(ddof=1)) if len(Y) > 1 else 0.0
    sdn = float((X.mean(0) - Y.mean(0)).std(ddof=1))
    return sdx, sdy, sdn


def mde(sdx, sdy, sdn, sx, sy, n, z=2.80):
    """Efecto mínimo detectable con potencia 0.8 y alfa 0.05 bilateral (z_0.975 + z_0.8 = 2.80)."""
    se = np.sqrt(sdx ** 2 / sx + (sdy ** 2 / sy if sy else 0.0) + sdn ** 2 / n)
    return float(z * se), float(se)


__all__ = ["bh", "holm", "hier_boot", "rank_biserial", "wilcoxon_p", "contrast", "verdict", "friedman_nemenyi",
           "iqm", "iqm_ci", "prob_improvement", "variance_components", "mde", "Q_NEMENYI"]
