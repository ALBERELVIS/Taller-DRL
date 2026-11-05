"""Evaluación: simulador vectorizado (equivalente exacto al entorno) y métricas.

Evaluar con el `SimpleTradingEnv` paso a paso y serie a serie es lento; aquí se simula
en paralelo un lote de series con la MISMA dinámica (comprobado en `tests_sanity.py`
frente al entorno original: balances y recompensas idénticos a precisión de máquina).
"""
import numpy as np
from scipy import stats

COST = 0.001
INITIAL = 1000.0


def obs_batch(P, t, pos):
    w = P[:, t - 5:t + 1]
    m = w.mean(axis=1)
    trend = np.clip((w[:, -1] - m) / m, -1.0, 1.0)
    return np.stack([trend, pos.astype(float)], axis=1).astype(np.float32)


def simulate(prices_list, policy, rng=None, cost=COST):
    """Simula `policy(obs (S,2), rng) -> acciones (S,)` sobre series de igual longitud.

    Si la política declara `needs_history=True` (estado enriquecido, extensión) se llama
    como `policy(obs, rng, P=P, t=t)`; solo debe usar P[:, :t+1] (causalidad, ver tests/).
    `cost` permite la extensión de sensibilidad a la comisión (0.001 = enunciado).

    Devuelve dict con rewards (S,T), balances (S,T+1), positions (S,T) (posición tras
    actuar), actions (S,T) y obs (S,T,2).
    """
    COST = cost
    hist = getattr(policy, "needs_history", False)
    P = np.asarray(prices_list, dtype=float)
    S, n = P.shape
    T = n - 1 - 5
    pos = np.zeros(S, dtype=int)
    cash = np.full(S, INITIAL)
    shares = np.zeros(S)
    rewards = np.zeros((S, T))
    balances = np.zeros((S, T + 1))
    balances[:, 0] = INITIAL
    positions = np.zeros((S, T), dtype=int)
    actions = np.zeros((S, T), dtype=int)
    observations = np.zeros((S, T, 2), dtype=np.float32)
    for k, t in enumerate(range(5, n - 1)):
        o = obs_batch(P, t, pos)
        a = np.asarray(policy(o, rng, P=P, t=t) if hist else policy(o, rng)).astype(int)
        prev, curr = P[:, t], P[:, t + 1]
        r = np.zeros(S)
        buy = (a == 2) & (pos == 0)
        sell = (a == 0) & (pos == 1)
        shares = np.where(buy, (cash - cash * COST) / prev, shares)
        cash = np.where(buy, 0.0, cash)
        sale = shares * prev
        cash = np.where(sell, sale - sale * COST, cash)
        shares = np.where(sell, 0.0, shares)
        pos = np.where(buy, 1, np.where(sell, 0, pos))
        r -= COST * (buy | sell)
        r += np.where(pos == 1, (curr - prev) / prev, 0.0)
        bal = np.where(pos == 1, cash + shares * curr, cash)
        rewards[:, k] = r
        balances[:, k + 1] = bal
        positions[:, k] = pos
        actions[:, k] = a
        observations[:, k] = o
    return dict(rewards=rewards, balances=balances, positions=positions,
                actions=actions, obs=observations)


def discounted(rewards, gamma):
    w = gamma ** np.arange(rewards.shape[1])
    return rewards @ w


def metrics(sim, gamma=0.99):
    b = sim["balances"]
    r = sim["rewards"]
    final = b[:, -1]
    rets = b[:, 1:] / b[:, :-1] - 1
    sd = rets.std(axis=1)
    sharpe = np.where(sd > 0, rets.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)
    peak = np.maximum.accumulate(b, axis=1)
    mdd = ((peak - b) / peak).max(axis=1)
    pos = sim["positions"]
    trades = np.abs(np.diff(np.concatenate([np.zeros((pos.shape[0], 1), int), pos], axis=1), axis=1)).sum(axis=1)
    return dict(
        final_balance=final,
        ret_pct=100 * (final / INITIAL - 1),
        log_wealth=np.log(np.maximum(final, 1e-12) / INITIAL),
        sum_reward=r.sum(axis=1),
        G=discounted(r, gamma),
        sharpe=sharpe,
        max_dd=100 * mdd,
        trades=trades,
        exposure=100 * pos.mean(axis=1),
    )


def summarize(m):
    """Resumen por conjunto de series (medias y medianas)."""
    return {
        "ret_median": float(np.median(m["ret_pct"])),
        "ret_mean": float(np.mean(m["ret_pct"])),
        "log_wealth": float(np.mean(m["log_wealth"])),
        "sum_reward": float(np.mean(m["sum_reward"])),
        "G": float(np.mean(m["G"])),
        "sharpe": float(np.mean(m["sharpe"])),
        "max_dd": float(np.mean(m["max_dd"])),
        "trades": float(np.mean(m["trades"])),
        "exposure": float(np.mean(m["exposure"])),
    }


def holm(pvals):
    """Corrección de Holm-Bonferroni (controla el error de familia) para un vector de p-valores."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def hier_bootstrap(X, Y, n_boot=4000, seed=0):
    """IC95% de mean(X) - mean(Y) con X (semillas_x, series), Y (semillas_y, series).

    Remuestrea semillas de cada agente (variabilidad del entrenamiento) y series de test de forma
    pareada (mismas series para ambos): incluye las dos fuentes de incertidumbre."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    out = np.empty(n_boot)
    for b in range(n_boot):
        sx = rng.integers(0, len(X), len(X))
        sy = rng.integers(0, len(Y), len(Y))
        j = rng.integers(0, n, n)
        out[b] = X[sx][:, j].mean() - Y[sy][:, j].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def paired_test(x, y, n_boot=5000, seed=0):
    """Compara x vs y emparejados (mismas series). Wilcoxon + IC95% bootstrap de la media de x-y."""
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return dict(mean_diff=float(d.mean()), ci_low=float(np.quantile(boots, 0.025)),
                ci_high=float(np.quantile(boots, 0.975)), p_wilcoxon=p,
                win_rate=float(np.mean(d > 0)))
