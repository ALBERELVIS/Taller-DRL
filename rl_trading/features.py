"""EXTENSIÓN: estado enriquecido con información causal del historial de precios.

El entorno no cambia: sigue devolviendo (tendencia, posición). El agente añade variables
calculadas SOLO con precios ya observados (P[:, :t+1]); `tests/` comprueba que alterar el
futuro no altera las variables. Conjuntos:

- base        : [z, pos]                              (z = tendencia / sigma_train)
- ventana     : base + últimos 10 log-rendimientos
- ma_vol      : base + (precio / MA30 - 1) + volatilidad de 20 pasos
- ventana_ma_vol : todo lo anterior

Todas las variables se normalizan con media y desviación típica de series de ENTRENAMIENTO
(bloque de semillas 99, como la codificación base).
"""
import numpy as np

from .encoding import CLIP
from .env import MULTI_GENS, make_series, train_seed

N_LAGS, MA, VOL = 10, 30, 20
SETS = ("base", "ventana", "ma_vol", "ventana_ma_vol")


def raw_features(P, t, feats):
    """Variables crudas en el instante t para un lote P (S, n). Solo usa P[:, :t+1]."""
    P = np.asarray(P, dtype=float)
    w = P[:, t - 5:t + 1]
    m = w.mean(axis=1)
    cols = [np.clip((w[:, -1] - m) / m, -1.0, 1.0)]
    if feats in ("ventana", "ventana_ma_vol"):
        seg = np.log(P[:, max(0, t - N_LAGS):t + 1])
        r = np.diff(seg, axis=1)
        if r.shape[1] < N_LAGS:
            r = np.concatenate([np.zeros((len(P), N_LAGS - r.shape[1])), r], axis=1)
        cols += list(r[:, ::-1].T)  # r_t, r_{t-1}, ...
    if feats in ("ma_vol", "ventana_ma_vol"):
        cols.append(P[:, t] / P[:, max(0, t - MA + 1):t + 1].mean(axis=1) - 1)
        lr = np.diff(np.log(P[:, max(0, t - VOL):t + 1]), axis=1)
        cols.append(lr.std(axis=1))
    return np.stack(cols, axis=1)


class FeatureEncoder:
    """Codificador con historial. API común con `Encoder`: n_in, needs_history, encode."""

    needs_history = True

    def __init__(self, gen, feats, n_series=30, stats=None):
        self.feats, self.gen = feats, gen
        if stats is None:
            gens = MULTI_GENS if gen == "multi" else (gen,)
            X = []
            for g in gens:
                for i in range(n_series):
                    p = make_series(g, train_seed(99, i))[None]
                    X += [raw_features(p, t, feats)[0] for t in range(5, p.shape[1] - 1, 3)]
            X = np.asarray(X)
            mu, sd = X.mean(0), X.std(0) + 1e-12
            mu[0] = 0.0  # la tendencia se escala sin centrar, como la codificación base
            stats = dict(mu=mu.tolist(), sd=sd.tolist())
        self.stats = stats
        self.mu, self.sd = np.asarray(stats["mu"]), np.asarray(stats["sd"])
        self.sigma = float(self.sd[0])
        self.n_in = len(self.mu) + 1

    def encode(self, obs, P, t):
        z = np.clip((raw_features(P, t, self.feats) - self.mu) / self.sd, -CLIP, CLIP)
        return np.concatenate([z, np.asarray(obs, np.float32)[:, 1:2]], axis=1).astype(np.float32)

    def from_env(self, obs, env):
        return self.encode(np.asarray(obs)[None], np.asarray(env.prices)[None], env.current_step)[0]

    def state(self):
        return dict(kind="features", gen=self.gen, feats=self.feats, stats=self.stats)


def make_encoder(gen, cfg):
    """Encoder base (enunciado) o con historial según cfg['features']."""
    from .encoding import Encoder
    feats = cfg.get("features", "base")
    if feats == "base" and gen != "multi":
        return Encoder(gen)
    if feats == "base":  # multi-mercado: sigma conjunta de los 4 generadores
        return FeatureEncoder("multi", "base")
    return FeatureEncoder(gen, feats)


def encoder_from_state(st):
    from .encoding import Encoder
    if st.get("kind") == "features":
        return FeatureEncoder(st["gen"], st["feats"], stats=st["stats"])
    return Encoder(sigma=st["sigma"])
