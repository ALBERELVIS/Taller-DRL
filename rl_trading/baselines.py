"""Políticas de referencia. Todas reciben obs (S,2)=[trend, posición] y rng."""
import numpy as np

from .encoding import trend_series
from .env import N_STEPS, make_series, train_seed


def random_policy(obs, rng):
    return rng.integers(0, 3, size=len(obs))


def buy_and_hold(obs, rng=None):
    # compra en el primer paso y no vuelve a operar (idéntico al B&H del notebook)
    return np.full(len(obs), 2)


def momentum(obs, rng=None):
    return np.where(obs[:, 0] > 0, 2, 0)


def contrarian(obs, rng=None):
    return np.where(obs[:, 0] < 0, 2, 0)


class MyopicBayes:
    """Regla 'oráculo miope': invierte si E[r_{t+1} | bin de tendencia] > 0.

    La esperanza condicionada se estima con 200 series de ENTRENAMIENTO (bloque de
    semillas 98) por cuantiles de la tendencia. Es una referencia informada (conoce la
    estadística del generador) y miope (gamma=0, ignora la comisión); no un óptimo exacto.
    """

    def __init__(self, gen, n_bins=20, n_series=200):
        T, R = [], []
        for i in range(n_series):
            p = make_series(gen, train_seed(98, i))
            T.append(trend_series(p))
            R.append(p[6:] / p[5:-1] - 1)
        T, R = np.concatenate(T), np.concatenate(R)
        self.edges = np.quantile(T, np.linspace(0, 1, n_bins + 1))[1:-1]
        idx = np.searchsorted(self.edges, T)
        self.mean_r = np.array([R[idx == b].mean() for b in range(n_bins)])

    def __call__(self, obs, rng=None):
        b = np.searchsorted(self.edges, obs[:, 0])
        return np.where(self.mean_r[b] > 0, 2, 0)


def regime_states(seed, n_steps=N_STEPS):
    """Régimen oculto (0 alcista, 1 bajista) de la serie `make_series('regimenes', seed)`.

    Repite exactamente las extracciones aleatorias de `generar_precios_regimenes`, así que el
    estado corresponde paso a paso a la serie generada (comprobado en tests/)."""
    np.random.seed(seed)
    estado, out = 0, np.empty(n_steps, dtype=int)
    for k in range(n_steps):
        if np.random.rand() < 0.02:
            estado = 1 - estado
        if estado == 0:
            np.random.normal(0.001, 0.01)
        else:
            np.random.normal(-0.0015, 0.025)
        out[k] = estado
    return out


class HiddenOracle:
    """Techo de referencia: política que conoce la variable OCULTA del generador (ningún agente la ve).

    - ciclo: conoce la deriva + fase del ciclo, E[r_{t+1}] = 0.0005·t + 0.05·sin(t); dentro si > 0.
    - gbm: no hay nada oculto que explotar; E[r] > 0, así que siempre dentro (= B&H).
    - reversion: conoce el nivel de equilibrio (100); compra bajo 100 - b1 y vende sobre 100 + b2,
      con (b1, b2) elegidos en series de ENTRENAMIENTO (no en test).
    - regimenes: conoce el régimen vigente en t; dentro solo en el alcista.
    Idea adaptada del trabajo de otro grupo (oráculos con información oculta)."""

    needs_history = True
    BANDS = (0, 1, 2, 3, 4, 6, 8, 10)

    def __init__(self, gen, seeds=None, band=None):
        self.gen = gen
        if gen == "ciclo":
            t = np.linspace(0, 50, N_STEPS)
            self.mu = 0.0005 * t + 0.05 * np.sin(t)
        elif gen == "regimenes":
            self.states = np.array([regime_states(s) for s in seeds])
        elif gen == "reversion":
            self.band = band if band is not None else self.fit_band()

    def fit_band(self, n_series=100):
        from .evaluate import metrics, simulate
        P = [make_series("reversion", train_seed(98, i)) for i in range(n_series)]
        best, arg = -np.inf, None
        for b1 in self.BANDS:
            for b2 in self.BANDS:
                self.band = (b1, b2)
                v = metrics(simulate(P, self))["sum_reward"].mean()
                if v > best:
                    best, arg = v, (b1, b2)
        return arg

    def __call__(self, obs, rng=None, P=None, t=None):
        pos = obs[:, 1]
        if self.gen == "gbm":
            return np.full(len(obs), 2)
        if self.gen == "ciclo":
            return np.full(len(obs), 2 if self.mu[t + 1] > 0 else 0)
        if self.gen == "regimenes":
            return np.where(self.states[:, t] == 0, 2, 0)
        b1, b2 = self.band
        p = P[:, t]
        return np.where(pos == 0, np.where(p < 100 - b1, 2, 1), np.where(p > 100 + b2, 0, 1))


BASELINES = {
    "Aleatorio": lambda gen: random_policy,
    "Buy & Hold": lambda gen: buy_and_hold,
    "Momentum": lambda gen: momentum,
    "Contraria": lambda gen: contrarian,
    "Oráculo miope": lambda gen: MyopicBayes(gen),
}
