"""Reglas de decisión parametrizadas sobre z = tendencia / sigma_train (misma información que el agente).

Familias (a, b en unidades de sigma):
- mom1  : invertido si z > a                          (momentum de un umbral)
- con1  : invertido si z < a                          (contraria de un umbral)
- mom_h : sin posición compra si z > a; con posición vende si z < b (b <= a)  -> histéresis
- con_h : sin posición compra si z < a; con posición vende si z > b (b >= a)  -> histéresis
- band  : invertido si a < z < b                      (banda, forma de U invertida)
- band_h: sin posición compra si a < z < b; con posición mantiene si c < z < d (c <= a, d >= b):
          banda con histéresis (4 umbrales, rejilla más gruesa de 0.5 sigma)

Los umbrales se ajustan por rejilla en series de ENTRENAMIENTO (bloque de semillas 97) con el
mismo criterio que optimizan los agentes (suma media de recompensas) y se evalúan una única
vez en test. Es el "listón" que debe batir un agente de RL con la misma observación.
"""
import numpy as np

from .encoding import trend_sigma
from .env import make_series, train_seed
from .evaluate import metrics, simulate

GRID = np.round(np.linspace(-3, 3, 25), 3)
GRID2 = np.round(np.linspace(-3, 3, 13), 3)
FAMILIES = ("mom1", "con1", "mom_h", "con_h", "band", "band_h")


class Rule:
    def __init__(self, sigma, family, a, b=None, c=None, d=None):
        self.sigma, self.family, self.a, self.b, self.c, self.d = sigma, family, a, b, c, d

    def __call__(self, obs, rng=None):
        z, pos = obs[:, 0] / self.sigma, obs[:, 1]
        f, a, b = self.family, self.a, self.b
        if f == "band_h":
            inside0 = (z > a) & (z < b)
            inside1 = (z > self.c) & (z < self.d)
            return np.where(pos == 0, np.where(inside0, 2, 1), np.where(inside1, 1, 0))
        if f == "mom1":
            return np.where(z > a, 2, 0)
        if f == "con1":
            return np.where(z < a, 2, 0)
        if f == "band":
            return np.where((z > a) & (z < b), 2, 0)
        if f == "mom_h":
            return np.where(pos == 0, np.where(z > a, 2, 1), np.where(z < b, 0, 1))
        if f == "con_h":
            return np.where(pos == 0, np.where(z < a, 2, 1), np.where(z > b, 0, 1))
        raise ValueError(f)

    def params(self):
        f = lambda v: None if v is None else float(v)  # noqa: E731
        return dict(family=self.family, a=f(self.a), b=f(self.b), c=f(self.c), d=f(self.d))


def candidates(family):
    if family == "band_h":
        return [(a, b, c, d) for a in GRID2 for b in GRID2 if b > a
                for c in GRID2 if c <= a for d in GRID2 if d >= b]
    if family in ("mom1", "con1"):
        return [(a, None) for a in GRID]
    if family == "mom_h":
        return [(a, b) for a in GRID for b in GRID if b <= a]
    if family == "con_h":
        return [(a, b) for a in GRID for b in GRID if b >= a]
    return [(a, b) for a in GRID for b in GRID if b > a]


def fit_series(gen, n=100):
    return [make_series(gen, train_seed(97, i)) for i in range(n)]


def fit_rule(gen, family, series=None, cost=0.001, sigma=None):
    """Mejor (a, b) de la familia por suma media de recompensas en `series`."""
    sigma = sigma or trend_sigma(gen)
    series = series if series is not None else fit_series(gen)
    best = (-np.inf, None)
    for cand in candidates(family):
        rule = Rule(sigma, family, *cand)
        v = float(metrics(simulate(series, rule, cost=cost))["sum_reward"].mean())
        if v > best[0]:
            best = (v, rule)
    return best[1], best[0]
