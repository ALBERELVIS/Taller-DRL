"""Codificación del estado como entrada de la red (preprocesado del agente).

El entorno no se modifica: la observación sigue siendo (tendencia, posición). Solo se
reescala la tendencia dividiéndola por su desviación típica, estimada exclusivamente
con series de ENTRENAMIENTO de cada generador. Sin este reescalado la tendencia vale
~0.02 en GBM/reversión/regímenes y ~0.15 en el ciclo, y la red recibe una entrada casi
constante frente a la posición (0/1), lo que ralentiza mucho el aprendizaje.
"""
import numpy as np

from .env import make_series, train_seed

_SIGMA = {}
CLIP = 5.0


def trend_series(prices):
    p = np.asarray(prices, dtype=float)
    out = []
    for t in range(5, len(p) - 1):
        w = p[t - 5:t + 1]
        m = w.mean()
        out.append(np.clip((w[-1] - m) / m, -1.0, 1.0))
    return np.array(out)


def trend_sigma(gen, n_series=50):
    """Desviación típica de la tendencia en series de entrenamiento (bloque de semillas 99)."""
    if gen not in _SIGMA:
        tr = np.concatenate([trend_series(make_series(gen, train_seed(99, i))) for i in range(n_series)])
        _SIGMA[gen] = float(tr.std())
    return _SIGMA[gen]


class Encoder:
    needs_history = False
    n_in = 2

    def __init__(self, gen=None, sigma=None):
        self.sigma = sigma if sigma is not None else trend_sigma(gen)

    def from_env(self, obs, env):
        return self(obs)

    def encode(self, obs, P=None, t=None):
        return self(obs)

    def state(self):
        return dict(kind="base", sigma=self.sigma)

    def __call__(self, obs):
        """obs: array (..., 2) con [trend, posicion] -> float32 (..., 2)."""
        obs = np.asarray(obs, dtype=np.float32)
        z = np.clip(obs[..., 0] / self.sigma, -CLIP, CLIP)
        return np.stack([z, obs[..., 1]], axis=-1).astype(np.float32)
