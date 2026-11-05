"""Q-learning tabular (referencia del taller de Luis): la tendencia se discretiza.

Estado discreto = (bin de tendencia normalizada, posición). Tabla Q de (n_bins*2) x 3.
alpha y epsilon decaen exponencialmente por episodio.
target="expected_sarsa": objetivo on-policy E_{a'~ε-greedy}[Q(s', a')] en lugar de max_a' Q(s', a')
(van Seijen et al., 2009); mismo flujo de números aleatorios que Q-learning.
"""
import time

import numpy as np

from .encoding import Encoder
from .env import SimpleTradingEnv, make_series, train_seed

DEFAULTS = dict(episodes=1000, n_bins=10, alpha=0.1, alpha_min=0.01, alpha_decay=0.997,
                gamma=0.9, eps_start=1.0, eps_min=0.02, eps_decay=0.99, reward_scale=100.0,
                timeout_bootstrap=True, eval_every=25, target="q")


class TabularPolicy:
    def __init__(self, Q, edges, encoder):
        self.Q, self.edges, self.enc = Q, edges, encoder

    def state(self, obs):
        z = self.enc(obs)
        return np.searchsorted(self.edges, z[..., 0]) * 2 + z[..., 1].astype(int)

    def q_values(self, obs):
        return self.Q[self.state(obs)]

    def __call__(self, obs, rng=None):
        return self.q_values(obs).argmax(axis=-1)


def train_qlearning(gen, cfg, run_seed, evaluator=None, log=print):
    c = {**DEFAULTS, **cfg}
    rng = np.random.default_rng(run_seed)
    enc = Encoder(gen)
    # bins equiprobables de una normal estándar sobre la tendencia normalizada
    from scipy.stats import norm
    edges = norm.ppf(np.linspace(0, 1, c["n_bins"] + 1)[1:-1])
    Q = np.zeros((c["n_bins"] * 2, 3))
    pol = TabularPolicy(Q, edges, enc)
    g = c["gamma"]
    hist = dict(episode=[], G=[], sum_reward=[], balance=[], eps=[], time=[])
    curve = dict(episode=[], val=[], time=[])
    best = (-np.inf, Q.copy(), 0)
    t0 = time.time()
    for ep in range(c["episodes"]):
        eps = max(c["eps_min"], c["eps_start"] * c["eps_decay"] ** ep)
        alpha = max(c["alpha_min"], c["alpha"] * c["alpha_decay"] ** ep)
        env = SimpleTradingEnv(make_series(gen, train_seed(run_seed, ep)))
        obs, _ = env.reset()
        s = int(pol.state(obs))
        G, R, k = 0.0, 0.0, 0
        while True:
            a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
            obs2, r, term, trunc, info = env.step(a)
            s2 = int(pol.state(obs2))
            if c["target"] == "expected_sarsa":
                v2 = (1 - eps) * Q[s2].max() + eps * Q[s2].mean()
            else:
                v2 = Q[s2].max()
            boot = 0.0 if (term and not c["timeout_bootstrap"]) else g * v2
            Q[s, a] += alpha * (r * c["reward_scale"] + boot - Q[s, a])
            G += g ** k * r
            R += r
            k += 1
            s = s2
            if term or trunc:
                break
        for key, v in zip(hist, (ep, G, R, info["balance"], eps, time.time() - t0)):
            hist[key].append(v)
        if evaluator is not None and ((ep + 1) % c["eval_every"] == 0 or ep == c["episodes"] - 1):
            val = evaluator(pol)
            curve["episode"].append(ep + 1)
            curve["val"].append(val)
            curve["time"].append(time.time() - t0)
            if val > best[0]:
                best = (val, Q.copy(), ep + 1)
    return dict(Q=Q, edges=edges, encoder=enc, hist=hist, curve=curve, best_state=best[1],
                best_episode=best[2], best_val=best[0], final_state=Q.copy(),
                train_time=time.time() - t0, cfg=c)
