"""Deep Q-Network (Mnih et al., 2015) con las mejoras vistas en clase.

- Experience replay con muestreo uniforme (rompe la correlación temporal).
- Red objetivo Q' copiada cada `target_update` pasos (target estable). Con
  `target_update=0` se usa la propia red (DQN 'vanilla' del notebook del profesor).
- Double DQN opcional (van Hasselt et al., 2016): a* = argmax Q(s'), valor Q'(s', a*).
- Pérdida Huber y recorte de gradiente (robustez frente a rendimientos extremos).
- epsilon-greedy con decaimiento exponencial por episodio.
- Recompensa escalada (x100): no cambia la política óptima y deja los Q en O(1).
- El fin del episodio es un límite de tiempo (el estado no contiene t): por defecto se
  sigue haciendo bootstrap en la última transición (Pardo et al., 2018).
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .env import make_env, train_series
from .features import make_encoder

DEFAULTS = dict(
    episodes=300, lr=1e-3, gamma=0.9, eps_start=1.0, eps_min=0.02, eps_decay=0.985,
    batch_size=64, buffer_size=50_000, warmup=1_000, train_freq=1, target_update=500,
    double=True, hidden=64, reward_scale=100.0, grad_clip=10.0, timeout_bootstrap=True,
    eval_every=10,
    # extensiones (los valores por defecto reproducen el protocolo principal)
    dueling=False, features="base", cost=0.001, reward_mode="pct", risk_lambda=0.0,
)


class QNet(nn.Module):
    def __init__(self, hidden=64, n_in=2, n_out=3, dueling=False):
        super().__init__()
        self.dueling = dueling
        if not dueling:
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_out))
        else:  # Wang et al. (2016): Q = V + A - mean(A)
            self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.v = nn.Linear(hidden, 1)
            self.adv = nn.Linear(hidden, n_out)

    def forward(self, x):
        if not self.dueling:
            return self.net(x)
        h = self.body(x)
        a = self.adv(h)
        return self.v(h) + a - a.mean(-1, keepdim=True)


def agent_reward(r, bal, prev_bal, c):
    """Recompensa que ve el agente (la evaluación usa siempre el balance real).
    pct: la del entorno; log: log(B_t/B_{t-1}); mv: r - lambda r^2 (media-varianza)."""
    if c["reward_mode"] == "log":
        return float(np.log(bal / prev_bal))
    if c["reward_mode"] == "mv":
        return r - c["risk_lambda"] * r * r
    return r


class Replay:
    def __init__(self, cap, n_in=2):
        self.s = np.zeros((cap, n_in), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, n_in), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.i, self.n = cap, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k, rng):
        idx = rng.integers(0, self.n, size=k)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


class GreedyPolicy:
    """Política greedy (sin exploración) para evaluar: argmax_a Q(s,a)."""

    def __init__(self, net, encoder):
        self.net, self.enc = net, encoder
        self.needs_history = encoder.needs_history

    def q_values(self, obs, P=None, t=None):
        x = self.enc.encode(obs, P, t) if self.needs_history else self.enc(obs)
        with torch.no_grad():
            return self.net(torch.from_numpy(x)).numpy()

    def __call__(self, obs, rng=None, P=None, t=None):
        return self.q_values(obs, P, t).argmax(axis=1)


def train_dqn(gen, cfg, run_seed, evaluator=None, log=print):
    """Entrena un agente DQN. `evaluator(policy) -> float` mide la política greedy en
    validación cada `eval_every` episodios (para curvas y parada temprana)."""
    c = {**DEFAULTS, **cfg}
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    enc = make_encoder(gen, c)
    q = QNet(c["hidden"], n_in=enc.n_in, dueling=c["dueling"])
    q_tgt = copy.deepcopy(q) if c["target_update"] > 0 else q
    opt = torch.optim.Adam(q.parameters(), lr=c["lr"])
    loss_fn = nn.SmoothL1Loss()
    buf = Replay(c["buffer_size"], n_in=enc.n_in)
    g = c["gamma"]
    hist = dict(episode=[], G=[], sum_reward=[], balance=[], eps=[], time=[])
    curve = dict(episode=[], val=[], time=[])
    best = (-np.inf, copy.deepcopy(q.state_dict()), 0)
    steps, t0 = 0, time.time()
    for ep in range(c["episodes"]):
        eps = max(c["eps_min"], c["eps_start"] * c["eps_decay"] ** ep)
        env = make_env(train_series(gen, run_seed, ep), c["cost"])
        obs, _ = env.reset()
        s = enc.from_env(obs, env)
        G, R, k = 0.0, 0.0, 0
        prev_bal = env.balance
        while True:
            if rng.random() < eps:
                a = int(rng.integers(3))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s[None]))[0].argmax())
            obs2, r, term, trunc, info = env.step(a)
            s2 = enc.from_env(obs2, env)
            done = float(term and not c["timeout_bootstrap"])
            ra = agent_reward(r, info["balance"], prev_bal, c)
            prev_bal = info["balance"]
            buf.add(s, a, ra * c["reward_scale"], s2, done)
            G += g ** k * r
            R += r
            k += 1
            steps += 1
            s = s2
            if buf.n >= c["warmup"] and steps % c["train_freq"] == 0:
                bs, ba, br, bs2, bd = buf.sample(c["batch_size"], rng)
                with torch.no_grad():
                    if c["double"]:
                        a_star = q(bs2).argmax(1, keepdim=True)
                        q_next = q_tgt(bs2).gather(1, a_star).squeeze(1)
                    else:
                        q_next = q_tgt(bs2).max(1).values
                    target = br + g * (1 - bd) * q_next
                pred = q(bs).gather(1, ba[:, None]).squeeze(1)
                loss = loss_fn(pred, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), c["grad_clip"])
                opt.step()
                if c["target_update"] > 0 and steps % c["target_update"] == 0:
                    q_tgt.load_state_dict(q.state_dict())
            if term or trunc:
                break
        for key, v in zip(hist, (ep, G, R, info["balance"], eps, time.time() - t0)):
            hist[key].append(v)
        if evaluator is not None and ((ep + 1) % c["eval_every"] == 0 or ep == c["episodes"] - 1):
            val = evaluator(GreedyPolicy(q, enc))
            curve["episode"].append(ep + 1)
            curve["val"].append(val)
            curve["time"].append(time.time() - t0)
            if val > best[0]:
                best = (val, copy.deepcopy(q.state_dict()), ep + 1)
    final_state = copy.deepcopy(q.state_dict())
    return dict(net=q, encoder=enc, hist=hist, curve=curve, best_state=best[1], best_episode=best[2],
                best_val=best[0], final_state=final_state, train_time=time.time() - t0, cfg=c)
