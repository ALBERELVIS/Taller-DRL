"""EXTENSIÓN: Proximal Policy Optimization (Schulman et al., 2017) con entornos vectorizados.

- `VecTradingEnv` replica EXACTAMENTE la dinámica de `SimpleTradingEnv` para un lote de series
  (verificado frente al entorno original en tests/): permite jugar N episodios a la vez.
- Actor-crítico de dos cabezas (la misma `ACNet` que A3C), ventaja GAE(lambda), objetivo
  recortado  L = E[min(rho A, clip(rho, 1-eps, 1+eps) A)], bonus de entropía, varias épocas
  de minibatch sobre cada lote de experiencia (on-policy, sin replay).
- Igual que DQN/A3C: recompensa x100 y bootstrap en el límite de tiempo.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a3c import ACNet, ActorPolicy
from .env import train_series
from .evaluate import obs_batch
from .features import make_encoder

DEFAULTS = dict(episodes=1600, n_envs=16, lr=3e-4, gamma=0.9, lam=0.95, clip=0.2, epochs=4,
                minibatch=1024, entropy=0.01, value_coef=0.5, hidden=64, reward_scale=100.0,
                grad_clip=0.5, eval_every=32, features="base", cost=0.001)


class VecTradingEnv:
    """Lote de S series de igual longitud con la dinámica del entorno original."""

    def __init__(self, prices, cost=0.001):
        self.P = np.asarray(prices, dtype=float)
        self.cost = cost

    def reset(self):
        S = len(self.P)
        self.t = 5
        self.pos = np.zeros(S, dtype=int)
        self.cash = np.full(S, 1000.0)
        self.shares = np.zeros(S)
        self.balance = np.full(S, 1000.0)
        return obs_batch(self.P, self.t, self.pos)

    def step(self, a):
        c, P, t = self.cost, self.P, self.t
        prev, curr = P[:, t], P[:, t + 1]
        buy = (a == 2) & (self.pos == 0)
        sell = (a == 0) & (self.pos == 1)
        self.shares = np.where(buy, (self.cash - self.cash * c) / prev, self.shares)
        self.cash = np.where(buy, 0.0, self.cash)
        sale = self.shares * prev
        self.cash = np.where(sell, sale - sale * c, self.cash)
        self.shares = np.where(sell, 0.0, self.shares)
        self.pos = np.where(buy, 1, np.where(sell, 0, self.pos))
        r = -c * (buy | sell) + np.where(self.pos == 1, (curr - prev) / prev, 0.0)
        self.balance = np.where(self.pos == 1, self.cash + self.shares * curr, self.cash)
        self.t += 1
        done = self.t >= P.shape[1] - 1
        return obs_batch(P, self.t, self.pos), r, done


def train_ppo(gen, cfg, run_seed, evaluator=None, log=print):
    c = {**DEFAULTS, **cfg}
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    enc = make_encoder(gen, c)
    net = ACNet(c["hidden"], n_in=enc.n_in)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
    g, lam, N = c["gamma"], c["lam"], c["n_envs"]
    hist = dict(episode=[], G=[], sum_reward=[], balance=[], time=[], exposure=[])
    curve = dict(episode=[], val=[], time=[])
    best = (-np.inf, copy.deepcopy(net.state_dict()), 0)
    t0, ep, next_eval = time.time(), 0, c["eval_every"]

    def encode(o, env):
        return enc.encode(o, env.P, env.t) if enc.needs_history else enc(o)

    while ep < c["episodes"]:
        env = VecTradingEnv([train_series(gen, run_seed, ep + i) for i in range(N)], c["cost"])
        o = env.reset()
        S, A, LP, V, R = [], [], [], [], []
        inside = np.zeros(N)
        done = False
        while not done:
            s = encode(o, env)
            with torch.no_grad():
                logits, v = net(torch.from_numpy(s))
                p = F.softmax(logits, -1).numpy()
            a = np.minimum((p.cumsum(1) < rng.random((N, 1))).sum(1), 2)
            o, r, done = env.step(a)
            inside += env.pos
            S.append(s), A.append(a), V.append(v.numpy()), R.append(r)
            LP.append(np.log(p[np.arange(N), a] + 1e-12))
        with torch.no_grad():
            v_last = net(torch.from_numpy(encode(o, env)))[1].numpy()  # bootstrap (límite de tiempo)
        R = np.asarray(R) * c["reward_scale"]
        V = np.asarray(V)
        T = len(R)
        adv = np.zeros_like(R)
        last = np.zeros(N)
        for k in range(T - 1, -1, -1):
            nv = v_last if k == T - 1 else V[k + 1]
            delta = R[k] + g * nv - V[k]
            last = delta + g * lam * last
            adv[k] = last
        ret = adv + V
        bS = torch.from_numpy(np.asarray(S).reshape(T * N, -1))
        bA = torch.from_numpy(np.asarray(A).reshape(-1))
        bLP = torch.from_numpy(np.asarray(LP, np.float32).reshape(-1))
        bAdv = torch.from_numpy(adv.reshape(-1).astype(np.float32))
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        bRet = torch.from_numpy(ret.reshape(-1).astype(np.float32))
        n = T * N
        for _ in range(c["epochs"]):
            perm = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, c["minibatch"]):
                idx = perm[i:i + c["minibatch"]]
                logits, v = net(bS[idx])
                logp = F.log_softmax(logits, -1)
                lp_a = logp.gather(1, bA[idx, None]).squeeze(1)
                ratio = torch.exp(lp_a - bLP[idx])
                pg = -torch.min(ratio * bAdv[idx], torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * bAdv[idx]).mean()
                ent = -(logp.exp() * logp).sum(1).mean()
                loss = pg + c["value_coef"] * (bRet[idx] - v).pow(2).mean() - c["entropy"] * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), c["grad_clip"])
                opt.step()
        raw = R / c["reward_scale"]
        Gs = (raw * (g ** np.arange(T))[:, None]).sum(0)
        for i in range(N):
            for key, val in zip(hist, (ep + i, Gs[i], raw[:, i].sum(), env.balance[i], time.time() - t0,
                                       inside[i] / T)):
                hist[key].append(val)
        ep += N
        if evaluator is not None and (ep >= next_eval or ep >= c["episodes"]):
            next_eval += c["eval_every"]
            val = evaluator(ActorPolicy(net, enc))
            curve["episode"].append(ep)
            curve["val"].append(val)
            curve["time"].append(time.time() - t0)
            if val > best[0]:
                best = (val, copy.deepcopy(net.state_dict()), ep)
    return dict(net=net, encoder=enc, hist=hist, curve=curve, best_state=best[1], best_episode=best[2],
                best_val=best[0], final_state=copy.deepcopy(net.state_dict()), train_time=time.time() - t0, cfg=c)
