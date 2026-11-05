"""Asynchronous Advantage Actor-Critic (A3C; Mnih et al., 2016).

- Red de dos cabezas sobre un tronco común (como el notebook de actor-crítico del
  profesor): actor -> pi(a|s) (softmax), crítico -> V(s).
- N procesos worker (torch.multiprocessing, 'spawn' en Windows). Cada uno tiene una
  copia local de la red, juega su propia serie de precios, calcula gradientes con
  retornos a n pasos y los aplica de forma ASÍNCRONA sobre los pesos globales
  compartidos (estilo Hogwild) con un Adam de estadísticos compartidos.
- Pérdida: -log pi(a|s) * A  +  c_v * (R - V)^2  -  beta * H(pi),  A = R^(n) - V(s).
- El proceso principal evalúa periódicamente la política greedy global en validación
  (curva de aprendizaje frente a episodios y tiempo real).
"""
import copy
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from .env import make_env, train_series
from .features import make_encoder

DEFAULTS = dict(episodes=1500, workers=8, lr=5e-4, gamma=0.9, n_steps=20, entropy=0.01,
                value_coef=0.5, hidden=64, reward_scale=100.0, grad_clip=5.0,
                timeout_bootstrap=True, eval_every=50, features="base", cost=0.001)


class ACNet(nn.Module):
    def __init__(self, hidden=64, n_in=2, n_out=3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_out)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class SharedAdam(torch.optim.Adam):
    """Adam cuyos momentos viven en memoria compartida entre procesos."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, foreach=False)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["step"] = torch.zeros(())
                st["exp_avg"] = torch.zeros_like(p.data)
                st["exp_avg_sq"] = torch.zeros_like(p.data)
                st["step"].share_memory_()
                st["exp_avg"].share_memory_()
                st["exp_avg_sq"].share_memory_()


class ActorPolicy:
    """Evaluación: greedy (argmax pi) o estocástica (muestreo de pi)."""

    def __init__(self, net, encoder, greedy=True):
        self.net, self.enc, self.greedy = net, encoder, greedy
        self.needs_history = encoder.needs_history

    def probs(self, obs, P=None, t=None):
        x = self.enc.encode(obs, P, t) if self.needs_history else self.enc(obs)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(x))
            return F.softmax(logits, -1).numpy()

    def __call__(self, obs, rng=None, P=None, t=None):
        p = self.probs(obs, P, t)
        if self.greedy:
            return p.argmax(axis=1)
        u = rng.random((len(p), 1))
        return (p.cumsum(axis=1) < u).sum(axis=1).clip(0, 2)


def _np_forward(W, s):
    """Forward de ACNet en NumPy (vistas de los pesos locales) para actuar sin construir grafo."""
    h = np.maximum(W[0] @ s + W[1], 0.0)
    h = np.maximum(W[2] @ h + W[3], 0.0)
    return W[4] @ h + W[5], float((W[6] @ h + W[7])[0])


def _worker(rank, gnet, opt, c, gen, enc, run_seed, counter, lock, queue):
    torch.set_num_threads(1)
    torch.manual_seed(run_seed * 1000 + rank)
    rng = np.random.default_rng(run_seed * 1000 + rank)
    lnet = ACNet(c["hidden"], n_in=enc.n_in)
    # vistas NumPy de los pesos locales: load_state_dict copia in-place, así que se mantienen al día
    W = [p.detach().numpy() for p in lnet.parameters()]
    g = c["gamma"]
    while True:
        with lock:
            ep = counter.value
            if ep >= c["episodes"]:
                break
            counter.value += 1
        env = make_env(train_series(gen, run_seed, ep), c["cost"])
        obs, _ = env.reset()
        s = enc.from_env(obs, env)
        lnet.load_state_dict(gnet.state_dict())
        G, R, k, done = 0.0, 0.0, 0, False
        while not done:
            S, A, rews = [], [], []
            for _ in range(c["n_steps"]):
                logits, _v = _np_forward(W, s)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                a = min(int((np.cumsum(p) < rng.random()).sum()), 2)
                obs2, r, term, trunc, info = env.step(a)
                S.append(s)
                A.append(a)
                rews.append(r * c["reward_scale"])
                G += g ** k * r
                R += r
                k += 1
                s = enc.from_env(obs2, env)
                done = term or trunc
                if done:
                    break
            ret = 0.0 if (done and not c["timeout_bootstrap"]) else _np_forward(W, s)[1]
            returns = np.zeros(len(rews), np.float32)
            for i in range(len(rews) - 1, -1, -1):
                ret = rews[i] + g * ret
                returns[i] = ret
            logits, vals = lnet(torch.from_numpy(np.asarray(S, np.float32)))
            logp = F.log_softmax(logits, -1)
            logp_a = logp.gather(1, torch.as_tensor(A)[:, None]).squeeze(1)
            ent = -(logp.exp() * logp).sum(1)
            adv = torch.from_numpy(returns) - vals
            loss = (-(logp_a * adv.detach()).mean() + c["value_coef"] * adv.pow(2).mean()
                    - c["entropy"] * ent.mean())
            lnet.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lnet.parameters(), c["grad_clip"])
            for lp, gp in zip(lnet.parameters(), gnet.parameters()):
                gp._grad = lp.grad.clone()
            opt.step()
            lnet.load_state_dict(gnet.state_dict())
        queue.put((ep, G, R, info["balance"], time.time()))


def train_a3c(gen, cfg, run_seed, evaluator=None, log=print):
    c = {**DEFAULTS, **cfg}
    torch.manual_seed(run_seed)
    enc = make_encoder(gen, c)
    gnet = ACNet(c["hidden"], n_in=enc.n_in)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=c["lr"])
    ctx = mp.get_context("spawn")
    counter, lock, queue = ctx.Value("i", 0), ctx.Lock(), ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=_worker, args=(i, gnet, opt, c, gen, enc, run_seed, counter, lock, queue))
             for i in range(c["workers"])]
    for p in procs:
        p.start()
    hist = dict(episode=[], G=[], sum_reward=[], balance=[], time=[])
    curve = dict(episode=[], val=[], time=[])
    best = (-np.inf, copy.deepcopy(gnet.state_dict()), 0)
    snap = ACNet(c["hidden"], n_in=enc.n_in)
    finished, next_eval = 0, c["eval_every"]

    def do_eval(n_done):
        nonlocal best
        snap.load_state_dict(gnet.state_dict())
        val = evaluator(ActorPolicy(snap, enc))
        curve["episode"].append(n_done)
        curve["val"].append(val)
        curve["time"].append(time.time() - t0)
        if val > best[0]:
            best = (val, copy.deepcopy(snap.state_dict()), n_done)

    import queue as _q
    while finished < c["episodes"]:
        try:
            ep, G, R, bal, tt = queue.get(timeout=5)
        except _q.Empty:
            if not any(p.is_alive() for p in procs):
                raise RuntimeError("Todos los workers de A3C han terminado inesperadamente")
            continue
        finished += 1
        for key, v in zip(hist, (ep, G, R, bal, tt - t0)):
            hist[key].append(v)
        if evaluator is not None and finished >= next_eval:
            do_eval(finished)
            next_eval += c["eval_every"]
    for p in procs:
        p.join()
    if evaluator is not None and (not curve["episode"] or curve["episode"][-1] != c["episodes"]):
        do_eval(c["episodes"])
    train_time = time.time() - t0
    return dict(net=gnet, encoder=enc, hist=hist, curve=curve, best_state=best[1], best_episode=best[2],
                best_val=best[0], final_state=copy.deepcopy(gnet.state_dict()), train_time=train_time, cfg=c)
