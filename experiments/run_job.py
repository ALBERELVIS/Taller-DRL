"""Ejecuta UN entrenamiento (algoritmo, generador, semilla, configuración) y guarda JSON.

Uso: python -m experiments.run_job --algo dqn --gen ciclo --seed 0 --cfg '{"lr":1e-3}'
       --out results/x.json [--test] [--save-model results/models/x.pt]

gen='multi' entrena un único agente con los 4 generadores (extensión) y lo evalúa en cada uno.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trading.env import MULTI_GENS, make_set  # noqa: E402
from rl_trading.evaluate import simulate, metrics, summarize  # noqa: E402
from rl_trading.encoding import trend_sigma  # noqa: E402


def build_policy(algo, res, state):
    enc = res["encoder"]
    if algo == "qlearning":
        from rl_trading.qlearning import TabularPolicy
        return TabularPolicy(state, res["edges"], enc)
    if algo.startswith("dqn"):
        from rl_trading.dqn import QNet, GreedyPolicy
        net = QNet(res["cfg"]["hidden"], n_in=enc.n_in, dueling=res["cfg"].get("dueling", False))
        net.load_state_dict(state)
        return GreedyPolicy(net, enc)
    from rl_trading.a3c import ACNet, ActorPolicy
    net = ACNet(res["cfg"]["hidden"], n_in=enc.n_in)
    net.load_state_dict(state)
    return ActorPolicy(net, enc)


def policy_map(algo, pol, gen):
    """Q(s,a) o pi(a|s) sobre una rejilla de tendencia, para posición 0 y 1 (solo estado base)."""
    sig = trend_sigma(gen)
    grid = np.linspace(-3 * sig, 3 * sig, 121)
    out = {"trend": grid.tolist()}
    for pos in (0, 1):
        obs = np.stack([grid, np.full_like(grid, pos)], 1).astype(np.float32)
        vals = pol.probs(obs) if algo in ("a3c", "ppo") else pol.q_values(obs)
        out[f"pos{pos}"] = np.asarray(vals).tolist()
        out[f"act{pos}"] = np.asarray(vals).argmax(1).tolist()
    return out


def make_evaluator(gen, cost):
    if gen != "multi":
        val_set = make_set(gen, "val")

        def evaluator(pol):
            return float(metrics(simulate(val_set, pol, np.random.default_rng(0), cost=cost))["sum_reward"].mean())
        return evaluator
    # multi-mercado: media de los excesos sobre B&H estandarizados por mercado (si no, domina el ciclo)
    from rl_trading.baselines import buy_and_hold
    sets = {g: make_set(g, "val") for g in MULTI_GENS}
    ref = {}
    for g, s in sets.items():
        r = metrics(simulate(s, buy_and_hold, cost=cost))["sum_reward"]
        ref[g] = (r.mean(), r.std() + 1e-9)

    def evaluator(pol):
        out = []
        for g, s in sets.items():
            r = metrics(simulate(s, pol, np.random.default_rng(0), cost=cost))["sum_reward"].mean()
            out.append((r - ref[g][0]) / ref[g][1])
        return float(np.mean(out))
    return evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)  # dqn | a3c | qlearning | ppo
    ap.add_argument("--gen", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--save-model", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads(a.cfg)
    cost = cfg.get("cost", 0.001)
    evaluator = make_evaluator(a.gen, cost)

    if a.algo == "dqn":
        from rl_trading.dqn import train_dqn as train
    elif a.algo == "a3c":
        from rl_trading.a3c import train_a3c as train
    elif a.algo == "qlearning":
        from rl_trading.qlearning import train_qlearning as train
    elif a.algo == "ppo":
        from rl_trading.ppo import train_ppo as train
    else:
        raise ValueError(a.algo)

    t0 = time.time()
    res = train(a.gen, cfg, a.seed, evaluator=evaluator)
    enc = res["encoder"]
    out = dict(algo=a.algo, gen=a.gen, seed=a.seed, cfg=res["cfg"], curve=res["curve"],
               hist={k: list(map(float, v)) for k, v in res["hist"].items()},
               best_episode=res["best_episode"], best_val=res["best_val"],
               train_time=res["train_time"], wall=time.time() - t0, sigma=enc.sigma,
               encoder=enc.state())
    if a.test:
        out["test"] = {}
        test_gens = MULTI_GENS if a.gen == "multi" else (a.gen,)
        for which in ("best", "final"):
            pol = build_policy(a.algo, res, res[f"{which}_state"])
            for g in test_gens:
                key = which if a.gen != "multi" else f"{which}_{g}"
                m = metrics(simulate(make_set(g, "test"), pol, np.random.default_rng(1), cost=cost))
                out["test"][key] = {k: v.tolist() for k, v in m.items()}
                out["test"][key + "_summary"] = summarize(m)
            if which == "best" and not enc.needs_history and a.gen != "multi":
                out["policy_map"] = policy_map(a.algo, pol, a.gen)
        if a.save_model:
            os.makedirs(os.path.dirname(a.save_model), exist_ok=True)
            payload = dict(algo=a.algo, gen=a.gen, cfg=res["cfg"], sigma=enc.sigma, encoder=enc.state(),
                           best_state=res["best_state"])
            if a.algo == "qlearning":
                payload["edges"] = res["edges"]
            torch.save(payload, a.save_model)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
