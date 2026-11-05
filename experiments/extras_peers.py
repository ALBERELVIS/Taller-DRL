"""Ideas tomadas de otros grupos, recalculadas con nuestro protocolo (sin reentrenar nada).

1. Oráculo con información oculta (techo de cada mercado) -> % del margen capturado.
2. Mejor punto (parada temprana por validación) frente a modelo final.
3. Ensemble de las 5 semillas (media de Q / pi / tablas) frente a cada semilla.

python -m experiments.extras_peers      -> results/peer_ideas.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_job import build_policy  # noqa: E402
from rl_trading.baselines import HiddenOracle  # noqa: E402
from rl_trading.encoding import Encoder  # noqa: E402
from rl_trading.env import N_TEST, TEST_SEED0, make_set  # noqa: E402
from rl_trading.evaluate import metrics, simulate  # noqa: E402

GENS = ["ciclo", "gbm", "reversion", "regimenes"]
ALGOS = ("qlearning", "dqn", "a3c", "ppo")
KEYS = ("sum_reward", "log_wealth", "trades", "exposure", "ret_pct")


def pick(m):
    return {k: np.asarray(m[k]).tolist() for k in KEYS}


def load_policies(algo, gen):
    pols, seeds = [], []
    for p in sorted(glob.glob(f"results/models/{algo}_{gen}_s*.pt")):
        d = torch.load(p, weights_only=False)
        res = dict(cfg=d["cfg"], encoder=Encoder(sigma=d["sigma"]), edges=d.get("edges"))
        pols.append(build_policy(algo, res, d["best_state"]))
        seeds.append(int(p.rsplit("_s", 1)[1].split(".")[0]))
    return pols, seeds


class Ensemble:
    """Política greedy sobre la media de los valores (Q o tabla) o de las probabilidades (actor)."""

    def __init__(self, pols, actor):
        self.pols, self.actor = pols, actor

    def __call__(self, obs, rng=None):
        f = (lambda p: p.probs(obs)) if self.actor else (lambda p: p.q_values(obs))
        return np.mean([f(p) for p in self.pols], axis=0).argmax(axis=1)


def main():
    out = {"oracle": {}, "ensemble": {}, "best_final": {}}
    for gen in GENS:
        ts = make_set(gen, "test")
        orc = HiddenOracle(gen, seeds=[TEST_SEED0 + i for i in range(N_TEST)])
        m = metrics(simulate(ts, orc))
        out["oracle"][gen] = dict(pick(m), band=getattr(orc, "band", None))
        print(gen, "oráculo Σr", m["sum_reward"].mean(), getattr(orc, "band", ""))
        for algo in ALGOS:
            pols, seeds = load_policies(algo, gen)
            if not pols:
                continue
            singles = [pick(metrics(simulate(ts, p))) for p in pols]
            ens = pick(metrics(simulate(ts, Ensemble(pols, actor=algo in ("a3c", "ppo")))))
            out["ensemble"][f"{algo}|{gen}"] = dict(seeds=seeds, singles=singles, ensemble=ens)
            print(gen, algo, "semillas", np.mean([s["sum_reward"] for s in singles]), "ensemble",
                  np.mean(ens["sum_reward"]))
            runs = [json.load(open(p)) for p in sorted(glob.glob(f"results/final/{algo}_{gen}_s*.json"))]
            if runs:
                out["best_final"][f"{algo}|{gen}"] = {
                    w: {k: [r["test"][w][k] for r in runs] for k in KEYS} for w in ("best", "final")}
    with open("results/peer_ideas.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
