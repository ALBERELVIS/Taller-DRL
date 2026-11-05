"""Búsqueda aleatoria de hiperparámetros evaluada SOLO en validación.

Las mismas configuraciones (muestreadas con semilla fija) se prueban en los 4
generadores. Puntuación = media de la G sin descontar en validación durante el último
30% del entrenamiento (rendimiento estable, no el pico puntual).
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.launcher import job, run_all  # noqa: E402

GENS = ["ciclo", "gbm", "reversion", "regimenes"]

SPACES = {
    "dqn": dict(lr=[3e-4, 1e-3, 3e-3], gamma=[0.5, 0.9, 0.99], eps_decay=[0.95, 0.98, 0.99],
                target_update=[250, 1000]),
    "a3c": dict(lr=[1e-4, 3e-4, 1e-3], gamma=[0.5, 0.9, 0.99], entropy=[0.001, 0.01, 0.05],
                n_steps=[5, 20]),
    "qlearning": dict(alpha=[0.05, 0.1, 0.3], gamma=[0.5, 0.9, 0.99], eps_decay=[0.99, 0.995],
                      n_bins=[6, 10, 16]),
    "ppo": dict(lr=[3e-4, 1e-3], gamma=[0.5, 0.9, 0.99], entropy=[0.0, 0.01]),
}
N_CONF = {"dqn": 12, "a3c": 10, "qlearning": 12, "ppo": 12}
FIXED = {"dqn": dict(episodes=250, eval_every=10),
         "a3c": dict(episodes=1000, workers=4, eval_every=40),
         "qlearning": dict(episodes=1000, eval_every=25),
         "ppo": dict(episodes=1600, eval_every=32)}
SEEDS = {"dqn": [0, 1], "a3c": [0], "qlearning": [0, 1], "ppo": [0]}


def configs(algo):
    space = SPACES[algo]
    grid = [dict(zip(space, v)) for v in itertools.product(*space.values())]
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(grid), size=N_CONF[algo], replace=False)
    return [{**grid[i], **FIXED[algo]} for i in idx]


def jobs_for(algo):
    js = []
    for ci, cfg in enumerate(configs(algo)):
        for gen in GENS:
            for s in SEEDS[algo]:
                js.append(job(algo, gen, s, cfg, f"results/tune/{algo}/{gen}_c{ci}_s{s}.json"))
    return js


def score(curve):
    v = np.asarray(curve["val"])
    k = max(1, int(np.ceil(0.3 * len(v))))
    return float(v[-k:].mean())


def collect(algo):
    rows = []
    for ci, cfg in enumerate(configs(algo)):
        for gen in GENS:
            sc = []
            for s in SEEDS[algo]:
                p = f"results/tune/{algo}/{gen}_c{ci}_s{s}.json"
                if os.path.exists(p):
                    d = json.load(open(p))
                    sc.append(score(d["curve"]))
            if sc:
                rows.append(dict(algo=algo, gen=gen, conf=ci, score=float(np.mean(sc)),
                                 **{k: cfg[k] for k in SPACES[algo]}))
    return rows


if __name__ == "__main__":
    algos = sys.argv[1].split(",")
    par = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    js = [j for a in algos for j in jobs_for(a)]
    run_all(js, max_parallel=par)
