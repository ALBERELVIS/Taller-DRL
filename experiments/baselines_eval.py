"""Evalúa las políticas de referencia en validación y test (mismas series que los agentes)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_trading.baselines import BASELINES  # noqa: E402
from rl_trading.env import make_set  # noqa: E402
from rl_trading.evaluate import simulate, metrics, summarize  # noqa: E402

GENS = ["ciclo", "gbm", "reversion", "regimenes", "sp500"]


def main():
    out = {}
    for gen in GENS:
        out[gen] = {}
        for name, make in BASELINES.items():
            if gen == "sp500" and name == "Oráculo miope":
                continue
            pol = make(gen)
            res = {}
            for kind in ("val", "test"):
                if name == "Aleatorio":
                    # el agente aleatorio se promedia sobre 20 realizaciones por serie
                    ms = [metrics(simulate(make_set(gen, kind), pol, np.random.default_rng(s))) for s in range(20)]
                    m = {k: np.mean([x[k] for x in ms], axis=0) for k in ms[0]}
                else:
                    m = metrics(simulate(make_set(gen, kind), pol, np.random.default_rng(1)))
                res[kind] = {k: v.tolist() for k, v in m.items()}
                res[kind + "_summary"] = summarize(m)
            out[gen][name] = res
            print(gen, name, {k: round(v, 3) for k, v in res["test_summary"].items()})
    os.makedirs("results", exist_ok=True)
    with open("results/baselines.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
