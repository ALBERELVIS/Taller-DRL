"""E1/E4: ajusta las reglas paramétricas (rl_trading/rules.py) y las evalúa en validación y test.

python -m experiments.rules_eval   -> results/rules/{gen}_c{cost}.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_trading.env import make_set  # noqa: E402
from rl_trading.evaluate import metrics, simulate, summarize  # noqa: E402
from rl_trading.rules import FAMILIES, fit_rule, fit_series  # noqa: E402

GENS = ("ciclo", "gbm", "reversion", "regimenes")
COSTS = (0.0, 0.0005, 0.001, 0.005)
COST_GENS = ("ciclo", "reversion", "regimenes")


def one(args):
    gen, cost = args
    out_path = f"results/rules/{gen}_c{cost}.json"
    if os.path.exists(out_path):
        return out_path
    fit = fit_series(gen)
    val, test = make_set(gen, "val"), make_set(gen, "test")
    res = {"gen": gen, "cost": cost, "families": {}}
    for fam in FAMILIES:
        rule, score = fit_rule(gen, fam, fit, cost=cost)
        mv = metrics(simulate(val, rule, cost=cost))
        mt = metrics(simulate(test, rule, cost=cost))
        res["families"][fam] = dict(params=rule.params(), fit_score=score, val=summarize(mv),
                                    test_summary=summarize(mt),
                                    test={k: v.tolist() for k, v in mt.items()})
    res["best_family"] = max(res["families"], key=lambda f: res["families"][f]["fit_score"])
    os.makedirs("results/rules", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f)
    return out_path


if __name__ == "__main__":
    tasks = [(g, 0.001) for g in GENS] + [(g, c) for g in COST_GENS for c in COSTS if c != 0.001]
    with Pool(min(16, len(tasks))) as pool:
        for p in pool.imap_unordered(one, tasks):
            print("ok", p, flush=True)
