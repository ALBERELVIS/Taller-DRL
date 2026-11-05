"""Genera memoria/memoria.html a partir de results/ y la convierte a PDF con Edge headless.

Orden: python -m experiments.extras_peers && python -m experiments.stats_report && python make_memoria.py
"""
import json
import os
import subprocess

import pandas as pd

from memoria_text import build_html
from rl_trading.encoding import trend_sigma

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "memoria")
os.makedirs(OUT, exist_ok=True)


def csv(name, **kw):
    return pd.read_csv(os.path.join(ROOT, "results", f"{name}.csv"), **kw)


def load_inputs():
    T = dict(main=csv("table_main"), contrasts=csv("table_all_contrasts"), verdicts=csv("table_verdicts"),
             tried=csv("table_tried"), oracle=csv("table_oracle"), power=csv("table_power"),
             friedman=csv("table_friedman"), pred=csv("table_predictability"), rules=csv("table_rules"),
             cost=csv("table_cost"), risk=csv("table_risk"), ablations=csv("table_ablations"),
             generalization=csv("table_generalization", index_col=0), sp500=csv("table_sp500"),
             sp500_null=csv("table_sp500_null"), ppo_explore=csv("table_ppo_explore"), features=csv("table_features"),
             algos=csv("table_algos"), tuning=csv("table_tuning"))
    T["best"] = json.load(open(os.path.join(ROOT, "results/best_configs.json")))
    T["timing"] = {(a, g): json.load(open(os.path.join(ROOT, f"results/timing/{a}_{g}_s10.json")))["train_time"]
                   for a in ("dqn", "qlearning", "a3c") for g in ("ciclo", "gbm", "reversion", "regimenes")}
    T["sigma"] = {g: trend_sigma(g) for g in ("ciclo", "gbm", "reversion", "regimenes")}
    return T


def main():
    html = build_html(load_inputs())
    path = os.path.join(OUT, "memoria.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf = os.path.join(OUT, "memoria.pdf")
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file:///" + path.replace("\\", "/")], check=True,
                   capture_output=True, timeout=120)
    print("PDF:", pdf)


if __name__ == "__main__":
    main()
