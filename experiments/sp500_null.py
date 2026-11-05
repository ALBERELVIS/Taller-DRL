"""¿Es suerte el resultado en el S&P 500 2011? Contraste frente a estrategias aleatorias equiparables.

Para cada ejecución de un agente se generan N políticas aleatorias de Markov (entrar/salir con
probabilidades fijas) con la misma exposición media y el mismo número esperado de operaciones que el
agente, y se calcula p = P(rendimiento aleatorio >= rendimiento del agente) en la serie de test de 2011.
Uso: python -m experiments.sp500_null
"""
import glob
import json

import numpy as np
import pandas as pd

from rl_trading.env import make_set
from rl_trading.evaluate import simulate

N_NULL = 20_000
LABELS = {"dqn": "DQN (Double)", "a3c": "A3C", "qlearning": "Q-learning (tabular)"}


def markov_policy(p_in, p_out):
    def pol(obs, rng):
        pos = obs[:, 1] > 0.5
        u = rng.random(len(obs))
        enter = (~pos) & (u < p_in)
        stay = pos & (u >= p_out)
        return np.where(enter | stay, 2, 0)
    return pol


def main():
    P = np.asarray(make_set("sp500", "test")[0], dtype=float)
    rows = []
    for path in sorted(glob.glob("results/sp500/*.json")):
        d = json.load(open(path))
        s = d["test"]["best_summary"]
        T = len(P) - 6
        e = np.clip(s["exposure"] / 100, 1e-3, 1 - 1e-3)
        tau = s["trades"] / T
        p_in, p_out = min(tau / (2 * (1 - e)), 1.0), min(tau / (2 * e), 1.0)
        sim = simulate(np.repeat(P[None], N_NULL, axis=0), markov_policy(p_in, p_out),
                       np.random.default_rng(12345 + d["seed"]))
        ret = (sim["balances"][:, -1] / sim["balances"][:, 0] - 1) * 100
        rows.append(dict(agente=LABELS[d["algo"]], seed=d["seed"], ret=s["ret_mean"], trades=s["trades"],
                         exposure=s["exposure"], null_ret_median=float(np.median(ret)),
                         null_ret_p95=float(np.percentile(ret, 95)),
                         null_trades=float(np.mean((np.diff(sim["positions"], axis=1) != 0).sum(1))),
                         null_exposure=float(sim["positions"].mean() * 100),
                         p_value=float((ret >= s["ret_mean"]).mean())))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_sp500_null.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
