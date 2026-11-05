"""Ejecución final con las mejores configuraciones (elegidas en validación) + ablaciones.

python -m experiments.main final      -> 5 semillas x 4 generadores x {DQN, A3C, Q-learning}
python -m experiments.main ablations  -> variantes DQN, sensibilidad a gamma, workers A3C
python -m experiments.main sp500      -> extensión con datos reales
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.launcher import job, run_all  # noqa: E402
from experiments.tune import GENS, collect, configs, SPACES  # noqa: E402

FINAL_SEEDS = [10, 11, 12, 13, 14]  # disjuntas de las usadas en el ajuste
FINAL_EPISODES = {"dqn": dict(episodes=400, eval_every=10),
                  "a3c": dict(episodes=1200, workers=4, eval_every=40),
                  "qlearning": dict(episodes=2000, eval_every=25),
                  "ppo": dict(episodes=2400, eval_every=32)}


def best_configs():
    best = {}
    for algo in ("dqn", "a3c", "qlearning", "ppo"):
        rows = collect(algo)
        cfgs = configs(algo)
        for gen in GENS:
            cand = [x for x in rows if x["gen"] == gen]
            if not cand:
                continue
            r = max(cand, key=lambda x: x["score"])
            best[(algo, gen)] = {**cfgs[r["conf"]], **FINAL_EPISODES[algo]}
    with open("results/best_configs.json", "w") as f:
        json.dump({f"{a}|{g}": c for (a, g), c in best.items()}, f, indent=1)
    return best


def final_jobs(best):
    js = []
    for (algo, gen), cfg in best.items():
        for s in FINAL_SEEDS:
            js.append(job(algo, gen, s, cfg, f"results/final/{algo}_{gen}_s{s}.json", test=True,
                          save_model=f"results/models/{algo}_{gen}_s{s}.pt"))
    return js


def ablation_jobs(best):
    js = []
    seeds = [10, 11]
    # 1) variantes de DQN
    variants = {"vanilla": dict(target_update=0, double=False),
                "target": dict(double=False),
                "double": dict(double=True)}
    for gen in GENS:
        for name, v in variants.items():
            cfg = {**best[("dqn", gen)], **v, "episodes": 250}
            if name != "vanilla" and cfg.get("target_update", 0) == 0:
                cfg["target_update"] = 500
            for s in seeds:
                js.append(job("dqn", gen, s, cfg, f"results/ablation/dqnvar_{name}_{gen}_s{s}.json", test=True))
    # 2) sensibilidad a gamma (DQN)
    for gen in GENS:
        for gm in (0.0, 0.5, 0.9, 0.99):
            cfg = {**best[("dqn", gen)], "gamma": gm, "episodes": 250}
            for s in seeds:
                js.append(job("dqn", gen, s, cfg, f"results/ablation/gamma_{gm}_{gen}_s{s}.json", test=True))
    return js


def iteration_jobs(best):
    """Iteración 2 sobre el punto débil (regímenes): mismos hiperparámetros, más episodios."""
    cfg = {**best[("dqn", "regimenes")], "episodes": 1000, "eval_every": 20}
    return [job("dqn", "regimenes", s, cfg, f"results/ablation/episodes_1000_regimenes_s{s}.json", test=True)
            for s in FINAL_SEEDS]


def iteration_a3c_jobs(best):
    """Iteración 2 de A3C en reversión: la curva seguía subiendo al final (mejor checkpoint = último)."""
    cfg = {**best[("a3c", "reversion")], "episodes": 3000, "eval_every": 60}
    return [job("a3c", "reversion", s, cfg, f"results/ablation/a3cepisodes_3000_reversion_s{s}.json", test=True)
            for s in FINAL_SEEDS]


A3C_REG_CONFS = {"A": dict(lr=3e-4, gamma=0.9, entropy=0.01, n_steps=20),
                 "B": dict(lr=3e-4, gamma=0.99, entropy=0.01, n_steps=20),
                 "C": dict(lr=1e-4, gamma=0.99, entropy=0.05, n_steps=20)}


def iteration_a3c_reg_jobs(best):
    """Iteración 2 de A3C en regímenes (convergía a 'siempre fuera'): más entropía/horizonte y 3000 episodios.
    La configuración se elige por validación (best_val); el test solo se mira para la elegida."""
    js = []
    for k, extra in A3C_REG_CONFS.items():
        cfg = {**best[("a3c", "regimenes")], **extra, "episodes": 3000, "eval_every": 60}
        js += [job("a3c", "regimenes", s, cfg, f"results/ablation/a3creg_{k}_regimenes_s{s}.json", test=True)
               for s in FINAL_SEEDS]
    return js


def worker_jobs(best):
    js = []
    for gen in ("ciclo", "reversion"):
        for w in (1, 2, 4, 8, 16):
            cfg = {**best[("a3c", gen)], "workers": w, "episodes": 800, "eval_every": 40}
            for s in (10,):
                js.append(job("a3c", gen, s, cfg, f"results/ablation/workers_{w}_{gen}_s{s}.json", test=True))
    return js


def sp500_jobs(best):
    js = []
    for algo in ("dqn", "a3c", "qlearning"):
        cfg = dict(best[(algo, "gbm")])
        if algo == "dqn":
            cfg["episodes"] = 300
        for s in (10, 11, 12):
            js.append(job(algo, "sp500", s, cfg, f"results/sp500/{algo}_s{s}.json", test=True))
    return js


def timing_jobs(best):
    """Una ejecución por algoritmo y mercado (semilla 10, configuración final) para medir tiempos sin carga."""
    return [job(algo, gen, 10, best[(algo, gen)], f"results/timing/{algo}_{gen}_s10.json")
            for gen in GENS for algo in ("dqn", "a3c", "qlearning") if (algo, gen) in best]


def terminal_jobs(best):
    """El entorno marca el último paso como terminated: DQN sin bootstrap en esa transición (como pide
    Gymnasium para un estado terminal) frente a tratarlo como límite de tiempo."""
    return [job("dqn", gen, s, {**best[("dqn", gen)], "timeout_bootstrap": False},
                f"{EXT}/terminal_dqn_{gen}_s{s}.json", test=True)
            for gen in GENS for s in FINAL_SEEDS[:3]]


# ----------------------------------------------------------------------------------
# Ampliación (E2-E6): mismos hiperparámetros elegidos en validación, un solo cambio cada vez
# ----------------------------------------------------------------------------------
EXT = "results/ext"
FEATURE_SETS = ("ventana", "ma_vol", "ventana_ma_vol")
MULTI_CFG = dict(lr=5e-4, gamma=0.5, eps_decay=0.995, target_update=1000, episodes=1600, eval_every=20)


def features_jobs(best, algo):
    seeds = FINAL_SEEDS if algo == "dqn" else FINAL_SEEDS[:3]
    return [job(algo, gen, s, {**best[(algo, gen)], "features": fs},
                f"{EXT}/features_{fs}_{algo}_{gen}_s{s}.json", test=True)
            for fs in FEATURE_SETS for gen in GENS for s in seeds]


def cost_jobs(best):
    return [job("dqn", gen, s, {**best[("dqn", gen)], "cost": c}, f"{EXT}/cost_{c}_dqn_{gen}_s{s}.json", test=True)
            for c in (0.0, 0.0005, 0.005) for gen in ("ciclo", "reversion", "regimenes") for s in FINAL_SEEDS[:3]]


RISK = {"log": dict(reward_mode="log"), "mv2": dict(reward_mode="mv", risk_lambda=2.0),
        "mv10": dict(reward_mode="mv", risk_lambda=10.0)}


def risk_jobs(best):
    return [job("dqn", gen, s, {**best[("dqn", gen)], **v}, f"{EXT}/risk_{k}_dqn_{gen}_s{s}.json", test=True)
            for k, v in RISK.items() for gen in GENS for s in FINAL_SEEDS[:3]]


def dueling_jobs(best):
    return [job("dqn", gen, s, {**best[("dqn", gen)], "dueling": True}, f"{EXT}/dueling_dqn_{gen}_s{s}.json",
                test=True) for gen in GENS for s in FINAL_SEEDS]


def multi_jobs():
    return [job("dqn", "multi", s, {**MULTI_CFG, "features": fs}, f"{EXT}/multi_{fs}_dqn_s{s}.json", test=True)
            for fs in ("base", "ma_vol") for s in FINAL_SEEDS]


def ppo_ext_jobs(best):
    """Colapso de exploración en regímenes (entropía) y entrenamiento largo donde la curva sigue subiendo."""
    js = [job("ppo", "regimenes", s, {**best[("ppo", "regimenes")], "entropy": e},
              f"{EXT}/ppo_entropy_{e}_regimenes_s{s}.json", test=True)
          for e in (0.0, 0.05, 0.2) for s in FINAL_SEEDS[:3]]
    js += [job("ppo", "ciclo", s, best[("ppo", "ciclo")], f"{EXT}/ppo_entropy_0.0_ciclo_s{s}.json", test=True)
           for s in FINAL_SEEDS[:2]]
    js += [job("ppo", gen, s, {**best[("ppo", gen)], "episodes": 4800}, f"{EXT}/ppo_long_{gen}_s{s}.json", test=True)
           for gen in ("ciclo", "reversion") for s in FINAL_SEEDS[:3]]
    return js


PPO_SNR = {"g05": dict(gamma=0.5, entropy=0.05),
           "g05_big": dict(gamma=0.5, entropy=0.05, n_envs=64, episodes=9600, eval_every=128)}


def ppo_snr_jobs(best):
    """Regímenes: ¿falla PPO por la varianza de la ventaja? Horizonte corto y lotes 4 veces mayores."""
    return [job("ppo", "regimenes", s, {**best[("ppo", "regimenes")], **v}, f"{EXT}/ppo_snr_{k}_regimenes_s{s}.json",
                test=True) for k, v in PPO_SNR.items() for s in FINAL_SEEDS[:3]]


def onpolicy_jobs(best):
    """On-policy frente a off-policy con la misma tabla: Expected SARSA frente a Q-learning, con exploración
    persistente (suelo de ε) en regímenes y, como control, en el ciclo."""
    js = []
    for gen, eps_list, seeds in (("regimenes", (0.02, 0.1, 0.3), FINAL_SEEDS[:3]), ("ciclo", (0.1,), FINAL_SEEDS[:2])):
        for tgt in ("q", "expected_sarsa"):
            for e in eps_list:
                js += [job("qlearning", gen, s, {**best[("qlearning", gen)], "target": tgt, "eps_min": e},
                           f"{EXT}/onpolicy_{tgt}_eps{e}_{gen}_s{s}.json", test=True) for s in seeds]
    return js


def dqn_feat_long_jobs(best):
    """Con 12 entradas, DQN no aprende en 400 episodios (curva de validación plana): mismo presupuesto que A3C."""
    return [job("dqn", gen, s, {**best[("dqn", gen)], "features": "ventana", "episodes": 1200, "eps_decay": 0.99,
                                "eval_every": 20}, f"{EXT}/featlong_ventana_dqn_{gen}_s{s}.json", test=True)
            for gen in ("reversion", "gbm") for s in FINAL_SEEDS[:3]]


if __name__ == "__main__":
    what = sys.argv[1]
    par = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    best = best_configs()
    algos = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dqn", "a3c", "qlearning"]
    best = {k: v for k, v in best.items() if k[0] in algos}
    if what == "final":
        js = final_jobs(best)
        # primero A3C (más lento y multiproceso), luego el resto
        js.sort(key=lambda j: j["algo"] != "a3c")
        run_all(js, max_parallel=par)
    elif what == "ablations":
        run_all(ablation_jobs(best), max_parallel=par)
    elif what == "iteration":
        run_all(iteration_jobs(best), max_parallel=par)
    elif what == "iteration_a3c":
        run_all(iteration_a3c_jobs(best), max_parallel=par)
    elif what == "iteration_a3c_reg":
        run_all(iteration_a3c_reg_jobs(best), max_parallel=par)
    elif what == "workers":
        run_all(worker_jobs(best), max_parallel=par)
    elif what == "sp500":
        run_all(sp500_jobs(best), max_parallel=par)
    elif what == "timing":
        run_all(timing_jobs(best), max_parallel=par)
    elif what == "terminal":
        run_all(terminal_jobs(best), max_parallel=par)
    elif what == "ext_dqn":
        run_all(multi_jobs() + features_jobs(best, "dqn") + cost_jobs(best) + dueling_jobs(best) + risk_jobs(best),
                max_parallel=par)
    elif what == "ext_a3c":
        run_all(features_jobs(best, "a3c"), max_parallel=par)
    elif what == "ext_ppo":
        run_all(ppo_ext_jobs(best), max_parallel=par)
    elif what == "ext_ppo_feat":
        run_all(features_jobs(best, "ppo"), max_parallel=par)
    elif what == "ext_onpolicy":
        run_all(onpolicy_jobs(best), max_parallel=par)
    elif what == "ext_cost":
        run_all(cost_jobs(best)[::-1], max_parallel=par)
    elif what == "ext_risk":
        run_all(risk_jobs(best)[::-1], max_parallel=par)
    elif what == "ext_ppo_snr":
        run_all(ppo_snr_jobs(best), max_parallel=par)
    elif what == "ext_dqn_feat_long":
        run_all(dqn_feat_long_jobs(best), max_parallel=par)
