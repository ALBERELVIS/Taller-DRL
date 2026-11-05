"""Análisis de la ampliación (E1-E8): tablas y figuras.

python -m experiments.analysis_ext [rules|features|multi|cost|risk|algos|peers]  (sin argumento = todo)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_trading.env import GEN_LABELS  # noqa: E402
from rl_trading.evaluate import hier_bootstrap, holm  # noqa: E402

GENS = ["ciclo", "gbm", "reversion", "regimenes"]
SHORT = {"ciclo": "Tendencia+ciclo", "gbm": "GBM", "reversion": "Reversión", "regimenes": "Regímenes"}
ALGOS = {"qlearning": "Q-learning", "dqn": "DQN", "a3c": "A3C", "ppo": "PPO"}
FIG = "results/figures"
EXT = "results/ext"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(pattern))]


def per_series(runs, key="sum_reward", which="best"):
    """Matriz (semillas, series) de una métrica de test."""
    return np.array([r["test"][which][key] for r in runs])


def finals(algo, gen):
    return load(f"results/final/{algo}_{gen}_s*.json")



def seed_sd(X):
    """Desviación típica muestral (ddof=1) entre semillas, como en table_main."""
    m = np.atleast_2d(X).mean(1)
    return float(m.std(ddof=1)) if len(m) > 1 else 0.0

def compare(X, Y):
    """X, Y: (semillas, series). Diferencia media, Wilcoxon sobre la media por serie e IC jerárquico."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    d = X.mean(0) - Y.mean(0)
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = np.nan
    lo, hi = hier_bootstrap(X, Y, n_boot=2000)
    return dict(diff=float(d.mean()), p=p, ci_low=lo, ci_high=hi, win=float(np.mean(d > 0)) * 100)


def summ(X):
    return float(X.mean()), seed_sd(X)


# ----------------------------------------------------------------------------------
# E1: reglas óptimas frente a los agentes
# ----------------------------------------------------------------------------------
def rules_table():
    rows, comp = [], []
    for gen in GENS:
        d = json.load(open(f"results/rules/{gen}_c0.001.json"))
        best = d["best_family"]
        for fam, v in d["families"].items():
            s = v["test_summary"]
            rows.append(dict(gen=gen, regla=fam, elegida=fam == best, **{k: v["params"][k] for k in "abcd"},
                             fit=v["fit_score"], val=v["val"]["sum_reward"], sum_reward=s["sum_reward"],
                             log_wealth=s["log_wealth"], trades=s["trades"], exposure=s["exposure"]))
        R = {k: np.array([d["families"][best]["test"][k]]) for k in ("sum_reward", "log_wealth")}
        for algo, lab in ALGOS.items():
            runs = finals(algo, gen)
            if not runs:
                continue
            for key in ("sum_reward", "log_wealth"):
                comp.append(dict(gen=gen, agente=lab, vs=f"regla {best}", metric=key,
                                 **compare(per_series(runs, key), R[key])))
    df, cp = pd.DataFrame(rows), pd.DataFrame(comp)
    cp["p_holm"] = holm(cp.p.to_numpy())
    df.to_csv("results/table_rules.csv", index=False)
    cp.to_csv("results/table_rules_vs_rl.csv", index=False)
    return df, cp


# ----------------------------------------------------------------------------------
# E2: estado enriquecido
# ----------------------------------------------------------------------------------
FSETS = ["base", "ventana", "ma_vol", "ventana_ma_vol"]


def feature_runs(fs, algo, gen):
    if fs == "base":
        runs = finals(algo, gen)
        return runs if algo == "dqn" else runs[:3]
    return load(f"{EXT}/features_{fs}_{algo}_{gen}_s*.json")


def features_table():
    rows = []
    for algo in ("dqn", "a3c", "ppo"):
        for gen in GENS:
            base = feature_runs("base", algo, gen)
            for fs in FSETS:
                runs = feature_runs(fs, algo, gen)
                if not runs or not base:
                    continue
                X = per_series(runs)
                m, sd = summ(X)
                row = dict(algo=algo, gen=gen, features=fs, n=len(runs), sum_reward=m, sum_reward_sd=sd,
                           log_wealth=per_series(runs, "log_wealth").mean(),
                           trades=per_series(runs, "trades").mean())
                if fs != "base":
                    row.update({f"vs_base_{k}": v for k, v in compare(X, per_series(base)).items()})
                d = json.load(open(f"results/rules/{gen}_c0.001.json"))
                rule = np.array([d["families"][d["best_family"]]["test"]["sum_reward"]])
                row.update({f"vs_rule_{k}": v for k, v in compare(X, rule).items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    if "vs_base_p" in df:
        df["vs_base_p_holm"] = holm(df["vs_base_p"].to_numpy())
        df["vs_rule_p_holm"] = holm(df["vs_rule_p"].to_numpy())
    df.to_csv("results/table_features.csv", index=False)
    if len(df):
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
        for ax, gen in zip(axes, GENS):
            sub = df[(df.gen == gen) & (df.features != "base")]
            for i, algo in enumerate(("dqn", "a3c", "ppo")):
                s = sub[sub.algo == algo]
                x = np.arange(len(s)) + (i - 1) * 0.27
                ax.bar(x, s.vs_base_diff, 0.25, label=algo.upper(),
                       yerr=[s.vs_base_diff - s.vs_base_ci_low, s.vs_base_ci_high - s.vs_base_diff], capsize=2)
            ax.set_xticks(range(3), ["ventana", "ma_vol", "ambas"], fontsize=7)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(SHORT[gen])
        axes[0].set_ylabel("Δ Σr test frente a estado base")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{FIG}/features.png", dpi=150)
        plt.close(fig)
    return df


# ----------------------------------------------------------------------------------
# E3: agente multi-mercado
# ----------------------------------------------------------------------------------
def multi_table():
    rows = []
    for fs in ("base", "ma_vol"):
        runs = load(f"{EXT}/multi_{fs}_dqn_s*.json")
        if not runs:
            continue
        for gen in GENS:
            X = np.array([r["test"][f"best_{gen}"]["sum_reward"] for r in runs])
            spec = per_series(finals("dqn", gen))
            rows.append(dict(agente=f"multi {fs}", gen=gen, n=len(runs), sum_reward=X.mean(),
                             sum_reward_sd=seed_sd(X),
                             log_wealth=np.mean([r["test"][f"best_{gen}"]["log_wealth"] for r in runs]),
                             trades=np.mean([r["test"][f"best_{gen}"]["trades"] for r in runs]),
                             **{f"vs_especializado_{k}": v for k, v in compare(X, spec).items()}))
    for gen in GENS:
        spec = finals("dqn", gen)
        rows.append(dict(agente="DQN especializado", gen=gen, n=len(spec), sum_reward=per_series(spec).mean(),
                         sum_reward_sd=seed_sd(per_series(spec)),
                         log_wealth=per_series(spec, "log_wealth").mean(), trades=per_series(spec, "trades").mean()))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_multi.csv", index=False)
    return df


# ----------------------------------------------------------------------------------
# E4: comisión
# ----------------------------------------------------------------------------------
def inaction_width(pmap):
    """Anchura (en sigma) de la zona de no operar: ni compra sin posición ni vende con posición."""
    z = np.asarray(pmap["trend"])
    step = (z[-1] - z[0]) / (len(z) - 1)
    a0, a1 = np.asarray(pmap["act0"]), np.asarray(pmap["act1"])
    sig = (z[-1] - z[0]) / 6
    return float(((a0 != 2) & (a1 != 0)).sum() * step / sig)


def rule_width(p):
    if p["family"] in ("mom_h", "con_h"):
        return abs(p["a"] - p["b"])
    if p["family"] == "band_h":
        return (p["a"] - p["c"]) + (p["d"] - p["b"])
    return 0.0


def cost_table():
    rows = []
    for gen in ("ciclo", "reversion", "regimenes"):
        for c in (0.0, 0.0005, 0.001, 0.005):
            runs = finals("dqn", gen) if c == 0.001 else load(f"{EXT}/cost_{c}_dqn_{gen}_s*.json")
            rp = f"results/rules/{gen}_c{c}.json"
            row = dict(gen=gen, cost_pct=c * 100)
            if runs:
                X = per_series(runs)
                row.update(dqn_sum_reward=X.mean(), dqn_sum_reward_sd=seed_sd(X),
                           dqn_log_wealth=per_series(runs, "log_wealth").mean(),
                           dqn_trades=per_series(runs, "trades").mean(),
                           dqn_width=np.mean([inaction_width(r["policy_map"]) for r in runs]))
            if os.path.exists(rp):
                d = json.load(open(rp))
                b = d["families"][d["best_family"]]
                row.update(rule=d["best_family"], rule_sum_reward=b["test_summary"]["sum_reward"],
                           rule_trades=b["test_summary"]["trades"], rule_width=rule_width(b["params"]))
                if runs:
                    row.update({f"dqn_vs_rule_{k}": v for k, v in
                                compare(per_series(runs), np.array([b["test"]["sum_reward"]])).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/table_cost.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, gen in zip(axes, ("ciclo", "reversion", "regimenes")):
        s = df[df.gen == gen]
        if "dqn_trades" in s:
            ax.plot(s.cost_pct, s.dqn_trades, "o-", label="DQN: operaciones")
        if "rule_trades" in s:
            ax.plot(s.cost_pct, s.rule_trades, "s--", label="Regla óptima: operaciones")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("Comisión (%)")
        ax.set_title(SHORT[gen])
        ax2 = ax.twinx()
        if "dqn_width" in s:
            ax2.plot(s.cost_pct, s.dqn_width, "^:", color="tab:green", label="DQN: zona sin operar (σ)")
        ax2.set_ylabel("Anchura zona sin operar (σ)", fontsize=7)
        if gen == "ciclo":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6)
    axes[0].set_ylabel("Operaciones por episodio")
    fig.tight_layout()
    fig.savefig(f"{FIG}/cost.png", dpi=150)
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------------
# E5: recompensa sensible al riesgo
# ----------------------------------------------------------------------------------
MODES = {"pct": "Σr (enunciado)", "log": "log-riqueza", "mv2": "media-varianza λ=2", "mv10": "media-varianza λ=10"}


def risk_table():
    rows = []
    for gen in GENS:
        base = finals("dqn", gen)
        for mode in MODES:
            runs = base[:3] if mode == "pct" else load(f"{EXT}/risk_{mode}_dqn_{gen}_s*.json")
            if not runs:
                continue
            s = pd.DataFrame([r["test"]["best_summary"] for r in runs]).mean()
            row = dict(gen=gen, modo=mode, n=len(runs), sum_reward=s.sum_reward, log_wealth=s.log_wealth,
                       sharpe=s.sharpe, max_dd=s.max_dd, trades=s.trades, exposure=s.exposure,
                       ret_median=s.ret_median)
            if mode != "pct":
                row.update({f"lw_vs_pct_{k}": v for k, v in
                            compare(per_series(runs, "log_wealth"), per_series(base[:3], "log_wealth")).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/table_risk.csv", index=False)
    if len(df):
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
        for ax, gen in zip(axes, GENS):
            s = df[df.gen == gen]
            for _, r in s.iterrows():
                ax.scatter(r.max_dd, r.log_wealth, s=30)
                ax.annotate(r.modo, (r.max_dd, r.log_wealth), fontsize=7)
            ax.set_xlabel("Drawdown máx. medio (%)")
            ax.set_title(SHORT[gen])
        axes[0].set_ylabel("Log-riqueza media")
        fig.tight_layout()
        fig.savefig(f"{FIG}/risk_frontier.png", dpi=150)
        plt.close(fig)
    return df


# ----------------------------------------------------------------------------------
# E6: PPO y Dueling frente a los algoritmos principales
# ----------------------------------------------------------------------------------
def algos_table():
    rows = []
    for gen in GENS:
        ref = per_series(finals("dqn", gen))
        variants = {"Q-learning": finals("qlearning", gen), "DQN (Double)": finals("dqn", gen),
                    "Dueling DQN": load(f"{EXT}/dueling_dqn_{gen}_s*.json"), "A3C": finals("a3c", gen),
                    "PPO": finals("ppo", gen), "PPO 4800 ep": load(f"{EXT}/ppo_long_{gen}_s*.json")}
        for name, runs in variants.items():
            if not runs:
                continue
            X = per_series(runs)
            conv = [r["best_episode"] for r in runs]
            row = dict(gen=gen, agente=name, n=len(runs), sum_reward=X.mean(), sum_reward_sd=seed_sd(X),
                       log_wealth=per_series(runs, "log_wealth").mean(), trades=per_series(runs, "trades").mean(),
                       train_time=np.mean([r["train_time"] for r in runs]), best_episode=np.mean(conv))
            if name != "DQN (Double)":
                row.update({f"vs_dqn_{k}": v for k, v in compare(X, ref).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/table_algos.csv", index=False)
    return df


def ppo_explore_table():
    """Colapso de exploración de PPO en regímenes: exposición de la política de comportamiento durante el
    entrenamiento y resultado en test, según el coeficiente de entropía (ciclo como control)."""
    rows, curves = [], {}
    cases = [(gen, f"γ ajustado, entropía {e}", f"{EXT}/ppo_entropy_{e}_{gen}_s*.json")
             for gen, ents in (("regimenes", (0.0, 0.05, 0.2)), ("ciclo", (0.0,))) for e in ents]
    cases += [("regimenes", "γ 0.5, entropía 0.05", f"{EXT}/ppo_snr_g05_regimenes_s*.json"),
              ("regimenes", "γ 0.5, entropía 0.05, lote ×4, 9600 ep", f"{EXT}/ppo_snr_g05_big_regimenes_s*.json")]
    for gen, e, pattern in cases:
        runs = load(pattern)
        if not runs:
            continue
        X = per_series(runs)
        expo = np.array([r["hist"]["exposure"] for r in runs])
        ep = np.array(runs[0]["hist"]["episode"])
        curves[(gen, e)] = (ep, expo.mean(0))
        q = len(ep) // 10
        rows.append(dict(gen=gen, variante=e, n=len(runs), sum_reward=X.mean(), sum_reward_sd=seed_sd(X),
                         trades=per_series(runs, "trades").mean(),
                         exposure_test=np.mean([r["test"]["best_summary"]["exposure"] for r in runs]),
                         expo_train_first10=expo[:, :q].mean() * 100, expo_train_last10=expo[:, -q:].mean() * 100,
                         best_val=np.mean([r["best_val"] for r in runs]),
                         best_episode=np.mean([r["best_episode"] for r in runs])))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_ppo_explore.csv", index=False)
    if curves:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        for (gen, e), (ep, ex) in curves.items():
            k = 32
            sm = np.convolve(ex, np.ones(k) / k, mode="valid")
            ax.plot(ep[k - 1:], sm * 100, label=f"{SHORT[gen]}, {e}", ls="-" if gen == "regimenes" else "--")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Tiempo invertido de la\npolítica estocástica (%)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{FIG}/ppo_explore.png", dpi=150)
        plt.close(fig)
    return df


def onpolicy_table():
    """Misma tabla, distinto objetivo: Q-learning (off-policy) frente a Expected SARSA (on-policy)."""
    rows = []
    for gen, eps_list in (("regimenes", (0.02, 0.1, 0.3)), ("ciclo", (0.1,))):
        for e in eps_list:
            q = load(f"{EXT}/onpolicy_q_eps{e}_{gen}_s*.json")
            for tgt, lab in (("q", "Q-learning"), ("expected_sarsa", "Expected SARSA")):
                runs = load(f"{EXT}/onpolicy_{tgt}_eps{e}_{gen}_s*.json")
                if not runs:
                    continue
                X = per_series(runs)
                row = dict(gen=gen, eps_min=e, objetivo=lab, n=len(runs), sum_reward=X.mean(),
                           sum_reward_sd=seed_sd(X), trades=per_series(runs, "trades").mean(),
                           exposure=np.mean([r["test"]["best_summary"]["exposure"] for r in runs]),
                           best_val=np.mean([r["best_val"] for r in runs]))
                if tgt != "q" and len(q) == len(runs):
                    row.update({f"vs_q_{k}": v for k, v in compare(X, per_series(q)).items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/table_onpolicy.csv", index=False)
    return df


def features_long_table():
    rows = []
    for gen in ("reversion", "gbm"):
        base = finals("dqn", gen)[:3]
        for name, runs in (("base 400 ep", base),
                           ("ventana 400 ep", load(f"{EXT}/features_ventana_dqn_{gen}_s1[012].json")),
                           ("ventana 1200 ep, ε 0.99", load(f"{EXT}/featlong_ventana_dqn_{gen}_s*.json"))):
            if not runs:
                continue
            X = per_series(runs)
            row = dict(gen=gen, variante=name, n=len(runs), sum_reward=X.mean(), sum_reward_sd=seed_sd(X),
                       trades=per_series(runs, "trades").mean(), best_val=np.mean([r["best_val"] for r in runs]),
                       best_episode=np.mean([r["best_episode"] for r in runs]))
            if name != "base 400 ep":
                row.update({f"vs_base_{k}": v for k, v in compare(X, per_series(base)).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/table_features_long.csv", index=False)
    return df


# ----------------------------------------------------------------------------------
# E8: comparación interna con otros grupos (mismas 100 series de test)
# ----------------------------------------------------------------------------------
def peers_table():
    if not os.path.exists("results/peers.json"):
        return None
    peers = json.load(open("results/peers.json"))
    rows = []
    for gen in GENS:
        ours = {f"Nuestro {lab}": finals(a, gen) for a, lab in ALGOS.items()}
        best_ours = max((k for k in ours if ours[k]), key=lambda k: per_series(ours[k]).mean())
        Xo = per_series(ours[best_ours])
        for k, runs in ours.items():
            if runs:
                rows.append(dict(gen=gen, grupo="nosotros", agente=k, n=len(runs), sum_reward=per_series(runs).mean(),
                                 log_wealth=per_series(runs, "log_wealth").mean(),
                                 trades=per_series(runs, "trades").mean()))
        for key, runs in peers.items():
            grp, algo, variant, g = key.split("|")
            if g != gen:
                continue
            X = np.array([r["sum_reward"] for r in runs])
            rows.append(dict(gen=gen, grupo=grp, agente=f"{algo} ({variant})", n=len(runs), sum_reward=X.mean(),
                             log_wealth=np.mean([r["log_wealth"] for r in runs]),
                             trades=np.mean([r["trades"] for r in runs]),
                             **{f"nuestro_mejor_vs_{k}": v for k, v in compare(Xo, X).items()},
                             nuestro_mejor=best_ours))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_peers.csv", index=False)
    return df


def memo_figure():
    """Figura compacta de la ampliación para la memoria (3 paneles)."""
    cost = pd.read_csv("results/table_cost.csv")
    risk = pd.read_csv("results/table_risk.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    ax = axes[0]
    for gen, col in (("ciclo", "tab:blue"), ("reversion", "tab:orange")):
        s = cost[cost.gen == gen]
        ax.plot(s.cost_pct, s.dqn_trades, "o-", color=col, label=f"DQN {SHORT[gen]}")
        ax.plot(s.cost_pct, s.rule_trades, "s:", color=col, alpha=0.7, label=f"Regla {SHORT[gen]}")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([0, 0.05, 0.1, 0.5], ["0", "0.05", "0.1", "0.5"])
    ax.set_xlabel("Comisión (%)")
    ax.set_ylabel("Operaciones / episodio")
    ax.set_title("(a) Comisión: el agente opera menos", fontsize=8)
    ax.legend(fontsize=6)
    ax = axes[1]
    mk = {"pct": "o", "log": "s", "mv2": "^", "mv10": "D"}
    for gen, col in (("reversion", "tab:orange"), ("regimenes", "tab:green")):
        s = risk[risk.gen == gen]
        for _, r in s.iterrows():
            ax.scatter(r.max_dd, r.log_wealth, marker=mk[r.modo], color=col, s=28,
                       label=f"{r.modo}" if gen == "reversion" else None)
        ax.plot(s.max_dd, s.log_wealth, color=col, lw=0.6, alpha=0.6)
        ax.annotate(SHORT[gen], (s.max_dd.iloc[0], s.log_wealth.iloc[0]), fontsize=6, color=col,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Drawdown máximo medio (%)")
    ax.set_ylabel("Log-riqueza media")
    ax.set_title("(b) Recompensa con aversión al riesgo", fontsize=8)
    ax.legend(fontsize=6, title="recompensa", title_fontsize=6)
    ax = axes[2]
    for gen, e, pattern, ls in (("regimenes", "Regímenes, entropía 0", f"{EXT}/ppo_entropy_0.0_regimenes_s*.json", "-"),
                                ("regimenes", "Regímenes, entropía 0.2", f"{EXT}/ppo_entropy_0.2_regimenes_s*.json", "-"),
                                ("ciclo", "Tendencia+ciclo, entropía 0", f"{EXT}/ppo_entropy_0.0_ciclo_s*.json", "--")):
        runs = load(pattern)
        if not runs:
            continue
        ex = np.array([r["hist"]["exposure"] for r in runs]).mean(0)
        ep = np.array(runs[0]["hist"]["episode"])
        k = 32
        ax.plot(ep[k - 1:], np.convolve(ex, np.ones(k) / k, mode="valid") * 100, ls=ls, label=e)
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("% tiempo invertido\n(política estocástica)")
    ax.set_title("(c) PPO en regímenes: colapso a «siempre fuera»", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/memo_ext.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["rules", "features", "features_long", "multi", "cost", "risk", "algos", "ppo_explore",
                            "onpolicy", "peers", "memo"]
    fns = dict(rules=rules_table, features=features_table, features_long=features_long_table, multi=multi_table,
               cost=cost_table, risk=risk_table, algos=algos_table, ppo_explore=ppo_explore_table, peers=peers_table,
               onpolicy=onpolicy_table, memo=memo_figure)
    for w in what:
        out = fns[w]()
        if isinstance(out, tuple):
            for o in out:
                print(o.round(3).to_string())
        elif out is not None:
            print(out.round(3).to_string())
