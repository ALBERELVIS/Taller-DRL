"""Evaluación estadística global de todo el trabajo con una única regla de decisión.

Recalcula TODOS los contrastes (mínimos, balance frente a referencias y cada experimento probado)
sobre la misma métrica (Σr por serie de test, la recompensa del enunciado), con:
  - Wilcoxon pareado sobre la media por serie + IC95% bootstrap jerárquico (semillas × series);
  - tamaños del efecto: rango-biserial, % de series ganadas y cociente geométrico de capital;
  - equivalencia (TOST, IC90% dentro de ±δ);
  - familia CONFIRMATORIA (mínimos del enunciado, Holm) y EXPLORATORIA (resto, Benjamini-Hochberg).
Además: Friedman + Nemenyi por mercado, potencia/tamaño muestral, IQM y P(mejora), % del techo.

python -m experiments.stats_report   (requiere results/peer_ideas.json: python -m experiments.extras_peers)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_trading.stats import (bh, contrast, friedman_nemenyi, holm, iqm_ci, mde, prob_improvement,  # noqa: E402
                              variance_components, verdict)

GENS = ["ciclo", "gbm", "reversion", "regimenes"]
SHORT = {"ciclo": "Tendencia+ciclo", "gbm": "GBM", "reversion": "Reversión", "regimenes": "Regímenes"}
EXT = "results/ext"
FIG = "results/figures"
BASE = json.load(open("results/baselines.json"))
PEER = json.load(open("results/peer_ideas.json"))
KEYS = ("sum_reward", "log_wealth")
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------- fuentes de datos (semillas × series)
def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(pattern))]


def from_runs(runs, which="best", sub=None):
    if not runs:
        return None
    k = which if sub is None else f"{which}_{sub}"
    return {key: np.array([r["test"][k][key] for r in runs]) for key in KEYS}


def finals(algo, gen):
    return load(f"results/final/{algo}_{gen}_s*.json")


def base_agent(gen, name):
    if name == "Siempre fuera":
        return {k: np.zeros((1, 100)) for k in KEYS}
    return {k: np.atleast_2d(BASE[gen][name]["test"][k]) for k in KEYS}


def rule(gen, cost=0.001):
    d = json.load(open(f"results/rules/{gen}_c{cost}.json"))
    return {k: np.atleast_2d(d["families"][d["best_family"]]["test"][k]) for k in KEYS}


def oracle(gen):
    return {k: np.atleast_2d(PEER["oracle"][gen][k]) for k in KEYS}


ALG = {"Q-learning": "qlearning", "DQN": "dqn", "A3C": "a3c", "PPO": "ppo"}
BASES = ("Aleatorio", "Buy & Hold", "Siempre fuera", "Momentum", "Contraria", "Oráculo miope")


def agent(name, gen):
    if name in ALG:
        return from_runs(finals(ALG[name], gen))
    if name in BASES:
        return base_agent(gen, name)
    if name == "Mejor regla":
        return rule(gen)
    if name == "Oráculo oculto":
        return oracle(gen)
    if name == "Dueling DQN":
        return from_runs(load(f"{EXT}/dueling_dqn_{gen}_s*.json"))
    if name.startswith("feat:"):  # feat:<algo>:<conjunto>
        _, algo, fs = name.split(":")
        return from_runs(load(f"{EXT}/features_{fs}_{algo}_{gen}_s*.json"))
    raise KeyError(name)


def delta(gen):
    """Diferencia mínima relevante: 0.02 de Σr (2 puntos de rendimiento acumulado) o el 2% de la Σr de
    B&H si es mayor (en el ciclo la escala de Σr es 60 veces mayor)."""
    return max(0.02, 0.02 * abs(base_agent(gen, "Buy & Hold")["sum_reward"].mean()))


# ---------------------------------------------------------------- registro de contrastes
ROWS = []


def add(family, block, exp, gen, A, B, a_lab, b_lab, metric="sum_reward", ref_gen=None):
    if A is None or B is None:
        return
    d = delta(ref_gen or gen)
    c = contrast(A[metric], B[metric], delta=d, LX=A["log_wealth"], LY=B["log_wealth"])
    ROWS.append(dict(familia=family, bloque=block, exp=exp, gen=gen, a=a_lab, b=b_lab, metrica=metric,
                     media_a=float(A[metric].mean()), media_b=float(B[metric].mean()), **c))


def minimums():
    for gen in GENS:
        D, A3, Q = agent("DQN", gen), agent("A3C", gen), agent("Q-learning", gen)
        for lab, X in (("DQN", D), ("A3C", A3)):
            for ref in ("Aleatorio", "Buy & Hold"):
                add("confirmatoria", "balance", f"{lab} vs {ref}", gen, X, base_agent(gen, ref), lab, ref)
        add("confirmatoria", "balance", "DQN vs Q-learning", gen, D, Q, "DQN", "Q-learning")


def balance():
    for gen in GENS:
        D, A3, Q, P = (agent(n, gen) for n in ("DQN", "A3C", "Q-learning", "PPO"))
        R, O = agent("Mejor regla", gen), agent("Oráculo oculto", gen)
        add("exploratoria", "balance", "A3C vs DQN", gen, A3, D, "A3C", "DQN")
        add("exploratoria", "balance", "A3C vs Q-learning", gen, A3, Q, "A3C", "Q-learning")
        for ref in ("Aleatorio", "Buy & Hold", "Oráculo miope", "Siempre fuera"):
            add("exploratoria", "balance", f"Q-learning vs {ref}", gen, Q, base_agent(gen, ref), "Q-learning", ref)
        for ref in ("Oráculo miope", "Siempre fuera"):
            for lab, X in (("DQN", D), ("A3C", A3)):
                add("exploratoria", "balance", f"{lab} vs {ref}", gen, X, base_agent(gen, ref), lab, ref)
        for lab, X in (("Q-learning", Q), ("DQN", D), ("A3C", A3), ("PPO", P),
                       ("PPO+ventana+vol./MA", agent("feat:ppo:ventana_ma_vol", gen))):
            add("exploratoria", "balance", f"{lab} vs Mejor regla", gen, X, R, lab, "Mejor regla")
            add("exploratoria", "balance", f"{lab} vs Oráculo oculto", gen, X, O, lab, "Oráculo oculto")


def tried():
    pf = PEER["best_final"]
    ens = PEER["ensemble"]
    for gen in GENS:
        D = agent("DQN", gen)
        # ablaciones DQN (2 semillas): modelo final, que es donde se nota la inestabilidad
        V = {v: from_runs(load(f"results/ablation/dqnvar_{v}_{gen}_s*.json"), "final") for v in
             ("vanilla", "target", "double")}
        add("exploratoria", "probado", "Quitar la red objetivo (final)", gen, V["vanilla"], V["double"], "vanilla",
            "Double")
        add("exploratoria", "probado", "Double frente a solo red objetivo (final)", gen, V["double"], V["target"],
            "Double", "red objetivo")
        G = {g: from_runs(load(f"results/ablation/gamma_{g}_{gen}_s*.json")) for g in ("0.0", "0.5", "0.9", "0.99")}
        for g in ("0.0", "0.9", "0.99"):
            add("exploratoria", "probado", f"γ={g} (frente a 0.5)", gen, G[g], G["0.5"], f"γ={g}", "γ=0.5")
        add("exploratoria", "probado", "Dueling DQN", gen, agent("Dueling DQN", gen), D, "Dueling", "DQN")
        term = load(f"{EXT}/terminal_dqn_{gen}_s*.json")
        if term:
            add("exploratoria", "probado", "Último paso terminal (sin bootstrap)", gen, from_runs(term), D,
                "DQN terminal", "DQN límite de tiempo")
        add("exploratoria", "probado", "PPO frente a DQN", gen, agent("PPO", gen), D, "PPO", "DQN")
        for algo, lab in (("qlearning", "Q-learning"), ("dqn", "DQN"), ("a3c", "A3C"), ("ppo", "PPO")):
            k = f"{algo}|{gen}"
            if k in pf:
                B = {m: np.array(pf[k]["best"][m]) for m in KEYS}
                F = {m: np.array(pf[k]["final"][m]) for m in KEYS}
                add("exploratoria", "probado", f"Parada temprana {lab}", gen, B, F, "mejor punto", "final")
            if k in ens:
                E = {m: np.atleast_2d(ens[k]["ensemble"][m]) for m in KEYS}
                S = {m: np.array([s[m] for s in ens[k]["singles"]]) for m in KEYS}
                add("exploratoria", "probado", f"Ensemble 5 semillas {lab}", gen, E, S, "ensemble", "cada semilla")
        for algo, lab in (("dqn", "DQN"), ("a3c", "A3C"), ("ppo", "PPO")):
            base = from_runs(finals(algo, gen) if algo == "dqn" else finals(algo, gen)[:3])
            for fs, fl in (("ventana", "ventana"), ("ma_vol", "vol./MA"), ("ventana_ma_vol", "ventana+vol./MA")):
                add("exploratoria", "probado", f"Estado + {fl} ({lab})", gen, agent(f"feat:{algo}:{fs}", gen), base,
                    f"{lab}+{fl}", f"{lab} base")
        for fs, fl in (("base", "estado base"), ("ma_vol", "vol./MA")):
            runs = load(f"{EXT}/multi_{fs}_dqn_s*.json")
            add("exploratoria", "probado", f"Agente único 4 mercados ({fl})", gen, from_runs(runs, sub=gen), D,
                "único", "especializado")
        base3 = from_runs(finals("dqn", gen)[:3])
        for mode, ml in (("log", "log-riqueza"), ("mv2", "media-varianza λ=2"), ("mv10", "media-varianza λ=10")):
            add("exploratoria", "probado", f"Recompensa {ml}", gen,
                from_runs(load(f"{EXT}/risk_{mode}_dqn_{gen}_s*.json")), base3, ml, "Σr", metric="log_wealth")
    # iteraciones de mejora (un mercado cada una)
    add("exploratoria", "probado", "Más episodios", "regimenes",
        from_runs(load("results/ablation/episodes_1000_regimenes_s*.json")), agent("DQN", "regimenes"),
        "DQN 1000 ep", "DQN 400 ep")
    add("exploratoria", "probado", "Más episodios", "reversion",
        from_runs(load("results/ablation/a3cepisodes_3000_reversion_s*.json")), agent("A3C", "reversion"),
        "A3C 3000 ep", "A3C 1200 ep")
    add("exploratoria", "probado", "Más episodios", "ciclo", from_runs(load(f"{EXT}/ppo_long_ciclo_s*.json")),
        agent("PPO", "ciclo"), "PPO 4800 ep", "PPO 2400 ep")
    add("exploratoria", "probado", "PPO 4800 ep frente a la mejor regla", "ciclo",
        from_runs(load(f"{EXT}/ppo_long_ciclo_s*.json")), rule("ciclo"), "PPO 4800 ep", "Mejor regla")
    reg = pd.read_csv("results/table_a3c_regimenes_iter.csv")
    chosen = reg[reg.elegida_por_validacion].name.iloc[0]
    add("exploratoria", "probado", "A3C regímenes: 3000 ep y config. elegida en validación", "regimenes",
        from_runs(load(f"results/ablation/{chosen}_s*.json")), agent("A3C", "regimenes"), "A3C iter.", "A3C")
    for gen in ("reversion", "gbm"):
        add("exploratoria", "probado", "DQN + ventana con 3× episodios", gen,
            from_runs(load(f"{EXT}/featlong_ventana_dqn_{gen}_s*.json")), from_runs(finals("dqn", gen)[:3]),
            "ventana 1200 ep", "DQN base")
    for gen, eps in (("regimenes", 0.02), ("regimenes", 0.1), ("regimenes", 0.3), ("ciclo", 0.1)):
        add("exploratoria", "probado", f"Expected SARSA frente a Q-learning (ε={eps})", gen,
            from_runs(load(f"{EXT}/onpolicy_expected_sarsa_eps{eps}_{gen}_s*.json")),
            from_runs(load(f"{EXT}/onpolicy_q_eps{eps}_{gen}_s*.json")), "Expected SARSA", "Q-learning")
    for gen in ("regimenes",):
        for e in ("0.0", "0.05", "0.2"):
            add("exploratoria", "probado", f"PPO entropía {e} (regímenes)", gen,
                from_runs(load(f"{EXT}/ppo_entropy_{e}_{gen}_s*.json")), base_agent(gen, "Siempre fuera"),
                f"PPO β={e}", "Siempre fuera")
    for gen in ("ciclo", "reversion", "regimenes"):
        for c in (0.0, 0.0005, 0.005):
            runs = load(f"{EXT}/cost_{c}_dqn_{gen}_s*.json")
            add("exploratoria", "probado", f"Comisión {c * 100:g}%: DQN frente a regla re-ajustada", gen,
                from_runs(runs), rule(gen, c), "DQN", "regla")
    # S&P 500: cada ejecución frente a 20 000 estrategias aleatorias con igual exposición y nº de operaciones
    sp = pd.read_csv("results/table_sp500_null.csv")
    for _, r in sp.iterrows():
        ROWS.append(dict(familia="exploratoria", bloque="probado", exp="S&P 500 2011 frente a azar equiparable",
                         gen="sp500", a=f"{r.agente} s{r.seed}", b="nulo aleatorio", metrica="ret_pct",
                         media_a=r.ret, media_b=r.null_ret_median, diff=r.ret - r.null_ret_median, ci_low=np.nan,
                         ci_high=np.nan, p=r.p_value))


def adjust(df):
    df["p_adj"] = np.nan
    c = df.familia == "confirmatoria"
    df.loc[c, "p_adj"] = holm(df.loc[c, "p"].to_numpy())
    df.loc[~c, "p_adj"] = bh(df.loc[~c, "p"].to_numpy())

    def v(r):
        if np.isnan(r.ci_low):
            return "gana" if r.p_adj < 0.05 else "no concluyente"
        return verdict(r)
    df["veredicto"] = df.apply(v, axis=1)
    return df


# ---------------------------------------------------------------- test global, potencia, rliable, techo
FRIED = ["Aleatorio", "Buy & Hold", "Siempre fuera", "Momentum", "Contraria", "Oráculo miope", "Mejor regla",
         "Q-learning", "DQN", "A3C", "PPO"]


def friedman():
    rows, res = [], {}
    for gen in GENS:
        M = np.stack([agent(n, gen)["sum_reward"].mean(0) for n in FRIED], axis=1)
        f = friedman_nemenyi(M)
        res[gen] = f
        for n, rk in zip(FRIED, f["avg_rank"]):
            rows.append(dict(gen=gen, agente=n, rango_medio=rk, chi2=f["chi2"], p=f["p"], cd=f["cd"], n=f["n"]))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_friedman.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.9))
    for ax, gen in zip(axes, GENS):
        f = res[gen]
        order = np.argsort(f["avg_rank"])
        y = np.arange(len(FRIED))
        cols = ["tab:green" if FRIED[i] in ALG else "tab:gray" for i in order]
        ax.scatter(f["avg_rank"][order], y, c=cols, s=10, zorder=3)
        best = f["avg_rank"][order[0]]
        ax.axvspan(best, best + f["cd"], color="tab:blue", alpha=0.12, lw=0)
        ax.set_yticks(y, [FRIED[i] for i in order], fontsize=5)
        ax.invert_yaxis()
        ax.set_xlabel("Rango medio (1 = mejor)", fontsize=5.5)
        ax.tick_params(axis="x", labelsize=5)
        ax.set_title(f"{SHORT[gen]} (CD={f['cd']:.2f})", fontsize=6.5)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(f"{FIG}/memo_cd.png", dpi=300)
    plt.close(fig)
    return df


def power():
    rows = []
    for gen in GENS:
        for a, b in (("DQN", "Q-learning"), ("A3C", "DQN"), ("DQN", "Buy & Hold")):
            A, B = agent(a, gen)["sum_reward"], agent(b, gen)["sum_reward"]
            sdx, sdy, sdn = variance_components(A, B)
            obs_se = contrast(A, B)["se"]
            for S in (3, 5, 10, 20):
                for N in (40, 100, 400):
                    m, se = mde(sdx, sdy, sdn, S, S if len(B) > 1 else 1, N)
                    seed_share = (sdx ** 2 / S + (sdy ** 2 / S if len(B) > 1 else 0)) / se ** 2
                    rows.append(dict(gen=gen, a=a, b=b, S=S, N=N, mde=m, se=se, seed_share=seed_share * 100,
                                     sd_seed_a=sdx, sd_seed_b=sdy, sd_series=sdn, se_bootstrap=obs_se,
                                     diff=float(A.mean() - B.mean()), delta=delta(gen)))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_power.csv", index=False)
    return df


def rliable():
    rows = []
    for gen in GENS:
        X = {n: agent(n, gen)["sum_reward"] for n in ALG}
        for n, M in X.items():
            m, lo, hi = iqm_ci(M)
            rows.append(dict(gen=gen, agente=n, iqm=m, iqm_low=lo, iqm_high=hi, media=float(M.mean())))
        for a, b in (("DQN", "Q-learning"), ("A3C", "DQN"), ("PPO", "DQN"), ("PPO", "A3C")):
            rows.append(dict(gen=gen, agente=f"P({a} > {b})", prob=prob_improvement(X[a], X[b])))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_rliable.csv", index=False)
    return df


def ceiling():
    rows = []
    for gen in GENS:
        o = agent("Oráculo oculto", gen)["sum_reward"].mean()
        bhv = agent("Buy & Hold", gen)["sum_reward"].mean()
        triv_name, triv = ("Buy & Hold", bhv) if bhv > 0 else ("Siempre fuera", 0.0)
        span = o - triv
        for n in ("Mejor regla", "Q-learning", "DQN", "A3C", "PPO", "feat:dqn:ma_vol", "feat:ppo:ventana_ma_vol"):
            x = agent(n, gen)["sum_reward"].mean()
            rows.append(dict(gen=gen, agente=n, sum_reward=x, oraculo=o, trivial=triv_name, trivial_sr=triv,
                             margen=span, captura=(x - triv) / span * 100 if span > 0.01 else np.nan,
                             band=str(PEER["oracle"][gen].get("band"))))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_oracle.csv", index=False)
    return df


# ---------------------------------------------------------------- tablas para la memoria
VERDICT_ROWS = ["DQN vs Aleatorio", "A3C vs Aleatorio", "DQN vs Buy & Hold", "A3C vs Buy & Hold",
                "DQN vs Q-learning", "A3C vs DQN", "Q-learning vs Buy & Hold", "DQN vs Mejor regla",
                "PPO vs Mejor regla", "PPO+ventana+vol./MA vs Mejor regla", "DQN vs Oráculo oculto"]
# fila de la tabla de la memoria -> contrastes de los que salen sus celdas (el primero que tenga ese mercado)
TRIED_ROWS = {
    "Quitar la red objetivo (final)": ["Quitar la red objetivo (final)"],
    "Double frente a solo red objetivo (final)": ["Double frente a solo red objetivo (final)"],
    "Dueling DQN": ["Dueling DQN"],
    "Último paso terminal (sin bootstrap)": ["Último paso terminal (sin bootstrap)"],
    "γ=0.0 (frente a 0.5)": ["γ=0.0 (frente a 0.5)"],
    "γ=0.9 (frente a 0.5)": ["γ=0.9 (frente a 0.5)"],
    "γ=0.99 (frente a 0.5)": ["γ=0.99 (frente a 0.5)"],
    "Más episodios": ["Más episodios"],
    "Parada temprana DQN": ["Parada temprana DQN"],
    "Parada temprana A3C": ["Parada temprana A3C"],
    "Ensemble 5 semillas DQN": ["Ensemble 5 semillas DQN"],
    "Ensemble 5 semillas A3C": ["Ensemble 5 semillas A3C"],
    "PPO frente a DQN": ["PPO frente a DQN"],
    "Estado + vol./MA (DQN)": ["Estado + vol./MA (DQN)"],
    "Estado + vol./MA (A3C)": ["Estado + vol./MA (A3C)"],
    "Estado + vol./MA (PPO)": ["Estado + vol./MA (PPO)"],
    "Estado + ventana+vol./MA (PPO)": ["Estado + ventana+vol./MA (PPO)"],
    "Estado + ventana (DQN)": ["Estado + ventana (DQN)"],
    "Agente único 4 mercados (vol./MA)": ["Agente único 4 mercados (vol./MA)"],
    "Expected SARSA frente a Q-learning": ["Expected SARSA frente a Q-learning (ε=0.02)",
                                           "Expected SARSA frente a Q-learning (ε=0.1)"],
    "Recompensa media-varianza λ=2": ["Recompensa media-varianza λ=2"],
    "Recompensa media-varianza λ=10": ["Recompensa media-varianza λ=10"],
    "Recompensa log-riqueza": ["Recompensa log-riqueza"],
    "Comisión 0%: DQN frente a regla re-ajustada": ["Comisión 0%: DQN frente a regla re-ajustada"],
    "Comisión 0.5%: DQN frente a regla re-ajustada": ["Comisión 0.5%: DQN frente a regla re-ajustada"],
}


def group_of(vs):
    w, l = sum(v == "gana" for v in vs), sum(v == "pierde" for v in vs)
    if w and not l:
        return "mejoró"
    if l and not w:
        return "empeoró"
    if w and l:
        return "depende del mercado"
    return "sin cambio"


def memo_tables(df):
    out = []
    rows = [("veredicto", e, [e]) for e in VERDICT_ROWS] + [("probado", k, v) for k, v in TRIED_ROWS.items()]
    for kind, exp, sources in rows:
        cells = {}
        for g in GENS:
            for s in sources:
                x = df[(df.exp == s) & (df.gen == g)]
                if len(x):
                    cells[g] = x.iloc[0]
                    break
        if not cells:
            print("FALTA", exp)
            continue
        vs = [c.veredicto for c in cells.values()]
        row = dict(tabla=kind, exp=exp, grupo=group_of(vs) if kind == "probado" else "",
                   n_x=int(max(c.n_x for c in cells.values())), n_y=int(max(c.n_y for c in cells.values())))
        for g in GENS:
            for col in ("veredicto", "diff", "ci_low", "ci_high", "p_adj", "familia", "exp"):
                row[f"{g}_{col}"] = cells[g][col] if g in cells else np.nan
        out.append(row)
    t = pd.DataFrame(out)
    t[t.tabla == "veredicto"].to_csv("results/table_verdicts.csv", index=False)
    t[t.tabla == "probado"].to_csv("results/table_tried.csv", index=False)
    return t


def predictability():
    """E[r_{t+1} | decil de la tendencia] en 400 series de entrenamiento (la "estrategia esperada")."""
    from rl_trading.baselines import MyopicBayes
    rows = []
    for gen in GENS:
        mb = MyopicBayes(gen, n_bins=10, n_series=400)
        for i, v in enumerate(mb.mean_r):
            rows.append(dict(gen=gen, decil=i + 1, mean_r=float(v)))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_predictability.csv", index=False)
    return df


def main():
    predictability()
    minimums()
    balance()
    tried()
    df = adjust(pd.DataFrame(ROWS))
    df.to_csv("results/table_all_contrasts.csv", index=False)
    print(df.groupby(["familia", "veredicto"]).size())
    memo_tables(df)
    print(friedman().groupby("gen").first()[["chi2", "p", "cd"]])
    power()
    rliable()
    print(ceiling().round(3).to_string())


if __name__ == "__main__":
    main()
