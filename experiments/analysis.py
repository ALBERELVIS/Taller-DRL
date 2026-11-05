"""Análisis: tablas (rendimiento, convergencia, tests pareados) y figuras.

python -m experiments.analysis
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
from rl_trading.env import GEN_LABELS, make_set  # noqa: E402
from rl_trading.evaluate import hier_bootstrap, holm, paired_test, simulate, metrics  # noqa: E402

GENS = ["ciclo", "gbm", "reversion", "regimenes"]
ALGOS = {"qlearning": "Q-learning (tabular)", "dqn": "DQN (Double)", "a3c": "A3C"}
COLORS = {"qlearning": "tab:green", "dqn": "tab:blue", "a3c": "tab:red", "Buy & Hold": "tab:orange",
          "Aleatorio": "gray", "Momentum": "tab:purple", "Contraria": "tab:brown", "Oráculo miope": "k"}
FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(pattern))]


def conv_episode(curve, frac=0.9, smooth=3):
    v = pd.Series(curve["val"]).rolling(smooth, min_periods=1).mean().to_numpy()
    ep = np.asarray(curve["episode"])
    tt = np.asarray(curve["time"])
    v0, vmax = v[0], v.max()
    if vmax - v0 <= 1e-12:
        return ep[0], tt[0]
    i = int(np.argmax(v >= v0 + frac * (vmax - v0)))
    return ep[i], tt[i]


def main_table(base, finals):
    rows, stat_rows = [], []
    for gen in GENS:
        bh_lw = np.asarray(base[gen]["Buy & Hold"]["test"]["log_wealth"])
        for name in ["Aleatorio", "Buy & Hold", "Momentum", "Contraria", "Oráculo miope"]:
            s = base[gen][name]["test_summary"]
            lw = np.asarray(base[gen][name]["test"]["log_wealth"])
            rows.append(dict(gen=gen, agente=name, tipo="baseline", ret_median=s["ret_median"],
                             log_wealth=s["log_wealth"], log_wealth_sd=0.0, sum_reward=s["sum_reward"],
                             sum_reward_sd=0.0, G=s["G"], sharpe=s["sharpe"], max_dd=s["max_dd"], trades=s["trades"],
                             exposure=s["exposure"], beats_bh=float(np.mean(lw > bh_lw)) * 100,
                             conv_ep=np.nan, conv_ep_sd=np.nan, train_time=np.nan))
        rows.append(dict(gen=gen, agente="Siempre fuera", tipo="baseline", ret_median=0.0, log_wealth=0.0,
                         log_wealth_sd=0.0, sum_reward=0.0, sum_reward_sd=0.0, G=0.0, sharpe=0.0, max_dd=0.0,
                         trades=0.0, exposure=0.0, beats_bh=float(np.mean(0 > bh_lw)) * 100,
                         conv_ep=np.nan, conv_ep_sd=np.nan, train_time=np.nan))
        for algo, label in ALGOS.items():
            runs = [r for r in finals if r["algo"] == algo and r["gen"] == gen]
            if not runs:
                continue
            summ = pd.DataFrame([r["test"]["best_summary"] for r in runs])
            lw = np.mean([r["test"]["best"]["log_wealth"] for r in runs], axis=0)
            conv = np.array([conv_episode(r["curve"]) for r in runs])
            rows.append(dict(gen=gen, agente=label, tipo="RL", ret_median=summ.ret_median.mean(),
                             log_wealth=summ.log_wealth.mean(), log_wealth_sd=summ.log_wealth.std(),
                             sum_reward=summ.sum_reward.mean(), sum_reward_sd=summ.sum_reward.std(),
                             G=summ.G.mean(), sharpe=summ.sharpe.mean(), max_dd=summ.max_dd.mean(),
                             trades=summ.trades.mean(), exposure=summ.exposure.mean(),
                             beats_bh=float(np.mean(lw > bh_lw)) * 100,
                             conv_ep=conv[:, 0].mean(), conv_ep_sd=conv[:, 0].std(), conv_time=conv[:, 1].mean(),
                             train_time=np.mean([r["train_time"] for r in runs])))
            for ref in ["Buy & Hold", "Aleatorio", "Oráculo miope"]:
                t = paired_test(lw, base[gen][ref]["test"]["log_wealth"])
                stat_rows.append(dict(gen=gen, agente=label, vs=ref, **t))
            ql = [r for r in finals if r["algo"] == "qlearning" and r["gen"] == gen]
            if algo != "qlearning" and ql:
                Yq = np.array([r["test"]["best"]["log_wealth"] for r in ql])
                t = paired_test(lw, Yq.mean(axis=0))
                lo, hi = hier_bootstrap(np.array([r["test"]["best"]["log_wealth"] for r in runs]), Yq)
                stat_rows.append(dict(gen=gen, agente=label, vs="Q-learning", **t, hb_low=lo, hb_high=hi))
    df = pd.DataFrame(rows)
    st = pd.DataFrame(stat_rows)
    st["p_holm"] = holm(st.p_wilcoxon.to_numpy())
    df.to_csv("results/table_main.csv", index=False)
    st.to_csv("results/table_stats.csv", index=False)
    return df, st


def fig_learning_curves(base, finals, compact=False):
    xkeys = ["episode"] if compact else ["episode", "time"]
    fig, axes = plt.subplots(len(xkeys), 4, figsize=(7.4, 1.9) if compact else (16, 6.5), squeeze=False)
    for j, gen in enumerate(GENS):
        for row, xkey in enumerate(xkeys):
            ax = axes[row, j]
            for algo, label in ALGOS.items():
                runs = [r for r in finals if r["algo"] == algo and r["gen"] == gen]
                if not runs:
                    continue
                L = min(len(r["curve"]["val"]) for r in runs)
                V = np.array([r["curve"]["val"][:L] for r in runs])
                X = np.mean([r["curve"][xkey][:L] for r in runs], axis=0)
                ax.plot(X, V.mean(0), color=COLORS[algo], label=label)
                ax.fill_between(X, V.mean(0) - V.std(0), V.mean(0) + V.std(0), color=COLORS[algo], alpha=0.15)
            for name in ["Buy & Hold", "Oráculo miope", "Momentum", "Contraria"]:
                v = base[gen][name]["val_summary"]["sum_reward"]
                ax.axhline(v, color=COLORS[name], ls="--", lw=0.8, label=name)
            ax.axhline(0, color="gray", ls=":", lw=1)
            if gen == "ciclo":
                ax.set_ylim(5.5, 8.8)
            if row == 0:
                ax.set_title(GEN_LABELS[gen])
                ax.set_xscale("log")
            ax.set_xlabel("Episodios (log)" if xkey == "episode" else "Tiempo real (s)")
            if j == 0:
                ax.set_ylabel("Σr validación")
    if compact:
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=7, fontsize=5.5, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(f"{FIG}/memo_curves.png", dpi=300)
    else:
        axes[0, 0].legend(fontsize=7, loc="lower right")
        fig.suptitle("Curvas de aprendizaje: recompensa total media en 40 series de validación (media ± sd, 5 semillas)")
        fig.tight_layout()
        fig.savefig(f"{FIG}/learning_curves.png", dpi=150)
    plt.close(fig)


def fig_train_G(finals):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for j, gen in enumerate(GENS):
        ax = axes[j]
        for algo, label in ALGOS.items():
            runs = [r for r in finals if r["algo"] == algo and r["gen"] == gen]
            if not runs:
                continue
            Gs = []
            for r in runs:
                h = pd.DataFrame(r["hist"]).sort_values("time")
                Gs.append(h["G"].rolling(50, min_periods=1).mean().to_numpy())
            L = min(map(len, Gs))
            Gs = np.array([g[:L] for g in Gs])
            ax.plot(np.arange(1, L + 1), Gs.mean(0), color=COLORS[algo], label=label)
        ax.set_xscale("log")
        ax.set_title(GEN_LABELS[gen])
        ax.set_xlabel("Episodio (log)")
    axes[0].set_ylabel("G de entrenamiento\n(media móvil 50 ep.)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/train_G.png", dpi=150)
    plt.close(fig)


def fig_policy_maps(finals, compact=False):
    from rl_trading.baselines import MyopicBayes
    from rl_trading.encoding import trend_sigma
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.95) if compact else (16, 3.6))
    algos = [("dqn", "DQN"), ("a3c", "A3C")] if compact else [("dqn", "DQN"), ("a3c", "A3C"), ("qlearning", "Q-learning")]
    for j, gen in enumerate(GENS):
        ax = axes[j]
        sig = trend_sigma(gen)
        for algo, label in algos:
            runs = [r for r in finals if r["algo"] == algo and r["gen"] == gen]
            if not runs:
                continue
            grid = np.asarray(runs[0]["policy_map"]["trend"]) / sig
            # fracción de semillas cuya política greedy deja al agente INVERTIDO tras actuar
            inv0 = np.mean([np.asarray(r["policy_map"]["act0"]) == 2 for r in runs], axis=0)
            inv1 = np.mean([np.asarray(r["policy_map"]["act1"]) != 0 for r in runs], axis=0)
            ax.plot(grid, inv0, color=COLORS[algo], lw=1.6, label=f"{label}: compra (sin posición)")
            ax.plot(grid, inv1, color=COLORS[algo], lw=1.2, ls="--", label=f"{label}: mantiene (con posición)")
        mb = MyopicBayes(gen)
        centers = np.r_[mb.edges[0] - 1e-9, (mb.edges[1:] + mb.edges[:-1]) / 2, mb.edges[-1] + 1e-9]
        ax2 = ax.twinx()
        ax2.plot(centers / sig, mb.mean_r * 1e3, "k:", lw=1.5, label="E[r(t+1)|tendencia]·1e3")
        ax2.axhline(0, color="k", lw=0.5)
        if not compact or j == 3:
            ax2.set_ylabel("E[r | tendencia] ×1e3", fontsize=5.5 if compact else 8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(GEN_LABELS[gen])
        ax.set_xlabel("Tendencia normalizada (σ)")
        if j == 0:
            ax.set_ylabel("Fracción invertida" if compact else "Fracción de semillas invertidas")
            if not compact:
                ax.legend(fontsize=6, loc="upper left")
                ax2.legend(fontsize=6, loc="lower right")
    if compact:
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h + [plt.Line2D([], [], color="k", ls=":")], lab + ["E[r(t+1)|tendencia]"], loc="lower center",
                   ncol=5, fontsize=5.5, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(f"{FIG}/memo_policy.png", dpi=300)
    else:
        fig.suptitle("Política aprendida frente a la estrategia teórica esperada (curva punteada)")
        fig.tight_layout()
        fig.savefig(f"{FIG}/policy_maps.png", dpi=150)
    plt.close(fig)


def fig_distributions(base, finals):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for j, gen in enumerate(GENS):
        ax = axes[j]
        bh = np.asarray(base[gen]["Buy & Hold"]["test"]["log_wealth"])
        data, labels = [], []
        for name in ["Aleatorio", "Momentum", "Contraria", "Oráculo miope"]:
            data.append(np.asarray(base[gen][name]["test"]["log_wealth"]) - bh)
            labels.append(name)
        for algo, label in [("qlearning", "Q-learn."), ("dqn", "DQN"), ("a3c", "A3C")]:
            runs = [r for r in finals if r["algo"] == algo and r["gen"] == gen]
            if runs:
                data.append(np.mean([r["test"]["best"]["log_wealth"] for r in runs], axis=0) - bh)
                labels.append(label)
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=35, fontsize=7)
        ax.axhline(0, color="tab:orange", ls="--")
        ax.set_title(GEN_LABELS[gen])
        if j == 0:
            ax.set_ylabel("log(riqueza) − log(riqueza B&H)\n(100 series de test)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/excess_vs_bh.png", dpi=150)
    plt.close(fig)


def fig_episode_examples():
    import torch
    from experiments.run_job import build_policy
    from rl_trading.encoding import Encoder
    fig, axes = plt.subplots(2, 4, figsize=(16, 5.5), sharex=True)
    for j, gen in enumerate(GENS):
        path = f"results/models/dqn_{gen}_s10.pt"
        if not os.path.exists(path):
            continue
        d = torch.load(path, weights_only=False)
        res = dict(cfg=d["cfg"], encoder=Encoder(sigma=d["sigma"]))
        pol = build_policy("dqn", res, d["best_state"])
        p = make_set(gen, "test")[0]
        sim = simulate([p], pol)
        bh = simulate([p], lambda o, r: np.full(len(o), 2))
        pr = p[5:-1]
        pos = sim["positions"][0]
        ax = axes[0, j]
        ax.plot(pr, color="k", lw=0.8)
        ax.plot(np.where(pos == 1, pr, np.nan), "g.", ms=2, label="invertido")
        ax.plot(np.where(pos == 0, pr, np.nan), "r.", ms=2, label="fuera")
        ax.set_title(GEN_LABELS[gen])
        if gen == "ciclo":
            ax.set_yscale("log")
        ax2 = axes[1, j]
        ax2.plot(sim["balances"][0], color="tab:blue", label="DQN")
        ax2.plot(bh["balances"][0], color="tab:orange", ls="--", label="Buy & Hold")
        ax2.axhline(1000, color="gray", ls=":")
        if gen == "ciclo":
            ax2.set_yscale("log")
        ax2.set_xlabel("Paso")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("Precio")
    axes[1, 0].set_ylabel("Balance ($)")
    fig.suptitle("Ejemplo: primera serie de test de cada generador (DQN, semilla 10)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/episode_examples.png", dpi=150)
    plt.close(fig)


def generalization(base):
    import torch
    from experiments.run_job import build_policy
    from rl_trading.encoding import Encoder
    out = {}
    for algo in ("dqn", "a3c"):
        M = np.full((4, 4), np.nan)
        for i, gtrain in enumerate(GENS):
            paths = sorted(glob.glob(f"results/models/{algo}_{gtrain}_s*.pt"))
            if not paths:
                continue
            pols = []
            for pth in paths:
                d = torch.load(pth, weights_only=False)
                pols.append(build_policy(algo, dict(cfg=d["cfg"], encoder=Encoder(sigma=d["sigma"])), d["best_state"]))
            for k, gtest in enumerate(GENS):
                ts = make_set(gtest, "test")
                bh = np.asarray(base[gtest]["Buy & Hold"]["test"]["log_wealth"])
                ex = [np.mean(metrics(simulate(ts, pol))["log_wealth"] - bh) for pol in pols]
                M[i, k] = np.mean(ex)
        out[algo] = M
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (algo, M) in zip(axes, out.items()):
        lim = np.nanmax(np.abs(np.clip(M, -3, 3)))
        im = ax.imshow(np.clip(M, -3, 3), cmap="RdYlGn", vmin=-lim, vmax=lim)
        for i in range(4):
            for k in range(4):
                ax.text(k, i, f"{M[i, k]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4), [GEN_LABELS[g].split(" ")[0] for g in GENS], rotation=20)
        ax.set_yticks(range(4), [GEN_LABELS[g].split(" ")[0] for g in GENS])
        ax.set_xlabel("Test en")
        ax.set_ylabel("Entrenado en")
        ax.set_title(f"{ALGOS[algo]}: exceso log-riqueza vs B&H")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{FIG}/generalization.png", dpi=150)
    plt.close(fig)
    pd.DataFrame({f"{a}_{GENS[i]}": M[i] for a, M in out.items() for i in range(4)}, index=GENS).T.to_csv(
        "results/table_generalization.csv")
    return out


def a3creg_choice(df):
    """Configuración de la iteración A3C-regímenes elegida SOLO por validación (best_val medio)."""
    sub = df[df.kind == "a3creg"]
    if sub.empty:
        return None
    return sub.groupby("name").best_val.mean().idxmax()


def ablations(base):
    rows = []
    for p in glob.glob("results/ablation/*.json"):
        d = json.load(open(p))
        name = os.path.basename(p).rsplit("_s", 1)[0]
        kind = name.split("_")[0]
        s = d["test"]["best_summary"]
        ce, ct = conv_episode(d["curve"])
        rows.append(dict(kind=kind, name=name, gen=d["gen"], seed=d["seed"], workers=d["cfg"].get("workers"),
                         gamma=d["cfg"].get("gamma"), sum_reward=s["sum_reward"], log_wealth=s["log_wealth"],
                         trades=s["trades"], exposure=s["exposure"], conv_ep=ce, conv_time=ct,
                         train_time=d["train_time"],
                         train_span=float(max(d["hist"]["time"]) - min(d["hist"]["time"])),
                         best_val=d["best_val"],
                         final_val=float(np.mean(d["curve"]["val"][-3:]))))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv("results/table_ablations_raw.csv", index=False)
    agg = df.groupby(["kind", "name", "gen"]).agg(
        sum_reward=("sum_reward", "mean"), sum_reward_sd=("sum_reward", "std"), log_wealth=("log_wealth", "mean"),
        trades=("trades", "mean"), exposure=("exposure", "mean"), conv_ep=("conv_ep", "mean"),
        conv_time=("conv_time", "mean"), train_time=("train_time", "mean"), train_span=("train_span", "mean"),
        final_val=("final_val", "mean"),
        final_val_sd=("final_val", "std")).reset_index()
    agg.to_csv("results/table_ablations.csv", index=False)
    reg = df[df.kind == "a3creg"]
    if len(reg):
        sel = reg.groupby("name").agg(val=("best_val", "mean"), test_sum_reward=("sum_reward", "mean"),
                                      test_sd=("sum_reward", "std"), trades=("trades", "mean"),
                                      exposure=("exposure", "mean"), n=("seed", "count")).reset_index()
        sel["elegida_por_validacion"] = sel.name == a3creg_choice(df)
        sel.to_csv("results/table_a3c_regimenes_iter.csv", index=False)
    # figuras
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    # iteraciones de mejora: protocolo principal vs iteración 2 (Σr en test, media ± sd de 5 semillas)
    items = []
    for label, algo, gen, it_name in [("DQN regímenes\n400→1000 ep", "dqn", "regimenes", "episodes_1000_regimenes"),
                                      ("A3C reversión\n1200→3000 ep", "a3c", "reversion", "a3cepisodes_3000_reversion"),
                                      ("A3C regímenes\nconf. elegida en val.", "a3c", "regimenes", a3creg_choice(df))]:
        base_runs = [json.load(open(p))["test"]["best_summary"]["sum_reward"]
                     for p in glob.glob(f"results/final/{algo}_{gen}_s*.json")]
        it_runs = df[df.name == it_name].sum_reward.to_numpy() if it_name else np.array([])
        items.append((label, np.mean(base_runs), np.std(base_runs),
                      it_runs.mean() if len(it_runs) else np.nan, it_runs.std() if len(it_runs) else 0))
    x = np.arange(len(items))
    axes[3].bar(x - 0.2, [i[1] for i in items], 0.4, yerr=[i[2] for i in items], capsize=2, label="Protocolo principal")
    axes[3].bar(x + 0.2, [i[3] for i in items], 0.4, yerr=[i[4] for i in items], capsize=2, label="Iteración 2")
    axes[3].set_xticks(x, [i[0] for i in items], fontsize=7)
    axes[3].set_title("Iteraciones de mejora: Σr en test")
    axes[3].legend(fontsize=7)
    axes[3].axhline(0, color="k", lw=0.5)
    sub = agg[agg.kind == "dqnvar"].copy()
    if len(sub):
        sub["var"] = sub.name.str.split("_").str[1]
        piv = sub.pivot(index="gen", columns="var", values="final_val").reindex(GENS)
        err = sub.pivot(index="gen", columns="var", values="final_val_sd").reindex(GENS)
        cols = [c for c in ["vanilla", "target", "double"] if c in piv]
        ref = piv["double"] if "double" in piv else 0
        piv[cols].sub(ref, axis=0).plot.bar(yerr=err[cols], ax=axes[0], rot=0, capsize=2)
        axes[0].set_title("Variantes DQN: Σr validación final − Double")
        axes[0].set_xticks(range(len(GENS)), [GEN_LABELS[g].split(" ")[0] for g in GENS], fontsize=7)
        axes[0].set_xlabel("")
        axes[0].axhline(0, color="k", lw=0.5)
    sub = agg[agg.kind == "gamma"].copy()
    if len(sub):
        sub["g"] = sub.name.str.split("_").str[1].astype(float)
        piv = sub.pivot(index="gen", columns="g", values="sum_reward").reindex(GENS)
        ref = piv[0.5] if 0.5 in piv else 0
        piv.sub(ref, axis=0).plot.bar(ax=axes[1], rot=0)
        axes[1].set_title("Sensibilidad a γ (DQN): Σr test − (γ=0.5)")
        axes[1].set_xticks(range(len(GENS)), [GEN_LABELS[g].split(" ")[0] for g in GENS], fontsize=7)
        axes[1].set_xlabel("")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].legend(title="γ", fontsize=7)
    sub = agg[agg.kind == "workers"].copy()
    if len(sub):
        ax2 = axes[2].twinx()
        for gen, g in sub.groupby("gen"):
            g = g.assign(w=g.name.str.split("_").str[1].astype(int)).sort_values("w")
            l, = axes[2].plot(g.w, g.train_time, "o-", label=f"{GEN_LABELS[gen]}: tiempo total")
            axes[2].plot(g.w, g.train_span, "^:", color=l.get_color(), label=f"{GEN_LABELS[gen]}: sin arranque")
            ax2.plot(g.w, g.sum_reward / g.sum_reward.max(), "s--", color=l.get_color(), alpha=0.6,
                     label=f"{GEN_LABELS[gen]}: Σr test / máx")
        axes[2].set_xscale("log", base=2)
        axes[2].set_xlabel("Nº de workers A3C (800 episodios)")
        axes[2].set_ylabel("Tiempo de entrenamiento (s)")
        ax2.set_ylabel("Σr test relativo")
        h1, l1 = axes[2].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axes[2].legend(h1 + h2, l1 + l2, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        axes[2].set_title("A3C: escalado con workers asíncronos")
    fig.tight_layout()
    fig.savefig(f"{FIG}/ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return agg


def sp500(base):
    rows = []
    for algo in ALGOS:
        runs = load(f"results/sp500/{algo}_s*.json")
        if runs:
            s = pd.DataFrame([r["test"]["best_summary"] for r in runs])
            rows.append(dict(agente=ALGOS[algo], ret=s.ret_median.mean(), ret_sd=s.ret_median.std(),
                             sum_reward=s.sum_reward.mean(), trades=s.trades.mean(), exposure=s.exposure.mean(),
                             sharpe=s.sharpe.mean(), max_dd=s.max_dd.mean()))
    for name in ["Aleatorio", "Buy & Hold", "Momentum", "Contraria"]:
        s = base["sp500"][name]["test_summary"]
        rows.append(dict(agente=name, ret=s["ret_median"], ret_sd=0, sum_reward=s["sum_reward"],
                         trades=s["trades"], exposure=s["exposure"], sharpe=s["sharpe"], max_dd=s["max_dd"]))
    df = pd.DataFrame(rows)
    df.to_csv("results/table_sp500.csv", index=False)
    return df


def fig_predictability():
    from rl_trading.baselines import MyopicBayes
    from rl_trading.encoding import trend_sigma
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, gen in zip(axes, GENS):
        mb = MyopicBayes(gen, n_bins=10, n_series=400)
        sig = trend_sigma(gen)
        ax.bar(range(1, 11), mb.mean_r * 1e3, color=np.where(mb.mean_r > 0, "tab:green", "tab:red"))
        ax.set_title(f"{GEN_LABELS[gen]} (σ_trend={sig:.3f})")
        ax.set_xlabel("Decil de la tendencia observada")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.axhline(-1.0, color="gray", ls=":", lw=1)
    axes[0].set_ylabel("E[r(t+1) | decil] ×1e3")
    fig.suptitle("Predictibilidad: rendimiento medio del paso siguiente por decil de tendencia "
                 "(líneas punteadas: ±comisión 0.1%)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/predictability.png", dpi=150)
    plt.close(fig)


def tuning_report():
    from experiments.tune import collect, SPACES
    allrows = []
    for algo in ("dqn", "a3c", "qlearning"):
        allrows += collect(algo)
    df = pd.DataFrame(allrows)
    df.to_csv("results/table_tuning.csv", index=False)
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    for i, algo in enumerate(("dqn", "a3c", "qlearning")):
        d = df[df.algo == algo].copy()
        if d.empty:
            continue
        # puntuación relativa por generador (z-score) para poder promediar entre mercados
        d["z"] = d.groupby("gen")["score"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
        for j, hp in enumerate(list(SPACES[algo])[:4]):
            ax = axes[i, j]
            for gen, g in d.groupby("gen"):
                m = g.groupby(hp)["z"].mean()
                ax.plot(range(len(m)), m.values, "o-", label=GEN_LABELS[gen])
                ax.set_xticks(range(len(m)), [str(x) for x in m.index])
            ax.set_title(f"{ALGOS[algo]}: {hp}")
            if j == 0:
                ax.set_ylabel("Puntuación validación (z)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Ajuste de hiperparámetros (búsqueda aleatoria): efecto marginal de cada hiperparámetro en validación")
    fig.tight_layout()
    fig.savefig(f"{FIG}/tuning.png", dpi=150)
    plt.close(fig)
    return df


if __name__ == "__main__":
    base = json.load(open("results/baselines.json"))
    if "pred" in sys.argv[1:]:
        fig_predictability()
    if "tune" in sys.argv[1:]:
        t = tuning_report()
        print(t.sort_values(["algo", "gen", "score"], ascending=[True, True, False]).groupby(["algo", "gen"]).head(3).to_string())
    finals = load("results/final/*.json")
    what = sys.argv[1:] or ["all"]
    if finals and ("all" in what or "main" in what):
        df, st = main_table(base, finals)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        print(df.round(3).to_string())
        print(st.round(4).to_string())
        fig_learning_curves(base, finals)
        fig_train_G(finals)
        fig_policy_maps(finals)
        fig_distributions(base, finals)
        fig_episode_examples()
        with plt.rc_context({"font.size": 6, "axes.titlesize": 6.5, "lines.linewidth": 1.0,
                             "xtick.labelsize": 5.5, "ytick.labelsize": 5.5}):
            fig_learning_curves(base, finals, compact=True)
            fig_policy_maps(finals, compact=True)
    if "all" in what or "gen" in what:
        print(generalization(base))
    if "all" in what or "abl" in what:
        a = ablations(base)
        if a is not None:
            print(a.round(3).to_string())
    if "all" in what or "sp500" in what:
        print(sp500(base).round(3).to_string())
