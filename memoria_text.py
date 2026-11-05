# -*- coding: utf-8 -*-
"""Texto y maquetación de la memoria (2 páginas A4).

Regla: ningún resultado se escribe a mano. Todas las cifras salen de results/*.csv a través de
las funciones de este módulo (lo comprueba check_memoria.py). Los únicos literales numéricos del
texto son constantes del enunciado o del diseño (comisión 0.1%, 500 precios, ×100, etc.).
"""
import numpy as np

GENS = ["ciclo", "gbm", "reversion", "regimenes"]
GL = {"ciclo": "Tendencia+ciclo", "gbm": "Browniano (GBM)", "reversion": "Reversión media", "regimenes": "Regímenes"}
AG = ["Aleatorio", "Buy & Hold", "Siempre fuera", "Momentum", "Contraria", "Oráculo miope",
      "Q-learning (tabular)", "DQN (Double)", "A3C"]
AGS = {"Q-learning (tabular)": "Q-learning", "DQN (Double)": "DQN", "Oráculo miope": "Oráculo miope*"}
SYM = {"gana": ("▲", "#17803a"), "pierde": ("▼", "#c0392b"), "equivalente": ("=", "#1f5fbf"),
       "no concluyente": ("·", "#666")}

CSS = """
@page { size: A4; margin: 8mm 9mm 7mm 9mm; }
body { font-family: 'Calibri', 'Segoe UI', sans-serif; font-size: 8.5pt; line-height: 1.1; color: #111; margin: 0; }
h1 { font-size: 12pt; margin: 0; }
.sub { font-size: 7.6pt; color: #444; margin-bottom: 1px; }
h2 { font-size: 9.2pt; margin: 3px 0 1px 0; color: #1f3b73; border-bottom: 0.6px solid #1f3b73; }
p { margin: 1px 0; text-align: justify; }
ul { margin: 0 0 1px 11px; padding: 0; } li { margin: 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 7.5pt; margin: 1px 0; line-height: 1.04; }
th, td { border-bottom: 0.4px solid #c8c8c8; padding: 0 2.5px; text-align: right; vertical-align: top; }
th { background: #e8edf6; } td:first-child, th:first-child { text-align: left; }
td.l { text-align: left; }
tr.rl td { background: #f4f8ee; } tr.grp td { background: #eef0f4; font-weight: bold; text-align: left; }
.b { font-weight: bold; }
.cap { font-size: 7.5pt; color: #333; margin: 0 0 2px 0; }
img { width: 76%; display: block; margin: 0 auto; }
h2 { break-after: avoid; }
table.keep { break-inside: avoid; }
.dec td { font-size: 7.5pt; } .dec td:first-child { width: 17%; font-weight: bold; }
"""


def build_html(T):
    """T: dict de DataFrames/valores cargados por make_memoria.py."""
    tab, AC, ver, tri = T["main"], T["contrasts"], T["verdicts"], T["tried"]
    orc, pw, fr, pred, rules = T["oracle"], T["power"], T["friedman"], T["pred"], T["rules"]
    cost, risk, abl, gen, sp, spn = T["cost"], T["risk"], T["ablations"], T["generalization"], T["sp500"], T["sp500_null"]
    pexp, feat, algos, best, tim, tune = T["ppo_explore"], T["features"], T["algos"], T["best"], T["timing"], T["tuning"]

    def r(g, a):
        x = tab[(tab.gen == g) & (tab.agente == a)]
        return x.iloc[0]

    def C(exp, g):
        x = AC[(AC.exp == exp) & (AC.gen == g)]
        return x.iloc[0]

    def nd(g):
        return 2 if g == "ciclo" else 3

    def ci(c, d=None):
        d = d if d is not None else 3
        return f"[{c.ci_low:+.{d}f}, {c.ci_high:+.{d}f}]"

    def pv(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    def cell(v, diff, g):
        if not isinstance(v, str):
            return "<td>–</td>"
        s, col = SYM[v]
        return f"<td style='color:{col}'>{s} {diff:+.{nd(g)}f}</td>"

    # ------------------------------------------------------------------ Tabla 1: decisiones
    sig = T["sigma"]
    d_c = C("DQN vs Aleatorio", "ciclo").delta
    d_o = C("DQN vs Aleatorio", "gbm").delta
    n_conf = int((AC.familia == "confirmatoria").sum())
    n_expl = int((AC.familia == "exploratoria").sum())
    ntune = {a: int(tune[tune.algo == a].groupby("gen").size().max()) for a in ("dqn", "a3c", "qlearning")}
    term_v = [C("Último paso terminal (sin bootstrap)", g).veredicto for g in GENS]
    term_txt = ", ".join(f"{v} en {n} mercado{'s' if n > 1 else ''}" for v, n in
                         sorted({v: term_v.count(v) for v in term_v}.items(), key=lambda x: -x[1]))
    dec = [
        ("Entorno", "Sin modificar: estado (tendencia de 6 precios, posición), vender/mantener/comprar, recompensa = rendimiento invertido − 0.1% "
         "por operación. Simulador vectorizado verificado idéntico al entorno."),
        ("Entrada a la red", f"Tendencia/σ y posición, <b>sin discretizar</b>; σ solo de series de entrenamiento ({sig['gbm']:.3f} en GBM, {sig['ciclo']:.2f} en ciclo). "
         f"Sin escalar, la tendencia varía ~{1 / sig['gbm']:.0f} veces menos que la posición y la red apenas la distingue. Es biyectiva: no cambia el MDP."),
        ("Último paso; ×100", "El entorno lo marca <i>terminated</i>; decidimos tratarlo como <i>límite de tiempo</i> (<i>bootstrap</i>): la serie no acaba por "
         f"nada observable y el estado no contiene t. Sin <i>bootstrap</i>, DQN: {term_txt} (Tabla 6). Recompensa ×100: Q en O(1), mismo argmax."),
        ("DQN", "Red 2-64-64-3; <i>replay</i> 50k (transiciones consecutivas correlacionadas); red objetivo (<i>target</i> fijo durante C pasos); Double "
         "(max sobre Q ruidosas sobreestima «estar dentro»); Huber y recorte de gradiente; ε<sub>k</sub>=max(0.02, λ<sub>ε</sub><sup>k</sup>) por episodio."),
        ("A3C", "Real: 4 procesos asíncronos (Hogwild, Adam compartido), red actor/crítico de dos cabezas como la del profesor, ventaja a n pasos y "
         "entropía β. <i>On-policy</i>, sin <i>replay</i>: decorrelan los workers, cada uno en series distintas."),
        ("Referencias", "Aleatorio, B&amp;H, siempre fuera, momentum/contraria, oráculo miope* (invierte si E[r<sub>t+1</sub>|tendencia]&gt;0), Q-learning tabular, "
         "<b>mejor regla de umbrales</b> (búsqueda exhaustiva de 1–4 umbrales en entrenamiento) y <b>oráculo oculto</b> (conoce fase, equilibrio o régimen: techo)."),
        ("Protocolo", f"Semillas disjuntas: entrenamiento (serie nueva por episodio), validación (40 series: ajuste y parada temprana), test (100 series "
         f"comunes: pareado). Ajuste aleatorio ({ntune['dqn']}/{ntune['a3c']}/{ntune['qlearning']} config. DQN/A3C/Q-learning; Tabla 2), 5 semillas nuevas. El test no elige nada."),
        ("Métricas", "Σr = recompensa total, lo que optimiza el agente (principal); log-riqueza (del balance) y <b>cociente de capital</b> exp(Δ log-riqueza), "
         "lo que le importa al inversor: Σr suma rendimientos aritméticos («G sube pero el PnL no»)."),
        ("Estadística", f"<b>Regla única</b> para los {n_conf + n_expl} contrastes: Wilcoxon pareado sobre 100 series con la media de las semillas en cada serie "
         "(inferencia sobre estos modelos) + IC95% <i>bootstrap</i> jerárquico (remuestrea semillas y series: añade la variabilidad entre entrenamientos). "
         "<b>Gana/pierde</b>: p ajustada &lt;0.05 y el IC excluye el 0; <b>equivalente</b>: IC90% dentro de ±δ (TOST); si no, <b>no concluyente</b>. "
         f"δ = 0.02 de Σr ({d_c:.2f} en el ciclo, 2% del B&amp;H). <b>Confirmatoria</b>: los "
         f"mínimos ({n_conf}, Holm); <b>exploratoria</b>: el resto ({n_expl}, Benjamini-Hochberg). Friedman-Nemenyi y potencia."),
    ]
    assert abs(d_o - 0.02) < 1e-12
    dec_table = "<table class=dec>" + "".join(f"<tr><td class=l>{a}</td><td class=l>{b}</td></tr>" for a, b in dec) + "</table>"

    # ------------------------------------------------------------------ predictibilidad
    P = {g: pred[pred.gen == g].sort_values("decil").mean_r.to_numpy() * 1e3 for g in GENS}
    pred_txt = (f"<b>¿Qué se puede aprender?</b> E[r<sub>t+1</sub>|decil de tendencia] en 400 series de entrenamiento: ciclo, <i>momentum</i> "
                f"({P['ciclo'][0]:+.0f}‰ a {P['ciclo'][-1]:+.0f}‰); GBM, nada (|E| ≤ {np.abs(P['gbm']).max():.2f}‰ &lt; comisión de 1‰); reversión, <i>contraria</i> "
                f"({P['reversion'][0]:+.1f}‰ a {P['reversion'][-1]:+.1f}‰); regímenes, <b>U invertida</b> ({P['regimenes'].max():+.2f}‰ en el centro, "
                f"{P['regimenes'][0]:+.1f}‰ en el extremo: la tendencia extrema delata el régimen bajista). Es la estrategia esperada (Fig. 1).")

    # ------------------------------------------------------------------ Tabla 2: hiperparámetros
    def hp(algo, g):
        c = best[f"{algo}|{g}"]
        if algo == "dqn":
            return f"α={c['lr']:g}, γ={c['gamma']}, λ<sub>ε</sub>={c['eps_decay']}, C={c['target_update']}"
        if algo == "a3c":
            return f"α={c['lr']:g}, γ={c['gamma']}, β={c['entropy']}, n={c['n_steps']}"
        return f"α<sub>0</sub>={c['alpha']}, γ={c['gamma']}, λ<sub>ε</sub>={c['eps_decay']}, bins={c['n_bins']}"
    eps = {a: best[f"{a}|ciclo"]["episodes"] for a in ("dqn", "a3c", "qlearning")}
    hh = "<tr><th>Hiperparámetros</th>" + "".join(f"<th>{GL[g]}</th>" for g in GENS) + "</tr>"
    hb = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{hp(a, g)}</td>" for g in GENS) + "</tr>"
                 for a, n in (("dqn", f"DQN ({eps['dqn']} ep)"), ("a3c", f"A3C ({eps['a3c']} ep, 4 w)"),
                              ("qlearning", f"Q-learn. ({eps['qlearning']} ep)")))
    hp_table = f"<table>{hh}{hb}</table>"
    g05 = sum(best[f"{a}|{g}"]["gamma"] == 0.5 for a in ("dqn", "a3c") for g in GENS)

    # ------------------------------------------------------------------ Tabla 3: resultados
    head = "<tr><th rowspan=2>Agente</th>" + "".join(f"<th colspan=3>{GL[g]}</th>" for g in GENS) + "</tr>"
    head += "<tr>" + "<th>Ret.med %</th><th>Σr</th><th>&gt;B&amp;H %</th>" * 4 + "</tr>"
    body = ""
    for a in AG:
        cls = " class=rl" if a in ("Q-learning (tabular)", "DQN (Double)", "A3C") else ""
        cells = ""
        for g in GENS:
            x = r(g, a)
            best_sr = max(tab[tab.gen == g].sum_reward)
            b = " class=b" if abs(x.sum_reward - best_sr) < 1e-9 else ""
            ret = f"{x.ret_median:,.0f}" if abs(x.ret_median) >= 1000 else f"{x.ret_median:+.1f}"
            sd = f"<span style='color:#666'>±{x.sum_reward_sd:.2f}</span>" if x.tipo == "RL" else ""
            cells += f"<td>{ret}</td><td{b}>{x.sum_reward:+.3f}{sd}</td><td>{x.beats_bh:.0f}</td>"
        body += f"<tr{cls}><td>{AGS.get(a, a)}</td>{cells}</tr>"
    main_table = f"<table><thead>{head}</thead>{body}</table>"

    ch = "<tr><th>Convergencia</th>" + "".join(f"<th>{GL[g]}</th>" for g in GENS) + "</tr>"
    cb = ""
    AK = {"Q-learning (tabular)": "qlearning", "DQN (Double)": "dqn", "A3C": "a3c"}
    for a, k in AK.items():
        cb += f"<tr class=rl><td>{AGS.get(a, a)}</td>" + "".join(
            f"<td>{r(g, a).conv_ep:.0f} ep ≈ {r(g, a).conv_ep * tim[(k, g)] / eps[k]:.0f} s / {tim[(k, g)]:.0f} s</td>"
            for g in GENS) + "</tr>"
    conv_table = f"<table>{ch}{cb}</table>"
    ratio_ep = np.mean([(tim[("dqn", g)] / eps["dqn"]) / (tim[("qlearning", g)] / eps["qlearning"]) for g in GENS])
    conv_s = {k: [r(g, a).conv_ep * tim[(k, g)] / eps[k] for g in GENS] for a, k in AK.items()}
    fastest = {"qlearning": "Q-learning", "dqn": "DQN", "a3c": "A3C"}[min(conv_s, key=lambda k: np.mean(conv_s[k]))]

    q = {g: r(g, "Q-learning (tabular)") for g in GENS}
    d = {g: r(g, "DQN (Double)") for g in GENS}
    a3 = {g: r(g, "A3C") for g in GENS}
    bh = {g: r(g, "Buy & Hold") for g in GENS}
    rr = rules[rules.elegida].set_index("gen")
    cw = cost.set_index(["gen", "cost_pct"])

    # ------------------------------------------------------------------ Tabla 5: balance
    vh = "<tr><th>A frente a B (Δ = Σr de A − Σr de B)</th>" + "".join(f"<th>{GL[g]}</th>" for g in GENS) + "</tr>"
    vb = ""
    for _, x in ver.iterrows():
        conf = "<sup>C</sup>" if x[f"{GENS[0]}_familia"] == "confirmatoria" else ""
        vb += f"<tr><td>{x.exp}{conf}</td>" + "".join(cell(x[f"{g}_veredicto"], x[f"{g}_diff"], g) for g in GENS) + "</tr>"
    oc = orc.set_index(["gen", "agente"])

    def cap(g, a):
        v = oc.loc[(g, a), "captura"]
        return "sin margen" if np.isnan(v) else f"{v:.0f}%"
    vb += ("<tr><td>% del margen hasta el oráculo oculto capturado: DQN → PPO+ventana+vol./MA</td>" +
           "".join(f"<td>{cap(g, 'DQN')}" + ("" if cap(g, 'DQN') == "sin margen" else f" → {cap(g, 'feat:ppo:ventana_ma_vol')}") + "</td>"
                   for g in GENS) + "</tr>")
    ver_table = f"<table class=keep><thead>{vh}</thead>{vb}</table>"

    # ------------------------------------------------------------------ Tabla 6: qué probamos
    LABEL = {
        "Quitar la red objetivo (final)": "Quitar la red objetivo (vanilla vs Double, modelo final)",
        "Double frente a solo red objetivo (final)": "Double DQN vs solo red objetivo (modelo final)",
        "Dueling DQN": "Dueling DQN vs DQN",
        "γ=0.0 (frente a 0.5)": "DQN con γ=0 vs γ=0.5", "γ=0.9 (frente a 0.5)": "DQN con γ=0.9 vs γ=0.5",
        "γ=0.99 (frente a 0.5)": "DQN con γ=0.99 vs γ=0.5",
        "Más episodios": "Más episodios (ciclo: PPO ×2; reversión: A3C ×2.5; regímenes: DQN ×2.5)",
        "Parada temprana DQN": "Parada temprana por validación vs modelo final (DQN)",
        "Parada temprana A3C": "Parada temprana por validación vs modelo final (A3C)",
        "Ensemble 5 semillas DQN": "Ensemble de las 5 semillas vs cada semilla (DQN)",
        "Ensemble 5 semillas A3C": "Ensemble de las 5 semillas vs cada semilla (A3C)",
        "PPO frente a DQN": "PPO (GAE + objetivo recortado) vs DQN",
        "Estado + vol./MA (DQN)": "Estado + vol./MA vs estado base (DQN)",
        "Estado + vol./MA (A3C)": "Estado + vol./MA vs estado base (A3C)",
        "Estado + vol./MA (PPO)": "Estado + vol./MA vs estado base (PPO)",
        "Estado + ventana+vol./MA (PPO)": "Estado + ventana + vol./MA vs base (PPO)",
        "Estado + ventana (DQN)": "Estado + ventana de 10 rendimientos vs base (DQN)",
        "Agente único 4 mercados (vol./MA)": "Un DQN para los 4 mercados (+vol./MA) vs especializado",
        "Expected SARSA frente a Q-learning": "Expected SARSA (on-policy) vs Q-learning, misma tabla",
        "Recompensa media-varianza λ=2": "Recompensa r−λr², λ=2 (en log-riqueza)",
        "Recompensa media-varianza λ=10": "Recompensa r−λr², λ=10 (en log-riqueza)",
        "Recompensa log-riqueza": "Recompensa log(B<sub>t</sub>/B<sub>t−1</sub>) (en log-riqueza)",
        "Último paso terminal (sin bootstrap)": "Último paso terminal, sin <i>bootstrap</i> (DQN)",
        "Comisión 0%: DQN frente a regla re-ajustada": "Comisión 0%: DQN vs regla re-ajustada",
        "Comisión 0.5%: DQN frente a regla re-ajustada": "Comisión 0.5%: DQN vs regla re-ajustada",
    }
    ROBUST = {"Comisión 0%: DQN frente a regla re-ajustada", "Comisión 0.5%: DQN frente a regla re-ajustada"}
    GROUPS = [("mejoró", "Mejoró (gana en algún mercado y no pierde en ninguno)"),
              ("depende del mercado", "Depende del mercado (gana en unos y pierde en otros)"),
              ("sin cambio", "Sin cambio demostrable (equivalente o no concluyente)"),
              ("empeoró", "Empeoró (pierde en algún mercado y no gana en ninguno)")]
    th = "<tr><th>Qué probamos (A vs B; semillas)</th>" + "".join(f"<th>{GL[g]}</th>" for g in GENS) + "</tr>"
    tb = ""
    for key, title in GROUPS:
        sub = tri[(tri.grupo == key) & (~tri.exp.isin(ROBUST))]
        if sub.empty:
            continue
        tb += f"<tr class=grp><td colspan=5>{title}</td></tr>"
        for _, x in sub.iterrows():
            ns = f"{x.n_x}" if x.n_x == x.n_y else f"{x.n_x} vs {x.n_y}"
            tb += f"<tr><td>{LABEL[x.exp]} <span style='color:#666'>({ns})</span></td>" + "".join(
                cell(x[f"{g}_veredicto"], x[f"{g}_diff"], g) for g in GENS) + "</tr>"
    tb += "<tr class=grp><td colspan=5>Robustez a la comisión (DQN y regla re-entrenados con otra comisión)</td></tr>"
    for _, x in tri[tri.exp.isin(ROBUST)].iterrows():
        tb += f"<tr><td>{LABEL[x.exp]} <span style='color:#666'>({x.n_x} vs regla)</span></td>" + "".join(
            cell(x[f"{g}_veredicto"], x[f"{g}_diff"], g) for g in GENS) + "</tr>"
    tried_table = f"<table><thead>{th}</thead>{tb}</table>"
    assert set(LABEL) == set(tri.exp), set(tri.exp) ^ set(LABEL)

    # ------------------------------------------------------------------ cifras para el texto
    def cv(exp, g):
        return C(exp, g)

    dq = {g: cv("DQN vs Q-learning", g) for g in GENS}
    pwr = pw.set_index(["gen", "a", "b", "S", "N"])

    def mdev(g, a, b, S, N=100, col="mde"):
        return pwr.loc[(g, a, b, S, N), col]
    share = [mdev(g, "DQN", "Q-learning", 5, col="seed_share") for g in ("ciclo", "reversion")]
    frr = fr.groupby("gen").first()
    fmax = frr.p.max()
    rank = fr.pivot(index="agente", columns="gen", values="rango_medio")
    top_rl = {g: rank[g].idxmin() for g in GENS}
    bf = {g: cv("Parada temprana DQN", g) for g in GENS}
    en_a = cv("Ensemble 5 semillas A3C", "ciclo")
    en_d = cv("Ensemble 5 semillas DQN", "regimenes")
    van = cv("Quitar la red objetivo (final)", "reversion")
    av = abl.set_index(["name", "gen"])
    vval = av.loc[("dqnvar_vanilla_reversion", "reversion"), "final_val"]
    dval = av.loc[("dqnvar_double_reversion", "reversion"), "final_val"]
    es = cv("Expected SARSA frente a Q-learning (ε=0.02)", "regimenes")
    es3 = cv("Expected SARSA frente a Q-learning (ε=0.3)", "regimenes")
    pe = pexp[pexp.gen == "regimenes"].set_index("variante")
    e0, e2 = pe.loc["γ ajustado, entropía 0.0"], pe.loc["γ ajustado, entropía 0.2"]
    rk = risk.set_index(["gen", "modo"])
    w = abl[abl.name.str.startswith("workers") & (abl.gen == "ciclo")].copy()
    w["w"] = w.name.str.split("_").str[1].astype(int)
    w = w.set_index("w")
    speed = w.train_span.loc[1] / w.train_span.min()
    diag = np.mean([gen.loc[f"dqn_{g}", g] for g in GENS])
    off = np.mean([gen.loc[f"dqn_{g}", h] for g in GENS for h in GENS if g != h])
    spr = sp.set_index("agente")
    sp_rows = AC[AC.exp.str.startswith("S&P")]
    sp_win = sp_rows[sp_rows.veredicto == "gana"]
    dv = cv("DQN vs Buy & Hold", "gbm")
    rv_d = cv("DQN vs Mejor regla", "ciclo")
    rv_p = cv("PPO vs Mejor regla", "ciclo")
    pl = cv("PPO 4800 ep frente a la mejor regla", "ciclo")
    fl = cv("DQN + ventana con 3× episodios", "reversion")
    feat_tr = feat[(feat.algo == "dqn") & (feat.features == "ventana") & (feat.gen == "reversion")].trades.iloc[0]
    a3i = cv("A3C regímenes: 3000 ep y config. elegida en validación", "regimenes")
    ql_reg = cv("Q-learning vs Mejor regla", "regimenes")
    mv2 = cv("Recompensa media-varianza λ=2", "regimenes")
    duel_n = int((tri.set_index("exp").loc["Dueling DQN", [f"{g}_veredicto" for g in GENS]] == "equivalente").sum())
    a3reg_bh = cv("A3C vs Buy & Hold", "regimenes")
    n_c_win = int(((AC.familia == "confirmatoria") & (AC.veredicto == "gana")).sum())
    cdv = frr.cd.iloc[0]
    cap_dqn = {g: oc.loc[(g, "DQN"), "captura"] for g in GENS}
    cap_best = {g: oc.loc[(g, "feat:ppo:ventana_ma_vol"), "captura"] for g in GENS}
    sw = cw.loc[("reversion", 0.0), "dqn_width"], cw.loc[("reversion", 0.5), "dqn_width"]
    tr_rev = cw.loc[("reversion", 0.0), "dqn_trades"], cw.loc[("reversion", 0.5), "dqn_trades"]
    bins = sorted({best[f"qlearning|{g}"]["n_bins"] for g in GENS})
    nbins = f"{bins[0]}–{bins[-1]}" if len(bins) > 1 else f"{bins[0]}"

    conf_rev = cv("DQN vs Q-learning", "reversion")

    def fx(v):
        return f"{v:+.3f}"

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>Taller RL Trader 2 — DQN y A3C para trading con un único activo</h1>
<div class=sub>Máster MIAX · Aprendizaje por Refuerzo · Grupo: Albert Martin y Jesús Bussión · Código y notebook reproducibles: <i>rl_trading/</i>, <i>experiments/</i>, <i>solucion_taller_trader2.ipynb</i></div>

<h2>1. Planteamiento: decisiones de diseño y por qué</h2>
{dec_table}
<p class=cap>Tabla 1. Decisiones de diseño y su justificación. vol./MA = volatilidad de 20 días y P/MA30−1; ventana = 10 últimos log-rendimientos (causales, verificado).</p>
<p>{pred_txt}</p>
{hp_table}
<p class=cap>Tabla 2. Hiperparámetros elegidos en validación. γ=0.5 gana en {g05} de 8 casos DQN/A3C: la señal predice el paso siguiente y un horizonte corto reduce la varianza del <i>target</i>.</p>

<h2>2. Resultados de los mínimos (test: 100 series × 5 semillas)</h2>
{main_table}
<p class=cap>Tabla 3. Ret.med = rendimiento final mediano; Σr = recompensa total media (±sd entre semillas); &gt;B&amp;H = % de series cuya riqueza final supera a B&amp;H. Negrita: mejor Σr.</p>
{conv_table}
<p class=cap>Tabla 4. Episodios hasta el 90% de la mejora en validación (media de 5 semillas) ≈ segundos hasta ese punto / tiempo total, <b>sin carga</b> y en serie
(1 semilla). Un episodio de DQN cuesta ≈{ratio_ep:.0f} de Q-learning, pero converge en {min(d[g].conv_ep for g in GENS):.0f}–{max(d[g].conv_ep for g in GENS):.0f}
episodios frente a {min(q[g].conv_ep for g in GENS):.0f}–{max(q[g].conv_ep for g in GENS):.0f}; en segundos gana {fastest}. DQN y Q-learning son deterministas; A3C no.</p>
<img src="../results/figures/memo_policy.png">
<p class=cap>Figura 1. Política greedy aprendida (fracción de semillas invertidas; continua: compra sin posición; discontinua: mantiene con posición) frente a E[r<sub>t+1</sub>|tendencia] (punteada).</p>
<ul>
<li><b>Ciclo</b> (Tabla 3, Fig. 1)<b>:</b> <i>momentum</i> con histéresis (compra con tendencia &gt;0 y mantiene aunque baje algo de 0); baten a B&amp;H en el {d['ciclo'].beats_bh:.0f}% de las series con {a3['ciclo'].trades:.0f}–{d['ciclo'].trades:.0f}
operaciones, frente a {r('ciclo', 'Oráculo miope').trades:.0f} del oráculo miope, que ignora la comisión. <b>GBM:</b> DQN converge a B&amp;H ({d['gbm'].trades:.0f} operación; <b>equivalente</b>, IC90%
[{dv.ci90_low:+.3f}, {dv.ci90_high:+.3f}]): sin señal, operar solo paga comisión; Q-learning sobreopera ({q['gbm'].trades:.0f} operaciones).</li>
<li><b>Reversión:</b> contraria con histéresis: DQN {d['reversion'].ret_median:+.1f}% mediano (B&amp;H {bh['reversion'].ret_median:+.1f}%) con {d['reversion'].trades:.0f} operaciones frente a
{r('reversion', 'Contraria').trades:.0f} de la regla contraria: solo opera en los extremos. <b>Regímenes:</b> Q-learning y DQN aprenden la <b>banda</b> no lineal esperada
({fx(q['regimenes'].sum_reward)} y {fx(d['regimenes'].sum_reward)}); A3C se queda en «siempre fuera» (Σr {a3['regimenes'].sum_reward:.3f}).</li>
</ul>

<h2>3. Balance: qué batimos y qué no</h2>
{ver_table}
<p class=cap>Tabla 5. Veredicto con la regla única (Tabla 1): <span style='color:#17803a'>▲ gana</span>, <span style='color:#c0392b'>▼ pierde</span>, <span style='color:#1f5fbf'>= equivalente (TOST)</span>,
· no concluyente; el número es Δ Σr. <sup>C</sup> = familia confirmatoria (Holm); resto, Benjamini-Hochberg. RL: 5 semillas (PPO+ventana+vol./MA: 3).</p>
<ul>
<li><b>Batimos</b> ({n_c_win} de {n_conf} contrastes confirmatorios): al aleatorio en los 4 mercados con DQN y A3C, y a B&amp;H donde hay señal; DQN multiplica el capital final de B&amp;H por
{cv('DQN vs Buy & Hold', 'ciclo').cap_ratio:.1f}, {cv('DQN vs Buy & Hold', 'reversion').cap_ratio:.2f} y {cv('DQN vs Buy & Hold', 'regimenes').cap_ratio:.2f} (ciclo, reversión, regímenes). En GBM, «equivalente a B&amp;H» es lo correcto.
A3C frente a B&amp;H en regímenes ({a3reg_bh['diff']:+.3f}) no resiste a Holm (p<sub>Holm</sub>={a3reg_bh.p_adj:.2f}). Friedman rechaza la igualdad de los agentes en los 4 mercados (p≤{fmax:.0e}; diferencia crítica de Nemenyi {cdv:.2f} rangos).</li>
<li><b>DQN frente a Q-learning</b> (petición del profesor): DQN gana en GBM ({dq['gbm']['diff']:+.3f}; Q-learning sobreopera) y pierde en regímenes ({dq['regimenes']['diff']:+.3f}, IC {ci(dq['regimenes'])}); en ciclo y
reversión es <b>no concluyente</b>, no un empate: en reversión el IC90% [{conf_rev.ci90_low:+.3f}, {conf_rev.ci90_high:+.3f}] se sale de ±δ. Potencia: el {share[0]:.0f}–{share[1]:.0f}% de la incertidumbre viene de las semillas; el
efecto mínimo detectable en reversión es {mdev('reversion', 'DQN', 'Q-learning', 5):.3f} con 5 semillas y {mdev('reversion', 'DQN', 'Q-learning', 20):.3f} con 20, aún &gt;δ: separar algoritmos exige más semillas, no más series.
Con una variable continua, {nbins} bins equiprobables apenas pierden información: la red no gana en representación, sí en eficiencia muestral (Tabla 4).</li>
<li><b>No batimos</b> a la <b>mejor regla de umbrales</b> en el ciclo (DQN {rv_d['diff']:+.2f}, IC {ci(rv_d, 2)}; PPO {rv_p['diff']:+.3f}, equivalente, y con el doble de episodios {pl['diff']:+.3f}), y nadie alcanza al <b>oráculo oculto</b>:
DQN captura el {cap_dqn['ciclo']:.0f}%, {cap_dqn['reversion']:.0f}% y {cap_dqn['regimenes']:.0f}% del margen entre la mejor política trivial y él. Con esta observación, la política óptima <i>es</i> una regla de dos umbrales si la ventaja de
estar dentro es monótona en la tendencia, y la búsqueda exhaustiva la optimiza directamente; en regímenes la regla de 4 umbrales sobreajusta ({rr.loc['regimenes', 'fit']:.3f} en ajuste, {rr.loc['regimenes', 'sum_reward']:.3f} en test)
y Q-learning la supera ({ql_reg['diff']:+.3f}). <b>Con más información sí</b>: PPO + ventana + vol./MA bate a la regla en ciclo, reversión y regímenes, y la captura sube al {cap_best['ciclo']:.0f}%, {cap_best['reversion']:.0f}% y
{cap_best['regimenes']:.0f}%. El cuello de botella es la observación, no el algoritmo.</li>
</ul>

<h2>4. Qué probamos: qué mejoró, qué empeoró y qué no cambió</h2>
{tried_table}
<p class=cap>Tabla 6. Cada experimento (A) frente a lo que sustituye (B), con la misma regla (Δ Σr; log-riqueza en las filas de recompensa); entre paréntesis, semillas: con 2–3, ▲ y = son indicios
(el IC jerárquico estima mal la variabilidad entre entrenamientos). Parada temprana, ensemble, oráculo oculto y cociente de capital son ideas de otros trabajos, recalculadas con nuestro protocolo.</p>
<ul>
<li><b>Mejoró.</b> La <b>parada temprana</b> rescata lo aprendido antes de que la política derive (DQN {bf['reversion']['diff']:+.3f} en reversión y {bf['regimenes']['diff']:+.3f} en regímenes). El <b>ensemble</b> de A3C anula la semilla
que colapsó a «siempre dentro» (ciclo {en_a['diff']:+.2f}, no concluyente por esa misma semilla). <b>PPO</b> reutiliza cada lote con el objetivo recortado: es el más estable y supera a DQN en el ciclo. <b>Más
información en el estado</b> ayuda a los tres algoritmos profundos donde hay estructura oculta. Con <b>λ=2</b> la penalización castiga más el régimen bajista (más volátil): actúa como filtro de régimen
(log-riqueza {mv2['diff']:+.3f}; exposición {rk.loc[('regimenes', 'pct'), 'exposure']:.0f}% → {rk.loc[('regimenes', 'mv2'), 'exposure']:.0f}%, <i>drawdown</i> {rk.loc[('regimenes', 'pct'), 'max_dd']:.0f}% → {rk.loc[('regimenes', 'mv2'), 'max_dd']:.0f}%).</li>
<li><b>Depende del mercado.</b> Más variables hacen que DQN <b>sobreopere donde no hay señal</b>: con la ventana, {feat_tr:.0f} operaciones en reversión, y triplicar los episodios no lo corrige ({fl['diff']:+.3f}). Un único agente
para los 4 mercados mejora en reversión y pierde en regímenes: con vol./MA distingue el mercado, pero no del todo. Con <b>λ=10</b>, en el ciclo (≈10% de volatilidad por paso) la penalización supera a la señal
y el agente sale del mercado (<i>drawdown</i> {rk.loc[('ciclo', 'pct'), 'max_dd']:.0f}% → {rk.loc[('ciclo', 'mv10'), 'max_dd']:.0f}%): λ debería escalarse con la varianza de cada mercado.</li>
<li><b>Sin cambio demostrable.</b> Sin red objetivo la validación final cae en reversión ({vval:.3f} frente a {dval:.3f}), pero en test, con 2 semillas, es no concluyente ({van['diff']:+.3f}, IC {ci(van)}): indicio, no prueba.
Dueling: equivalente en {duel_n} mercado y no concluyente en el resto (con 3 acciones separar V y A apenas aporta). Ningún γ mejora a 0.5 (γ=0.9 empeora el ciclo). log(B<sub>t</sub>/B<sub>t−1</sub>) ≈ r − r²/2 (salvo al operar: la comisión es multiplicativa) equivale a λ≈0.5. El ensemble de
DQN en regímenes promedia bandas distintas y acaba «siempre fuera» ({en_d['diff']:+.3f}): promediar valores no promedia políticas.</li>
<li><b>Empeoró: <i>on-policy</i> con señal débil.</b> A3C y PPO acaban «siempre fuera» en regímenes. Sin entropía, la política estocástica de PPO pasa de estar invertida el {e0.expo_train_first10:.0f}% del tiempo
al {e0.expo_train_last10:.2f}%; con entropía 0.2 sigue explorando ({e2.expo_train_last10:.0f}%) y tampoco aprende ({fx(e2.sum_reward)}); A3C con 3000 episodios y la configuración elegida en validación, tampoco ({a3i['diff']:+.3f}).
Evidencia a favor (no aísla todas las diferencias con A3C y PPO): con la misma tabla, cambiar solo el objetivo de Q-learning (max, <i>off-policy</i>) por Expected SARSA (<i>on-policy</i>) hunde el resultado ({es['diff']:+.3f}, IC {ci(es)}; con ε=0.3, {es3['diff']:+.3f}, no concluyente).
On-policy valora «comprar» bajo su propia exploración, que paga comisiones al deshacer la posición (<i>cliff walking</i>): con una ventaja (≈{P['regimenes'].max():.1f}‰) menor que el coste de explorar, «fuera» es un óptimo local.</li>
<li><b>Comisión.</b> Con más comisión DQN opera menos (reversión {tr_rev[0]:.0f} → {tr_rev[1]:.0f} operaciones) y ensancha la zona de no operar ({sw[0]:.1f}σ → {sw[1]:.1f}σ), como predicen los modelos con costes
proporcionales; sin comisión la histéresis no desaparece (la posición hace de memoria en una observación no markoviana). Al 0.5%, DQN supera en regímenes a la regla re-ajustada, que sobreajusta.</li>
<li><b>Otras pruebas</b> (sin contraste por serie): A3C escala ×{speed:.1f} con 16 workers en el ciclo (sin el arranque de procesos); las políticas no generalizan entre mercados (exceso de log-riqueza sobre B&amp;H
{diag:+.2f} en su mercado, {off:+.2f} fuera); en el S&amp;P 500 de 2011 (una serie, fuera de muestra) DQN gana {spr.loc['DQN (Double)', 'ret']:+.1f}% frente a {spr.loc['Buy & Hold', 'ret']:+.1f}% de B&amp;H, pero frente a
20 000 estrategias aleatorias de igual exposición y nº de operaciones solo {len(sp_win)} de {len(sp_rows)} ejecuciones es significativa tras Benjamini-Hochberg: indicio, no evidencia.</li>
</ul>

<h2>5. Conclusiones y limitaciones</h2>
<p>(1) DQN <b>redescubre la estrategia esperada</b> en los 4 mercados, con la histéresis que impone la comisión; A3C, en ciclo y reversión (en regímenes se queda fuera).
Ambos baten al aleatorio y a B&amp;H donde hay señal. En GBM, DQN converge a B&amp;H ({d['gbm'].trades:.0f} operación), mientras A3C ({a3['gbm'].trades:.0f}) y Q-learning ({q['gbm'].trades:.0f}) operan ruido. (2) Con la
observación del enunciado no superan a la mejor regla de dos umbrales (DQN pierde en el ciclo, PPO es equivalente) salvo en regímenes, donde la regla sobreajusta, y quedan lejos del oráculo oculto: el límite es la <b>observación</b>; con
volatilidad y media móvil el RL profundo supera a la regla y la red aventaja a la tabla. (3) Con señal débil (regímenes) los métodos <b>on-policy</b> colapsan a «siempre fuera»; Q-learning y DQN no. (4) La
<b>metodología</b> cambia las conclusiones: parada temprana, semillas, test pareado, corrección múltiple y TOST convierten «DQN empata con Q-learning» en «no concluyente con 5 semillas». <b>Limitaciones:</b>
datos sintéticos estacionarios; comisión sin deslizamiento; búsqueda de hiperparámetros limitada; las variantes de las Tablas 5–6 heredan los hiperparámetros del estado base y usan 2–5 semillas (poca
potencia para efectos pequeños); el S&amp;P 500 es una sola serie. Descartados por coste/beneficio: REINFORCE y A2C síncrono (PPO y A3C los cubren), MSE frente a Huber,
más mercados sintéticos y otros activos reales.</p>
</body></html>"""
    return html
