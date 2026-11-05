# -*- coding: utf-8 -*-
"""Comprobaciones automáticas de consistencia de la memoria (python check_memoria.py).

1. Sin cifras escritas a mano: los literales numéricos del texto de memoria_text.py deben ser
   constantes del enunciado o del diseño (lista ALLOWED); todo resultado sale de una plantilla.
2. Vocabulario único: sin sinónimos de las métricas ni de los veredictos, y «empate» solo negado.
3. Significación solo con el esquema global (Holm confirmatorio / BH exploratorio).
4. Cada «equivalente» de las tablas está respaldado por TOST y cada «gana/pierde» por la regla.
5. Cada elemento de results/inventory.csv aparece en la memoria.
6. Cada tabla y figura se cita en el texto.
7. Exactamente 2 páginas y tamaños de letra mínimos.
"""
import ast
import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "memoria" / "memoria.html"
PDF = ROOT / "memoria" / "memoria.pdf"

# Constantes del enunciado y del diseño (no son resultados).
ALLOWED = {
    "0", "1", "2", "3", "4", "5", "6", "8", "10", "16", "20", "30", "40", "50", "64", "90", "95", "100",
    "400", "500", "2011", "3000", "20 000",
    "0.0", "0.001", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "0.9", "0.98", "0.99", "1e3", "2.5",
}
FORBIDDEN = {
    r"ratio geom": "usar «cociente de capital»",
    r"techo capturado": "usar «% del margen capturado»",
    r"\bBonferroni\b": "usar el esquema global (Holm/BH)",
    r"44 contrastes": "usar el esquema global (Holm/BH)",
    r"estadísticamente significativ": "usar gana/pierde/equivalente/no concluyente",
    r"\bempata\b(?!.{0,40}»)": "«empate» solo si TOST lo respalda",
    r"(?<!no es )(?<!no un )\bempate\b": "«empate» solo si TOST lo respalda",
    r"\bcheckpoint\b": "usar «parada temprana» / «modelo final»",
}

errors = []


def fail(msg):
    errors.append(msg)
    print("  ✗", msg)


def template_strings():
    """Texto literal de las plantillas (sin CSS ni expresiones {…})."""
    tree = ast.parse((ROOT / "memoria_text.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CSS" for t in node.targets):
            node.value.value = ""
        if isinstance(node, ast.JoinedStr):
            out += [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def clean(s):
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
    s = re.sub(r"style='[^']*'|style=\"[^\"]*\"", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s)


def check_literals():
    print("1. Literales numéricos en las plantillas")
    seen = {}
    keys = set(pd.read_csv(ROOT / "results" / "table_all_contrasts.csv").exp)
    for s in template_strings():
        if s in keys or s.startswith(("results/", "../")):
            continue
        t = clean(s)
        t = re.sub(r"#[0-9a-fA-F]{3,6}\b", " ", t)
        t = re.sub(r"\b\d+-\d+-\d+-\d+\b", " ", t)          # arquitectura 2-64-64-3
        t = re.sub(r"\b(MA30|r_?t\+1|t\+1|O\(1\))", " ", t)
        for m in re.finditer(r"(?<![\w.])(\d{1,3}(?: \d{3})+|\d+(?:\.\d+)?(?:e\d+)?)", t):
            tok = m.group(1)
            if tok not in ALLOWED:
                seen.setdefault(tok, t[max(0, m.start() - 40):m.end() + 25].replace("\n", " "))
    for tok, ctx in seen.items():
        fail(f"literal {tok!r} no permitido: …{ctx}…")


def check_vocabulary(text):
    print("2-3. Vocabulario y esquema de significación")
    for pat, why in FORBIDDEN.items():
        for m in re.finditer(pat, text):
            fail(f"«{m.group(0)}» ({why}): …{text[max(0, m.start() - 50):m.end() + 30]}…")
    for term in ("Σr", "log-riqueza", "cociente de capital", "gana", "pierde", "equivalente", "no concluyente",
                 "confirmatoria", "exploratoria"):
        if term not in text:
            fail(f"falta el término definido «{term}»")


def check_verdicts():
    print("4. Veredictos respaldados por la regla")
    ac = pd.read_csv(ROOT / "results" / "table_all_contrasts.csv")
    eq = ac[ac.veredicto == "equivalente"]
    bad = eq[~((eq.ci90_low > -eq.delta) & (eq.ci90_high < eq.delta))]
    for _, x in bad.iterrows():
        fail(f"equivalente sin TOST: {x.exp} / {x.gen}")
    sig = ac[ac.veredicto.isin(["gana", "pierde"])]
    no_ci = sig.ci_low.isna()  # S&P 500: una sola serie, solo hay p frente al azar
    bad = sig[~((sig.p_adj < 0.05) & (no_ci | (sig.ci_low > 0) | (sig.ci_high < 0)))]
    for _, x in bad.iterrows():
        fail(f"gana/pierde sin p ajustada<0.05 e IC que excluya 0: {x.exp} / {x.gen}")
    sign = sig[((sig.veredicto == "gana") & (sig["diff"] < 0)) | ((sig.veredicto == "pierde") & (sig["diff"] > 0))]
    for _, x in sign.iterrows():
        fail(f"signo incoherente: {x.exp} / {x.gen}")
    fam = ac.groupby("familia").size().to_dict()
    print(f"   {len(ac)} contrastes; familias {fam}; equivalentes {len(eq)}; gana/pierde {len(sig)}")


def check_inventory(text):
    print("5. Cobertura del inventario")
    inv = pd.read_csv(ROOT / "results" / "inventory.csv")
    flat = re.sub(r"\s+", " ", text)
    for _, x in inv.iterrows():
        if x.frase_clave not in flat:
            fail(f"inventario {x.id} ({x.elemento}): no aparece «{x.frase_clave}»")
        if isinstance(x.ubicacion, str) and x.ubicacion.startswith(("Tabla", "Figura")) and x.ubicacion not in flat:
            fail(f"inventario {x.id}: no existe {x.ubicacion}")
    print(f"   {len(inv)} elementos")


def check_refs(text):
    print("6. Tablas y figuras citadas en el texto")
    flat = re.sub(r"\s+", " ", text)
    caps = set(re.findall(r"(Tabla|Figura) (\d+)\.", flat))
    body = re.sub(r"(Tabla|Figura) \d+\. ", "", flat)
    for kind, n in sorted(caps):
        n = int(n)
        refs = re.findall(rf"{kind}s? (\d+)(?:[–-](\d+))?", body)
        if kind == "Figura":
            refs += re.findall(r"Fig\. (\d+)()", body)
        hit = any(int(a) == n or (b and int(a) <= n <= int(b)) for a, b in refs)
        if not hit:
            fail(f"{kind} {n} no se cita en el texto")
    print(f"   {len(caps)} tablas/figuras")


def check_pdf():
    print("7. Longitud y tipografía")
    from pypdf import PdfReader
    n = len(PdfReader(str(PDF)).pages)
    if n != 2:
        fail(f"el PDF tiene {n} páginas (deben ser 2)")
    import pymupdf
    subsup = {html.unescape(s).strip() for s in re.findall(r"<su[bp]>(.*?)</su[bp]>", HTML.read_text(encoding="utf-8"))}
    sizes = []
    for page in pymupdf.open(str(PDF)):
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    if len(sp["text"].strip()) > 3 and not sp["flags"] & 1 and sp["text"].strip() not in subsup:
                        sizes.append(round(sp["size"], 2))
    small = [s for s in sizes if s < 7.45]
    if small:
        fail(f"{len(small)} fragmentos con letra < 7.5 pt (mín. {min(small)})")
    print(f"   {n} páginas; letra mínima {min(sizes)} pt, mediana {np.median(sizes)} pt")


def main():
    text = clean(HTML.read_text(encoding="utf-8"))
    check_literals()
    check_vocabulary(text)
    check_verdicts()
    check_inventory(text)
    check_refs(text)
    check_pdf()
    if errors:
        print(f"\n{len(errors)} problema(s)")
        sys.exit(1)
    print("\nMemoria consistente")


if __name__ == "__main__":
    main()
