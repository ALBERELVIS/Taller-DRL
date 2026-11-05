"""Genera `solucion_taller_trader2.ipynb` (teoría + decisiones + resultados) a partir de los
resultados guardados en `results/`. Ejecutar después de los experimentos:

    python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace solucion_taller_trader2.ipynb
"""
import nbformat as nbf

from notebook_text import CELLS

nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(src.strip("\n")) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n"))
               for kind, src in CELLS]
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "solucion_taller_trader2.ipynb")
print("Notebook escrito con", len(CELLS), "celdas")
