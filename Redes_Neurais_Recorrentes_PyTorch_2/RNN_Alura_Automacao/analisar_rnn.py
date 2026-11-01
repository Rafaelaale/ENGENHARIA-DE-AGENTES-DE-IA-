import ast
import sys
from pathlib import Path
import nbformat

notebook = Path(sys.argv[1])
relatorio = Path(sys.argv[2])

nb = nbformat.read(str(notebook), as_version=4)

linhas = [
    "ANALISE DO NOTEBOOK RNN ALURA",
    "=" * 75,
    f"Total de celulas: {len(nb.cells)}",
    ""
]

for numero, celula in enumerate(nb.cells, 1):
    if celula.cell_type != "code":
        continue

    fonte = celula.source

    linhas.append("=" * 75)
    linhas.append(f"CELULA {numero}")
    linhas.append("=" * 75)

    for n, texto in enumerate(fonte.splitlines(), 1):
        linhas.append(f"{n:03d} | {texto}")

    linhas.append("")

    try:
        arvore = ast.parse(fonte)
        nomes = sorted({
            node.id
            for node in ast.walk(arvore)
            if isinstance(node, ast.Name)
        })

        linhas.append("NOMES REFERENCIADOS:")
        linhas.append(", ".join(nomes))
    except SyntaxError as erro:
        linhas.append(
            f"SINTAXE PYTHON: linha {erro.lineno}: {erro.msg}"
        )

    if "split" in fonte:
        linhas.append("")
        linhas.append("*** REFERENCIA A SPLIT ENCONTRADA ***")

    linhas.append("")

relatorio.write_text(
    "\n".join(linhas),
    encoding="utf-8"
)

print("Analise gerada:", relatorio)
print("Celulas analisadas:", len(nb.cells))
print("Nenhum treinamento executado.")