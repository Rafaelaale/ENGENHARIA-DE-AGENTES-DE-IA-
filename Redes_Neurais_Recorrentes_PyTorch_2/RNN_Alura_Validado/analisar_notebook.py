import ast
import json
import re
import sys
from pathlib import Path

original = Path(sys.argv[1])
saida = Path(sys.argv[2])
destino = saida / "Classificacao_Original_Corrigido.ipynb"
relatorio = saida / "analise_notebook.txt"

with original.open(encoding="utf-8") as arquivo:
    notebook = json.load(arquivo)

alteracoes = []
pendencias = []

for indice, celula in enumerate(notebook["cells"]):
    if celula.get("cell_type") != "code":
        continue

    fonte = celula.get("source", [])
    codigo = "".join(fonte) if isinstance(fonte, list) else fonte

    # Correcoes conhecidas do exercicio
    substituicoes = [
        (
            r"nn\.LogSoftmax\(\s*\)",
            "nn.LogSoftmax(dim=1)",
            "LogSoftmax"
        ),
        (
            r"\.decode\(\s*['\"]utf-8['\"]\s*\)",
            "",
            "decode em strings"
        )
    ]

    for padrao, substituto, descricao in substituicoes:
        codigo, quantidade = re.subn(
            padrao,
            substituto,
            codigo
        )

        if quantidade:
            alteracoes.append(
                f"Celula {indice}: {descricao} ({quantidade})"
            )

    try:
        ast.parse(codigo)
    except SyntaxError as erro:
        pendencias.append(
            f"Celula {indice}: linha {erro.lineno}: {erro.msg}"
        )

    celula["source"] = codigo.splitlines(keepends=True)
    celula["execution_count"] = None
    celula["outputs"] = []

notebook.setdefault("metadata", {})["kernelspec"] = {
    "display_name": "RNN Alura - Python 3.11",
    "language": "python",
    "name": "rnn-alura311"
}

destino.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

linhas = [
    "ANALISE DO NOTEBOOK ORIGINAL",
    f"Total de celulas: {len(notebook['cells'])}",
    f"Correcoes: {len(alteracoes)}",
    f"Pendencias de sintaxe: {len(pendencias)}",
    "",
    "CORRECOES:"
] + alteracoes + ["", "PENDENCIAS:"] + pendencias

relatorio.write_text(
    "\n".join(linhas),
    encoding="utf-8"
)

print("\n".join(linhas))
print("\nNotebook corrigido:", destino)