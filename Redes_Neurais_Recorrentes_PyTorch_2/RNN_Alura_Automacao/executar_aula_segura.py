import ast
import json
import re
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

raiz = Path(sys.argv[1])
entrada = Path(sys.argv[2])
saida = Path(sys.argv[3])
relatorio = Path(sys.argv[4])

nb = nbformat.read(str(entrada), as_version=4)

total = len(nb.cells)
executadas = []
puladas = []
erros = []

def eh_treinamento(codigo):
    padroes = [
        r"\.backward\s*\(",
        r"\.step\s*\(",
        r"\boptimizer\.step\s*\(",
        r"\botimizador\.step\s*\(",
        r"\bfor\s+\w+\s+in\s+range\s*\(\s*(?:n_epochs|n_epocas|epochs|epocas)",
    ]
    return any(re.search(p, codigo, re.I) for p in padroes)

def contem_chamada_treino(codigo):
    return bool(re.search(
        r"^\s*(?:treinar|train|treinamento|executar_treino)\s*\(",
        codigo,
        re.I | re.M
    ))

def preparar(celula, numero):
    if celula.cell_type != "code":
        return False

    codigo = celula.source

    if numero == 18 and re.search(r"(?<!\w)split\s*\(", codigo):
        codigo = (
            "from sklearn.model_selection import train_test_split\n"
            + re.sub(r"(?<!\w)split\s*\(", "train_test_split(", codigo)
        )

    if eh_treinamento(codigo) or contem_chamada_treino(codigo):
        puladas.append((numero, "Treinamento omitido para preservar o checkpoint"))
        celula.source = (
            f"print('Celula original {numero}: treinamento preservado, nao reexecutado')"
        )
        return True

    celula.source = codigo
    return False

for numero, celula in enumerate(nb.cells, 1):
    preparar(celula, numero)

try:
    cliente = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(raiz)}},
    )

    cliente.execute()

except Exception as exc:
    erros.append(("Jupyter", type(exc).__name__, str(exc)))

finally:
    for numero, celula in enumerate(nb.cells, 1):
        if celula.cell_type != "code":
            continue

        if celula.execution_count is not None:
            executadas.append(numero)

        for output in celula.get("outputs", []):
            if output.output_type == "error":
                erros.append((
                    numero,
                    output.get("ename", "Erro"),
                    output.get("evalue", "")
                ))

    nbformat.write(nb, str(saida))

    linhas = [
        "RELATORIO DE EXECUCAO - AULA RNN",
        "=" * 55,
        f"Total de celulas: {total}",
        f"Celulas de codigo executadas: {len(executadas)}",
        f"Celulas de codigo com tentativa de execucao: {executadas}",
        "",
        "CELULAS DE TREINAMENTO OMITIDAS:",
    ]

    if puladas:
        linhas.extend(f"Celula {n}: {motivo}" for n, motivo in puladas)
    else:
        linhas.append("Nenhuma identificada")

    linhas.extend(["", "ERROS IDENTIFICADOS:"])

    if erros:
        linhas.extend(
            f"Celula {numero}: {tipo} - {mensagem}"
            for numero, tipo, mensagem in erros
        )
    else:
        linhas.append("Nenhum erro registrado")

    linhas.extend([
        "",
        "IMPORTANTE:",
        "Uma celula executada pode conter erro.",
        "Celulas de treinamento substituidas foram omitidas.",
        "O checkpoint existente nao foi sobrescrito.",
        f"Notebook de resultados: {saida}",
    ])

    relatorio.write_text("\n".join(linhas), encoding="utf-8")

    print("\n".join(linhas))