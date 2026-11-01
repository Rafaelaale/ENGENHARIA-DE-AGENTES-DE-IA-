import json
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path.cwd()
pasta = root / "RNN_Alura_Automacao"

origem = pasta / "Classificacao_Sequencias_Corrigido.ipynb"
destino = pasta / "Classificacao_Execucao_Jupyter.ipynb"
relatorio = pasta / "diagnostico_jupyter.txt"

mensagens = []

def log(texto):
    print(texto, flush=True)
    mensagens.append(str(texto))

nb = nbformat.read(str(origem), as_version=4)
nbformat.validate(nb)

# O kernel usa o Python do ambiente virtual.
kernel = "python3"

cliente = NotebookClient(
    nb,
    timeout=60,
    kernel_name=kernel,
    allow_errors=True,
    resources={
        "metadata": {
            "path": str(root)
        }
    }
)

log("INICIANDO EXECUCAO JUPYTER")
log(f"CELULAS: {len(nb.cells)}")

falhas = []

try:
    cliente.execute()

    for numero, celula in enumerate(nb.cells, 1):
        if celula.cell_type != "code":
            continue

        erros = [
            saida for saida in celula.get("outputs", [])
            if saida.get("output_type") == "error"
        ]

        if erros:
            falhas.append(numero)
            log(f"CELULA {numero}: ERRO")

            for erro in erros:
                log("\n".join(
                    erro.get("traceback", [])
                ))
        elif celula.get("execution_count") is not None:
            log(f"CELULA {numero}: EXECUTADA")

finally:
    nbformat.write(nb, str(destino))

    log(f"FALHAS: {falhas}")
    log(f"NOTEBOOK EXECUTADO: {destino}")

    relatorio.write_text(
        "\n".join(mensagens) + "\n",
        encoding="utf-8"
    )

if falhas:
    raise SystemExit(1)

log("VALIDACAO JUPYTER CONCLUIDA")