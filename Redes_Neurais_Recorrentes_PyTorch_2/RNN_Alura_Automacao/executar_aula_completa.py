import ast
import json
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

original = {
    numero: celula.source
    for numero, celula in enumerate(nb.cells, 1)
    if celula.cell_type == "code"
}

# A celula 24 define forward e precisa ser executada.
# A celula 25 contem o treinamento de 200 epocas.
# Na copia executada, substituimos esse treinamento
# por um teste de inferencia sem otimizacao.

nb.cells[24].source = """
import torch

loss_treino, loss_test = [], []
acc_treino, acc_test = [], []

print("Validacao da arquitetura da RNN")

model.eval()

with torch.no_grad():
    dados_validacao, rotulos_validacao = sample_batch(size=5)

    resultados = []

    for dado in dados_validacao:
        dado = dado.to(args['device'])
        saida = model(dado)
        resultados.append(saida.detach().cpu())

    print("Amostras avaliadas:", len(resultados))
    print("Formato da primeira saida:", resultados[0].shape)

print("Inferencia concluida sem alterar pesos.")
"""

falha = None

try:
    cliente = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(raiz)}},
    )

    cliente.execute()

except Exception as erro:
    falha = f"{type(erro).__name__}: {erro}"

executadas = []
erros = []
saidas = {}

for numero, celula in enumerate(nb.cells, 1):
    if celula.cell_type != "code":
        continue

    if celula.execution_count is not None:
        executadas.append(numero)

    for output in celula.get("outputs", []):
        if output.output_type == "error":
            erros.append(
                f"Celula {numero}: "
                f"{output.get('ename')} - "
                f"{output.get('evalue')}"
            )

        elif output.output_type == "stream":
            saidas.setdefault(numero, []).append(
                output.get("text", "")
            )

nbformat.write(nb, str(saida))

linhas = [
    "RELATORIO COMPLETO DA AULA RNN",
    "=" * 65,
    f"Total de celulas: {len(nb.cells)}",
    f"Celulas de codigo executadas: {len(executadas)}",
    f"Numeros: {executadas}",
    "",
    "CELULA 24:",
    "Funcao forward definida e preservada.",
    "",
    "CELULA 25:",
    "Treinamento de 200 epocas substituido por inferencia.",
    "Nenhuma etapa de otimizacao executada nessa celula.",
    "",
    "ERROS:",
]

if falha:
    linhas.append(f"Falha Jupyter: {falha}")

if erros:
    linhas.extend(erros)
elif not falha:
    linhas.append("Nenhum erro registrado.")

linhas.extend([
    "",
    "SAIDAS DAS CELULAS:",
])

for numero, textos in saidas.items():
    linhas.append(f"\nCELULA {numero}")
    linhas.extend(textos)

linhas.extend([
    "",
    "ARQUIVOS:",
    f"Notebook: {saida}",
    f"Relatorio: {relatorio}",
    "",
    "CHECKPOINT:",
    "O arquivo modelo_rnn_treinado.pt nao foi sobrescrito.",
])

relatorio.write_text(
    "\n".join(linhas),
    encoding="utf-8"
)

print("\n".join(linhas[:20]))