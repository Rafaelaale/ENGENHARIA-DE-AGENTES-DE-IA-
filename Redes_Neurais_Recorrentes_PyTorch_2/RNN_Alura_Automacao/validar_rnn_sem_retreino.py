import sys
import nbformat
from pathlib import Path
from nbclient import NotebookClient

raiz = Path(sys.argv[1])
entrada = Path(sys.argv[2])
saida = Path(sys.argv[3])
relatorio = Path(sys.argv[4])

nb = nbformat.read(str(entrada), as_version=4)

# A celula 24 define a funcao forward().
# Ela deve ser executada normalmente.

# A celula 25 executa 200 epocas de treinamento.
# Substituimos somente na copia de execucao.
original_25 = nb.cells[24].source

nb.cells[24].source = (
    "print('Treinamento de 200 epocas omitido para preservar o modelo.')\n"
    "loss_treino, loss_test = [], []\n"
    "acc_treino, acc_test = [], []"
)

falha_jupyter = None

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
    falha_jupyter = f"{type(exc).__name__}: {exc}"

erros = []
executadas = []

for numero, celula in enumerate(nb.cells, 1):
    if celula.cell_type != "code":
        continue

    if celula.execution_count is not None:
        executadas.append(numero)

    for output in celula.get("outputs", []):
        if output.output_type == "error":
            erros.append(
                f"Celula {numero}: "
                f"{output.get('ename')} - {output.get('evalue')}"
            )

nbformat.write(nb, str(saida))

linhas = [
    "VALIDACAO DA AULA RNN",
    "=" * 55,
    f"Total de celulas: {len(nb.cells)}",
    f"Celulas de codigo executadas: {len(executadas)}",
    f"Numeros das celulas: {executadas}",
    "",
    "Celula 24: definicao de forward preservada.",
    "Celula 25: treinamento de 200 epocas omitido.",
    "Checkpoint original nao foi sobrescrito.",
    "",
    "ERROS:",
]

if falha_jupyter:
    linhas.append(f"Falha Jupyter: {falha_jupyter}")

if erros:
    linhas.extend(erros)
elif not falha_jupyter:
    linhas.append("Nenhum erro registrado.")

linhas.extend([
    "",
    "OBSERVACAO:",
    "A validacao nao comprova a execucao das 200 epocas.",
    "A celula 25 foi substituida somente na copia executada.",
])

relatorio.write_text("\n".join(linhas), encoding="utf-8")
print("\n".join(linhas))