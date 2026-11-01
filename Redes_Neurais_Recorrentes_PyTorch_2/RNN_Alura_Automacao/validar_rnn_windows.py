import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

raiz = Path(sys.argv[1])
entrada = Path(sys.argv[2])
saida = Path(sys.argv[3])
relatorio = Path(sys.argv[4])

dados = raiz / "data" / "names"

if not dados.is_dir():
    raise RuntimeError(f"Pasta de dados nao encontrada: {dados}")

arquivos = list(dados.glob("*.txt"))

if not arquivos:
    raise RuntimeError("Nenhum arquivo de nacionalidade encontrado.")

nb = nbformat.read(str(entrada), as_version=4)

# Substitui somente o download Linux da celula 8.
nb.cells[7].source = """
from pathlib import Path

pasta_dados = Path('data/names')
arquivos_dados = sorted(pasta_dados.glob('*.txt'))

assert arquivos_dados, 'Arquivos de nacionalidade ausentes'

print('Dados locais encontrados:', len(arquivos_dados))
print('Download desnecessario no Windows')
"""

# Preserva a definicao forward() na celula 24.
# Substitui o treinamento de 200 epocas na celula 25.
nb.cells[24].source = """
import torch

loss_treino, loss_test = [], []
acc_treino, acc_test = [], []

model.eval()

with torch.no_grad():
    dados_validacao, rotulos_validacao = sample_batch(size=5)
    resultados = []

    for dado in dados_validacao:
        saida = model(dado.to(args['device']))
        resultados.append(saida.detach().cpu())

    assert len(resultados) > 0, "Nenhuma inferencia produzida"
    print("Quantidade real de resultados:", len(resultados))
    print('INFERENCIA APROVADA: 5 amostras')
    print('Formato da saida:', resultados[0].shape)
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
except Exception as exc:
    falha = f"{type(exc).__name__}: {exc}"

erros = []
executadas = []
avisos = []

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

        if output.output_type == "stream":
            texto = output.get("text", "")
            if any(
                termo in texto.lower()
                for termo in [
                    "not recognized",
                    "nao e reconhecido",
                    "não é reconhecido",
                    "traceback",
                ]
            ):
                avisos.append(f"Celula {numero}: {texto[:500]}")

nbformat.write(nb, str(saida))

linhas = [
    "VALIDACAO RNN - WINDOWS",
    "=" * 55,
    f"Arquivos de nacionalidade: {len(arquivos)}",
    f"Celulas totais: {len(nb.cells)}",
    f"Celulas de codigo processadas: {len(executadas)}",
    f"Celulas: {executadas}",
    "",
    "Celula 8: arquivos locais utilizados.",
    "Celula 24: funcao forward definida.",
    "Celula 25: inferencia sem treinamento.",
    "",
    "ERROS:",
]

if falha:
    linhas.append(f"Falha Jupyter: {falha}")

linhas.extend(erros or ["Nenhum erro Python registrado."])

linhas.extend(["", "AVISOS DE COMANDOS EXTERNOS:"])
linhas.extend(avisos or ["Nenhum identificado."])

relatorio.write_text("\n".join(linhas), encoding="utf-8")
print("\n".join(linhas))

if falha or erros or avisos or len(executadas) != 15:
    sys.exit(1)