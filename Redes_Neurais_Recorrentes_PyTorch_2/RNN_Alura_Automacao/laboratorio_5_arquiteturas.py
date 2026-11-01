import torch
from torch import nn
from pathlib import Path
import sys

torch.manual_seed(42)
torch.set_num_threads(2)

relatorio = Path(sys.argv[1])
resultados = []

def registrar(titulo, entrada, saida, esperado):
    assert tuple(saida.shape) == esperado, (
        f"{titulo}: esperado {esperado}, recebido {tuple(saida.shape)}"
    )

    texto = (
        f"{titulo}\n"
        f"Entrada: {tuple(entrada.shape)}\n"
        f"Saida: {tuple(saida.shape)}\n"
        f"TESTE: APROVADO\n"
    )
    print(texto)
    resultados.append(texto)

with torch.no_grad():

    # 1. ONE-TO-ONE
    entrada = torch.randn(2, 10)
    rede = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    saida = rede(entrada)
    registrar(
        "1. ONE-TO-ONE",
        entrada, saida, (2, 3)
    )

    # 2. ONE-TO-MANY
    # Uma entrada gera uma sequencia de 5 saidas.
    entrada = torch.randn(2, 10)
    encoder = nn.Linear(10, 16)
    decoder = nn.GRU(16, 16, batch_first=True)
    classificador = nn.Linear(16, 3)

    contexto = encoder(entrada).unsqueeze(1)
    sequencia = contexto.repeat(1, 5, 1)
    estados, _ = decoder(sequencia)
    saida = classificador(estados)

    registrar(
        "2. ONE-TO-MANY",
        entrada, saida, (2, 5, 3)
    )

    # 3. MANY-TO-ONE
    # Cinco elementos, uma classificacao.
    entrada = torch.randn(2, 5, 10)
    rnn = nn.RNN(10, 16, batch_first=True)
    classificador = nn.Linear(16, 3)

    estados, _ = rnn(entrada)
    saida = classificador(estados[:, -1, :])

    registrar(
        "3. MANY-TO-ONE",
        entrada, saida, (2, 3)
    )

    # 4. MANY-TO-MANY SINCRONIZADO
    # Uma classificacao para cada elemento.
    entrada = torch.randn(2, 5, 10)
    rnn = nn.RNN(10, 16, batch_first=True)
    classificador = nn.Linear(16, 3)

    estados, _ = rnn(entrada)
    saida = classificador(estados)

    registrar(
        "4. MANY-TO-MANY SINCRONIZADO",
        entrada, saida, (2, 5, 3)
    )

    # 5. MANY-TO-MANY NAO SINCRONIZADO
    # Entrada com 7 elementos, saida com 4.
    entrada = torch.randn(2, 7, 10)
    encoder = nn.GRU(10, 16, batch_first=True)
    decoder = nn.GRU(16, 16, batch_first=True)
    classificador = nn.Linear(16, 3)

    _, memoria = encoder(entrada)
    entrada_decoder = torch.zeros(2, 4, 16)
    estados, _ = decoder(entrada_decoder, memoria)
    saida = classificador(estados)

    registrar(
        "5. MANY-TO-MANY NAO SINCRONIZADO",
        entrada, saida, (2, 4, 3)
    )

resumo = (
    "\nLABORATORIO RNN - VARIACOES NA ARQUITETURA\n"
    + "=" * 50
    + "\n"
    + "\n".join(resultados)
    + "\n5 ARQUITETURAS VALIDADAS\n"
    + "Nenhum treinamento executado.\n"
    + "Modelo anterior preservado.\n"
)

relatorio.write_text(resumo, encoding="utf-8")
print("=" * 50)
print("RESULTADO FINAL: 5/5 TESTES APROVADOS")
print("Relatorio:", relatorio)