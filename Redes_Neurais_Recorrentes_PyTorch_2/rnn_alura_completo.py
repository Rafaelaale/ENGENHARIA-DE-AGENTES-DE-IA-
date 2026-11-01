import os
import string
import unicodedata
import random
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)
random.seed(42)

ROOT = "data/names"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isdir(ROOT):
    raise FileNotFoundError(
        f"Pasta de dados nao encontrada: {os.path.abspath(ROOT)}"
    )

def normalizar(texto):
    return "".join(
        caractere for caractere in
        unicodedata.normalize("NFKD", texto)
        if caractere in string.ascii_letters + " .'-"
    )

caracteres_validos = string.ascii_letters + " .'-"
tam_dicionario = len(caracteres_validos)

categorias = []
dados = []
rotulos = []

arquivos = sorted(
    arquivo for arquivo in os.listdir(ROOT)
    if arquivo.lower().endswith(".txt")
)

for arquivo in arquivos:
    categoria = os.path.splitext(arquivo)[0]
    caminho = os.path.join(ROOT, arquivo)

    with open(caminho, encoding="utf-8") as f:
        nomes = [
            normalizar(linha.strip())
            for linha in f
        ]

    nomes = [nome for nome in nomes if nome]

    categorias.append(categoria)
    dados.append(nomes)
    rotulos.append([categoria] * len(nomes))

print("Categorias:", len(categorias))
print("Amostras por classe:")
for categoria, nomes in zip(categorias, dados):
    print(categoria, len(nomes))

def nome2tensor(nome):
    nome = normalizar(nome)
    tensor = torch.zeros(len(nome), tam_dicionario)

    for indice, letra in enumerate(nome):
        posicao = caracteres_validos.find(letra)
        tensor[indice, posicao] = 1

    return tensor.to(DEVICE)

class RNN(nn.Module):
    def __init__(self, entrada, features, saida):
        super().__init__()

        self.rnn = nn.RNNCell(entrada, features)
        self.linear = nn.Linear(features, saida)
        self.softmax = nn.LogSoftmax(dim=1)
        self.features = features

    def forward(self, nome):
        hidden = torch.zeros(
            1, self.features,
            device=nome.device
        )

        for letra in nome:
            hidden = self.rnn(
                letra.unsqueeze(0),
                hidden
            )

        return self.softmax(self.linear(hidden))

if len(categorias) < 2:
    raise ValueError("Sao necessarias pelo menos duas categorias.")

model = RNN(
    tam_dicionario,
    256,
    len(categorias)
).to(DEVICE)

criterio = nn.NLLLoss()
otimizador = torch.optim.Adam(model.parameters(), lr=0.001)

amostras = [
    (nome, indice)
    for indice, nomes in enumerate(dados)
    for nome in nomes
]

print("Total de amostras:", len(amostras))
print("Dispositivo:", DEVICE)

# Treinamento demonstrativo
EPOCAS = 3
AMOSTRAS_POR_EPOCA = min(1000, len(amostras))

model.train()

for epoca in range(EPOCAS):
    lote = random.sample(amostras, AMOSTRAS_POR_EPOCA)
    perda_total = 0.0
    acertos = 0

    for nome, categoria in lote:
        entrada = nome2tensor(nome)
        alvo = torch.tensor(
            [categoria],
            dtype=torch.long,
            device=DEVICE
        )

        otimizador.zero_grad()
        saida = model(entrada)
        perda = criterio(saida, alvo)

        perda.backward()
        otimizador.step()

        perda_total += perda.item()
        acertos += int(saida.argmax(dim=1).item() == categoria)

    print(
        f"Epoca {epoca + 1}/{EPOCAS} | "
        f"Perda: {perda_total / len(lote):.4f} | "
        f"Acuracia de treino: {acertos / len(lote):.2%}"
    )

def prever(nome, top_k=3):
    model.eval()

    with torch.no_grad():
        saida = model(nome2tensor(nome))
        probabilidades = torch.exp(saida)
        valores, indices = probabilidades.topk(
            min(top_k, len(categorias)),
            dim=1
        )

    print(f"\nSobrenome: {nome}")

    for valor, indice in zip(
        valores[0].tolist(),
        indices[0].tolist()
    ):
        print(f"{categorias[indice]}: {valor:.2%}")

prever("Araujo")
prever("Cabral")
prever("Schmidt")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "categorias": categorias,
        "caracteres_validos": caracteres_validos
    },
    "modelo_rnn_alura.pt"
)

print("\nModelo salvo: modelo_rnn_alura.pt")