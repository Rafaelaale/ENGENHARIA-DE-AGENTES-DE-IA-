# ==========================================================
# ALURA - AULA 12: MULTIPLOS CANAIS
# Curso: Redes Neurais Convolucionais com PyTorch
# ==========================================================

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F


# ==========================================================
# 1. CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados"
SAIDA.mkdir(parents=True, exist_ok=True)


def salvar(fig, nome):
    caminho = SAIDA / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=145)
    plt.close(fig)
    print("Imagem salva:", caminho)


def painel(ax, matriz, titulo, cmap="viridis"):
    ax.imshow(matriz, cmap=cmap)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


print("=" * 65)
print("AULA 12 - MULTIPLOS CANAIS | ALURA")
print("=" * 65)


# ==========================================================
# 2. IMAGEM RGB - TRES CANAIS
# ==========================================================

# Uma imagem colorida possui:
# R = vermelho
# G = verde
# B = azul
#
# Cada canal contem uma matriz 6x6.

r = np.tile(
    np.array(
        [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
        dtype=np.float32
    ),
    (6, 1)
)

g = np.tile(
    np.array(
        [0.1, 0.1, 0.2, 0.7, 0.8, 0.9],
        dtype=np.float32
    )[:, None],
    (1, 6)
)

b = np.zeros(
    (6, 6),
    dtype=np.float32
)

for i in range(6):
    for j in range(6):

        if i == j or i + j == 5:
            b[i, j] = 0.9
        else:
            b[i, j] = 0.1


# Formato usado para visualizar:
# altura, largura, canais.

rgb = np.stack(
    [r, g, b],
    axis=2
)


# PyTorch utiliza:
# N = quantidade de imagens
# C = canais
# H = altura
# W = largura
#
# Formato N,C,H,W.

entrada = (
    torch.from_numpy(rgb.copy())
    .permute(2, 0, 1)
    .unsqueeze(0)
)


print("\n1 - FORMATO DOS DADOS")

print(
    "Imagem RGB (H,W,C):",
    rgb.shape
)

print(
    "Tensor PyTorch (N,C,H,W):",
    tuple(entrada.shape)
)

assert rgb.shape == (6, 6, 3)

assert tuple(entrada.shape) == (
    1, 3, 6, 6
)


# ==========================================================
# 3. VISUALIZAR OS CANAIS
# ==========================================================

fig, axs = plt.subplots(
    1, 4,
    figsize=(13, 3.5)
)

painel(
    axs[0],
    rgb,
    "Imagem RGB 6x6"
)

painel(
    axs[1],
    r,
    "Canal vermelho",
    "Reds"
)

painel(
    axs[2],
    g,
    "Canal verde",
    "Greens"
)

painel(
    axs[3],
    b,
    "Canal azul",
    "Blues"
)

salvar(
    fig,
    "42_multiplos_canais_entrada_rgb.png"
)


# ==========================================================
# 4. CRIAR QUATRO FILTROS RGB
# ==========================================================

# Cada filtro possui tres matrizes 3x3.
#
# Formato dos pesos:
# (filtros, canais_entrada, altura, largura)
#
# 4 filtros, 3 canais, kernels 3x3.

pesos = np.zeros(
    (4, 3, 3, 3),
    dtype=np.float32
)


# Filtro 1 - bordas verticais no vermelho.

pesos[0, 0] = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)


# Filtro 2 - bordas horizontais no verde.

pesos[1, 1] = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
], dtype=np.float32)


# Filtro 3 - padrao diagonal no azul.

pesos[2, 2] = np.array([
    [1, 0, -1],
    [0, 1, 0],
    [-1, 0, 1]
], dtype=np.float32)


# Filtro 4 - combina os tres canais RGB.

pesos[3, 0] = pesos[0, 0] / 3.0
pesos[3, 1] = pesos[1, 1] / 3.0
pesos[3, 2] = pesos[2, 2] / 3.0


print("\n2 - FILTROS CONVOLUCIONAIS")

print(
    "Formato dos pesos:",
    pesos.shape
)

print(
    "Quatro filtros geram quatro mapas."
)


# ==========================================================
# 5. CONVOLUCAO MULTICANAL MANUAL
# ==========================================================

# A convolucao desloca o filtro nas
# dimensoes altura e largura.
#
# Para cada posicao:
# 1. Multiplica cada canal.
# 2. Soma os valores de cada canal.
# 3. Soma as contribuicoes dos canais.
# 4. Produz um pixel de saida.
#
# PyTorch Conv2d usa correlacao cruzada,
# por isso o calculo manual usa os pesos
# sem inversao.

manual = np.zeros(
    (4, 4, 4),
    dtype=np.float32
)

entrada_chw = entrada[0].numpy()


for filtro in range(4):

    for y in range(4):

        for x in range(4):

            janela = entrada_chw[
                :,
                y:y + 3,
                x:x + 3
            ]

            produtos = (
                janela * pesos[filtro]
            )

            manual[filtro, y, x] = (
                np.sum(produtos)
            )


# ==========================================================
# 6. CALCULAR UM PIXEL DE SAIDA
# ==========================================================

# Vamos examinar o quarto filtro.
# Ele utiliza informacoes dos tres canais.

janela = entrada_chw[
    :,
    0:3,
    0:3
]

contribuicoes = np.sum(
    janela * pesos[3],
    axis=(1, 2)
)


print("\n3 - SOMA DOS CANAIS")

for nome, valor in zip(
    ["R", "G", "B"],
    contribuicoes
):

    print(
        "Contribuicao",
        nome,
        ":",
        round(float(valor), 5)
    )


print(
    "Soma total:",
    round(
        float(contribuicoes.sum()),
        5
    )
)

print(
    "Pixel de saida:",
    round(
        float(manual[3, 0, 0]),
        5
    )
)


np.testing.assert_allclose(
    manual[3, 0, 0],
    contribuicoes.sum(),
    atol=1e-6
)


# ==========================================================
# 7. CONVOLUCAO MULTICANAL COM PYTORCH
# ==========================================================

pesos_torch = torch.from_numpy(
    pesos
)


with torch.no_grad():

    # Apenas um filtro RGB.
    um_mapa = F.conv2d(
        entrada,
        pesos_torch[3:4],
        stride=1,
        padding=0
    )

    # Quatro filtros RGB.
    quatro_mapas = F.conv2d(
        entrada,
        pesos_torch,
        stride=1,
        padding=0
    )


print("\n4 - CONVOLUCAO PYTORCH")

print(
    "Saida com um filtro:",
    tuple(um_mapa.shape)
)

print(
    "Saida com quatro filtros:",
    tuple(quatro_mapas.shape)
)


assert tuple(um_mapa.shape) == (
    1, 1, 4, 4
)

assert tuple(quatro_mapas.shape) == (
    1, 4, 4, 4
)


np.testing.assert_allclose(
    quatro_mapas[0].numpy(),
    manual,
    atol=1e-6,
    rtol=1e-6
)

print(
    "VALIDACAO MANUAL X PYTORCH: OK"
)


# ==========================================================
# 8. VISUALIZAR SOMA DOS CANAIS
# ==========================================================

fig, axs = plt.subplots(
    2, 4,
    figsize=(13, 6)
)


for c, nome in enumerate(
    ["R", "G", "B"]
):

    painel(
        axs[0, c],
        pesos[3, c],
        "Pesos do canal " + nome,
        "coolwarm"
    )

    produto = (
        janela[c] * pesos[3, c]
    )

    painel(
        axs[1, c],
        produto,
        "Produto do canal " + nome,
        "coolwarm"
    )


    for i in range(3):
        for j in range(3):

            axs[1, c].text(
                j,
                i,
                f"{produto[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none"
                }
            )


axs[0, 3].axis("off")

axs[0, 3].text(
    0.05,
    0.65,
    "1 filtro RGB\n= 3 matrizes 3x3",
    fontsize=12,
    transform=axs[0, 3].transAxes
)

axs[1, 3].axis("off")

axs[1, 3].text(
    0.05,
    0.65,
    "Soma R + G + B\n"
    "= 1 pixel de saida\n\n"
    f"{contribuicoes.sum():.4f}",
    fontsize=12,
    transform=axs[1, 3].transAxes
)


salvar(
    fig,
    "43_filtro_rgb_soma_canais.png"
)


# ==========================================================
# 9. QUATRO FILTROS, QUATRO MAPAS
# ==========================================================

fig, axs = plt.subplots(
    2, 4,
    figsize=(14, 6.5)
)

nomes = [
    "Bordas verticais R",
    "Bordas horizontais G",
    "Padrao diagonal B",
    "Combinacao RGB"
]


for i in range(4):

    canal = min(i, 2)

    painel(
        axs[0, i],
        pesos[i, canal],
        f"Filtro {i + 1}: canal {canal}",
        "coolwarm"
    )

    painel(
        axs[1, i],
        manual[i],
        f"Mapa {i + 1}\n{nomes[i]}"
    )


salvar(
    fig,
    "44_quatro_filtros_quatro_mapas.png"
)


# ==========================================================
# 10. ARQUITETURA DE UMA CNN
# ==========================================================

# A transcricao apresenta:
#
# Conv1 -> 6 filtros
# Pool1
# Conv2 -> 12 filtros
# Pool2
# Conv3 -> 18 filtros
# Pool3
# FC1 -> FC2
#
# Tambem cita Batch Normalization.
#
# Vamos construir uma demonstracao da
# dimensionalidade, sem treinamento.

torch.manual_seed(7)


rede = nn.ModuleDict({

    "conv1": nn.Conv2d(
        3, 6,
        kernel_size=3
    ),

    "pool1": nn.MaxPool2d(
        kernel_size=2
    ),

    "bn1": nn.BatchNorm2d(6),

    "conv2": nn.Conv2d(
        6, 12,
        kernel_size=3
    ),

    "pool2": nn.MaxPool2d(
        kernel_size=2
    ),

    "bn2": nn.BatchNorm2d(12),

    "conv3": nn.Conv2d(
        12, 18,
        kernel_size=3
    ),

    "pool3": nn.MaxPool2d(
        kernel_size=2
    ),

    "bn3": nn.BatchNorm2d(18),

    "fc1": nn.Linear(
        18 * 6 * 6,
        32
    ),

    "fc2": nn.Linear(
        32,
        2
    )
})


# Modo avaliacao.
# Nao estamos treinando a rede.

rede.eval()


# Uma imagem RGB 64x64.
exemplo = torch.rand(
    1, 3, 64, 64
)

etapas = [
    (
        "Entrada",
        tuple(exemplo.shape)
    )
]


# ==========================================================
# 11. PASSAGEM PELA CNN
# ==========================================================

with torch.no_grad():

    x = exemplo

    # Primeira convolucao.
    x = F.relu(
        rede["conv1"](x)
    )

    etapas.append(
        ("Conv1: 6 filtros", tuple(x.shape))
    )

    x = rede["pool1"](x)

    etapas.append(
        ("Pool1", tuple(x.shape))
    )

    x = rede["bn1"](x)

    etapas.append(
        ("BatchNorm1", tuple(x.shape))
    )


    # Segunda convolucao.
    x = F.relu(
        rede["conv2"](x)
    )

    etapas.append(
        ("Conv2: 12 filtros", tuple(x.shape))
    )

    x = rede["pool2"](x)

    etapas.append(
        ("Pool2", tuple(x.shape))
    )

    x = rede["bn2"](x)

    etapas.append(
        ("BatchNorm2", tuple(x.shape))
    )


    # Terceira convolucao.
    x = F.relu(
        rede["conv3"](x)
    )

    etapas.append(
        ("Conv3: 18 filtros", tuple(x.shape))
    )

    x = rede["pool3"](x)

    etapas.append(
        ("Pool3", tuple(x.shape))
    )

    x = rede["bn3"](x)

    etapas.append(
        ("BatchNorm3", tuple(x.shape))
    )


    # Camadas totalmente conectadas.
    x = torch.flatten(x, 1)

    etapas.append(
        ("Flatten", tuple(x.shape))
    )

    x = F.relu(
        rede["fc1"](x)
    )

    etapas.append(
        ("FC1", tuple(x.shape))
    )

    x = rede["fc2"](x)

    etapas.append(
        ("FC2", tuple(x.shape))
    )


print("\n5 - ARQUITETURA CNN")

for nome, dimensoes in etapas:

    print(
        f"{nome:22s}: {dimensoes}"
    )


# ==========================================================
# 12. VALIDAR OS CANAIS
# ==========================================================

assert etapas[-1][1] == (1, 2)

assert etapas[2][1][1] == 6
assert etapas[5][1][1] == 12
assert etapas[8][1][1] == 18

# Pooling preserva o numero de canais.

assert etapas[3][1] == etapas[2][1]
assert etapas[6][1] == etapas[5][1]
assert etapas[9][1] == etapas[8][1]


print(
    "\nVALIDACAO DOS CANAIS E DIMENSOES: OK"
)


# ==========================================================
# 13. DESENHAR A ARQUITETURA
# ==========================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)

ax.set_xlim(0, 13)
ax.set_ylim(-0.6, len(etapas) - 0.2)
ax.invert_yaxis()


for i, (nome, forma) in enumerate(etapas):

    if len(forma) == 4:

        texto = (
            f"{forma[2]}x"
            f"{forma[3]}x"
            f"{forma[1]}"
        )

    else:

        texto = (
            str(forma[1]) + " valores"
        )

    ax.text(
        0.3,
        i,
        nome,
        ha="left",
        va="center",
        fontsize=10
    )

    ax.text(
        7.0,
        i,
        texto,
        ha="left",
        va="center",
        fontsize=10,
        family="monospace"
    )

    ax.plot(
        [0.2, 11.5],
        [i + 0.42, i + 0.42],
        color="lightgray",
        linewidth=0.5
    )


ax.set_xticks([])
ax.set_yticks([])

for borda in ax.spines.values():
    borda.set_visible(False)


ax.set_title(
    "Arquitetura CNN\n"
    "Conv -> Pool -> BatchNorm -> FC"
)


salvar(
    fig,
    "45_arquitetura_cnn_multiplos_canais.png"
)


# ==========================================================
# 14. RESUMO
# ==========================================================

print("\nCONCEITOS DA AULA")

print(
    "1 - Conv2d percorre altura e largura."
)

print(
    "2 - Uma entrada RGB possui 3 canais."
)

print(
    "3 - Um filtro RGB possui pesos para 3 canais."
)

print(
    "4 - As contribuicoes dos canais sao somadas."
)

print(
    "5 - Um filtro produz um mapa de ativacao."
)

print(
    "6 - Quatro filtros produzem quatro mapas."
)

print(
    "7 - Pooling reduz dimensoes espaciais."
)

print(
    "8 - BatchNorm preserva as dimensoes."
)

print(
    "9 - Camadas FC produzem as saidas finais."
)

print(
    "10 - Nenhum filtro foi treinado nesta aula."
)


print(
    "\nAULA 12 EXECUTADA COM SUCESSO!"
)
