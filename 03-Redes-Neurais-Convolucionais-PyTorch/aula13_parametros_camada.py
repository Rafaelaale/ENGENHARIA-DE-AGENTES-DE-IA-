# ALURA - AULA 13: PARAMETROS DA CAMADA CONVOLUCIONAL
# Curso: Redes Neurais Convolucionais com PyTorch

from pathlib import Path
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torch import nn
import torch.nn.functional as F


# =====================================================
# 1. CONFIGURACAO DO PROJETO
# =====================================================

BASE = Path(__file__).resolve().parent

SAIDA = BASE / "resultados"

SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


def salvar(fig, arquivo):

    caminho = SAIDA / arquivo

    fig.tight_layout()

    fig.savefig(
        caminho,
        dpi=145
    )

    plt.close(fig)

    print("Imagem salva:", caminho)


def dimensao(n, kernel, stride=1, padding=0):

    # Formula para Conv2d com dilation = 1.
    # Usamos floor para obter a parte inteira.

    return (
        math.floor(
            (n - kernel + 2 * padding) / stride
        ) + 1
    )


def desenhar_matriz(
    ax,
    matriz,
    titulo,
    caixa=None
):

    ax.imshow(
        matriz,
        cmap="gray",
        vmin=min(
            0,
            float(matriz.min())
        ),
        vmax=max(
            1,
            float(matriz.max())
        )
    )

    ax.set_title(
        titulo,
        fontsize=10
    )

    h, w = matriz.shape

    if h <= 7 and w <= 7:

        for i in range(h):

            for j in range(w):

                ax.text(
                    j,
                    i,
                    f"{matriz[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.75,
                        "edgecolor": "none",
                        "pad": 1
                    }
                )

        ax.set_xticks(
            np.arange(-0.5, w, 1),
            minor=True
        )

        ax.set_yticks(
            np.arange(-0.5, h, 1),
            minor=True
        )

        ax.grid(
            which="minor",
            color="gray",
            alpha=0.5
        )

    ax.set_xticks([])
    ax.set_yticks([])

    if caixa is not None:

        y, x, k = caixa

        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                k,
                k,
                fill=False,
                edgecolor="red",
                linewidth=3
            )
        )


print("=" * 64)
print("AULA 13 - PARAMETROS DA CAMADA | PYTORCH")
print("=" * 64)


# =====================================================
# 2. IN_CHANNELS E OUT_CHANNELS
# =====================================================

# A transcricao apresenta a sequencia:
#
# Conv1: 1 canal de entrada -> 6 filtros
# Conv2: 6 canais de entrada -> 12 filtros
# Conv3: 12 canais de entrada -> 18 filtros

print("\n1 - CANAIS E FILTROS")

cinza = torch.rand(
    1, 1, 25, 25
)

conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=6,
    kernel_size=3,
    stride=1,
    padding=1
)

conv2 = nn.Conv2d(
    in_channels=6,
    out_channels=12,
    kernel_size=3,
    stride=1,
    padding=1
)

conv3 = nn.Conv2d(
    in_channels=12,
    out_channels=18,
    kernel_size=3,
    stride=1,
    padding=1
)

with torch.no_grad():

    c1 = conv1(cinza)
    c2 = conv2(c1)
    c3 = conv3(c2)


for nome, camada, tens in [

    ("conv1", conv1, c1),
    ("conv2", conv2, c2),
    ("conv3", conv3, c3)

]:

    print(
        f"{nome}: "
        f"pesos={tuple(camada.weight.shape)} "
        f"saida={tuple(tens.shape)}"
    )


assert tuple(c1.shape) == (
    1, 6, 25, 25
)

assert tuple(c2.shape) == (
    1, 12, 25, 25
)

assert tuple(c3.shape) == (
    1, 18, 25, 25
)


# =====================================================
# 3. IMAGEM RGB COM 16 FILTROS
# =====================================================

rgb = torch.rand(
    1, 3, 25, 25
)

conv_rgb = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)

with torch.no_grad():

    saida_rgb = conv_rgb(rgb)


print(
    "\nRGB (3 canais) com 16 filtros:",
    tuple(saida_rgb.shape)
)

print(
    "Pesos dos filtros RGB:",
    tuple(conv_rgb.weight.shape)
)


assert tuple(conv_rgb.weight.shape) == (
    16, 3, 3, 3
)


# =====================================================
# 4. STRIDE 1 E STRIDE 2
# =====================================================

# A professora demonstra:
#
# Entrada: 5x5
# Kernel: 3x3
#
# Stride 1 -> saida 3x3
# Stride 2 -> saida 2x2

print("\n2 - STRIDE: DESLOCAMENTO DA JANELA")

imagem5 = np.tile(

    np.array(
        [10, 15, 20, 80, 85],
        dtype=np.float32
    ),

    (5, 1)
)


kernel = torch.tensor(

    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ],

    dtype=torch.float32

).reshape(1, 1, 3, 3)


entrada5 = torch.from_numpy(
    imagem5
).reshape(1, 1, 5, 5)


with torch.no_grad():

    saida_stride1 = F.conv2d(
        entrada5,
        kernel,
        stride=1,
        padding=0
    )[0, 0].numpy()

    saida_stride2 = F.conv2d(
        entrada5,
        kernel,
        stride=2,
        padding=0
    )[0, 0].numpy()


print(
    "Entrada:",
    imagem5.shape
)

print(
    "Stride 1:",
    saida_stride1.shape
)

print(
    "Stride 2:",
    saida_stride2.shape
)

print(
    "\nMapa stride 1:"
)

print(
    saida_stride1.astype(int)
)

print(
    "\nMapa stride 2:"
)

print(
    saida_stride2.astype(int)
)


np.testing.assert_array_equal(

    saida_stride1,

    np.array([
        [-30, -195, -195],
        [-30, -195, -195],
        [-30, -195, -195]
    ])

)


np.testing.assert_array_equal(

    saida_stride2,

    np.array([
        [-30, -195],
        [-30, -195]
    ])

)


assert dimensao(5, 3, 1, 0) == 3

assert dimensao(5, 3, 2, 0) == 2


# =====================================================
# 5. VISUALIZAR DESLOCAMENTOS
# =====================================================

fig, axs = plt.subplots(
    2, 3,
    figsize=(12, 7)
)


desenhar_matriz(
    axs[0, 0],
    imagem5,
    "Entrada 5x5 - stride 1",
    caixa=(0, 0, 3)
)

desenhar_matriz(
    axs[0, 1],
    imagem5,
    "Proxima janela: +1 pixel",
    caixa=(0, 1, 3)
)

desenhar_matriz(
    axs[0, 2],
    saida_stride1,
    "Saida stride 1: 3x3"
)

desenhar_matriz(
    axs[1, 0],
    imagem5,
    "Entrada 5x5 - stride 2",
    caixa=(0, 0, 3)
)

desenhar_matriz(
    axs[1, 1],
    imagem5,
    "Proxima janela: +2 pixels",
    caixa=(0, 2, 3)
)

desenhar_matriz(
    axs[1, 2],
    saida_stride2,
    "Saida stride 2: 2x2"
)


salvar(
    fig,
    "46_stride_1_e_2.png"
)


# =====================================================
# 6. PADDING 0 E PADDING 1
# =====================================================

# O padding acrescenta zeros nas bordas.
#
# Com kernel 3x3:
#
# Padding 0: 5x5 -> 3x3
# Padding 1: 5x5 -> 5x5

print("\n3 - PADDING: BORDAS ARTIFICIAIS")


com_borda = F.pad(
    entrada5,
    (1, 1, 1, 1),
    mode="constant",
    value=0
)


with torch.no_grad():

    saida_padding0 = F.conv2d(
        entrada5,
        kernel,
        padding=0
    )[0, 0].numpy()

    saida_padding1 = F.conv2d(
        entrada5,
        kernel,
        padding=1
    )[0, 0].numpy()


print(
    "Imagem original:",
    tuple(entrada5.shape)
)

print(
    "Imagem com borda de zeros:",
    tuple(com_borda.shape)
)

print(
    "Saida padding 0:",
    saida_padding0.shape
)

print(
    "Saida padding 1:",
    saida_padding1.shape
)


assert saida_padding0.shape == (
    3, 3
)

assert saida_padding1.shape == (
    5, 5
)


# =====================================================
# 7. VISUALIZAR PADDING
# =====================================================

fig, axs = plt.subplots(
    1, 4,
    figsize=(17, 4.5)
)


desenhar_matriz(
    axs[0],
    imagem5,
    "Entrada 5x5"
)

desenhar_matriz(
    axs[1],
    com_borda[0, 0].numpy(),
    "Padding 1: matriz 7x7"
)

desenhar_matriz(
    axs[2],
    saida_padding0,
    "Padding 0: saida 3x3"
)

desenhar_matriz(
    axs[3],
    saida_padding1,
    "Padding 1: saida 5x5"
)


salvar(
    fig,
    "47_padding_0_e_1.png"
)


# =====================================================
# 8. KERNEL_SIZE E PADDING
# =====================================================

# Campo de visao:
#
# Kernel 3x3 -> padding 1
# Kernel 5x5 -> padding 2
# Kernel 7x7 -> padding 3
#
# Todos preservam a resolucao espacial
# quando stride = 1.

print(
    "\n4 - KERNEL_SIZE E PADDING"
)

casos = []


for tamanho_kernel in [3, 5, 7]:

    for padding in [
        0,
        (tamanho_kernel - 1) // 2
    ]:

        previsto = dimensao(
            25,
            tamanho_kernel,
            1,
            padding
        )

        camada = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=tamanho_kernel,
            stride=1,
            padding=padding
        )

        with torch.no_grad():

            real = camada(
                cinza
            ).shape[-1]

        assert real == previsto

        casos.append(
            (
                tamanho_kernel,
                padding,
                real
            )
        )

        print(
            f"25x25 | "
            f"kernel={tamanho_kernel}x{tamanho_kernel} "
            f"padding={padding} "
            f"stride=1 -> {real}x{real}"
        )


# =====================================================
# 9. GRAFICO DO CAMPO DE VISAO
# =====================================================

fig, ax = plt.subplots(
    figsize=(10, 4.5)
)


nomes = [
    f"K{k}, P{p}"
    for k, p, _ in casos
]

valores = [
    valor
    for _, _, valor in casos
]


barras = ax.bar(
    nomes,
    valores
)

ax.axhline(
    25,
    color="black",
    linestyle="--",
    label="Resolucao original = 25"
)


for barra, valor in zip(
    barras,
    valores
):

    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.25,
        str(valor),
        ha="center"
    )


ax.set_ylim(0, 29)

ax.set_ylabel(
    "Altura e largura da saida"
)

ax.set_title(
    "Kernel size e padding alteram a resolucao"
)

ax.legend()


salvar(
    fig,
    "48_kernel_size_padding.png"
)


# =====================================================
# 10. EXERCICIO DA ALURA
# =====================================================

# Pergunta:
#
# Entrada: 224x224
# Kernel: 7x7
# Stride: 7
# Padding: 0
#
# Qual a resolucao da saida?

print(
    "\n5 - EXERCICIO DA TRANSCRICAO"
)

resolucao = dimensao(
    n=224,
    kernel=7,
    stride=7,
    padding=0
)

camada224 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=7,
    stride=7,
    padding=0
)


with torch.no_grad():

    saida224 = camada224(
        torch.rand(
            1, 3, 224, 224
        )
    )


print(
    "Formula: "
    "floor((224 - 7 + 2*0)/7) + 1 =",
    resolucao
)

print(
    "Entrada:",
    (1, 3, 224, 224)
)

print(
    "Saida:",
    tuple(saida224.shape)
)


assert resolucao == 32

assert tuple(saida224.shape) == (
    1, 16, 32, 32
)


print(
    "EXERCICIO ALURA VALIDADO: OK"
)


# =====================================================
# 11. ARQUITETURA CNN E DIMENSOES
# =====================================================

# Como vimos anteriormente:
#
# Entrada -> Conv1 -> Pool1
#         -> Conv2 -> Pool2
#         -> Conv3 -> Pool3
#
# Vamos verificar as dimensoes.

print(
    "\n6 - CNN: CONVOLUCAO E POOLING"
)


rede = nn.ModuleList([

    nn.Conv2d(
        3, 6, 3,
        padding=1
    ),

    nn.MaxPool2d(2),

    nn.Conv2d(
        6, 12, 3,
        padding=1
    ),

    nn.MaxPool2d(2),

    nn.Conv2d(
        12, 18, 3,
        padding=1
    ),

    nn.MaxPool2d(2)

])


x = torch.rand(
    1, 3, 64, 64
)

etapas = [
    (
        "Entrada",
        tuple(x.shape)
    )
]


nomes_camadas = [
    "Conv1",
    "Pool1",
    "Conv2",
    "Pool2",
    "Conv3",
    "Pool3"
]


with torch.no_grad():

    for nome, camada in zip(
        nomes_camadas,
        rede
    ):

        x = camada(x)

        etapas.append(
            (
                nome,
                tuple(x.shape)
            )
        )


for nome, forma in etapas:

    print(
        f"{nome:10s} -> {forma}"
    )


assert etapas[-1][1] == (
    1, 18, 8, 8
)


# =====================================================
# 12. GRAFICO DAS DIMENSOES DA CNN
# =====================================================

fig, ax = plt.subplots(
    figsize=(11, 5.5)
)

ax.axis("off")

ax.set_title(
    "Canais e resolucao espacial ao longo da CNN"
)


for i, (nome, forma) in enumerate(etapas):

    texto = (
        f"{nome:10s} "
        f"N={forma[0]} "
        f"C={forma[1]:2d} "
        f"H={forma[2]:2d} "
        f"W={forma[3]:2d}"
    )

    ax.text(
        0.07,
        0.91 - i * 0.12,
        texto,
        transform=ax.transAxes,
        fontsize=13,
        family="monospace",
        va="center"
    )


salvar(
    fig,
    "49_canais_arquitetura.png"
)


# =====================================================
# 13. RESUMO FINAL
# =====================================================

print("\nFORMULA DA CONVOLUCAO:")

print(
    "X_out = floor((X_in - F + 2*P) / S) + 1"
)

print(
    "F = kernel_size"
)

print(
    "S = stride"
)

print(
    "P = padding"
)

print(
    "\nOBS: para stride > 1, usar floor."
)

print(
    "in_channels = canais de entrada."
)

print(
    "out_channels = filtros e canais de saida."
)

print(
    "kernel_size = campo de visao."
)

print(
    "stride = tamanho do passo."
)

print(
    "padding = preenchimento das bordas."
)

print(
    "\nAULA 13 EXECUTADA COM SUCESSO!"
)
