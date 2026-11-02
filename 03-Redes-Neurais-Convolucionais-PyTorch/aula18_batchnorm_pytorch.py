# ==========================================================
# ALURA - AULA 18: BATCHNORM NO PYTORCH
# Bloco: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
# ==========================================================

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from skimage import data

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent

RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

torch.manual_seed(18)


def salvar(fig, nome):

    destino = RESULTADOS / nome

    fig.tight_layout()

    fig.savefig(
        destino,
        dpi=125
    )

    plt.close(fig)

    print(
        "Imagem salva:",
        destino
    )


def mostrar(ax, imagem, titulo, cmap="viridis"):

    ax.imshow(
        imagem,
        cmap=cmap
    )

    ax.set_title(
        titulo,
        fontsize=10
    )

    ax.axis("off")


def plano(tensor, n=0, c=0):

    return (
        tensor[n, c]
        .detach()
        .cpu()
        .numpy()
    )


print("=" * 65)
print("AULA 18 - BATCHNORM NO PYTORCH | ALURA")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR A IMAGEM ASTRONAUT
# ==========================================================

# Imagem colorida:
# altura, largura, canais
#
# O PyTorch precisa de:
# batch, canais, altura, largura

foto = data.astronaut()

rgb_tns = (
    torch.from_numpy(
        foto.copy()
    )
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float()
    / 255.0
)


print("\n1 - IMAGEM ASTRONAUT")

print(
    "Imagem NumPy (H,W,C):",
    foto.shape
)

print(
    "Tensor PyTorch (N,C,H,W):",
    tuple(rgb_tns.shape)
)


assert tuple(rgb_tns.shape) == (
    1, 3, 512, 512
)


# ==========================================================
# 3 - CRIAR O BLOCO CONVOLUCIONAL
# ==========================================================

# nn.Sequential funciona como um container.
#
# A entrada passa automaticamente
# por cada camada na ordem definida.
#
# Conv2d: 3 canais -> 32 filtros
# BatchNorm2d: normaliza 32 canais
# ReLU: ativacao nao linear
# MaxPool2d: reduz a resolucao

bloco = nn.Sequential(

    nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        padding=1
    ),

    nn.BatchNorm2d(
        num_features=32
    ),

    nn.ReLU(),

    nn.MaxPool2d(
        kernel_size=2
    )

)


print("\n2 - BLOCO CONVOLUCIONAL")

print(bloco)


# Quando o stride nao e informado,
# MaxPool2d utiliza stride=kernel_size.

assert bloco[3].stride in (
    2,
    (2, 2)
)

assert bloco[1].num_features == 32


# ==========================================================
# 4 - PASSAGEM PELAS CAMADAS
# ==========================================================

# Para uma unica imagem, utilizamos
# eval() para observar a inferencia.
#
# Em eval(), BatchNorm utiliza as
# estatisticas acumuladas.
#
# Como o bloco ainda nao foi treinado,
# estas estatisticas estao nos valores
# iniciais da camada.

bloco.eval()


with torch.no_grad():

    # Primeira operacao: convolucao.
    a_conv = bloco[0](
        rgb_tns
    )

    # Segunda operacao: normalizacao.
    a_bn = bloco[1](
        a_conv
    )

    # Terceira operacao: ReLU.
    a_relu = bloco[2](
        a_bn
    )

    # Executa o Sequential completo.
    saida = bloco(
        rgb_tns
    )


print("\n3 - DIMENSOES DAS CAMADAS")

print(
    "Entrada:",
    tuple(rgb_tns.shape)
)

print(
    "Apos Conv2d:",
    tuple(a_conv.shape)
)

print(
    "Apos BatchNorm2d:",
    tuple(a_bn.shape)
)

print(
    "Apos ReLU:",
    tuple(a_relu.shape)
)

print(
    "Apos MaxPool2d(2):",
    tuple(saida.shape)
)


# ==========================================================
# 5 - VALIDAR A SAIDA DO BLOCO
# ==========================================================

assert tuple(a_conv.shape) == (
    1, 32, 512, 512
)

assert tuple(a_bn.shape) == (
    1, 32, 512, 512
)

assert tuple(a_relu.shape) == (
    1, 32, 512, 512
)

assert tuple(saida.shape) == (
    1, 32, 256, 256
)


with torch.no_grad():

    saida_manual = F.max_pool2d(
        a_relu,
        kernel_size=2
    )


torch.testing.assert_close(
    saida,
    saida_manual
)


print(
    "VALIDACAO DO BLOCO SEQUENCIAL: OK"
)


# ==========================================================
# 6 - TESTAR MAXPOOLING COM KERNEL 10
# ==========================================================

# A professora modifica o tamanho da
# janela do Pooling para 10.
#
# Como stride nao foi informado,
# o deslocamento tambem sera 10.
#
# Entrada 512x512
# Saida 51x51

pool10 = nn.MaxPool2d(
    kernel_size=10
)


with torch.no_grad():

    saida_pool10 = pool10(
        a_relu
    )


print("\n4 - COMPARACAO DE POOLING")

print(
    "MaxPool kernel 2:",
    tuple(saida.shape)
)

print(
    "MaxPool kernel 10:",
    tuple(saida_pool10.shape)
)


assert tuple(saida_pool10.shape) == (
    1, 32, 51, 51
)


print(
    "VALIDACAO POOLING 2 E 10: OK"
)


# ==========================================================
# 7 - GRAFICO DO BLOCO SEQUENCIAL
# ==========================================================

fig, ax = plt.subplots(
    figsize=(13, 3.3)
)

ax.set_xlim(0, 13)
ax.set_ylim(0, 3)
ax.axis("off")


blocos = [

    (
        "Entrada RGB\n3 x 512 x 512",
        0.1
    ),

    (
        "Conv2d\n32 x 512 x 512",
        2.75
    ),

    (
        "BatchNorm2d\n32 x 512 x 512",
        5.4
    ),

    (
        "ReLU\n32 x 512 x 512",
        8.05
    ),

    (
        "MaxPool2d\n32 x 256 x 256",
        10.7
    )

]


for rotulo, x in blocos:

    ax.add_patch(

        FancyBboxPatch(

            (x, 1.05),

            2.1,
            1.2,

            boxstyle="round,pad=0.1",

            linewidth=1.2,

            facecolor="lightsteelblue"

        )

    )

    ax.text(
        x + 1.05,
        1.65,
        rotulo,
        ha="center",
        va="center",
        fontsize=9
    )


for x in [
    2.28,
    4.93,
    7.58,
    10.23
]:

    ax.annotate(

        "",

        xy=(x + 0.42, 1.65),

        xytext=(x, 1.65),

        arrowprops={
            "arrowstyle": "->",
            "lw": 1.5
        }

    )


ax.set_title(
    "nn.Sequential: bloco convolucional completo"
)


salvar(
    fig,
    "71_batchnorm_pytorch_sequencial.png"
)


# ==========================================================
# 8 - VISUALIZAR CADA ETAPA
# ==========================================================

fig, axs = plt.subplots(
    1, 5,
    figsize=(16, 4)
)


mostrar(
    axs[0],
    foto,
    "Entrada RGB",
    cmap=None
)


for ax, tensor, titulo in zip(

    axs[1:],

    [
        a_conv,
        a_bn,
        a_relu,
        saida
    ],

    [
        "Conv2d - mapa 1",
        "BatchNorm - mapa 1",
        "ReLU - mapa 1",
        "Pool2 - mapa 1"
    ]

):

    mostrar(
        ax,
        plano(tensor),
        titulo
    )


salvar(
    fig,
    "72_batchnorm_pytorch_etapas.png"
)


# ==========================================================
# 9 - VISUALIZAR POOLING 2 E 10
# ==========================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(12, 4)
)


mostrar(
    axs[0],
    plano(a_relu),
    "Antes do Pooling\n512 x 512"
)

mostrar(
    axs[1],
    plano(saida),
    "MaxPool2d(2)\n256 x 256"
)

mostrar(
    axs[2],
    plano(saida_pool10),
    "MaxPool2d(10)\n51 x 51"
)


salvar(
    fig,
    "73_batchnorm_pytorch_pool2_pool10.png"
)


# ==========================================================
# 10 - CRIAR MINI-BATCH COM 12 IMAGENS
# ==========================================================

# A professora concatena o mesmo tensor
# varias vezes para produzir 12 imagens.
#
# Vamos fazer a mesma operacao.
#
# Para economizar memoria e facilitar
# a execucao na CPU, reduzimos antes
# cada imagem para 128x128.
#
# Isso nao altera o conceito do exercicio.

foto128 = F.interpolate(
    rgb_tns,
    size=(128, 128),
    mode="bilinear",
    align_corners=False
)


minibatch12 = torch.cat(

    [foto128] * 12,

    dim=0

)


print("\n5 - MINI-BATCH DE 12 IMAGENS")

print(
    "Formato do minibatch:",
    tuple(minibatch12.shape)
)


assert tuple(minibatch12.shape) == (
    12, 3, 128, 128
)


# ==========================================================
# 11 - FORWARD COM BATCH NORMALIZATION
# ==========================================================

# No modo train(), BatchNorm utiliza
# as estatisticas do lote atual.
#
# O bloco passa a ter 12 imagens
# na dimensao N.
#
# torch.no_grad() evita armazenar
# gradientes porque ainda nao vamos
# treinar uma rede neural.

bloco.train()


media_antes = (
    bloco[1]
    .running_mean
    .detach()
    .clone()
)


with torch.no_grad():

    saida12 = bloco(
        minibatch12
    )


media_depois = (
    bloco[1]
    .running_mean
    .detach()
    .clone()
)


print(
    "Entrada do lote:",
    tuple(minibatch12.shape)
)

print(
    "Saida do lote:",
    tuple(saida12.shape)
)

print(
    "running_mean atualizada:",
    not torch.equal(
        media_antes,
        media_depois
    )
)


assert tuple(saida12.shape) == (
    12, 32, 64, 64
)

assert not torch.equal(
    media_antes,
    media_depois
)


print(
    "VALIDACAO MINI-BATCH COM 12 IMAGENS: OK"
)


# ==========================================================
# 12 - VISUALIZAR O MINI-BATCH
# ==========================================================

fig, axs = plt.subplots(
    2, 4,
    figsize=(12, 6)
)


for i, ax in enumerate(axs.flat):

    mostrar(

        ax,

        plano(
            saida12,
            n=i,
            c=0
        ),

        f"Exemplo {i + 1}: mapa 1"

    )


fig.suptitle(
    "12 copias da mesma imagem - sem diversidade de dados"
)


salvar(
    fig,
    "74_batchnorm_pytorch_minibatch12.png"
)


# ==========================================================
# 13 - VISUALIZAR OS MAPAS CONVOLUCIONAIS
# ==========================================================

# A saida continua com 32 canais.
#
# Vamos visualizar 16 deles.

fig, axs = plt.subplots(
    4, 4,
    figsize=(12, 11)
)


for i, ax in enumerate(axs.flat):

    mostrar(

        ax,

        plano(
            saida,
            c=i
        ),

        f"Canal {i + 1} / 32"

    )


fig.suptitle(
    "MaxPooling preserva 32 mapas de caracteristicas"
)


salvar(
    fig,
    "75_batchnorm_pytorch_16_mapas.png"
)


# ==========================================================
# 14 - CONTAR PARAMETROS
# ==========================================================

# BatchNorm possui gamma e beta
# para cada um dos 32 canais.
#
# Total:
# 32 gamma + 32 beta = 64 parametros

params_bn = sum(

    p.numel()

    for p in bloco[1].parameters()

)


params_pool = sum(

    p.numel()

    for p in bloco[3].parameters()

)


assert params_bn == 64
assert params_pool == 0


print("\n6 - PARAMETROS DAS CAMADAS")

print(
    "BatchNorm2d:",
    params_bn,
    "parametros aprendiveis"
)

print(
    "MaxPool2d:",
    params_pool,
    "parametros aprendiveis"
)


# ==========================================================
# 15 - RESUMO DA AULA
# ==========================================================

print("\nCONCEITOS APRENDIDOS")

print(
    "1 - nn.Sequential agrupa as camadas."
)

print(
    "2 - Conv2d(3,32,3,padding=1) "
    "gera 32 mapas de ativacao."
)

print(
    "3 - BatchNorm2d(32) normaliza "
    "os 32 canais."
)

print(
    "4 - ReLU aplica ativacao nao linear."
)

print(
    "5 - MaxPool2d(2) reduz "
    "512x512 para 256x256."
)

print(
    "6 - MaxPool2d(10) reduz "
    "512x512 para 51x51."
)

print(
    "7 - Um minibatch de 12 imagens "
    "gera 12 conjuntos de mapas."
)

print(
    "8 - BatchNorm em train usa "
    "estatisticas do lote."
)

print(
    "9 - BatchNorm em eval usa "
    "estatisticas acumuladas."
)

print(
    "10 - Nenhum otimizador foi executado."
)

print(
    "11 - Ainda nao houve treinamento "
    "dos pesos da CNN."
)


print(
    "\nOBSERVACAO:"
)

print(
    "Repetir 12 vezes a mesma imagem "
    "nao cria exemplos independentes."
)

print(
    "Com batch=1, BatchNorm2d tambem "
    "pode usar as posicoes espaciais "
    "H x W para normalizar."
)


print(
    "\nAULA 18 EXECUTADA COM SUCESSO!"
)
