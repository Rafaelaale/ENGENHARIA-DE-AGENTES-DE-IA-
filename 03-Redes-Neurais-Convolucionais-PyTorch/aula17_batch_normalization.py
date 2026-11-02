# ==========================================================
# ALURA - AULA 17: BATCH NORMALIZATION
# Curso: Redes Neurais Convolucionais com PyTorch
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
SAIDA = BASE / "resultados"
SAIDA.mkdir(exist_ok=True)

torch.manual_seed(17)


def salvar(fig, nome):

    caminho = SAIDA / nome

    fig.tight_layout()
    fig.savefig(caminho, dpi=135)
    plt.close(fig)

    print("Imagem salva:", caminho)


print("=" * 65)
print("AULA 17 - BATCH NORMALIZATION | ALURA")
print("=" * 65)


# ==========================================================
# 2 - PRE-PROCESSAMENTO E NORMALIZACAO
# ==========================================================

# A professora comeca explicando que dados
# centralizados e normalizados podem tornar
# o treinamento mais estavel.
#
# Vamos criar duas caracteristicas numericas
# com medias e escalas diferentes.

pontos = torch.randn(400, 2)

pontos[:, 0] = (
    pontos[:, 0] * 4 + 12
)

pontos[:, 1] = (
    pontos[:, 0] * 0.5
    + pontos[:, 1] * 2
    + 3
)


# Centralizar significa subtrair a media.
central = (
    pontos
    - pontos.mean(dim=0, keepdim=True)
)


# Padronizar significa dividir pelo
# desvio padrao de cada caracteristica.
padronizado = (
    central
    / pontos.std(
        dim=0,
        unbiased=False,
        keepdim=True
    )
)


fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 4)
)


for ax, valores, nome in zip(

    axs,

    [
        pontos,
        central,
        padronizado
    ],

    [
        "Dado original",
        "Subtracao da media",
        "Media 0 e desvio 1"
    ]

):

    a = valores.numpy()

    ax.scatter(
        a[:, 0],
        a[:, 1],
        s=9,
        alpha=0.5
    )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.7
    )

    ax.axvline(
        0,
        color="gray",
        linewidth=0.7
    )

    ax.set_title(nome)

    ax.set_xlabel(
        "Caracteristica 1"
    )

    ax.set_ylabel(
        "Caracteristica 2"
    )


salvar(
    fig,
    "66_batchnorm_centralizar_padronizar.png"
)

print(
    "\n1 - PRE-PROCESSAMENTO:"
    " distribuicao original, centrada e padronizada"
)


# ==========================================================
# 3 - BATCH NORMALIZATION POR CANAL
# ==========================================================

# O PyTorch espera:
#
# N = quantidade de exemplos no batch
# C = numero de canais
# H = altura
# W = largura
#
# Vamos criar um tensor com:
#
# 4 exemplos
# 3 canais
# 8 pixels de altura
# 8 pixels de largura
#
# Cada canal tera uma distribuicao diferente.

base = torch.randn(
    4, 3, 8, 8
)

escalas = torch.tensor([
    2.0,
    5.0,
    0.5
]).view(1, 3, 1, 1)

deslocamentos = torch.tensor([
    10.0,
    -5.0,
    3.0
]).view(1, 3, 1, 1)

x = (
    base * escalas
    + deslocamentos
)


# Criar a camada.
# num_features deve corresponder
# ao numero de canais.

bn = nn.BatchNorm2d(
    num_features=3
)


# Modo de treinamento:
# utiliza as estatisticas do batch.
#
# Nao significa que estamos treinando
# uma rede com otimizador.

bn.train()


with torch.no_grad():

    y = bn(x)


# ==========================================================
# 4 - VALIDACAO MATEMATICA MANUAL
# ==========================================================

# A normalizacao acontece por canal.
#
# Para cada canal, calculamos a media
# sobre N, H e W.
#
# Formula:
#
# y = gamma * (x - media)
#     / sqrt(variancia + eps) + beta
#
# Inicialmente:
# gamma = 1
# beta = 0

media = x.mean(
    dim=(0, 2, 3),
    keepdim=True
)

variancia = x.var(
    dim=(0, 2, 3),
    unbiased=False,
    keepdim=True
)

manualmente = (
    (x - media)
    / torch.sqrt(
        variancia + bn.eps
    )
)


# Conferir a formula manual
# com o resultado do PyTorch.

torch.testing.assert_close(
    y,
    manualmente,
    rtol=1e-5,
    atol=1e-5
)


assert tuple(y.shape) == (
    4, 3, 8, 8
)

assert torch.all(
    torch.abs(
        y.mean(dim=(0, 2, 3))
    ) < 1e-5
)

assert torch.all(
    torch.abs(
        y.var(
            dim=(0, 2, 3),
            unbiased=False
        ) - 1
    ) < 1e-4
)


print(
    "\n2 - BATCH NORMALIZATION POR CANAL"
)

print(
    "Entrada:",
    tuple(x.shape)
)

print(
    "Saida:",
    tuple(y.shape)
)

print(
    "Medias antes:",
    x.mean(
        (0, 2, 3)
    ).tolist()
)

print(
    "Medias depois:",
    y.mean(
        (0, 2, 3)
    ).tolist()
)

print(
    "Desvios depois:",
    y.std(
        (0, 2, 3),
        unbiased=False
    ).tolist()
)

print(
    "VALIDACAO FORMULA MANUAL X BATCHNORM2D: OK"
)


# ==========================================================
# 5 - VISUALIZAR OS TRES CANAIS
# ==========================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 4)
)


for c, ax in enumerate(axs):

    ax.hist(
        x[:, c].flatten().numpy(),
        bins=25,
        alpha=0.5,
        label="Antes"
    )

    ax.hist(
        y[:, c].flatten().numpy(),
        bins=25,
        alpha=0.5,
        label="Depois da BN"
    )

    ax.set_title(
        f"Canal {c + 1}: normalizacao independente"
    )

    ax.legend(fontsize=8)


salvar(
    fig,
    "67_batchnorm_por_canal.png"
)


# ==========================================================
# 6 - GAMMA E BETA
# ==========================================================

# BatchNorm possui parametros aprendiveis:
#
# gamma: escala
# beta: deslocamento
#
# Cada canal tem seu proprio gamma e beta.
#
# Inicialmente:
# gamma = 1
# beta = 0
#
# Vamos modificar os valores manualmente
# apenas para visualizar o comportamento.

bn_ajustada = nn.BatchNorm2d(
    num_features=3
)


with torch.no_grad():

    bn_ajustada.weight.copy_(
        torch.tensor([
            1.5,
            0.5,
            2.0
        ])
    )

    bn_ajustada.bias.copy_(
        torch.tensor([
            -1.0,
            0.5,
            1.0
        ])
    )

    z = bn_ajustada(x)


# ==========================================================
# 7 - VALIDAR GAMMA E BETA
# ==========================================================

manual_z = (
    (x - media)
    / torch.sqrt(
        variancia + bn_ajustada.eps
    )
)

manual_z = (
    manual_z
    * bn_ajustada.weight.view(
        1, 3, 1, 1
    )
    + bn_ajustada.bias.view(
        1, 3, 1, 1
    )
)


torch.testing.assert_close(
    z,
    manual_z,
    rtol=1e-5,
    atol=1e-5
)


print("\n3 - GAMMA E BETA")

print(
    "Gamma inicial:",
    bn.weight.detach().tolist()
)

print(
    "Beta inicial:",
    bn.bias.detach().tolist()
)

print(
    "Gamma ajustado manualmente:",
    bn_ajustada.weight.detach().tolist()
)

print(
    "Beta ajustado manualmente:",
    bn_ajustada.bias.detach().tolist()
)

print(
    "Parametros treinaveis de BN com 3 canais:",
    sum(
        p.numel()
        for p in bn.parameters()
    )
)

print(
    "VALIDACAO GAMMA E BETA: OK"
)


# ==========================================================
# 8 - VISUALIZAR O EFEITO DOS PARAMETROS
# ==========================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 4)
)


for c, ax in enumerate(axs):

    ax.hist(
        y[:, c].flatten().numpy(),
        bins=25,
        alpha=0.55,
        label="gamma=1, beta=0"
    )

    ax.hist(
        z[:, c].flatten().numpy(),
        bins=25,
        alpha=0.55,
        label=(
            f"gamma={bn_ajustada.weight[c].item():g}, "
            f"beta={bn_ajustada.bias[c].item():g}"
        )
    )

    ax.set_title(
        f"Canal {c + 1}"
    )

    ax.legend(fontsize=8)


salvar(
    fig,
    "68_batchnorm_gamma_beta.png"
)


# ==========================================================
# 9 - BLOCO CONV -> BN -> RELU
# ==========================================================

# A professora apresenta a ordem:
#
# Convolucao
#     |
# Batch Normalization
#     |
# ReLU
#
# Vamos demonstrar esse fluxo sem
# treinar os filtros da convolucao.


# Carregar a imagem astronauta.
imagem = torch.from_numpy(
    data.astronaut().copy()
).permute(
    2, 0, 1
).unsqueeze(
    0
).float() / 255.0


# Reduzir para acelerar o exemplo.
imagem = F.interpolate(
    imagem,
    size=(64, 64),
    mode="bilinear",
    align_corners=False
)


# Criar um batch com quatro versoes
# da fotografia.

lote = torch.cat([

    imagem,

    imagem * 0.85,

    (imagem * 1.1).clamp(0, 1),

    (imagem * 0.9 + 0.05).clamp(0, 1)

], dim=0)


# Uma convolucao com seis filtros.
conv = nn.Conv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3,
    padding=1
)


# Batch Normalization com seis canais.
bn6 = nn.BatchNorm2d(
    num_features=6
)


# Exemplo de outra camada BN,
# que receberia doze canais.
#
# Ela possui parametros proprios.

bn12 = nn.BatchNorm2d(
    num_features=12
)


conv.eval()
bn6.train()


with torch.no_grad():

    a = conv(lote)

    b = bn6(a)

    c = F.relu(b)


assert tuple(a.shape) == (
    4, 6, 64, 64
)

assert tuple(b.shape) == (
    4, 6, 64, 64
)

assert tuple(c.shape) == (
    4, 6, 64, 64
)


# Confirmar que sao instancias
# independentes.

assert bn6.weight is not bn12.weight


assert sum(
    p.numel()
    for p in bn6.parameters()
) == 12


assert sum(
    p.numel()
    for p in bn12.parameters()
) == 24


print(
    "\n4 - BLOCO CONV -> BN -> RELU"
)

print(
    "Entrada:",
    tuple(lote.shape)
)

print(
    "Apos Conv2d:",
    tuple(a.shape)
)

print(
    "Apos BatchNorm2d:",
    tuple(b.shape)
)

print(
    "Apos ReLU:",
    tuple(c.shape)
)

print(
    "BN6: 6 canais e 12 parametros aprendiveis"
)

print(
    "BN12: 12 canais e 24 parametros aprendiveis"
)

print(
    "As duas instancias possuem pesos independentes: OK"
)


# ==========================================================
# 10 - VISUALIZAR CONV -> BN -> RELU
# ==========================================================

fig, axs = plt.subplots(
    1, 4,
    figsize=(15, 4)
)


axs[0].imshow(
    imagem[0].permute(
        1, 2, 0
    ).numpy()
)

axs[0].set_title(
    "Entrada RGB"
)


for ax, tensor, nome in zip(

    axs[1:],

    [a, b, c],

    [
        "Conv: 6 mapas",
        "BatchNorm: 6 mapas",
        "ReLU: 6 mapas"
    ]

):

    ax.imshow(
        tensor[0, 0].numpy(),
        cmap="viridis"
    )

    ax.set_title(nome)


for ax in axs:
    ax.axis("off")


salvar(
    fig,
    "69_batchnorm_conv_bn_relu.png"
)


# ==========================================================
# 11 - TRAIN() E EVAL()
# ==========================================================

# Em modo train, a BatchNorm utiliza
# media e variancia do batch atual.
#
# Ela tambem atualiza as estatisticas
# acumuladas running_mean e running_var.
#
# Em modo eval, usa as estatisticas
# acumuladas.
#
# Aqui nao fazemos treinamento de pesos.
# Apenas demonstramos esses dois modos.

bn_exemplo = nn.BatchNorm2d(
    num_features=3
)

bn_exemplo.train()


with torch.no_grad():

    for k in range(12):

        exemplo = (
            torch.randn(
                8, 3, 8, 8
            ) * 2 + 4
        )

        _ = bn_exemplo(exemplo)


    media_acumulada = (
        bn_exemplo.running_mean.clone()
    )

    variancia_acumulada = (
        bn_exemplo.running_var.clone()
    )


    bn_exemplo.eval()


    teste = (
        torch.randn(
            4, 3, 8, 8
        ) * 2 + 4
    )

    teste_bn = bn_exemplo(teste)


    esperado_eval = (

        (
            teste
            - media_acumulada.view(
                1, 3, 1, 1
            )
        )

        / torch.sqrt(

            variancia_acumulada.view(
                1, 3, 1, 1
            ) + bn_exemplo.eps

        )

    )


torch.testing.assert_close(
    teste_bn,
    esperado_eval,
    rtol=1e-5,
    atol=1e-5
)


print("\n5 - MODOS TRAIN E EVAL")

print(
    "running_mean:",
    media_acumulada.tolist()
)

print(
    "running_var:",
    variancia_acumulada.tolist()
)

print(
    "No modo eval, a camada usa "
    "estatisticas acumuladas: OK"
)

print(
    "Nenhum peso foi otimizado ou treinado nesta aula."
)


# ==========================================================
# 12 - DIAGRAMA DO BLOCO CONVOLUCIONAL
# ==========================================================

fig, ax = plt.subplots(
    figsize=(12, 3.6)
)

ax.set_xlim(0, 12)
ax.set_ylim(0, 2.8)
ax.axis("off")


blocos = [

    ("Entrada RGB", 0.3),

    ("Conv2d\n3 -> 6 canais", 3.2),

    ("BatchNorm2d\n6 canais", 6.1),

    ("ReLU\n6 canais", 9.0)

]


for rotulo, x0 in blocos:

    ax.add_patch(

        FancyBboxPatch(

            (x0, 0.85),

            2.5,
            1.2,

            boxstyle="round,pad=0.15",

            linewidth=1.3,

            facecolor="lightsteelblue"

        )

    )

    ax.text(
        x0 + 1.25,
        1.45,
        rotulo,
        ha="center",
        va="center",
        fontsize=11
    )


for x0 in [
    2.8,
    5.7,
    8.6
]:

    ax.annotate(

        "",

        xy=(
            x0 + 0.4,
            1.45
        ),

        xytext=(
            x0,
            1.45
        ),

        arrowprops={
            "arrowstyle": "->",
            "lw": 1.6
        }

    )


ax.set_title(
    "Bloco da transcricao: "
    "Conv -> BatchNorm -> ReLU"
)


salvar(
    fig,
    "70_batchnorm_fluxo_camadas.png"
)


# ==========================================================
# 13 - RESUMO FINAL
# ==========================================================

print("\nCONCEITOS APRENDIDOS")

print(
    "1 - Normalizacao de entrada e "
    "normalizacao de ativacoes sao distintas."
)

print(
    "2 - BatchNorm2d calcula estatisticas "
    "por canal sobre N, H e W."
)

print(
    "3 - Gamma e beta sao parametros "
    "aprendiveis de cada camada."
)

print(
    "4 - BatchNorm preserva o formato "
    "de entrada (N,C,H,W)."
)

print(
    "5 - Cada bloco precisa de sua propria "
    "instancia BatchNorm2d."
)

print(
    "6 - Ordem da transcricao: "
    "Conv -> BatchNorm -> ReLU."
)

print(
    "7 - Train usa estatisticas do batch; "
    "eval usa estatisticas acumuladas."
)

print(
    "8 - Ainda nao houve treinamento da CNN."
)

print(
    "\nAULA 17 EXECUTADA COM SUCESSO!"
)
