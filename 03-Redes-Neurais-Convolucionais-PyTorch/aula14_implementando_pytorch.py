# ALURA - AULA 14: IMPLEMENTANDO CONV2D EM PYTORCH
# Curso: Redes Neurais Convolucionais com PyTorch

from pathlib import Path

import numpy as np
import torch
from torch import nn
from skimage import data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# 1. CONFIGURACAO
# =====================================================

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(14)


def salvar(fig, nome):
    caminho = RESULTADOS / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=125)
    plt.close(fig)
    print("Imagem gerada:", caminho)


def mostrar(ax, matriz, titulo, cmap="gray"):
    ax.imshow(matriz, cmap=cmap)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")


def visualizar_16_mapas(tensor, nome):
    fig, eixos = plt.subplots(
        4, 4,
        figsize=(12, 12)
    )

    for i, ax in enumerate(eixos.flat):

        mapa = (
            tensor[0, i]
            .detach()
            .cpu()
            .numpy()
        )

        mostrar(
            ax,
            mapa,
            f"Filtro {i + 1} | {mapa.shape}"
        )

    fig.suptitle(
        "16 mapas de ativacao - sem treinamento",
        fontsize=14
    )

    salvar(fig, nome)


print("=" * 65)
print("AULA 14 - IMPLEMENTANDO EM PYTORCH | ALURA")
print("=" * 65)


# =====================================================
# 2. CARREGAR AS DUAS IMAGENS
# =====================================================

# Imagem preto e branco.
pb = data.brick()

# Imagem colorida RGB.
rgb = data.astronaut()

print("\n1 - IMAGENS")

print(
    "Preto e branco:",
    pb.shape
)

print(
    "RGB:",
    rgb.shape
)

print(
    "Tipo dos dados:",
    type(pb),
    type(rgb)
)

assert pb.shape == (512, 512)
assert rgb.shape == (512, 512, 3)


fig, axs = plt.subplots(
    1, 2,
    figsize=(11, 5)
)

mostrar(
    axs[0],
    pb,
    "Tijolos - 1 canal"
)

mostrar(
    axs[1],
    rgb,
    "Astronauta - RGB"
)

salvar(
    fig,
    "50_aula14_imagens_entrada.png"
)


# =====================================================
# 3. CRIAR PRIMEIRA CAMADA CONV2D
# =====================================================

# in_channels = 1
# out_channels = 16
# kernel_size = 3
# stride = 1
# padding = 0

conv_pb = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3
)

print("\n2 - CAMADA CONVOLUCIONAL PB")

print(
    "Camada:",
    conv_pb
)

print(
    "Pesos:",
    tuple(conv_pb.weight.shape)
)


# =====================================================
# 4. CONVERTER A IMAGEM EM TENSOR
# =====================================================

# Array: (512, 512)
# Tensor: (1, 1, 512, 512)
#
# Ordem N, C, H, W:
# N = quantidade de imagens
# C = quantidade de canais
# H = altura
# W = largura

pb_tns = torch.tensor(
    pb,
    dtype=torch.float32
)

pb_tns = pb_tns.view(
    1,
    1,
    pb.shape[0],
    pb.shape[1]
)

print("\n3 - TENSOR PRETO E BRANCO")

print(
    "Formato:",
    tuple(pb_tns.shape)
)


# =====================================================
# 5. EXECUTAR O FORWARD
# =====================================================

# A camada recebe o tensor e produz
# os 16 mapas de caracteristicas.

with torch.no_grad():

    mapa_pb_sem_padding = conv_pb(
        pb_tns
    )

print(
    "Saida PB kernel 3 padding 0:",
    tuple(mapa_pb_sem_padding.shape)
)

assert tuple(
    mapa_pb_sem_padding.shape
) == (1, 16, 510, 510)


# =====================================================
# 6. TESTAR PADDING = 1
# =====================================================

conv_pb_padding = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    padding=1
)

with torch.no_grad():

    mapa_pb_com_padding = conv_pb_padding(
        pb_tns
    )

print(
    "Saida PB kernel 3 padding 1:",
    tuple(mapa_pb_com_padding.shape)
)

assert tuple(
    mapa_pb_com_padding.shape
) == (1, 16, 512, 512)


# =====================================================
# 7. VISUALIZAR A CONVOLUCAO PB
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

mostrar(
    axs[0],
    pb,
    "Original 512x512"
)

mostrar(
    axs[1],
    mapa_pb_sem_padding[0, 0].numpy(),
    "Conv2d 3x3 - sem padding"
)

mostrar(
    axs[2],
    mapa_pb_com_padding[0, 0].numpy(),
    "Conv2d 3x3 - padding 1"
)

salvar(
    fig,
    "51_aula14_pb_padding.png"
)


# =====================================================
# 8. CONVERTER IMAGEM RGB EM TENSOR
# =====================================================

# A imagem RGB vem como:
# (altura, largura, canais)
#
# Precisamos converter para:
# (batch, canais, altura, largura)

print("\n4 - CONVERSAO RGB")

rgb_tns = torch.tensor(
    rgb,
    dtype=torch.float32
)

print(
    "Antes:",
    tuple(rgb_tns.shape)
)

# Colocar canais na primeira dimensao.
rgb_tns = rgb_tns.permute(
    2, 0, 1
)

print(
    "Depois do permute:",
    tuple(rgb_tns.shape)
)

# Criar dimensao do batch.
rgb_tns = rgb_tns.unsqueeze(0)

print(
    "Depois do unsqueeze:",
    tuple(rgb_tns.shape)
)

assert tuple(rgb_tns.shape) == (
    1, 3, 512, 512
)


# =====================================================
# 9. REPRODUZIR O ERRO DA TRANSCRICAO
# =====================================================

# A camada conv_pb espera somente 1 canal.
# Mas a imagem RGB possui 3 canais.
#
# Vamos capturar o erro para que o
# programa continue normalmente.

print("\n5 - ERRO DE CANAIS")

try:

    with torch.no_grad():

        conv_pb(rgb_tns)

except RuntimeError as erro:

    print(
        "ERRO ESPERADO:"
    )

    print(
        "Camada espera 1 canal, "
        "mas recebeu 3 canais."
    )

    print(
        "Mensagem do PyTorch:"
    )

    print(
        str(erro).splitlines()[0]
    )

else:

    raise AssertionError(
        "Era esperado um erro de canais."
    )

print(
    "\nCorrigindo in_channels para 3..."
)


# =====================================================
# 10. CRIAR CAMADA CORRETA PARA RGB
# =====================================================

conv_rgb = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=0
)

with torch.no_grad():

    mapa_rgb3 = conv_rgb(
        rgb_tns
    )

print("\n6 - CONVOLUCAO RGB")

print(
    "Pesos:",
    tuple(conv_rgb.weight.shape)
)

print(
    "Saida 3x3 padding 0:",
    tuple(mapa_rgb3.shape)
)

assert tuple(
    conv_rgb.weight.shape
) == (16, 3, 3, 3)

assert tuple(
    mapa_rgb3.shape
) == (1, 16, 510, 510)


# =====================================================
# 11. TESTAR RGB COM PADDING = 1
# =====================================================

conv_rgb_p1 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)

with torch.no_grad():

    mapa_rgb3_p1 = conv_rgb_p1(
        rgb_tns
    )

print(
    "Saida 3x3 padding 1:",
    tuple(mapa_rgb3_p1.shape)
)

assert tuple(
    mapa_rgb3_p1.shape
) == (1, 16, 512, 512)


# =====================================================
# 12. KERNEL 5X5 SEM PADDING
# =====================================================

conv_rgb5 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=5,
    padding=0
)

with torch.no_grad():

    mapa_rgb5 = conv_rgb5(
        rgb_tns
    )

print("\n7 - KERNEL 5X5")

print(
    "Saida 5x5 padding 0:",
    tuple(mapa_rgb5.shape)
)

assert tuple(
    mapa_rgb5.shape
) == (1, 16, 508, 508)


# =====================================================
# 13. KERNEL 5X5 COM PADDING = 2
# =====================================================

conv_rgb5_p2 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=5,
    padding=2
)

with torch.no_grad():

    mapa_rgb5_p2 = conv_rgb5_p2(
        rgb_tns
    )

print(
    "Saida 5x5 padding 2:",
    tuple(mapa_rgb5_p2.shape)
)

assert tuple(
    mapa_rgb5_p2.shape
) == (1, 16, 512, 512)


# =====================================================
# 14. VISUALIZAR KERNEL RGB 3X3
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

mostrar(
    axs[0],
    rgb,
    "Imagem RGB original"
)

mostrar(
    axs[1],
    mapa_rgb3[0, 0].numpy(),
    "Kernel 3x3 padding 0"
)

mostrar(
    axs[2],
    mapa_rgb3_p1[0, 0].numpy(),
    "Kernel 3x3 padding 1"
)

salvar(
    fig,
    "52_aula14_rgb_kernel3.png"
)


# =====================================================
# 15. VISUALIZAR KERNEL RGB 5X5
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

mostrar(
    axs[0],
    rgb,
    "Imagem RGB original"
)

mostrar(
    axs[1],
    mapa_rgb5[0, 0].numpy(),
    "Kernel 5x5 padding 0"
)

mostrar(
    axs[2],
    mapa_rgb5_p2[0, 0].numpy(),
    "Kernel 5x5 padding 2"
)

salvar(
    fig,
    "53_aula14_rgb_kernel5.png"
)


# =====================================================
# 16. VISUALIZAR OS 16 MAPAS DE ATIVACAO
# =====================================================

visualizar_16_mapas(
    mapa_rgb3,
    "54_aula14_16_mapas_rgb.png"
)


# =====================================================
# 17. VALIDACAO FINAL
# =====================================================

print("\n8 - RESUMO DAS DIMENSOES")

casos = [
    ("PB 3x3 P0", mapa_pb_sem_padding),
    ("PB 3x3 P1", mapa_pb_com_padding),
    ("RGB 3x3 P0", mapa_rgb3),
    ("RGB 3x3 P1", mapa_rgb3_p1),
    ("RGB 5x5 P0", mapa_rgb5),
    ("RGB 5x5 P2", mapa_rgb5_p2)
]

for nome, tensor in casos:

    print(
        f"{nome:12s}: {tuple(tensor.shape)}"
    )

print(
    "\nOBSERVACAO:"
)

print(
    "Os 16 filtros ainda possuem pesos aleatorios."
)

print(
    "Nao houve treinamento nesta aula."
)

print(
    "\nAULA 14 EXECUTADA COM SUCESSO!"
)
