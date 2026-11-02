# ==========================================================
# ALURA - AULA 15: POOLING
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
from matplotlib.patches import Rectangle


# ==========================================================
# 1. CONFIGURACAO DO PROJETO
# ==========================================================

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados"

SAIDA.mkdir(exist_ok=True)

torch.manual_seed(15)


def salvar(fig, nome):

    destino = SAIDA / nome

    fig.tight_layout()
    fig.savefig(destino, dpi=135)
    plt.close(fig)

    print("Imagem gerada:", destino)


def matriz(ax, valores, titulo, caixa=None):

    a = np.asarray(valores)

    ax.imshow(
        a,
        cmap="gray",
        vmin=min(0.0, float(a.min())),
        vmax=max(1.0, float(a.max()))
    )

    ax.set_title(titulo, fontsize=10)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            ax.text(
                j, i,
                f"{a[i, j]:g}",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                    pad=1
                )
            )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticks(
        np.arange(-0.5, a.shape[1], 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, a.shape[0], 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="gray",
        linewidth=0.7
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    if caixa is not None:

        y, x, tam = caixa

        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                tam, tam,
                fill=False,
                edgecolor="red",
                linewidth=2.5
            )
        )


print("=" * 64)
print("AULA 15 - POOLING | ALURA")
print("=" * 64)


# ==========================================================
# 2. MATRIZ 4 X 4 DA TRANSCRICAO
# ==========================================================

original = np.array([
    [1, 1, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, 3, 4]
], dtype=np.float32)

entrada = torch.from_numpy(
    original
).view(1, 1, 4, 4)

print("\n1 - MAXPOOLING 2X2 COM STRIDE 2")

print("Entrada:")
print(original.astype(int))


# ==========================================================
# 3. DEFINIR CAMADA MAXPOOL2D
# ==========================================================

maxpool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

# Comparacao adicional: media da regiao.
media_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)

with torch.no_grad():

    maximo = maxpool(
        entrada
    )[0, 0].numpy()

    media = media_pool(
        entrada
    )[0, 0].numpy()


print("\nSaida MaxPooling:")
print(maximo.astype(int))

print("\nSaida Average Pooling:")
print(media)


# ==========================================================
# 4. VALIDAR O RESULTADO DA PROFESSORA
# ==========================================================

esperado = np.array([
    [6, 8],
    [3, 4]
], dtype=np.float32)

np.testing.assert_array_equal(
    maximo,
    esperado
)

print("\nCALCULO JANELA POR JANELA")

for i in range(0, 4, 2):

    for j in range(0, 4, 2):

        janela = original[
            i:i + 2,
            j:j + 2
        ]

        print(
            f"Janela ({i},{j})",
            janela.astype(int).tolist(),
            "-> maximo",
            int(janela.max())
        )


print(
    "\nVALIDACAO DOS VALORES 6, 8, 3, 4: OK"
)


# ==========================================================
# 5. VISUALIZAR MAXPOOLING E AVERAGE POOLING
# ==========================================================

fig, axs = plt.subplots(
    1, 4,
    figsize=(14, 4)
)

matriz(
    axs[0],
    original,
    "Entrada 4x4"
)

matriz(
    axs[1],
    original,
    "Primeira janela 2x2",
    caixa=(0, 0, 2)
)

matriz(
    axs[2],
    maximo,
    "MaxPool2d -> 2x2"
)

matriz(
    axs[3],
    media,
    "AvgPool2d -> 2x2"
)

salvar(
    fig,
    "55_pooling_max_matriz.png"
)


# ==========================================================
# 6. KERNEL_SIZE E STRIDE
# ==========================================================

# Segundo exemplo da transcricao:
#
# Uma imagem 4x4 com colunas de valores
# 35, 60, 275 e -280.
#
# Kernel 2, stride 1 -> 3x3
# Kernel 2, stride 2 -> 2x2

colunas = np.tile(
    np.array(
        [35, 60, 275, -280],
        dtype=np.float32
    ),
    (4, 1)
)

t2 = torch.from_numpy(
    colunas
).view(1, 1, 4, 4)


with torch.no_grad():

    stride1 = nn.MaxPool2d(
        kernel_size=2,
        stride=1
    )(t2)[0, 0].numpy()

    stride2 = nn.MaxPool2d(
        kernel_size=2,
        stride=2
    )(t2)[0, 0].numpy()

    k3s1 = nn.MaxPool2d(
        kernel_size=3,
        stride=1
    )(t2)[0, 0].numpy()

    k3padrao = nn.MaxPool2d(
        kernel_size=3
    )(t2)[0, 0].numpy()


np.testing.assert_array_equal(
    stride1,
    np.tile([60, 275, 275], (3, 1))
)

np.testing.assert_array_equal(
    stride2,
    np.tile([60, 275], (2, 1))
)

assert stride1.shape == (3, 3)
assert stride2.shape == (2, 2)
assert k3s1.shape == (2, 2)
assert k3padrao.shape == (1, 1)


print("\n2 - KERNEL_SIZE E STRIDE")

casos = [
    ("Kernel 2, stride 1", stride1),
    ("Kernel 2, stride 2", stride2),
    ("Kernel 3, stride 1", k3s1),
    ("Kernel 3, stride padrao 3", k3padrao)
]

for nome, resultado in casos:

    print(
        nome,
        ":",
        resultado.shape
    )

    print(resultado.astype(int))


print(
    "\nOBS: em MaxPool2d, quando stride "
    "e omitido, ele usa kernel_size."
)


# ==========================================================
# 7. VISUALIZAR STRIDE E KERNEL_SIZE
# ==========================================================

fig, axs = plt.subplots(
    1, 5,
    figsize=(16, 3.5)
)

valores = [
    colunas,
    stride1,
    stride2,
    k3s1,
    k3padrao
]

titulos = [
    "Entrada 4x4",
    "K2 S1 -> 3x3",
    "K2 S2 -> 2x2",
    "K3 S1 -> 2x2",
    "K3 S3 -> 1x1"
]

for ax, valor, titulo in zip(
    axs,
    valores,
    titulos
):

    matriz(
        ax,
        valor,
        titulo
    )

salvar(
    fig,
    "56_pooling_stride_kernel.png"
)


# ==========================================================
# 8. SUBAMOSTRAGEM EM UMA FOTOGRAFIA
# ==========================================================

# Na aula, a professora utiliza uma
# fotografia de Paulo Silveira.
#
# Aqui utilizaremos data.astronaut(),
# que ja acompanha o Scikit-image.
#
# Vamos reduzir:
# 400x400 -> 80x80 -> 20x20.
#
# Esta parte usa interpolacao para ilustrar
# a reducao de resolucao de uma fotografia.
# O MaxPooling em mapas vem depois.

print("\n3 - SUBAMOSTRAGEM DE FOTOGRAFIA")

fotografia = data.astronaut()

foto_tns = (
    torch.from_numpy(
        fotografia.copy()
    )
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float() / 255.0
)


with torch.no_grad():

    foto400 = F.interpolate(
        foto_tns,
        size=(400, 400),
        mode="bilinear",
        align_corners=False
    )

    foto80 = F.interpolate(
        foto400,
        size=(80, 80),
        mode="area"
    )

    foto20 = F.interpolate(
        foto400,
        size=(20, 20),
        mode="area"
    )


def para_rgb(tensor):

    return (
        tensor[0]
        .permute(1, 2, 0)
        .numpy()
        .clip(0, 1)
    )


for nome, tens in [
    ("Original reduzida para 400x400", foto400),
    ("80x80", foto80),
    ("20x20", foto20)
]:

    print(
        nome,
        ":",
        tuple(tens.shape)
    )


# ==========================================================
# 9. VISUALIZAR PERDA DE RESOLUCAO
# ==========================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 5)
)

for ax, tensor, titulo in zip(
    axs,
    [foto400, foto80, foto20],
    ["400x400", "80x80", "20x20"]
):

    ax.imshow(
        para_rgb(tensor),
        interpolation="nearest"
    )

    ax.set_title(titulo)
    ax.axis("off")


salvar(
    fig,
    "57_pooling_subamostragem_fotografia.png"
)


# ==========================================================
# 10. AMPLIAR IMAGENS PARA VER A PIXELIZACAO
# ==========================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 5)
)

for ax, tensor, titulo in zip(
    axs,
    [foto400, foto80, foto20],
    [
        "Original 400x400",
        "80x80 ampliada",
        "20x20 ampliada"
    ]
):

    ax.imshow(
        para_rgb(tensor),
        interpolation="nearest",
        extent=(0, 400, 400, 0)
    )

    ax.set_title(titulo)
    ax.axis("off")


salvar(
    fig,
    "58_pooling_detalhes_perdidos.png"
)


# ==========================================================
# 11. POOLING EM MAPAS DE CARACTERISTICAS
# ==========================================================

# Agora aplicaremos o MaxPooling
# a um mapa de caracteristicas real.
#
# Primeiro criamos bordas com Sobel.
# Depois reduzimos as ativacoes com Pooling.

print("\n4 - POOLING EM MAPAS DE CARACTERISTICAS")

pb = torch.tensor(
    data.brick().astype(np.float32) / 255.0
).view(1, 1, 512, 512)


# Filtro manual de bordas verticais.
sobel = torch.tensor([
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0]
]).view(1, 1, 3, 3)


with torch.no_grad():

    bordas = F.conv2d(
        pb,
        sobel
    ).abs()

    bordas_p1 = nn.MaxPool2d(
        2, 2
    )(bordas)

    bordas_p2 = nn.MaxPool2d(
        2, 2
    )(bordas_p1)


for nome, tensor in [
    ("Conv2d - bordas", bordas),
    ("Pool1", bordas_p1),
    ("Pool2", bordas_p2)
]:

    print(
        nome,
        tuple(tensor.shape)
    )


# ==========================================================
# 12. COMPARAR MAPAS DE CARACTERISTICAS
# ==========================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(14, 5)
)

mapas = [
    bordas,
    bordas_p1,
    bordas_p2
]

titulos = [
    "Convolucao: 510x510",
    "Pool1: 255x255",
    "Pool2: 127x127"
]


for ax, titulo, tensor in zip(
    axs,
    titulos,
    mapas
):

    ax.imshow(
        tensor[0, 0].numpy(),
        cmap="gray"
    )

    ax.set_title(titulo)
    ax.axis("off")


salvar(
    fig,
    "59_pooling_mapas_caracteristicas.png"
)


# ==========================================================
# 13. ARQUITETURA CNN COM POOLING
# ==========================================================

# Estrutura apresentada na transcricao:
#
# Entrada RGB
# Conv1 - 6 filtros
# Pool1
# Conv2 - 12 filtros
# Pool2
# Conv3 - 18 filtros
# Pool3
#
# Os filtros desta rede nao foram treinados.

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

rede.eval()


entrada_cnn = F.interpolate(
    foto400,
    size=(128, 128),
    mode="area"
)


# ==========================================================
# 14. CAMPO RECEPTIVO
# ==========================================================

# Campo receptivo teorico:
# regiao da imagem original que pode
# influenciar uma ativacao.
#
# Pooling aumenta o campo efetivo
# das camadas seguintes.

etapas = [
    (
        "Entrada",
        tuple(entrada_cnn.shape),
        1
    )
]

campo_receptivo = 1
passo_efetivo = 1

x = entrada_cnn


nomes = [
    "Conv1 (6 filtros)",
    "Pool1",
    "Conv2 (12 filtros)",
    "Pool2",
    "Conv3 (18 filtros)",
    "Pool3"
]


with torch.no_grad():

    for nome, camada in zip(
        nomes,
        rede
    ):

        x = camada(x)

        if isinstance(
            camada,
            nn.Conv2d
        ):

            campo_receptivo += (
                2 * passo_efetivo
            )

            x = F.relu(x)

        else:

            campo_receptivo += (
                passo_efetivo
            )

            passo_efetivo *= 2

        etapas.append(
            (
                nome,
                tuple(x.shape),
                campo_receptivo
            )
        )


assert etapas[-1][1] == (
    1, 18, 16, 16
)

assert etapas[-1][2] == 22


print("\n5 - ARQUITETURA CNN SEM TREINAMENTO")

for nome, formato, campo in etapas:

    print(
        f"{nome:20s} -> "
        f"{str(formato):20s} | "
        f"campo receptivo: {campo}x{campo}"
    )


# ==========================================================
# 15. REDUCAO DE CUSTO COMPUTACIONAL
# ==========================================================

antes = 6 * 128 * 128
depois = 6 * 64 * 64

reducao = (
    1 - depois / antes
) * 100


print("\n6 - REDUCAO DAS ATIVACOES")

print(
    "Conv1:",
    antes,
    "elementos"
)

print(
    "Pool1:",
    depois,
    "elementos"
)

print(
    "Reducao:",
    f"{reducao:.1f}%"
)

print(
    "Parametros treinaveis do MaxPooling: 0"
)


# ==========================================================
# 16. GRAFICO DA ARQUITETURA
# ==========================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.axis("off")

ax.set_title(
    "CNN: convolucao alternada com Pooling",
    fontsize=13
)


for i, (nome, forma, campo) in enumerate(etapas):

    texto = (
        f"{nome:20s} | "
        f"{forma[1]:2d} canais x "
        f"{forma[2]:3d} x "
        f"{forma[3]:3d} | "
        f"campo {campo}x{campo}"
    )

    ax.text(
        0.06,
        0.90 - i * 0.12,
        texto,
        transform=ax.transAxes,
        family="monospace",
        fontsize=11
    )


salvar(
    fig,
    "60_pooling_arquitetura_cnn.png"
)


# ==========================================================
# 17. VALIDACOES FINAIS
# ==========================================================

assert maximo.shape == (2, 2)
assert stride1.shape == (3, 3)
assert stride2.shape == (2, 2)

assert tuple(bordas.shape) == (
    1, 1, 510, 510
)

assert tuple(bordas_p1.shape) == (
    1, 1, 255, 255
)

assert tuple(bordas_p2.shape) == (
    1, 1, 127, 127
)


print("\n7 - RESUMO DA AULA")

print(
    "1 - Pooling realiza subamostragem."
)

print(
    "2 - MaxPooling escolhe a maior ativacao."
)

print(
    "3 - Kernel_size define a janela."
)

print(
    "4 - Stride define o deslocamento."
)

print(
    "5 - Pooling preserva os canais."
)

print(
    "6 - Pooling reduz a resolucao espacial."
)

print(
    "7 - Pooling nao possui pesos treinaveis."
)

print(
    "8 - Pooling amplia o campo receptivo "
    "das camadas seguintes."
)

print(
    "9 - A rede demonstrativa nao foi treinada."
)

print("\nAULA 15 EXECUTADA COM SUCESSO!")
