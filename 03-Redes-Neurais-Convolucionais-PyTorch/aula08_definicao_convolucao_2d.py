# ==========================================================
# ALURA - AULA 08: DEFINICAO DE CONVOLUCAO 2D
# Curso: Redes Neurais Convolucionais com PyTorch
# ==========================================================

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from scipy.signal import convolve2d

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------
# 1. CONFIGURACAO DO PROJETO
# ----------------------------------------------------------

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AULA 08 - CONVOLUCAO 2D")
print("=" * 60)


# ----------------------------------------------------------
# 2. IMAGEM 6 X 6 - EXEMPLO DA ALURA
# ----------------------------------------------------------

# Intensidades dos pixels:
# maior valor = mais claro
# menor valor = mais escuro

imagem = np.array([
    [90, 85, 80, 20, 15, 10],
    [90, 85, 80, 20, 15, 10],
    [90, 85, 80, 20, 15, 10],
    [90, 85, 80, 20, 15, 10],
    [90, 85, 80, 20, 15, 10],
    [90, 85, 80, 20, 15, 10]
], dtype=np.float32)

print("\nIMAGEM ORIGINAL:")
print(imagem.astype(int))
print("Dimensoes:", imagem.shape)


# ----------------------------------------------------------
# 3. KERNEL 3 X 3
# ----------------------------------------------------------

# Kernel invertido apresentado na aula:
# procura bordas verticais do claro para o escuro.

kernel_invertido = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

# A convolucao matematica inverte o kernel.
kernel_original = np.flip(
    kernel_invertido,
    axis=(0, 1)
).copy()

print("\nKERNEL ORIGINAL:")
print(kernel_original.astype(int))

print("\nKERNEL INVERTIDO:")
print(kernel_invertido.astype(int))


# ----------------------------------------------------------
# 4. DIMENSOES DA CONVOLUCAO
# ----------------------------------------------------------

altura, largura = imagem.shape
kh, kw = kernel_invertido.shape

# Sem padding e com stride = 1.
saida_altura = altura - kh + 1
saida_largura = largura - kw + 1

mapa_manual = np.zeros(
    (saida_altura, saida_largura),
    dtype=np.float32
)

print("\nDIMENSOES:")
print("Entrada:", imagem.shape)
print("Kernel:", kernel_invertido.shape)
print("Saida:", mapa_manual.shape)


# ----------------------------------------------------------
# 5. CONVOLUCAO 2D MANUAL
# ----------------------------------------------------------

print("\nCONVOLUCAO MANUAL")

for linha in range(saida_altura):

    for coluna in range(saida_largura):

        # Extrair janela 3 x 3.
        janela = imagem[
            linha:linha + kh,
            coluna:coluna + kw
        ]

        # Multiplicacao pixel a pixel.
        produtos = janela * kernel_invertido

        # Somatorio.
        resultado = np.sum(produtos)

        mapa_manual[linha, coluna] = resultado

        print(
            f"Posicao ({linha}, {coluna}) "
            f"= {resultado:.0f}"
        )

        if (linha, coluna) in [(0, 0), (0, 1)]:

            print("Janela:")
            print(janela.astype(int))

            print("Multiplicacoes:")
            print(produtos.astype(int))

            print("Soma:", int(resultado))


# ----------------------------------------------------------
# 6. MAPA DE CARACTERISTICAS
# ----------------------------------------------------------

print("\nMAPA DE CARACTERISTICAS:")
print(mapa_manual.astype(int))

mapa_esperado = np.array([
    [30, 195, 195, 30],
    [30, 195, 195, 30],
    [30, 195, 195, 30],
    [30, 195, 195, 30]
], dtype=np.float32)

np.testing.assert_array_equal(
    mapa_manual,
    mapa_esperado
)

print("\nVALIDACAO ALURA: OK")


# ----------------------------------------------------------
# 7. COMPARACAO COM SCIPY
# ----------------------------------------------------------

resultado_scipy = convolve2d(
    imagem,
    kernel_original,
    mode="valid"
)

np.testing.assert_allclose(
    mapa_manual,
    resultado_scipy,
    atol=1e-6
)

print("VALIDACAO SCIPY: OK")


# ----------------------------------------------------------
# 8. CONVOLUCAO 2D COM PYTORCH
# ----------------------------------------------------------

# Formato PyTorch:
# (batch, canais, altura, largura)

tensor_imagem = (
    torch.from_numpy(imagem.copy())
    .unsqueeze(0)
    .unsqueeze(0)
)

tensor_kernel = (
    torch.from_numpy(kernel_invertido.copy())
    .unsqueeze(0)
    .unsqueeze(0)
)

# Conv2d calcula correlacao cruzada.
# Utilizamos o kernel invertido para reproduzir
# a convolucao matematica da transcricao.

with torch.no_grad():

    resultado_torch = F.conv2d(
        tensor_imagem,
        tensor_kernel,
        stride=1,
        padding=0
    )

np.testing.assert_allclose(
    mapa_manual,
    resultado_torch[0, 0].numpy(),
    atol=1e-5
)

print("\nVALIDACAO PYTORCH: OK")
print("Entrada:", tuple(tensor_imagem.shape))
print("Saida:", tuple(resultado_torch.shape))


# ----------------------------------------------------------
# 9. FUNCAO PARA DESENHAR MATRIZES
# ----------------------------------------------------------

def desenhar_matriz(
    ax,
    matriz,
    titulo,
    minimo=None,
    maximo=None
):

    ax.imshow(
        matriz,
        cmap="gray",
        vmin=minimo,
        vmax=maximo
    )

    ax.set_title(titulo)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):

            ax.text(
                j, i,
                f"{matriz[i, j]:.0f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none"
                }
            )

    ax.set_xticks(range(matriz.shape[1]))
    ax.set_yticks(range(matriz.shape[0]))

    ax.set_xticks(
        np.arange(-0.5, matriz.shape[1], 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, matriz.shape[0], 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="gray",
        linewidth=0.6
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )


# ----------------------------------------------------------
# 10. GRAFICO: IMAGEM, KERNEL E RESULTADO
# ----------------------------------------------------------

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

desenhar_matriz(
    axs[0],
    imagem,
    "Imagem original 6 x 6",
    minimo=0,
    maximo=100
)

desenhar_matriz(
    axs[1],
    kernel_invertido,
    "Kernel invertido 3 x 3",
    minimo=-1,
    maximo=1
)

desenhar_matriz(
    axs[2],
    mapa_manual,
    "Mapa de caracteristicas 4 x 4",
    minimo=0,
    maximo=200
)

fig.tight_layout()

arquivo1 = (
    RESULTADOS / "22_convolucao_2d_matrizes.png"
)

fig.savefig(arquivo1, dpi=150)
plt.close(fig)


# ----------------------------------------------------------
# 11. GRAFICO: DESLOCAMENTO DO KERNEL EM X E Y
# ----------------------------------------------------------

posicoes = [
    (0, 0),
    (0, 1),
    (1, 0),
    (3, 3)
]

fig, axs = plt.subplots(
    2, 4,
    figsize=(16, 8)
)

for indice, (linha, coluna) in enumerate(posicoes):

    ax1 = axs[0, indice]

    desenhar_matriz(
        ax1,
        imagem,
        f"Janela ({linha},{coluna})",
        minimo=0,
        maximo=100
    )

    ax1.add_patch(
        Rectangle(
            (coluna - 0.5, linha - 0.5),
            3, 3,
            fill=False,
            edgecolor="red",
            linewidth=3
        )
    )

    ax2 = axs[1, indice]

    desenhar_matriz(
        ax2,
        mapa_manual,
        f"Saida = {mapa_manual[linha, coluna]:.0f}",
        minimo=0,
        maximo=200
    )

    ax2.add_patch(
        Rectangle(
            (coluna - 0.5, linha - 0.5),
            1, 1,
            fill=False,
            edgecolor="red",
            linewidth=3
        )
    )

fig.tight_layout()

arquivo2 = (
    RESULTADOS / "23_convolucao_2d_deslocamentos.png"
)

fig.savefig(arquivo2, dpi=150)
plt.close(fig)


# ----------------------------------------------------------
# 12. CAMADA CNN COM PESOS ALEATORIOS
# ----------------------------------------------------------

# Exemplo introdutorio de filtros aprendiveis.
# A rede ainda nao sera treinada nesta aula.

torch.manual_seed(7)

camada = nn.Conv2d(
    in_channels=1,
    out_channels=2,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

with torch.no_grad():
    mapas = camada(tensor_imagem)

print("\nCNN - FILTROS APRENDIVEIS")
print("Pesos:", tuple(camada.weight.shape))
print("Mapas:", tuple(mapas.shape))

fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 4)
)

axs[0].imshow(
    mapa_manual,
    cmap="gray"
)
axs[0].set_title("Filtro de borda manual")

axs[1].imshow(
    mapas[0, 0].numpy(),
    cmap="viridis"
)
axs[1].set_title("Filtro aleatorio 1")

axs[2].imshow(
    mapas[0, 1].numpy(),
    cmap="viridis"
)
axs[2].set_title("Filtro aleatorio 2")

for ax in axs:
    ax.axis("off")

fig.tight_layout()

arquivo3 = (
    RESULTADOS / "24_convolucao_2d_filtros.png"
)

fig.savefig(arquivo3, dpi=150)
plt.close(fig)


# ----------------------------------------------------------
# 13. FINALIZACAO
# ----------------------------------------------------------

print("\nRESUMO:")
print("Imagem 6 x 6")
print("Kernel 3 x 3")
print("Stride = 1")
print("Padding = 0")
print("Mapa resultante 4 x 4")
print("Formula: 6 - 3 + 1 = 4")

print("\nIMAGENS GERADAS:")
print(arquivo1)
print(arquivo2)
print(arquivo3)

print("\nAULA 08 EXECUTADA COM SUCESSO!")
