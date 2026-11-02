# ALURA - AULA 09: APLICACAO COM SCIPY (CONVOLUCAO 2D)
# Curso: Redes Neurais Convolucionais com PyTorch

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from skimage import data

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


# =====================================================
# CONFIGURACAO DO PROJETO
# =====================================================

PASTA = Path(__file__).resolve().parent / "resultados"
PASTA.mkdir(parents=True, exist_ok=True)


def salvar(fig, arquivo):
    caminho = PASTA / arquivo
    fig.tight_layout()
    fig.savefig(caminho, dpi=135)
    plt.close(fig)
    print("Imagem salva:", caminho)


def exibir_kernel(ax, matriz, titulo):
    ax.imshow(matriz, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(titulo)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(int(matriz[i, j])),
                ha="center",
                va="center",
                fontsize=13,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none"
                }
            )


def pintar_ativacao(ax, mapa, titulo, absoluto=False):

    escala = max(
        1.0,
        float(np.percentile(np.abs(mapa), 98))
    )

    if absoluto:
        ax.imshow(
            np.abs(mapa),
            cmap="gray",
            vmin=0,
            vmax=escala
        )
    else:
        ax.imshow(
            mapa,
            cmap="seismic",
            vmin=-escala,
            vmax=escala
        )

    ax.set_title(titulo)
    ax.axis("off")


print("=" * 60)
print("AULA 09 - APLICACAO COM SCIPY: CONVOLUCAO 2D")
print("=" * 60)


# =====================================================
# ETAPA 1 - CARREGAR IMAGEM DE TIJOLOS
# =====================================================

# A imagem ja vem incluida no scikit-image.
# Nao precisa baixar nada da internet.

imagem = data.brick().astype(np.float32)

print("\nIMAGEM DE TIJOLOS")
print("Formato:", imagem.shape)
print("Tipo:", imagem.dtype)
print("Menor intensidade:", imagem.min())
print("Maior intensidade:", imagem.max())


# =====================================================
# ETAPA 2 - DEFINIR KERNELS
# =====================================================

# Procura transicoes verticais.
# As colunas possuem pesos diferentes.

kernel_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)


# Procura transicoes horizontais.
# As linhas possuem pesos diferentes.

kernel_horizontal = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)


print("\nKERNEL VERTICAL:")
print(kernel_vertical)

print("\nKERNEL HORIZONTAL:")
print(kernel_horizontal)

print("\nKERNEL VERTICAL INVERTIDO:")
print(np.flip(kernel_vertical, (0, 1)))

print("\nKERNEL HORIZONTAL INVERTIDO:")
print(np.flip(kernel_horizontal, (0, 1)))


# =====================================================
# ETAPA 3 - VISUALIZAR IMAGEM E KERNELS
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

axs[0].imshow(
    imagem,
    cmap="gray",
    vmin=0,
    vmax=255
)

axs[0].set_title("Imagem de tijolos")
axs[0].axis("off")

exibir_kernel(
    axs[1],
    kernel_vertical,
    "Kernel vertical 3x3"
)

exibir_kernel(
    axs[2],
    kernel_horizontal,
    "Kernel horizontal 3x3"
)

salvar(
    fig,
    "25_tijolos_e_kernels.png"
)


# =====================================================
# ETAPA 4 - CONVOLUCAO COM SCIPY
# =====================================================

# mode="valid":
# opera somente nas janelas inteiramente
# contidas na imagem, sem padding.

mapa_vertical = convolve2d(
    imagem,
    kernel_vertical,
    mode="valid"
)

mapa_horizontal = convolve2d(
    imagem,
    kernel_horizontal,
    mode="valid"
)


print("\nCONVOLUCAO 2D COM SCIPY")

print("Entrada:", imagem.shape)
print("Mapa vertical:", mapa_vertical.shape)
print("Mapa horizontal:", mapa_horizontal.shape)

assert mapa_vertical.shape == (
    imagem.shape[0] - 2,
    imagem.shape[1] - 2
)

assert mapa_horizontal.shape == mapa_vertical.shape


# =====================================================
# ETAPA 5 - CALCULO MANUAL DE UMA JANELA 3x3
# =====================================================

janela = imagem[:3, :3]

# Inversao nos dois eixos.
kernel_invertido = np.flip(
    kernel_vertical,
    (0, 1)
)

produtos = janela * kernel_invertido

resultado_manual = float(produtos.sum())

print("\nPRIMEIRA JANELA 3x3:")
print(janela)

print("\nMULTIPLICACOES:")
print(produtos)

print("\nSOMA MANUAL:", resultado_manual)

print(
    "RESULTADO SCIPY:",
    float(mapa_vertical[0, 0])
)

np.testing.assert_allclose(
    resultado_manual,
    mapa_vertical[0, 0],
    atol=1e-4
)

print("VALIDACAO MANUAL X SCIPY: OK")


# =====================================================
# ETAPA 6 - VALIDACAO COM PYTORCH
# =====================================================

# PyTorch utiliza N, C, H, W.
# Uma imagem, um canal, 512x512.

entrada = (
    torch.from_numpy(imagem)
    .unsqueeze(0)
    .unsqueeze(0)
)

# O Conv2d do PyTorch usa correlacao cruzada.
# Inverter o kernel reproduz a convolucao
# matematica realizada pelo SciPy.

peso_vertical = torch.from_numpy(
    np.flip(kernel_vertical, (0, 1)).copy()
).unsqueeze(0).unsqueeze(0)

peso_horizontal = torch.from_numpy(
    np.flip(kernel_horizontal, (0, 1)).copy()
).unsqueeze(0).unsqueeze(0)


with torch.no_grad():

    saida_vertical = F.conv2d(
        entrada,
        peso_vertical
    )

    saida_horizontal = F.conv2d(
        entrada,
        peso_horizontal
    )


np.testing.assert_allclose(
    saida_vertical[0, 0].numpy(),
    mapa_vertical,
    atol=1e-4
)

np.testing.assert_allclose(
    saida_horizontal[0, 0].numpy(),
    mapa_horizontal,
    atol=1e-4
)

print("\nVALIDACAO SCIPY X PYTORCH: OK")

print("Tensor entrada:", tuple(entrada.shape))
print("Tensor saida:", tuple(saida_vertical.shape))


# =====================================================
# ETAPA 7 - BORDAS VERTICAIS E HORIZONTAIS
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(16, 5)
)

axs[0].imshow(
    imagem,
    cmap="gray"
)

axs[0].set_title("Imagem original")
axs[0].axis("off")

pintar_ativacao(
    axs[1],
    mapa_vertical,
    "Bordas verticais: ativacoes"
)

pintar_ativacao(
    axs[2],
    mapa_horizontal,
    "Bordas horizontais: ativacoes"
)

salvar(
    fig,
    "26_tijolos_bordas_verticais_horizontais.png"
)


# =====================================================
# ETAPA 8 - VALOR ABSOLUTO DAS ATIVACOES
# =====================================================

# O valor absoluto remove a diferenca de sinal,
# destacando bordas nos dois sentidos.

fig, axs = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

pintar_ativacao(
    axs[0],
    mapa_vertical,
    "Bordas verticais: valor absoluto",
    absoluto=True
)

pintar_ativacao(
    axs[1],
    mapa_horizontal,
    "Bordas horizontais: valor absoluto",
    absoluto=True
)

salvar(
    fig,
    "27_tijolos_bordas_valor_absoluto.png"
)


# =====================================================
# ETAPA 9 - TESTAR ATIVACOES POSITIVAS E NEGATIVAS
# =====================================================

# A professora utiliza a logo da Alura.
# Aqui criamos uma imagem didatica com texto
# de alto contraste, sem depender de download.
# Nao se trata do logotipo oficial.

ilustracao = Image.new(
    "L",
    (640, 240),
    color=245
)

desenho = ImageDraw.Draw(ilustracao)

try:
    fonte = ImageFont.truetype(
        "arialbd.ttf",
        105
    )
except OSError:
    fonte = ImageFont.load_default(size=95)

desenho.text(
    (65, 55),
    "ALURA",
    font=fonte,
    fill=15
)

desenho.rectangle(
    (12, 12, 626, 225),
    outline=25,
    width=5
)

imagem_texto = np.asarray(
    ilustracao,
    dtype=np.float32
)


ativacao_texto = convolve2d(
    imagem_texto,
    kernel_vertical,
    mode="valid"
)

positivas = np.maximum(
    ativacao_texto,
    0
)

negativas = np.minimum(
    ativacao_texto,
    0
)


print("\nATIVACOES POSITIVAS E NEGATIVAS")

print(
    "Menor ativacao:",
    float(ativacao_texto.min())
)

print(
    "Maior ativacao:",
    float(ativacao_texto.max())
)


fig, axs = plt.subplots(
    2, 2,
    figsize=(12, 7)
)

axs[0, 0].imshow(
    imagem_texto,
    cmap="gray",
    vmin=0,
    vmax=255
)

axs[0, 0].set_title(
    "Texto ilustrativo - nao e logotipo oficial"
)

axs[0, 0].axis("off")

pintar_ativacao(
    axs[0, 1],
    ativacao_texto,
    "Ativacoes positivas e negativas"
)

pintar_ativacao(
    axs[1, 0],
    positivas,
    "Somente ativacoes positivas",
    absoluto=True
)

pintar_ativacao(
    axs[1, 1],
    negativas,
    "Somente ativacoes negativas",
    absoluto=True
)

salvar(
    fig,
    "28_bordas_positivas_negativas.png"
)


# =====================================================
# ETAPA 10 - COMPARAR MAPA ASSINADO E MODULO
# =====================================================

mapa_absoluto = np.abs(
    ativacao_texto
)

fig, axs = plt.subplots(
    1, 3,
    figsize=(16, 4)
)

axs[0].imshow(
    imagem_texto,
    cmap="gray",
    vmin=0,
    vmax=255
)

axs[0].set_title("Imagem original")
axs[0].axis("off")

pintar_ativacao(
    axs[1],
    ativacao_texto,
    "Mapa de ativacoes com sinal"
)

pintar_ativacao(
    axs[2],
    mapa_absoluto,
    "Valor absoluto: ambas as direcoes",
    absoluto=True
)

salvar(
    fig,
    "29_bordas_valor_absoluto_texto.png"
)


# =====================================================
# RESUMO DA AULA
# =====================================================

print("\nRESUMO FINAL")

print("Imagem de entrada:", imagem.shape)
print("Kernel vertical:", kernel_vertical.shape)
print("Kernel horizontal:", kernel_horizontal.shape)
print("Stride: 1")
print("Padding: 0")

print("Mapa vertical:", mapa_vertical.shape)
print("Mapa horizontal:", mapa_horizontal.shape)

print(
    "Menor ativacao vertical:",
    float(mapa_vertical.min())
)

print(
    "Maior ativacao vertical:",
    float(mapa_vertical.max())
)

print(
    "Menor ativacao horizontal:",
    float(mapa_horizontal.min())
)

print(
    "Maior ativacao horizontal:",
    float(mapa_horizontal.max())
)

print("\nCONCEITOS:")
print("1 - Kernels identificam padroes.")
print("2 - Filtros diferentes detectam bordas diferentes.")
print("3 - O sinal indica o sentido da transicao.")
print("4 - O valor absoluto destaca ambos os sentidos.")
print("5 - A convolucao valid reduz a dimensao espacial.")

print("\nAULA 09 EXECUTADA COM SUCESSO!")
