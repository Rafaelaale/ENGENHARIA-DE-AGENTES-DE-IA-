# ALURA - AULA 10: FILTROS CONVOLUCIONAIS
# Curso: Redes Neurais Convolucionais com PyTorch
# Media, Sobel vertical e horizontal, Laplace

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import data


# =====================================================
# 1. CONFIGURACAO DO PROJETO
# =====================================================

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados"
SAIDA.mkdir(parents=True, exist_ok=True)


def salvar(fig, nome):

    caminho = SAIDA / nome

    fig.tight_layout()
    fig.savefig(caminho, dpi=135)
    plt.close(fig)

    print("Imagem salva:", caminho)


def mostrar(
    ax,
    imagem,
    titulo,
    *,
    abs_val=False,
    cmap="gray"
):

    valores = (
        np.abs(imagem)
        if abs_val
        else imagem
    )

    if abs_val:

        escala = float(
            np.percentile(valores, 99)
        )

        ax.imshow(
            valores,
            cmap="gray",
            vmin=0,
            vmax=max(escala, 1e-6)
        )

    elif np.min(valores) < 0:

        escala = float(
            np.percentile(
                np.abs(valores),
                99
            )
        )

        ax.imshow(
            valores,
            cmap="seismic",
            vmin=-max(escala, 1e-6),
            vmax=max(escala, 1e-6)
        )

    else:

        ax.imshow(
            valores,
            cmap=cmap
        )

    ax.set_title(titulo)
    ax.axis("off")


def mostrar_kernel(ax, kernel, titulo):

    escala = float(
        np.max(np.abs(kernel))
    )

    ax.imshow(
        kernel,
        cmap="seismic",
        vmin=-escala,
        vmax=escala
    )

    ax.set_title(titulo)

    for linha in range(kernel.shape[0]):

        for coluna in range(kernel.shape[1]):

            numero = float(
                kernel[linha, coluna]
            )

            texto = (
                f"{numero:.3f}"
                if abs(numero) < 1
                else f"{numero:.0f}"
            )

            ax.text(
                coluna,
                linha,
                texto,
                ha="center",
                va="center",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 1
                }
            )

    ax.set_xticks([])
    ax.set_yticks([])


print("=" * 65)
print("AULA 10 - FILTROS CONVOLUCIONAIS | ALURA")
print("=" * 65)


# =====================================================
# 2. CARREGAR UMA IMAGEM COLORIDA
# =====================================================

# A Alura utiliza GUI.jpg.
# Nesta pratica, usamos uma fotografia
# que ja acompanha o scikit-image.

rgb = data.astronaut()

# Selecionar o canal vermelho (indice 0).
# A imagem passa de H,W,C para H,W.

imagem = rgb[:, :, 0].astype(
    np.float64
)

print("\n1. IMAGEM ORIGINAL")

print(
    "Imagem RGB:",
    rgb.shape
)

print(
    "Canal vermelho:",
    imagem.shape
)

print(
    "Faixa de pixels:",
    imagem.min(),
    "a",
    imagem.max()
)


# =====================================================
# 3. VISUALIZAR RGB E CANAL VERMELHO
# =====================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(11, 5)
)

axs[0].imshow(rgb)
axs[0].set_title(
    "Fotografia RGB do scikit-image"
)
axs[0].axis("off")

mostrar(
    axs[1],
    imagem,
    "Canal vermelho em tons de cinza"
)

salvar(
    fig,
    "30_fotografia_canal_vermelho.png"
)


# =====================================================
# 4. FILTRO DA MEDIA 3 X 3
# =====================================================

# Cada peso recebe 1/9.
# A convolucao calcula a media da vizinhanca.

media3 = np.zeros(
    (3, 3),
    dtype=float
)

media3[:] = 1.0 / (3 ** 2)


# =====================================================
# 5. FILTRO DA MEDIA 9 X 9
# =====================================================

# Agora cada peso recebe 1/81.
# Regiao maior produz suavizacao mais forte.

media9 = np.zeros(
    (9, 9),
    dtype=float
)

media9[:] = 1.0 / (9 ** 2)

assert np.isclose(
    media3.sum(),
    1
)

assert np.isclose(
    media9.sum(),
    1
)


# =====================================================
# 6. APLICAR A CONVOLUCAO DA MEDIA
# =====================================================

borrada3 = convolve2d(
    imagem,
    media3,
    mode="valid"
)

borrada9 = convolve2d(
    imagem,
    media9,
    mode="valid"
)

print("\n2. FILTRO DA MEDIA")

print(
    "Kernel 3x3 - peso:",
    round(float(media3[0, 0]), 6)
)

print(
    "Kernel 9x9 - peso:",
    round(float(media9[0, 0]), 6)
)

print(
    "Imagem:",
    imagem.shape
)

print(
    "Saida media 3x3:",
    borrada3.shape
)

print(
    "Saida media 9x9:",
    borrada9.shape
)


# =====================================================
# 7. VALIDAR A MEDIA MANUALMENTE
# =====================================================

primeira_media = float(
    imagem[:3, :3].mean()
)

np.testing.assert_allclose(
    primeira_media,
    borrada3[0, 0],
    atol=1e-10
)

print(
    "VALIDACAO MEDIA MANUAL X SCIPY: OK"
)


# =====================================================
# 8. VISUALIZAR O DESFOQUE
# =====================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

mostrar(
    axs[0],
    imagem,
    "Imagem original"
)

mostrar(
    axs[1],
    borrada3,
    "Media 3x3: suavizacao leve"
)

mostrar(
    axs[2],
    borrada9,
    "Media 9x9: maior suavizacao"
)

salvar(
    fig,
    "31_media_3x3_9x9.png"
)


# =====================================================
# 9. MOSTRAR OS KERNELS DA MEDIA
# =====================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(10, 5)
)

mostrar_kernel(
    axs[0],
    media3,
    "Pesos do kernel media 3x3"
)

mostrar_kernel(
    axs[1],
    media9,
    "Pesos do kernel media 9x9"
)

salvar(
    fig,
    "32_kernels_media.png"
)


# =====================================================
# 10. CONSTRUIR O SOBEL VERTICAL
# =====================================================

# Colunas com valores:
# -1  0  +1
#
# A linha central recebe pesos maiores.

sobel_v = np.zeros(
    (3, 3),
    dtype=float
)

sobel_v[:, 0] = -1
sobel_v[:, 2] = 1

sobel_v[1, 0] = -2
sobel_v[1, 2] = 2

print("\n3. SOBEL VERTICAL")

print(
    sobel_v.astype(int)
)


# =====================================================
# 11. CONSTRUIR O SOBEL HORIZONTAL
# =====================================================

# Linhas com valores negativos, zero e positivos.

sobel_h = np.zeros(
    (3, 3),
    dtype=float
)

sobel_h[0, :] = -1
sobel_h[2, :] = 1

sobel_h[0, 1] = -2
sobel_h[2, 1] = 2

print("\n4. SOBEL HORIZONTAL")

print(
    sobel_h.astype(int)
)


# =====================================================
# 12. APLICAR OS FILTROS SOBEL
# =====================================================

ativ_v = convolve2d(
    imagem,
    sobel_v,
    mode="valid"
)

ativ_h = convolve2d(
    imagem,
    sobel_h,
    mode="valid"
)

print(
    "Mapa Sobel vertical:",
    ativ_v.shape
)

print(
    "Mapa Sobel horizontal:",
    ativ_h.shape
)


# =====================================================
# 13. VISUALIZAR OS KERNELS SOBEL
# =====================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(14, 5)
)

mostrar(
    axs[0],
    imagem,
    "Imagem original"
)

mostrar_kernel(
    axs[1],
    sobel_v,
    "Sobel vertical"
)

mostrar_kernel(
    axs[2],
    sobel_h,
    "Sobel horizontal"
)

salvar(
    fig,
    "33_kernels_sobel.png"
)


# =====================================================
# 14. COMPARAR OS MAPAS DE ATIVACAO SOBEL
# =====================================================

fig, axs = plt.subplots(
    2,
    3,
    figsize=(16, 9)
)

mostrar(
    axs[0, 0],
    imagem,
    "Imagem original"
)

mostrar(
    axs[0, 1],
    ativ_v,
    "Sobel vertical: ativacoes com sinal"
)

mostrar(
    axs[0, 2],
    ativ_h,
    "Sobel horizontal: ativacoes com sinal"
)

mostrar(
    axs[1, 0],
    imagem,
    "Imagem original"
)

mostrar(
    axs[1, 1],
    ativ_v,
    "Sobel vertical: valor absoluto",
    abs_val=True
)

mostrar(
    axs[1, 2],
    ativ_h,
    "Sobel horizontal: valor absoluto",
    abs_val=True
)

salvar(
    fig,
    "34_sobel_bordas.png"
)


# =====================================================
# 15. CONSTRUIR O FILTRO LAPLACE
# =====================================================

# Todas as posicoes recebem -1.
# O pixel central recebe +8.
#
# Este kernel destaca variacoes em
# diferentes direcoes da imagem.

laplace = (
    np.ones((3, 3), dtype=float) * -1
)

laplace[1, 1] = 8

assert laplace.sum() == 0

print("\n5. LAPLACE")

print(
    laplace.astype(int)
)


# =====================================================
# 16. APLICAR O FILTRO LAPLACE
# =====================================================

ativ_l = convolve2d(
    imagem,
    laplace,
    mode="valid"
)

print(
    "Mapa Laplace:",
    ativ_l.shape
)


# =====================================================
# 17. COMPARAR LAPLACE COM E SEM ABSOLUTO
# =====================================================

fig, axs = plt.subplots(
    1,
    4,
    figsize=(18, 4.5)
)

mostrar(
    axs[0],
    imagem,
    "Imagem original"
)

mostrar_kernel(
    axs[1],
    laplace,
    "Kernel Laplace"
)

mostrar(
    axs[2],
    ativ_l,
    "Laplace com sinal"
)

mostrar(
    axs[3],
    ativ_l,
    "Laplace em valor absoluto",
    abs_val=True
)

salvar(
    fig,
    "35_laplace_bordas.png"
)


# =====================================================
# 18. COMPARAR AS DIRECOES DAS BORDAS
# =====================================================

# Exercicio adicional:
# a magnitude combina Sobel vertical e horizontal.

magnitude_sobel = np.hypot(
    ativ_v,
    ativ_h
)

fig, axs = plt.subplots(
    1,
    4,
    figsize=(18, 4.5)
)

mostrar(
    axs[0],
    imagem,
    "Imagem original"
)

mostrar(
    axs[1],
    ativ_v,
    "Sobel vertical - absoluto",
    abs_val=True
)

mostrar(
    axs[2],
    ativ_h,
    "Sobel horizontal - absoluto",
    abs_val=True
)

mostrar(
    axs[3],
    magnitude_sobel,
    "Magnitude combinada Sobel",
    abs_val=True
)

salvar(
    fig,
    "36_comparacao_filtros.png"
)


# =====================================================
# 19. RESUMO FINAL
# =====================================================

print("\n6. RESUMO")

print(
    "Media 3x3:",
    borrada3.shape
)

print(
    "Media 9x9:",
    borrada9.shape
)

print(
    "Sobel vertical:",
    ativ_v.shape
)

print(
    "Sobel horizontal:",
    ativ_h.shape
)

print(
    "Laplace:",
    ativ_l.shape
)

print("\nCONCEITOS APRENDIDOS:")

print(
    "1 - Filtro da media suaviza imagens."
)

print(
    "2 - Kernels maiores ampliam a vizinhanca."
)

print(
    "3 - Sobel vertical detecta bordas verticais."
)

print(
    "4 - Sobel horizontal detecta bordas horizontais."
)

print(
    "5 - Laplace detecta variacoes em varias direcoes."
)

print(
    "6 - Valor absoluto destaca ambas as polaridades."
)

print(
    "7 - Nao houve treinamento de rede neural."
)

print(
    "\nAULA 10 EXECUTADA COM SUCESSO!"
)
