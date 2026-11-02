# AULA 11 - PADROES COMPLEXOS - ALURA
# Curso: Redes Neurais Convolucionais com PyTorch

from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve
from skimage import data, io


BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados"
SAIDA.mkdir(exist_ok=True)


def salvar(fig, nome):
    caminho = SAIDA / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=145)
    plt.close(fig)
    print("Imagem salva:", caminho)


def mostrar(ax, imagem, titulo, cmap="gray"):
    ax.imshow(imagem, cmap=cmap)
    ax.set_title(titulo)
    ax.axis("off")


# =====================================================
# ETAPA 1 - CONFIGURAR A IMAGEM
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--imagem",
    default="plane.jpg"
)

parser.add_argument(
    "--x",
    type=int,
    default=None
)

parser.add_argument(
    "--y",
    type=int,
    default=None
)

parser.add_argument(
    "--tamanho",
    type=int,
    default=None
)

args = parser.parse_args()

print("=" * 65)
print("AULA 11 - PADROES COMPLEXOS | CONVOLUCAO 2D")
print("=" * 65)

foto_path = Path(args.imagem)

if not foto_path.is_absolute():
    foto_path = BASE / foto_path


# =====================================================
# ETAPA 2 - CARREGAR FOTO DO AVIAO OU EXEMPLO
# =====================================================

if foto_path.is_file():

    foto = io.imread(str(foto_path))

    x = 250 if args.x is None else args.x
    y = 109 if args.y is None else args.y

    tamanho = (
        21
        if args.tamanho is None
        else args.tamanho
    )

    print("\nImagem fornecida:", foto_path)
    print("Confira se o retangulo seleciona a turbina.")

else:

    foto = data.astronaut()

    x = 205 if args.x is None else args.x
    y = 88 if args.y is None else args.y

    tamanho = (
        31
        if args.tamanho is None
        else args.tamanho
    )

    print("\nplane.jpg nao encontrado.")
    print("Usando fotografia do Scikit-image.")
    print("Padrao escolhido: detalhe do rosto.")


# =====================================================
# ETAPA 3 - SELECIONAR UM CANAL DE COR
# =====================================================

if foto.ndim == 3:

    # Mesmo procedimento apresentado na Alura.
    img = foto[:, :, 0].astype(np.float64)

else:

    img = foto.astype(np.float64)


altura, largura = img.shape

if tamanho < 3 or tamanho % 2 != 1:

    raise ValueError(
        "O tamanho do patch deve ser impar e >= 3."
    )

if (
    x < 0 or y < 0
    or x + tamanho > largura
    or y + tamanho > altura
):

    raise ValueError(
        "Coordenadas do patch fora da imagem."
    )


print("\n1 - IMAGEM")

print("Dimensoes:", img.shape)
print("Coordenadas X, Y:", x, y)


# =====================================================
# ETAPA 4 - EXTRAIR O PATCH
# =====================================================

patch = img[
    y:y + tamanho,
    x:x + tamanho
].copy()

print("\n2 - PATCH")

print("Dimensoes:", patch.shape)

print(
    "Media original:",
    round(float(patch.mean()), 4)
)


# =====================================================
# ETAPA 5 - CENTRALIZAR O PATCH
# =====================================================

# Subtrair a media de todos os pixels.
# Isso reduz a influencia da intensidade
# media de brilho no casamento de padroes.

patch_centrado = patch - patch.mean()

print("\n3 - PATCH CENTRALIZADO")

print(
    "Media depois da centralizacao:",
    round(float(patch_centrado.mean()), 10)
)


# =====================================================
# ETAPA 6 - INVERTER O KERNEL
# =====================================================

# A convolucao matematica inverte o kernel.
# Por isso invertemos o patch antes de operar.

kernel = np.flip(
    patch_centrado,
    axis=(0, 1)
).copy()

print("\n4 - KERNEL INVERTIDO")

print("Dimensoes:", kernel.shape)


# =====================================================
# ETAPA 7 - VISUALIZAR A REGIAO PROCURADA
# =====================================================

fig, axs = plt.subplots(
    1, 2,
    figsize=(11, 5)
)

mostrar(
    axs[0],
    img,
    "Imagem original - canal 0"
)

axs[0].add_patch(
    Rectangle(
        (x - 0.5, y - 0.5),
        tamanho,
        tamanho,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
)

mostrar(
    axs[1],
    patch,
    "Patch extraido da imagem"
)

salvar(
    fig,
    "37_imagem_e_patch_complexo.png"
)


# =====================================================
# ETAPA 8 - VISUALIZAR AS TRANSFORMACOES
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(12, 4)
)

mostrar(
    axs[0],
    patch,
    "Patch original"
)

mostrar(
    axs[1],
    patch_centrado,
    "Patch menos a media",
    cmap="seismic"
)

mostrar(
    axs[2],
    kernel,
    "Kernel invertido em 180 graus",
    cmap="seismic"
)

salvar(
    fig,
    "38_patch_centralizado_invertido.png"
)


# =====================================================
# ETAPA 9 - CONVOLUCAO COM SCIPY
# =====================================================

# fftconvolve calcula a convolucao usando FFT.
# E uma alternativa eficiente da biblioteca SciPy
# para kernels maiores.

ativ_valid = fftconvolve(
    img,
    kernel,
    mode="valid"
)

ativ_same = fftconvolve(
    img,
    kernel,
    mode="same"
)

print("\n5 - CONVOLUCAO")

print("Entrada:", img.shape)
print("Saida valid:", ativ_valid.shape)
print("Saida same:", ativ_same.shape)


# =====================================================
# ETAPA 10 - VALIDACAO MANUAL
# =====================================================

# No ponto de origem do patch, a convolucao
# deve corresponder a multiplicacao e soma
# da regiao pelo patch centralizado.

janela_original = img[
    y:y + tamanho,
    x:x + tamanho
]

valor_manual = float(
    np.sum(
        janela_original * patch_centrado
    )
)

np.testing.assert_allclose(
    ativ_valid[y, x],
    valor_manual,
    rtol=1e-8,
    atol=1e-5
)

centro_y = y + tamanho // 2
centro_x = x + tamanho // 2

np.testing.assert_allclose(
    ativ_same[centro_y, centro_x],
    valor_manual,
    rtol=1e-8,
    atol=1e-5
)

print("\n6 - VALIDACAO")

print(
    "Produto e soma manual:",
    round(valor_manual, 3)
)

print(
    "Resultado SciPy:",
    round(float(ativ_valid[y, x]), 3)
)

print("VALIDACAO MANUAL X SCIPY: OK")


# =====================================================
# ETAPA 11 - MAPAS DE ATIVACAO
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

mostrar(
    axs[0],
    img,
    "Imagem original"
)

mostrar(
    axs[1],
    ativ_valid,
    "Mapa de ativacao - valid",
    cmap="inferno"
)

mostrar(
    axs[2],
    ativ_same,
    "Mapa de ativacao - same",
    cmap="inferno"
)

salvar(
    fig,
    "39_mapas_valid_same.png"
)


# =====================================================
# ETAPA 12 - CASAMENTO DE PADROES COMPLEXOS
# =====================================================

# Exercicio adicional:
# Correlacao cruzada normalizada (NCC).
#
# A ativacao bruta pode ser mais forte em
# outra regiao que tenha maior contraste.
#
# A NCC normaliza a semelhanca local,
# tornando o reconhecimento do patch
# menos dependente do brilho e contraste.

uns = np.ones(
    patch.shape,
    dtype=np.float64
)

somas = fftconvolve(
    img,
    uns,
    mode="valid"
)

somas_quadrados = fftconvolve(
    img ** 2,
    uns,
    mode="valid"
)

energia_patch = float(
    np.sum(
        patch_centrado ** 2
    )
)

variancia_local = np.maximum(
    somas_quadrados
    - (somas ** 2 / patch.size),
    0
)

denominador = np.sqrt(
    variancia_local * energia_patch
)

ncc = np.divide(
    ativ_valid,
    denominador,
    out=np.zeros_like(ativ_valid),
    where=denominador > 1e-12
)

ncc = np.clip(ncc, -1, 1)


# =====================================================
# ETAPA 13 - LOCALIZAR A MELHOR CORRESPONDENCIA
# =====================================================

melhor_y, melhor_x = np.unravel_index(
    int(np.argmax(ncc)),
    ncc.shape
)

assert ncc[y, x] > 0.99999

print("\n7 - CASAMENTO DE PADROES")

print(
    "NCC na posicao original:",
    round(float(ncc[y, x]), 6)
)

print(
    "Melhor posicao encontrada (X, Y):",
    int(melhor_x),
    int(melhor_y)
)

print(
    "Posicao original (X, Y):",
    x,
    y
)


# =====================================================
# ETAPA 14 - MOSTRAR ONDE O PADRAO FOI ENCONTRADO
# =====================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

mostrar(
    axs[0],
    img,
    "Regiao original procurada"
)

axs[0].add_patch(
    Rectangle(
        (x - 0.5, y - 0.5),
        tamanho,
        tamanho,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
)

mostrar(
    axs[1],
    ncc,
    "Mapa de semelhanca normalizada",
    cmap="inferno"
)

mostrar(
    axs[2],
    img,
    "Melhor correspondencia encontrada"
)

axs[2].add_patch(
    Rectangle(
        (melhor_x - 0.5, melhor_y - 0.5),
        tamanho,
        tamanho,
        fill=False,
        edgecolor="lime",
        linewidth=2
    )
)

salvar(
    fig,
    "40_casamento_padroes_localizado.png"
)


# =====================================================
# ETAPA 15 - COMPARAR CONVOLUCAO E NCC
# =====================================================

fig, axs = plt.subplots(
    1, 2,
    figsize=(11, 5)
)

mostrar(
    axs[0],
    ativ_valid,
    "Ativacao bruta da convolucao",
    cmap="inferno"
)

mostrar(
    axs[1],
    ncc,
    "Correlacao normalizada - NCC",
    cmap="inferno"
)

salvar(
    fig,
    "41_convolucao_vs_ncc.png"
)


# =====================================================
# ETAPA 16 - RESUMO
# =====================================================

print("\n8 - CONCEITOS APRENDIDOS")

print("1 - Extracao de patches de imagens.")
print("2 - Centralizacao pela media.")
print("3 - Inversao do kernel.")
print("4 - Convolucao com SciPy.")
print("5 - Modos valid e same.")
print("6 - Casamento de padroes complexos.")
print("7 - Localizacao do padrao na imagem.")
print("8 - Comparacao com NCC.")

print("\nOBSERVACAO:")
print("Esta aula nao treinou uma CNN.")
print("Os filtros foram definidos manualmente.")

print("\nAULA 11 EXECUTADA COM SUCESSO!")
