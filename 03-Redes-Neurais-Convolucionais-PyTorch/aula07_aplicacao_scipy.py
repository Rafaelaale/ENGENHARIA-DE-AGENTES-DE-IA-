# ALURA - AULA 07: APLICACAO COM SCIPY
# Convolucao 1D em sinal de acelerometro simulado

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve


BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados"
SAIDA.mkdir(parents=True, exist_ok=True)


def salvar(figura, nome):
    caminho = SAIDA / nome
    figura.savefig(
        caminho,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(figura)
    print("Imagem salva:", caminho)


def faixa_pixels(
    ax,
    valores,
    titulo,
    cmap="gray",
    minimo=None,
    maximo=None
):
    """Representa cada valor do sinal como um pixel."""

    valores = np.asarray(valores, dtype=float)

    if minimo is None:
        minimo = float(valores.min())

    if maximo is None:
        maximo = float(valores.max())

    if maximo == minimo:
        maximo = minimo + 1

    ax.imshow(
        valores[np.newaxis, :],
        cmap=cmap,
        vmin=minimo,
        vmax=maximo,
        aspect="auto",
        interpolation="nearest",
        extent=(
            -0.5,
            len(valores) - 0.5,
            -0.5,
            0.5
        )
    )

    ax.set_title(titulo)
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(valores)))

    ax.set_xlim(
        -0.5,
        len(valores) - 0.5
    )

    ax.tick_params(
        axis="x",
        labelsize=8
    )

    if len(valores) <= 15:

        for indice, valor in enumerate(valores):

            rel = (
                (valor - minimo) /
                (maximo - minimo)
            )

            cor = (
                "black"
                if rel > 0.53
                else "white"
            )

            ax.text(
                indice,
                0,
                f"{valor:.1f}",
                ha="center",
                va="center",
                color=cor,
                fontsize=9,
                fontweight="bold"
            )


print("=" * 65)
print("AULA 07 - APLICACAO COM SCIPY")
print("CONVOLUCAO 1D")
print("=" * 65)


# =====================================================
# ETAPA 1 - SIMULAR O ACELEROMETRO
# =====================================================

# Semente para gerar resultados reproduziveis.
rng = np.random.default_rng(42)

# 100 amostras entre 0 e 100.
x = np.linspace(0, 100, 100)

# Senoide com amplitude multiplicada por ruido.
y = (
    10 *
    np.sin(x) *
    rng.random(x.shape[0])
)

print("\n1) SINAL SIMULADO")

print("Pontos:", len(x))
print("Minimo:", round(float(y.min()), 3))
print("Maximo:", round(float(y.max()), 3))


fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.plot(
    x,
    y,
    color="black",
    linewidth=1.7
)

ax.set_title(
    "Sinal de acelerometro simulado"
)

ax.set_xlabel(
    "Tempo / amostra"
)

ax.set_ylabel(
    "Amplitude"
)

ax.grid(alpha=0.25)

salvar(
    fig,
    "16_sinal_acelerometro.png"
)


# =====================================================
# ETAPA 2 - SELECIONAR UM TRECHO
# =====================================================

sinal = y[5:15]

print("\n2) RECORTE DO SINAL")

print(
    "Sinal:",
    np.round(sinal, 2)
)

print(
    "Quantidade de pontos:",
    len(sinal)
)


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12, 6),
    gridspec_kw={
        "height_ratios": [1, 2]
    }
)

faixa_pixels(
    ax1,
    sinal,
    "Sinal como pixels"
)

ax2.plot(
    np.arange(len(sinal)),
    sinal,
    "-o",
    color="black",
    linewidth=2
)

ax2.axhline(
    0,
    color="gray",
    linewidth=0.8
)

ax2.set_title(
    "Os mesmos valores como grafico"
)

ax2.set_xlabel(
    "Posicao"
)

ax2.set_ylabel(
    "Amplitude"
)

ax2.set_xticks(
    np.arange(len(sinal))
)

ax2.grid(alpha=0.25)

fig.tight_layout()

salvar(
    fig,
    "17_sinal_como_pixels.png"
)


# =====================================================
# ETAPA 3 - CRIAR O KERNEL
# =====================================================

# Queremos detectar intervalos crescentes.
# A convolucao matematica inverte o filtro.
# Por isso definimos este kernel decrescente.

kernel = np.asarray([
    1.0,
    0.0,
    -1.0
])

kernel_invertido = np.flip(kernel)

print("\n3) KERNEL")

print(
    "Kernel original:",
    kernel
)

print(
    "Kernel invertido:",
    kernel_invertido
)


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(9, 4.5)
)

faixa_pixels(
    ax1,
    kernel,
    "Kernel original",
    minimo=-1,
    maximo=1
)

faixa_pixels(
    ax2,
    kernel_invertido,
    "Kernel invertido",
    minimo=-1,
    maximo=1
)

fig.tight_layout()

salvar(
    fig,
    "18_kernel_inversao.png"
)


# =====================================================
# ETAPA 4 - CONVOLUCAO MANUAL
# =====================================================

print("\n4) CONVOLUCAO MANUAL")

ativacao_manual = []

# Sem adicionar zeros: modo valido.
quantidade = (
    len(sinal) -
    len(kernel) +
    1
)

for deslocamento in range(quantidade):

    janela = sinal[
        deslocamento:
        deslocamento + len(kernel)
    ]

    # Multiplicacao elemento a elemento.
    produtos = (
        janela *
        kernel_invertido
    )

    # Soma dos produtos.
    resultado = float(
        produtos.sum()
    )

    ativacao_manual.append(
        resultado
    )

    print(
        f"\nDeslocamento {deslocamento}"
    )

    print(
        "Janela:",
        np.round(janela, 2)
    )

    print(
        "Kernel invertido:",
        kernel_invertido
    )

    print(
        "Produtos:",
        np.round(produtos, 2)
    )

    print(
        "Somatorio:",
        round(resultado, 3)
    )


ativacao_manual = np.asarray(
    ativacao_manual
)


# =====================================================
# ETAPA 5 - CONVOLUCAO AUTOMATICA COM SCIPY
# =====================================================

print("\n5) CONVOLUCAO COM SCIPY")

ativacao_scipy = convolve(
    sinal,
    kernel,
    mode="valid",
    method="direct"
)

print(
    "Resultado manual:",
    np.round(
        ativacao_manual,
        3
    )
)

print(
    "Resultado SciPy:",
    np.round(
        ativacao_scipy,
        3
    )
)

# Confirma que a implementacao manual esta correta.
np.testing.assert_allclose(
    ativacao_manual,
    ativacao_scipy,
    rtol=0,
    atol=1e-10
)

print(
    "VALIDACAO MANUAL X SCIPY: OK"
)


# =====================================================
# ETAPA 6 - VISUALIZAR OS DESLOCAMENTOS
# =====================================================

fig, axs = plt.subplots(
    3,
    1,
    figsize=(11, 9)
)

melhor = int(
    np.argmax(ativacao_scipy)
)

exemplos = [
    0,
    1,
    melhor
]

for ax, indice in zip(
    axs,
    exemplos
):

    janela = sinal[
        indice:
        indice + 3
    ]

    produto = (
        janela *
        kernel_invertido
    )

    ax.bar(
        np.arange(3) - 0.17,
        janela,
        width=0.33,
        label="Sinal",
        alpha=0.85
    )

    ax.bar(
        np.arange(3) + 0.17,
        produto,
        width=0.33,
        label="Produto pelo kernel",
        alpha=0.85
    )

    ax.set_xticks(
        np.arange(3)
    )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_title(
        f"Deslocamento {indice}: "
        f"soma = {produto.sum():+.3f}"
    )

    ax.legend(
        fontsize=9
    )

fig.tight_layout()

salvar(
    fig,
    "19_convolucao_manual.png"
)


# =====================================================
# ETAPA 7 - MAPA DE ATIVACAO DO TRECHO
# =====================================================

print("\n6) MAPA DE ATIVACAO")

print(
    "Ativacoes:",
    np.round(
        ativacao_scipy,
        3
    )
)

print(
    "Maior ativacao:",
    round(
        float(ativacao_scipy[melhor]),
        3
    )
)

print(
    "Janela de maior ativacao:",
    np.round(
        sinal[melhor:melhor + 3],
        2
    )
)

# O modo valid com kernel de tamanho 3
# produz saidas centradas nas posicoes 1 ate 8.
indices_centrais = np.arange(
    1,
    len(sinal) - 1
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12, 6.5),
    gridspec_kw={
        "height_ratios": [3, 1]
    }
)

ax1.plot(
    np.arange(len(sinal)),
    sinal,
    color="black",
    linewidth=2.2
)

limite = max(
    1,
    float(
        np.max(
            np.abs(ativacao_scipy)
        )
    )
)

pontos = ax1.scatter(
    indices_centrais,
    sinal[indices_centrais],
    c=ativacao_scipy,
    cmap="RdBu_r",
    vmin=-limite,
    vmax=limite,
    s=115,
    zorder=3
)

ax1.set_title(
    "Vermelho: crescente | Azul: decrescente"
)

ax1.set_xlabel(
    "Posicao no sinal"
)

ax1.set_ylabel(
    "Amplitude"
)

ax1.set_xticks(
    np.arange(len(sinal))
)

ax1.grid(alpha=0.25)

fig.colorbar(
    pontos,
    ax=ax1,
    label="Ativacao"
)

faixa_pixels(
    ax2,
    ativacao_scipy,
    "Mapa de ativacao",
    cmap="RdBu_r",
    minimo=-limite,
    maximo=limite
)

fig.tight_layout()

salvar(
    fig,
    "20_ativacao_trecho.png"
)


# =====================================================
# ETAPA 8 - APLICAR AO SINAL COMPLETO
# =====================================================

print("\n7) SINAL COMPLETO")

ativacao_completa = convolve(
    y,
    kernel,
    mode="valid",
    method="direct"
)

# Equivale a zerar resultados negativos.
ativacao_positiva = np.maximum(
    ativacao_completa,
    0
)

centros = np.arange(
    1,
    len(y) - 1
)

print(
    "Quantidade de amostras:",
    len(y)
)

print(
    "Quantidade de ativacoes:",
    len(ativacao_completa)
)

print(
    "Ativacoes positivas:",
    int(
        np.count_nonzero(
            ativacao_positiva
        )
    )
)

print(
    "Maior ativacao:",
    round(
        float(
            ativacao_positiva.max()
        ),
        3
    )
)


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(14, 6.5),
    gridspec_kw={
        "height_ratios": [3, 1]
    }
)

ax1.plot(
    np.arange(len(y)),
    y,
    color="black",
    linewidth=1.7
)

maximo = max(
    1,
    float(
        ativacao_positiva.max()
    )
)

pontos = ax1.scatter(
    centros,
    y[centros],
    c=ativacao_positiva,
    cmap="Reds",
    vmin=0,
    vmax=maximo,
    s=32,
    zorder=3
)

ax1.set_title(
    "Deteccao de intervalos crescentes"
)

ax1.set_xlabel(
    "Indice da amostra"
)

ax1.set_ylabel(
    "Amplitude"
)

ax1.grid(alpha=0.2)

fig.colorbar(
    pontos,
    ax=ax1,
    label="Ativacao positiva"
)

ax2.imshow(
    ativacao_positiva[np.newaxis, :],
    aspect="auto",
    cmap="Reds",
    interpolation="nearest",
    extent=(
        0.5,
        len(y) - 1.5,
        0,
        1
    ),
    vmin=0,
    vmax=maximo
)

ax2.set_title(
    "Mapa de ativacao do sinal completo"
)

ax2.set_xlabel(
    "Posicao central da janela"
)

ax2.set_yticks([])

fig.tight_layout()

salvar(
    fig,
    "21_ativacao_sinal_completo.png"
)


# =====================================================
# FINALIZACAO
# =====================================================

print("\nOBSERVACAO TECNICA:")

print(
    "O kernel mede a diferenca entre "
    "o ultimo e o primeiro valor da janela."
)

print(
    "Ativacao positiva indica crescimento "
    "entre esses dois valores."
)

print(
    "Isso nao garante que todos os pontos "
    "intermediarios estejam crescendo."
)

print(
    "\nAULA 07 EXECUTADA COM SUCESSO!"
)
