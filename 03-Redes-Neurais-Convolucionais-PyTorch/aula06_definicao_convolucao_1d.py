# ALURA - AULA 06: DEFINICAO DE CONVOLUCAO 1D
# Curso: Redes Neurais Convolucionais com PyTorch

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados"
SAIDA.mkdir(exist_ok=True)

print("=" * 65)
print("AULA 06 - DEFINICAO: CONVOLUCAO 1D")
print("=" * 65)


# =====================================================
# ETAPA 1 - DEFININDO AS FUNCOES f(x) E g(x)
# =====================================================

# f(x): sinal de entrada, ou seja, os dados.
f = np.array(
    [0, 0, 0, 0, 0, 0.3, 0.9, 1.6,
     1.0, 0.4, 0, 0, 0, 0, 0],
    dtype=np.float32
)

# g(x): filtro ou kernel.
# Usamos um filtro assimetrico para visualizar a inversao.
g = np.array(
    [0, 0.3, 1.0, 0.6, 0],
    dtype=np.float32
)

n = len(f)
k = len(g)

print("\nFUNCAO DE ENTRADA f(x):")
print(f)

print("\nFILTRO g(x):")
print(g)

print("\nFORMULA MATEMATICA:")
print("h[x] = somatorio de f[u] * g[x-u]")


# =====================================================
# ETAPA 2 - INVERTENDO O FILTRO
# =====================================================

g_invertido = g[::-1].copy()

print("\nFILTRO ORIGINAL:")
print(g)

print("\nFILTRO INVERTIDO:")
print(g_invertido)


# =====================================================
# ETAPA 3 - CONVOLUCAO MANUAL
# =====================================================

# Zeros adicionais permitem calcular as sobreposicoes
# parciais, nas extremidades do sinal.

f_com_zeros = np.pad(
    f,
    (k - 1, k - 1)
)

h_manual = np.zeros(
    n + k - 1,
    dtype=np.float32
)

# O filtro percorre o sinal de entrada.
for x in range(len(h_manual)):

    # Seleciona uma janela do sinal.
    trecho = f_com_zeros[x:x + k]

    # Multiplicacao elemento a elemento.
    produtos = trecho * g_invertido

    # Somatorio dos produtos.
    h_manual[x] = np.sum(produtos)

print("\nRESULTADO DA CONVOLUCAO MANUAL:")
print(np.round(h_manual, 3))


# =====================================================
# ETAPA 4 - VALIDANDO COM NUMPY
# =====================================================

h_numpy = np.convolve(
    f,
    g,
    mode="full"
)

np.testing.assert_allclose(
    h_manual,
    h_numpy,
    rtol=0,
    atol=1e-6
)

print("\nVALIDACAO NUMPY: OK")


# =====================================================
# ETAPA 5 - CONVOLUCAO COM PYTORCH
# =====================================================

# Formato esperado pelo Conv1d:
# (batch, canais, comprimento)

entrada = torch.tensor(f).reshape(1, 1, -1)

# O Conv1d usa correlacao cruzada.
# Para reproduzir a convolucao matematica,
# fornecemos o kernel invertido.

peso_convolucao = (
    torch.tensor(g_invertido)
    .reshape(1, 1, -1)
)

peso_original = (
    torch.tensor(g)
    .reshape(1, 1, -1)
)

with torch.no_grad():

    saida_pytorch = F.conv1d(
        entrada,
        peso_convolucao,
        padding=k - 1
    )

    saida_correlacao = F.conv1d(
        entrada,
        peso_original,
        padding=k - 1
    )

h_torch = (
    saida_pytorch
    .reshape(-1)
    .numpy()
)

h_correlacao = (
    saida_correlacao
    .reshape(-1)
    .numpy()
)

np.testing.assert_allclose(
    h_manual,
    h_torch,
    rtol=0,
    atol=1e-6
)

print("\nVALIDACAO PYTORCH: OK")

print("Formato da entrada:", tuple(entrada.shape))
print("Formato do kernel:", tuple(peso_convolucao.shape))
print("Formato da saida:", tuple(saida_pytorch.shape))


# =====================================================
# ETAPA 6 - CASAMENTO DE PADROES
# =====================================================

pico = int(np.argmax(h_manual))

print("\nCASAMENTO DE PADROES")
print("Posicao do maior resultado:", pico)
print("Valor maximo:", round(float(h_manual[pico]), 3))

# Tres momentos do deslocamento:
# antes do encontro, melhor encontro e depois.

passos = [
    4,
    pico,
    len(h_manual) - 5
]

for posicao in passos:

    trecho = f_com_zeros[posicao:posicao + k]
    produtos = trecho * g_invertido

    print("\nDeslocamento:", posicao)

    print(
        "Janela:",
        np.round(trecho, 2)
    )

    print(
        "Produtos:",
        np.round(produtos, 2)
    )

    print(
        "Somatorio:",
        round(float(h_manual[posicao]), 3)
    )


# =====================================================
# ETAPA 7 - GRAFICO DAS TRES FUNCOES
# =====================================================

fig, axs = plt.subplots(
    3,
    1,
    figsize=(11, 9)
)

axs[0].stem(
    np.arange(n),
    f
)

axs[0].set_title(
    "f(x): sinal de entrada"
)

axs[1].stem(
    np.arange(k),
    g
)

axs[1].set_title(
    "g(x): filtro original"
)

axs[2].stem(
    np.arange(len(h_manual)),
    h_manual
)

axs[2].axvline(
    pico,
    color="red",
    linestyle="--",
    label=f"Pico: x={pico}"
)

axs[2].legend()

axs[2].set_title(
    "h(x) = f * g: resultado da convolucao"
)

for ax in axs:
    ax.set_xlabel("Posicao")
    ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    SAIDA / "13_convolucao_1d_funcoes.png",
    dpi=150
)

plt.close(fig)


# =====================================================
# ETAPA 8 - GRAFICO DOS DESLOCAMENTOS
# =====================================================

fig, axs = plt.subplots(
    3,
    1,
    figsize=(11, 10)
)

for ax, x in zip(axs, passos):

    ax.stem(
        np.arange(n),
        f,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
        label="Dados f(x)"
    )

    posicoes = np.arange(
        x - (k - 1),
        x + 1
    )

    ax.stem(
        posicoes,
        g_invertido,
        linefmt="C1-",
        markerfmt="C1s",
        basefmt=" ",
        label="Filtro invertido"
    )

    ax.set_xlim(-2, n + k - 2)
    ax.set_ylim(-0.1, 1.9)

    ax.set_title(
        f"Deslocamento x={x} | "
        f"Resultado h[x]={h_manual[x]:.3f}"
    )

    ax.set_xlabel("Posicao no sinal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

fig.tight_layout()

fig.savefig(
    SAIDA / "14_convolucao_1d_deslocamentos.png",
    dpi=150
)

plt.close(fig)


# =====================================================
# ETAPA 9 - COMPARACAO MATEMATICA / PYTORCH
# =====================================================

fig, ax = plt.subplots(
    figsize=(11, 4)
)

ax.plot(
    h_manual,
    "o-",
    label="Convolucao manual / NumPy"
)

ax.plot(
    h_torch,
    "--",
    linewidth=2,
    label="PyTorch com filtro invertido"
)

ax.plot(
    h_correlacao,
    "s:",
    label="PyTorch com filtro original"
)

ax.set_title(
    "Convolucao matematica x correlacao cruzada"
)

ax.set_xlabel("Posicao")
ax.set_ylabel("Somatorio dos produtos")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    SAIDA / "15_convolucao_1d_comparacao.png",
    dpi=150
)

plt.close(fig)


# =====================================================
# RESULTADOS
# =====================================================

print("\nIMAGENS GERADAS:")

for nome in [
    "13_convolucao_1d_funcoes.png",
    "14_convolucao_1d_deslocamentos.png",
    "15_convolucao_1d_comparacao.png"
]:
    print(SAIDA / nome)

print("\nAULA 06 EXECUTADA COM SUCESSO!")
