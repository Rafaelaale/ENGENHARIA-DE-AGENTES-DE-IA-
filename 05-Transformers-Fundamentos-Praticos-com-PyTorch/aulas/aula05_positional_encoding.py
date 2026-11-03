"""
PROJETO 05 - TRANSFORMERS COM PYTORCH
AULA 05 - POSITIONAL ENCODING

1. Calculo matematico com NumPy
2. Visualizacao da matriz posicional
3. Implementacao em PyTorch
4. Integracao com tokens reais da Aula 04
5. Testes e resultados
"""

import json
import math

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 1 - EXEMPLO DO PROFESSOR
# ============================================================

d_model = 8

frase = "O gato brinca com o cachorro"

palavras = frase.split(" ")

seq_len = len(palavras)

print("=" * 60)
print("AULA 05 - POSITIONAL ENCODING")
print("=" * 60)

print("\nFrase:", frase)
print("Palavras:", palavras)
print("Quantidade de palavras:", seq_len)
print("Dimensao do modelo:", d_model)


# ============================================================
# 2 - MATRIZ COM NUMPY
# ============================================================

pe = np.zeros(
    (seq_len, d_model),
    dtype=np.float64
)

for pos in range(seq_len):

    for i in range(0, d_model, 2):

        div_term = 10000 ** (i / d_model)

        pe[pos, i] = math.sin(
            pos / div_term
        )

        pe[pos, i + 1] = math.cos(
            pos / div_term
        )


print("\nMatriz Positional Encoding:")
print(np.round(pe, 4))

print("\nShape da matriz:", pe.shape)

assert pe.shape == (6, 8)

# Na posicao zero:
# seno(0) = 0
# cosseno(0) = 1

np.testing.assert_allclose(
    pe[0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    atol=1e-8
)

print("\nTeste matematico: OK")


# ============================================================
# 3 - VISUALIZACAO DA MATRIZ
# ============================================================

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

im = ax[0].imshow(
    pe,
    cmap="RdBu_r",
    aspect="auto",
    vmin=-1,
    vmax=1
)

ax[0].set_title(
    "Positional Encoding\n"
    "Azul = negativo, Vermelho = positivo"
)

ax[0].set_xlabel("Dimensoes")
ax[0].set_ylabel("Palavras (posicao)")

ax[0].set_xticks(
    np.arange(d_model)
)

ax[0].set_yticks(
    np.arange(seq_len)
)

ax[0].set_yticklabels(
    palavras
)

fig.colorbar(
    im,
    ax=ax[0],
    fraction=0.046,
    pad=0.04
)

for k in range(3):

    ax[1].plot(
        np.arange(seq_len),
        pe[:, k],
        label=f"dim {k}"
    )

ax[1].set_title(
    "Sinais por dimensao"
)

ax[1].set_xlabel(
    "Posicao na sequencia"
)

ax[1].set_ylabel("Valor")

ax[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)

ax[1].legend()

plt.tight_layout()

grafico = (
    RESULTADOS
    / "aula05_positional_encoding.png"
)

plt.savefig(
    grafico,
    dpi=160,
    bbox_inches="tight"
)

plt.close(fig)

print("\nGrafico salvo:")
print(grafico)


# ============================================================
# 4 - CLASSE POSITIONAL ENCODING PYTORCH
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000
    ):

        super().__init__()

        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(
                "d_model deve ser positivo e par."
            )

        if max_len <= 0:
            raise ValueError(
                "max_len deve ser positivo."
            )

        self.d_model = d_model

        # Matriz inicial
        pe = torch.zeros(
            max_len,
            d_model
        )

        # Posicoes 0, 1, 2, ...
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        # Frequencias
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            ) * (
                -math.log(10000.0)
                / d_model
            )
        )

        # Dimensoes pares: seno
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Dimensoes impares: cosseno
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Buffer nao treinavel
        # Shape: [1, max_len, d_model]
        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Esperado tensor [batch, seq_len, d_model]."
            )

        B, T, C = x.shape

        if C != self.d_model:
            raise ValueError(
                "Dimensao dos embeddings incorreta."
            )

        if T > self.pe.shape[1]:
            raise ValueError(
                "Sequencia excede max_len."
            )

        posicoes = self.pe[
            :, :T, :
        ].to(dtype=x.dtype)

        return x + posicoes


# ============================================================
# 5 - VALIDAR NUMPY VERSUS PYTORCH
# ============================================================

pos_encoding = PositionalEncoding(
    d_model=d_model,
    max_len=5000
)

pe_torch = pos_encoding.pe[
    0, :seq_len, :
].detach().cpu().numpy()

np.testing.assert_allclose(
    pe,
    pe_torch,
    atol=1e-6
)

print("\nNumPy e PyTorch: resultados equivalentes.")


# ============================================================
# 6 - CARREGAR TOKENS REAIS DA AULA 04
# ============================================================

arquivo_tokens = (
    RESULTADOS
    / "aula04_tokenizacao_bert.json"
)

if not arquivo_tokens.exists():

    raise FileNotFoundError(
        "Execute a Aula 04 antes da Aula 05."
    )

dados = json.loads(
    arquivo_tokens.read_text(
        encoding="utf-8-sig"
    )
)

input_ids = torch.tensor(
    dados["input_ids"],
    dtype=torch.long
)

attention_mask = torch.tensor(
    dados["attention_mask"],
    dtype=torch.long
)

vocab_size = int(
    dados["vocab_size"]
)

pad_id = int(
    dados["pad_id"]
)

print("\nTokens reais da Aula 04:")
print("input_ids:", tuple(input_ids.shape))
print("attention_mask:", tuple(attention_mask.shape))


# ============================================================
# 7 - EMBEDDING + POSITIONAL ENCODING
# ============================================================

# Embedding de demonstracao, ainda sem treinamento.
# Nao estamos carregando pesos do modelo BERT.

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
    padding_idx=pad_id
)

with torch.no_grad():

    # IDs numericos -> vetores
    vetores = embedding(
        input_ids
    )

    # Somar informacao posicional
    saida = pos_encoding(
        vetores
    )


print("\nVetores de entrada:", tuple(vetores.shape))
print("Vetores com posicao:", tuple(saida.shape))


# ============================================================
# 8 - TESTES DA INTEGRACAO
# ============================================================

B, T = input_ids.shape

assert saida.shape == (
    B,
    T,
    d_model
)

assert attention_mask.shape == (
    B,
    T
)

np.testing.assert_allclose(
    (saida - vetores).numpy(),
    pos_encoding.pe[:, :T, :].expand(
        B, T, d_model
    ).numpy(),
    atol=1e-6
)

print("\nPrimeiro token - antes:")
print(
    vetores[0, 0].numpy().round(4)
)

print("\nPrimeiro token - depois:")
print(
    saida[0, 0].numpy().round(4)
)

print("\nTokens validos na mascara:")
print(int(attention_mask.sum()))

print("\nTeste de integracao: OK")


# ============================================================
# 9 - SALVAR RESULTADOS
# ============================================================

relatorio = {

    "aula": 5,

    "tema": "Positional Encoding",

    "frase": frase,

    "palavras": palavras,

    "seq_len_exemplo": seq_len,

    "d_model": d_model,

    "matriz_posicional": pe.tolist(),

    "grafico": grafico.name,

    "input_ids_shape": list(
        input_ids.shape
    ),

    "embedding_shape": list(
        vetores.shape
    ),

    "saida_shape": list(
        saida.shape
    ),

    "tokens_validos": int(
        attention_mask.sum()
    ),

    "testes": {
        "matematica": True,
        "numpy_pytorch": True,
        "integracao": True
    },

    "treinamento_realizado": False
}

arquivo_relatorio = (
    RESULTADOS
    / "aula05_positional_encoding.json"
)

arquivo_relatorio.write_text(
    json.dumps(
        relatorio,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\nRelatorio salvo:")
print(arquivo_relatorio)

print("\nAULA 05 - EXECUCAO CONCLUIDA!")
