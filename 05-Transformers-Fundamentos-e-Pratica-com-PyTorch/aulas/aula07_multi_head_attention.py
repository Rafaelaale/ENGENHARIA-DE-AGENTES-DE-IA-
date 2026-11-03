"""
PROJETO 05 - TRANSFORMERS COM PYTORCH
AULA 07 - MULTI-HEAD ATTENTION
"""

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aula06_self_attention import ScaledDotProductAttention


BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)


# ============================================================
# 1 - CLASSE MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    """
    Multi-Head Self-Attention.

    Projeta entrada em Q, K e V.
    Separa as cabecas.
    Calcula a atencao em paralelo.
    Concatena as cabecas.
    Realiza a projecao de saida.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):

        super().__init__()

        if (
            d_model <= 0
            or n_heads <= 0
            or d_model % n_heads != 0
        ):
            raise ValueError(
                "d_model deve ser divisivel por n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projecoes treinaveis
        self.q_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        self.k_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        self.v_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        # Projecao final
        self.out_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        # Reutilizacao da Aula 06
        self.sdpa = ScaledDotProductAttention(
            self.d_k,
            dropout=dropout
        )

        self.proj_drop = nn.Dropout(dropout)

    def _split_heads(self, x):

        B, T, C = x.shape

        return x.reshape(
            B,
            T,
            self.n_heads,
            self.d_k
        ).transpose(1, 2)

    def _merge_heads(self, x):

        B, H, T, D = x.shape

        return x.transpose(
            1, 2
        ).contiguous().reshape(
            B,
            T,
            H * D
        )

    def forward(self, x, mask=None):

        if (
            x.ndim != 3
            or x.shape[-1] != self.d_model
        ):
            raise ValueError(
                "Esperado [B, T, d_model]."
            )

        Q = self._split_heads(
            self.q_proj(x)
        )

        K = self._split_heads(
            self.k_proj(x)
        )

        V = self._split_heads(
            self.v_proj(x)
        )

        y, pesos = self.sdpa(
            Q, K, V, mask=mask
        )

        y = self._merge_heads(y)

        y = self.out_proj(y)

        y = self.proj_drop(y)

        return y, pesos


# ============================================================
# 2 - EXEMPLO PRATICO DO PROFESSOR
# ============================================================

def main():

    print("=" * 60)
    print("AULA 07 - MULTI-HEAD ATTENTION")
    print("=" * 60)

    palavras = [
        "O",
        "gato",
        "brinca",
        "com",
        "o",
        "cachorro"
    ]

    seq_len = len(palavras)
    n_heads = 2
    d_model = 8

    d_k = d_model // n_heads

    # ========================================================
    # 3 - REUTILIZAR POSITIONAL ENCODING DA AULA 05
    # ========================================================

    arquivo_pe = (
        RESULTADOS
        / "aula05_positional_encoding.json"
    )

    if not arquivo_pe.exists():
        raise FileNotFoundError(
            "Execute a Aula 05 antes da Aula 07."
        )

    dados_pe = json.loads(
        arquivo_pe.read_text(
            encoding="utf-8-sig"
        )
    )

    pe = torch.tensor(
        dados_pe["matriz_posicional"],
        dtype=torch.float32
    )

    assert pe.shape == (
        seq_len,
        d_model
    )

    # Embeddings aleatorios
    word_embeddings = torch.randn(
        seq_len,
        d_model
    ) * 0.2

    # Adicionar informacao posicional
    x = word_embeddings + pe

    # Adicionar batch
    x = x.unsqueeze(0)

    print("\nPalavras:", palavras)
    print("Entrada:", tuple(x.shape))
    print("Numero de cabecas:", n_heads)
    print("Dimensao por cabeca:", d_k)

    # ========================================================
    # 4 - PROJECOES MANUAIS Q, K E V
    # ========================================================

    W_Q = nn.Linear(
        d_model, d_model, bias=False
    )

    W_K = nn.Linear(
        d_model, d_model, bias=False
    )

    W_V = nn.Linear(
        d_model, d_model, bias=False
    )

    def split_heads(tensor):

        B, T, C = tensor.shape

        return tensor.reshape(
            B, T, n_heads, d_k
        ).transpose(1, 2)

    Q = split_heads(W_Q(x))
    K = split_heads(W_K(x))
    V = split_heads(W_V(x))

    print("\nQ:", tuple(Q.shape))
    print("K:", tuple(K.shape))
    print("V:", tuple(V.shape))

    # ========================================================
    # 5 - SCALED DOT-PRODUCT ATTENTION
    # ========================================================

    scores = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(d_k)

    pesos = F.softmax(
        scores,
        dim=-1
    )

    saida_cabecas = pesos @ V

    print("\nScores:", tuple(scores.shape))
    print("Pesos:", tuple(pesos.shape))
    print("Saida por cabeca:", tuple(saida_cabecas.shape))

    # ========================================================
    # 6 - CONCATENAR CABECAS
    # ========================================================

    saida_concatenada = (
        saida_cabecas
        .transpose(1, 2)
        .contiguous()
        .reshape(
            1,
            seq_len,
            d_model
        )
    )

    print(
        "\nSaida concatenada:",
        tuple(saida_concatenada.shape)
    )

    torch.testing.assert_close(
        pesos.sum(dim=-1),
        torch.ones_like(
            pesos.sum(dim=-1)
        )
    )

    print("\nTeste de Softmax: OK")

    # ========================================================
    # 7 - VALIDAR CLASSE MULTI-HEAD ATTENTION
    # ========================================================

    modelo = MultiHeadAttention(
        d_model=8,
        n_heads=2,
        dropout=0.0
    )

    modelo.eval()

    # Copiar as mesmas projecoes do exemplo manual.
    # A projecao final identidade permite comparar
    # a concatenacao diretamente.

    with torch.no_grad():

        modelo.q_proj.weight.copy_(
            W_Q.weight
        )

        modelo.k_proj.weight.copy_(
            W_K.weight
        )

        modelo.v_proj.weight.copy_(
            W_V.weight
        )

        nn.init.eye_(
            modelo.out_proj.weight
        )

        saida_classe, pesos_classe = modelo(x)

    torch.testing.assert_close(
        pesos_classe,
        pesos,
        atol=1e-6,
        rtol=1e-5
    )

    torch.testing.assert_close(
        saida_classe,
        saida_concatenada,
        atol=1e-6,
        rtol=1e-5
    )

    print("Calculo manual versus classe: OK")

    # ========================================================
    # 8 - TESTAR MASCARA CAUSAL
    # ========================================================

    mascara_causal = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool
        )
    )[None, None]

    _, pesos_causais = modelo(
        x,
        mask=mascara_causal
    )

    bloqueados = ~mascara_causal.expand_as(
        pesos_causais
    )

    assert torch.all(
        pesos_causais.masked_select(
            bloqueados
        ) == 0
    )

    print("Mascara causal nas 2 cabecas: OK")

    # ========================================================
    # 9 - TESTAR GRADIENTES TREINAVEIS
    # ========================================================

    modelo_treino = MultiHeadAttention(
        d_model=8,
        n_heads=2,
        dropout=0.1
    )

    entrada_treino = torch.randn(
        2, 5, 8
    )

    saida_treino, _ = modelo_treino(
        entrada_treino
    )

    perda = saida_treino.square().mean()

    perda.backward()

    for nome, parametro in modelo_treino.named_parameters():

        assert parametro.grad is not None, (
            f"Sem gradiente: {nome}"
        )

        assert torch.isfinite(
            parametro.grad
        ).all(), (
            f"Gradiente invalido: {nome}"
        )

    print("Gradientes das projecoes: OK")

    # ========================================================
    # 10 - INTEGRAR COM ARTIGOS DA AULA 04
    # ========================================================

    arquivo_tokens = (
        RESULTADOS
        / "aula04_tokenizacao_bert.json"
    )

    if not arquivo_tokens.exists():
        raise FileNotFoundError(
            "Nao encontrei os tokens da Aula 04."
        )

    dados = json.loads(
        arquivo_tokens.read_text(
            encoding="utf-8-sig"
        )
    )

    # Apenas 16 tokens por artigo para teste na CPU.
    ids = torch.tensor(
        dados["input_ids"],
        dtype=torch.long
    )[:, :16]

    mascara_tokens = torch.tensor(
        dados["attention_mask"],
        dtype=torch.bool
    )[:, :16]

    embeddings = nn.Embedding(
        int(dados["vocab_size"]),
        8,
        padding_idx=int(dados["pad_id"])
    )

    x_real = embeddings(ids)

    T = ids.shape[1]

    causal = torch.tril(
        torch.ones(
            T,
            T,
            dtype=torch.bool
        )
    )[None, None]

    # Permitir apenas tokens existentes e passados.
    mascara = (
        mascara_tokens[:, None, None, :]
        & causal
    )

    modelo_real = MultiHeadAttention(
        d_model=8,
        n_heads=2,
        dropout=0.1
    )

    modelo_real.eval()

    with torch.no_grad():

        y_real, attn_real = modelo_real(
            x_real,
            mask=mascara
        )

    assert y_real.shape == (
        ids.shape[0],
        T,
        8
    )

    assert attn_real.shape == (
        ids.shape[0],
        2,
        T,
        T
    )

    assert torch.all(
        attn_real.masked_select(
            ~mascara.expand_as(attn_real)
        ) == 0
    )

    print("\nIntegracao com Aula 04: OK")

    print("Tokens:", tuple(ids.shape))
    print("Saida:", tuple(y_real.shape))
    print("Atencao:", tuple(attn_real.shape))

    # ========================================================
    # 11 - VISUALIZAR AS DUAS CABECAS
    # ========================================================

    fig, eixos = plt.subplots(
        1,
        2,
        figsize=(11, 4.5)
    )

    matriz = pesos.detach().numpy()[0]

    for cabeca, ax in enumerate(eixos):

        im = ax.imshow(
            matriz[cabeca],
            cmap="Blues",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"Cabeca {cabeca + 1} - Pesos iniciais"
        )

        ax.set_xlabel("Key - palavra observada")
        ax.set_ylabel("Query - palavra consultando")

        ax.set_xticks(
            range(seq_len),
            palavras,
            rotation=45,
            ha="right"
        )

        ax.set_yticks(
            range(seq_len),
            palavras
        )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

    fig.tight_layout()

    imagem = (
        RESULTADOS
        / "aula07_multi_head_attention.png"
    )

    fig.savefig(
        imagem,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # 12 - SALVAR RESULTADOS
    # ========================================================

    relatorio = {

        "aula": 7,

        "tema": "Multi-Head Attention",

        "palavras": palavras,

        "d_model": d_model,

        "n_heads": n_heads,

        "d_k": d_k,

        "shape_qkv": list(Q.shape),

        "shape_pesos": list(pesos.shape),

        "shape_saida_concatenada": list(
            saida_concatenada.shape
        ),

        "pesos_por_cabeca": (
            pesos.detach().tolist()
        ),

        "integracao": {
            "tokens": list(ids.shape),
            "saida": list(y_real.shape),
            "pesos": list(attn_real.shape)
        },

        "testes": {
            "softmax": True,
            "manual_vs_classe": True,
            "mascara_causal": True,
            "gradientes": True,
            "aula04": True
        },

        "treinamento_realizado": False,

        "grafico": imagem.name
    }

    arquivo_json = (
        RESULTADOS
        / "aula07_multi_head_attention.json"
    )

    arquivo_json.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nGrafico salvo:")
    print(imagem)

    print("\nRelatorio salvo:")
    print(arquivo_json)

    print("\nAULA 07 VALIDADA!")


if __name__ == "__main__":
    main()
