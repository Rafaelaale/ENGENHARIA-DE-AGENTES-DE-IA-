"""
PROJETO 05 - TRANSFORMERS COM PYTORCH
AULA 06 - SCALED DOT-PRODUCT SELF-ATTENTION
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 1 - CLASSE DE ATENCAO EM PYTORCH
# ============================================================

class ScaledDotProductAttention(nn.Module):

    """
    Q, K, V: [batch, heads, seq_len, d_k]

    Mascara:
        True = permitir
        False = bloquear

    Retorna:
        saida da atencao
        pesos da atencao
    """

    def __init__(self, d_k: int, dropout: float = 0.1):

        super().__init__()

        if d_k <= 0:
            raise ValueError("d_k deve ser positivo.")

        self.scale = 1.0 / math.sqrt(d_k)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):

        # Produto escalar escalado
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) * self.scale

        # Mascara antes do softmax
        if mask is not None:

            mask = mask.to(
                device=scores.device,
                dtype=torch.bool
            )

            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

            if torch.isneginf(scores).all(dim=-1).any():
                raise ValueError(
                    "Uma consulta possui todas "
                    "as chaves bloqueadas."
                )

        # Pesos normalizados
        pesos = F.softmax(
            scores,
            dim=-1
        )

        # Dropout aplicado na combinacao
        saida = torch.matmul(
            self.drop(pesos),
            V
        )

        return saida, pesos


# ============================================================
# 2 - EXEMPLO DO PROFESSOR COM NUMPY
# ============================================================

def main():

    print("=" * 60)
    print("AULA 06 - SCALED DOT-PRODUCT ATTENTION")
    print("=" * 60)

    palavras = ["Eu", "Jogo", "Tibia"]

    # QUERY
    Q = np.array([
        [0.62, 0.27, 0.11],
        [0.33, 0.41, 0.26],
        [0.18, 0.29, 0.53]
    ], dtype=np.float64)

    # KEY
    K = np.array([
        [0.65, 0.22, 0.13],
        [0.28, 0.44, 0.28],
        [0.15, 0.25, 0.60]
    ], dtype=np.float64)

    # VALUE
    V = np.array([
        [0.58, 0.30, 0.12],
        [0.29, 0.37, 0.34],
        [0.20, 0.31, 0.49]
    ], dtype=np.float64)

    print("\nPalavras:", palavras)

    # ========================================================
    # 3 - CALCULAR SCORES
    # ========================================================

    score = Q @ K.T

    d_k = Q.shape[1]

    scaled_score = (
        score / math.sqrt(d_k)
    )

    print("\nScore Q @ K.T:")
    print(np.round(score, 4))

    print("\nScore escalado:")
    print(np.round(scaled_score, 4))

    # ========================================================
    # 4 - SOFTMAX
    # ========================================================

    # Subtrair o maximo melhora a estabilidade
    # numerica sem alterar o resultado do softmax.

    exps = np.exp(
        scaled_score
        - scaled_score.max(
            axis=1,
            keepdims=True
        )
    )

    attention_weights = (
        exps
        / exps.sum(
            axis=1,
            keepdims=True
        )
    )

    # Combinacao ponderada dos valores
    out_numpy = attention_weights @ V

    print("\nPesos de atencao:")
    print(np.round(attention_weights, 4))

    print("\nSaida da atencao:")
    print(np.round(out_numpy, 4))

    # ========================================================
    # 5 - RELACOES ENTRE PALAVRAS
    # ========================================================

    for i, palavra in enumerate(palavras):

        print(f"\n{palavra} observa:")

        for j, destino in enumerate(palavras):

            print(
                f"  {destino}: "
                f"{attention_weights[i, j]:.3f}"
            )

    # As linhas devem somar 1
    np.testing.assert_allclose(
        attention_weights.sum(axis=1),
        1.0,
        atol=1e-12
    )

    assert attention_weights.shape == (3, 3)

    print("\nTeste do Softmax: OK")

    # ========================================================
    # 6 - VISUALIZACAO DAS MATRIZES
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    graficos = [
        (
            axes[0],
            scaled_score,
            "Similaridade escalada"
        ),
        (
            axes[1],
            attention_weights,
            "Pesos de Self-Attention"
        )
    ]

    for ax, matriz, titulo in graficos:

        im = ax.imshow(
            matriz,
            cmap="Blues",
            aspect="equal"
        )

        ax.set_xticks(
            range(3),
            palavras
        )

        ax.set_yticks(
            range(3),
            palavras
        )

        ax.set_xlabel(
            "Palavra observada (Key)"
        )

        ax.set_ylabel(
            "Palavra consultando (Query)"
        )

        ax.set_title(titulo)

        for i in range(3):
            for j in range(3):

                ax.text(
                    j,
                    i,
                    f"{matriz[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9
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
        / "aula06_self_attention.png"
    )

    fig.savefig(
        imagem,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nGrafico gerado.")

    # ========================================================
    # 7 - COMPARAR NUMPY E PYTORCH
    # ========================================================

    modulo = ScaledDotProductAttention(
        d_k=3,
        dropout=0.1
    )

    # Modo avaliacao desativa o dropout
    modulo.eval()

    Qt = torch.tensor(
        Q,
        dtype=torch.float64
    )[None, None, :, :]

    Kt = torch.tensor(
        K,
        dtype=torch.float64
    )[None, None, :, :]

    Vt = torch.tensor(
        V,
        dtype=torch.float64
    )[None, None, :, :]

    with torch.no_grad():

        saida_torch, pesos_torch = modulo(
            Qt,
            Kt,
            Vt
        )

    np.testing.assert_allclose(
        pesos_torch[0, 0].numpy(),
        attention_weights,
        atol=1e-12
    )

    np.testing.assert_allclose(
        saida_torch[0, 0].numpy(),
        out_numpy,
        atol=1e-12
    )

    print("\nNumPy e PyTorch equivalentes: OK")

    # ========================================================
    # 8 - TESTAR MASCARA CAUSAL
    # ========================================================

    # Aposicao i nao pode consultar
    # posicoes futuras j > i.

    mascara_causal = torch.tril(
        torch.ones(
            3,
            3,
            dtype=torch.bool
        )
    )[None, None]

    with torch.no_grad():

        _, pesos_causais = modulo(
            Qt,
            Kt,
            Vt,
            mask=mascara_causal
        )

    assert torch.all(
        pesos_causais[0, 0].triu(
            diagonal=1
        ) == 0
    )

    torch.testing.assert_close(
        pesos_causais.sum(dim=-1),
        torch.ones(
            1,
            1,
            3,
            dtype=torch.float64
        )
    )

    print("Mascara causal: OK")

    # ========================================================
    # 9 - INTEGRACAO COM A AULA 04
    # ========================================================

    arquivo_anterior = (
        RESULTADOS
        / "aula04_tokenizacao_bert.json"
    )

    integracao = None

    if arquivo_anterior.exists():

        dados = json.loads(
            arquivo_anterior.read_text(
                encoding="utf-8-sig"
            )
        )

        # Usar apenas 16 tokens de cada artigo
        # para demonstracao eficiente na CPU.

        ids = torch.tensor(
            dados["input_ids"],
            dtype=torch.long
        )[:, :16]

        mask_tokens = torch.tensor(
            dados["attention_mask"],
            dtype=torch.bool
        )[:, :16]

        # Embedding de demonstracao.
        # Nao carrega pesos BERT.
        embedding = nn.Embedding(
            int(dados["vocab_size"]),
            8,
            padding_idx=int(dados["pad_id"])
        )

        with torch.no_grad():

            x = embedding(ids).unsqueeze(1)

            # [B, 1, 1, T]
            mask_keys = mask_tokens[
                :, None, None, :
            ]

            modulo_real = ScaledDotProductAttention(
                d_k=8,
                dropout=0.1
            ).eval()

            saida_real, pesos_reais = modulo_real(
                x,
                x,
                x,
                mask=mask_keys
            )

        print("\nIntegracao com tokens reais:")
        print("IDs:", tuple(ids.shape))
        print("Q/K/V:", tuple(x.shape))
        print("Saida:", tuple(saida_real.shape))
        print("Pesos:", tuple(pesos_reais.shape))

        # Validar dimensoes
        assert saida_real.shape == (
            ids.shape[0],
            1,
            ids.shape[1],
            8
        )

        assert pesos_reais.shape == (
            ids.shape[0],
            1,
            ids.shape[1],
            ids.shape[1]
        )

        # Cada linha de pesos soma 1
        torch.testing.assert_close(
            pesos_reais.sum(dim=-1),
            torch.ones_like(
                pesos_reais.sum(dim=-1)
            )
        )

        # Posicoes de padding devem receber peso zero
        assert torch.all(
            pesos_reais.masked_select(
                ~mask_keys.expand_as(
                    pesos_reais
                )
            ) == 0
        )

        integracao = {
            "input_ids_shape": list(ids.shape),
            "attention_output_shape": list(
                saida_real.shape
            ),
            "attention_weights_shape": list(
                pesos_reais.shape
            )
        }

        print("\nIntegracao Aula 04: OK")

    else:

        print(
            "\nAviso: dados da Aula 04 nao encontrados."
        )

    # ========================================================
    # 10 - SALVAR RELATORIO
    # ========================================================

    relatorio = {

        "aula": 6,

        "tema": "Scaled Dot-Product Attention",

        "palavras": palavras,

        "Q": Q.tolist(),
        "K": K.tolist(),
        "V": V.tolist(),

        "scores": score.tolist(),

        "scaled_scores": scaled_score.tolist(),

        "attention_weights": (
            attention_weights.tolist()
        ),

        "attention_output": (
            out_numpy.tolist()
        ),

        "grafico": imagem.name,

        "integracao_aula04": integracao,

        "treinamento_realizado": False,

        "testes": {
            "softmax": True,
            "numpy_vs_torch": True,
            "mascara_causal": True
        }
    }

    arquivo_json = (
        RESULTADOS
        / "aula06_self_attention.json"
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

    print("\nAULA 06 VALIDADA!")


if __name__ == "__main__":
    main()
