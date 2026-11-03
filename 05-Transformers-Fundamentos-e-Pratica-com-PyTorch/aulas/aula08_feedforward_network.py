"""
PROJETO 05 - TRANSFORMERS COM PYTORCH
AULA 08 - FEED-FORWARD NETWORK

Objetivos:
1. Comparar ReLU e GELU
2. Visualizar suas derivadas
3. Implementar FeedForward com PyTorch
4. Testar dimensoes e gradientes
5. Integrar com Multi-Head Attention
6. Salvar grafico e relatorio
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aula07_multi_head_attention import MultiHeadAttention


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 2 - CLASSE FEEDFORWARD
# ============================================================

class FeedForward(nn.Module):

    """
    Feed-Forward Network posicao-wise.

    Arquitetura:

    Linear(d_model, d_ff)
    GELU ou ReLU
    Dropout
    Linear(d_ff, d_model)
    Dropout

    A mesma rede e aplicada a cada posicao.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):

        super().__init__()

        if d_model <= 0:
            raise ValueError(
                "d_model deve ser positivo."
            )

        if d_ff is None:
            d_ff = 4 * d_model

        if d_ff <= 0:
            raise ValueError(
                "d_ff deve ser positivo."
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout deve estar entre 0 e 1."
            )

        activation = activation.lower()

        if activation == "gelu":
            act = nn.GELU()

        elif activation == "relu":
            act = nn.ReLU(inplace=True)

        else:
            raise ValueError(
                "Ativacao deve ser gelu ou relu."
            )

        self.d_model = d_model
        self.d_ff = d_ff

        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                d_ff
            ),

            act,

            nn.Dropout(dropout),

            nn.Linear(
                d_ff,
                d_model
            ),

            nn.Dropout(dropout)

        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if (
            x.ndim != 3
            or x.shape[-1] != self.d_model
        ):
            raise ValueError(
                "Esperado [batch, seq_len, d_model]."
            )

        return self.net(x)


# ============================================================
# 3 - COMPARAR RELU E GELU
# ============================================================

def comparar_ativacoes():

    print("\n1 - COMPARANDO RELU E GELU")

    x = np.linspace(
        -5,
        5,
        2000
    )

    # ReLU
    relu = np.maximum(
        0.0,
        x
    )

    # Aproximacao tanh do GELU
    gelu = (
        0.5 * x *
        (
            1 + np.tanh(
                np.sqrt(2 / np.pi)
                * (
                    x
                    + 0.044715 * np.power(x, 3)
                )
            )
        )
    )

    # Derivadas numericas
    drelu = np.gradient(
        relu,
        x
    )

    dgelu = np.gradient(
        gelu,
        x
    )

    print("Pontos calculados:", len(x))
    print("ReLU:", relu.shape)
    print("GELU:", gelu.shape)

    assert len(x) == 2000

    assert np.isfinite(
        gelu
    ).all()

    # Validar GELU contra PyTorch
    x_torch = torch.tensor(
        x,
        dtype=torch.float64
    )

    gelu_torch = F.gelu(
        x_torch,
        approximate="tanh"
    ).numpy()

    np.testing.assert_allclose(
        gelu,
        gelu_torch,
        atol=1e-10
    )

    print("GELU NumPy x PyTorch: OK")

    # ========================================================
    # 4 - GRAFICOS
    # ========================================================

    fig, axe = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    axe[0].plot(
        x,
        relu,
        label="ReLU"
    )

    axe[0].plot(
        x,
        gelu,
        label="GELU"
    )

    axe[0].set_title(
        "Funcoes de ativacao"
    )

    axe[0].set_xlabel("x")
    axe[0].set_ylabel("f(x)")

    axe[0].grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    axe[0].legend()

    # Derivadas
    axe[1].plot(
        x,
        drelu,
        label="Derivada ReLU"
    )

    axe[1].plot(
        x,
        dgelu,
        label="Derivada GELU"
    )

    axe[1].set_title(
        "Derivadas das funcoes"
    )

    axe[1].set_xlabel("x")
    axe[1].set_ylabel("f'(x)")

    axe[1].grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    axe[1].legend()

    plt.tight_layout()

    imagem = (
        RESULTADOS
        / "aula08_relu_gelu.png"
    )

    fig.savefig(
        imagem,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Grafico salvo:", imagem)

    return imagem


# ============================================================
# 5 - TESTAR CLASSE FEEDFORWARD
# ============================================================

def testar_feedforward():

    print("\n2 - TESTANDO FEEDFORWARD")

    d_model = 8
    d_ff = 4 * d_model

    modelo = FeedForward(
        d_model=d_model,
        d_ff=d_ff,
        dropout=0.1,
        activation="gelu"
    )

    entrada = torch.randn(
        2,
        6,
        d_model
    )

    modelo.eval()

    with torch.no_grad():

        saida = modelo(entrada)

    print(
        "Entrada:",
        tuple(entrada.shape)
    )

    print(
        "Saida:",
        tuple(saida.shape)
    )

    print("d_model:", d_model)
    print("d_ff:", d_ff)

    assert saida.shape == (
        2,
        6,
        8
    )

    # Parametros treinaveis
    parametros = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print(
        "Parametros treinaveis:",
        parametros
    )

    assert parametros == 552

    # ========================================================
    # 6 - VERIFICAR PROCESSAMENTO POR POSICAO
    # ========================================================

    # Modificar apenas uma posicao da sequencia.

    alterada = entrada.clone()

    alterada[:, 2, :] += 3.0

    with torch.no_grad():

        saida_alterada = modelo(
            alterada
        )

    # FFN nao mistura posicoes diferentes.
    for pos in range(6):

        if pos == 2:
            continue

        torch.testing.assert_close(
            saida[:, pos, :],
            saida_alterada[:, pos, :]
        )

    print(
        "Processamento independente por posicao: OK"
    )

    # ========================================================
    # 7 - VERIFICAR BACKPROPAGATION
    # ========================================================

    modelo.train()

    entrada_treino = torch.randn(
        2,
        6,
        8
    )

    predicao = modelo(
        entrada_treino
    )

    loss = predicao.square().mean()

    loss.backward()

    for nome, parametro in modelo.named_parameters():

        assert parametro.grad is not None, (
            f"Sem gradiente: {nome}"
        )

        assert torch.isfinite(
            parametro.grad
        ).all(), (
            f"Gradiente invalido: {nome}"
        )

    print(
        "Gradientes das camadas lineares: OK"
    )

    print(
        "Loss de teste:",
        round(loss.item(), 6)
    )

    # ========================================================
    # 8 - TESTAR ATIVACAO RELU
    # ========================================================

    modelo_relu = FeedForward(
        d_model=8,
        activation="relu",
        dropout=0.0
    )

    saida_relu = modelo_relu(
        torch.randn(2, 6, 8)
    )

    assert saida_relu.shape == (
        2,
        6,
        8
    )

    print("FeedForward com ReLU: OK")

    return parametros


# ============================================================
# 9 - INTEGRAR MULTI-HEAD COM FEEDFORWARD
# ============================================================

def integrar_transformer():

    print(
        "\n3 - INTEGRANDO MULTI-HEAD ATTENTION + FFN"
    )

    caminho = (
        RESULTADOS
        / "aula04_tokenizacao_bert.json"
    )

    if not caminho.exists():

        raise FileNotFoundError(
            "Dados da Aula 04 nao encontrados."
        )

    dados = json.loads(
        caminho.read_text(
            encoding="utf-8-sig"
        )
    )

    # Apenas 16 tokens por artigo para teste CPU
    input_ids = torch.tensor(
        dados["input_ids"],
        dtype=torch.long
    )[:, :16]

    attention_mask = torch.tensor(
        dados["attention_mask"],
        dtype=torch.bool
    )[:, :16]

    vocab_size = int(
        dados["vocab_size"]
    )

    pad_id = int(
        dados["pad_id"]
    )

    d_model = 8

    # Embedding demonstrativo
    embedding = nn.Embedding(
        vocab_size,
        d_model,
        padding_idx=pad_id
    )

    # Multi-Head da Aula 07
    multihead = MultiHeadAttention(
        d_model=8,
        n_heads=2,
        dropout=0.0
    ).eval()

    # FeedForward da Aula 08
    feedforward = FeedForward(
        d_model=8,
        d_ff=32,
        dropout=0.0,
        activation="gelu"
    ).eval()

    # Mascara causal + padding das chaves
    T = input_ids.shape[1]

    causal = torch.tril(
        torch.ones(
            T,
            T,
            dtype=torch.bool
        )
    )[None, None, :, :]

    mascara = (
        causal
        & attention_mask[:, None, None, :]
    )

    # Evitar linhas sem chave valida
    if not attention_mask[:, 0].all():
        raise ValueError(
            "Primeiro token invalido para mascara causal."
        )

    with torch.no_grad():

        x = embedding(
            input_ids
        )

        saida_attention, pesos = multihead(
            x,
            mask=mascara
        )

        saida_ffn = feedforward(
            saida_attention
        )

    print(
        "Tokens:",
        tuple(input_ids.shape)
    )

    print(
        "Embeddings:",
        tuple(x.shape)
    )

    print(
        "Multi-Head Attention:",
        tuple(saida_attention.shape)
    )

    print(
        "FeedForward:",
        tuple(saida_ffn.shape)
    )

    print(
        "Pesos de atencao:",
        tuple(pesos.shape)
    )

    assert saida_ffn.shape == (
        input_ids.shape[0],
        T,
        d_model
    )

    assert pesos.shape == (
        input_ids.shape[0],
        2,
        T,
        T
    )

    assert torch.isfinite(
        saida_ffn
    ).all()

    print("Integracao Multi-Head + FFN: OK")

    return {
        "tokens": list(input_ids.shape),
        "embeddings": list(x.shape),
        "multihead": list(saida_attention.shape),
        "feedforward": list(saida_ffn.shape),
        "pesos_atencao": list(pesos.shape)
    }


# ============================================================
# 10 - EXECUTAR AULA
# ============================================================

def main():

    print("=" * 60)
    print("AULA 08 - FEED-FORWARD NETWORK")
    print("=" * 60)

    imagem = comparar_ativacoes()

    parametros = testar_feedforward()

    integracao = integrar_transformer()

    # ========================================================
    # 11 - SALVAR RELATORIO
    # ========================================================

    relatorio = {

        "aula": 8,

        "tema": "Feed-Forward Network",

        "d_model": 8,

        "d_ff": 32,

        "activation": "gelu",

        "parametros_treinaveis_ffn": parametros,

        "grafico": imagem.name,

        "integracao": integracao,

        "testes": {
            "gelu_numpy_pytorch": True,
            "dimensoes": True,
            "processamento_por_posicao": True,
            "gradientes": True,
            "relu": True,
            "multihead_ffn": True
        },

        "treinamento_realizado": False
    }

    arquivo = (
        RESULTADOS
        / "aula08_feedforward_network.json"
    )

    arquivo.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nRelatorio salvo:")
    print(arquivo)

    print("\nAULA 08 VALIDADA!")


if __name__ == "__main__":
    main()
