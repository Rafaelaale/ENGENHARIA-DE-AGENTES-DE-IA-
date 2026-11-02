# ==========================================================
# HUGGING FACE - AULA 06
# ZERO-SHOT CLASSIFICATION MULTILINGUE COM T5 E LIQFIT
# ==========================================================

from pathlib import Path
import json
import math

import pandas as pd
import torch

from liqfit.pipeline import ZeroShotClassificationPipeline
from liqfit.models import T5ForZeroShotClassification
from transformers import T5Tokenizer


BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = (
    "knowledgator/"
    "comprehend_it-multilingual-t5-base"
)

CATEGORIAS = [
    "beleza",
    "cozinha",
    "livros"
]


# ==========================================================
# 1 - INICIALIZACAO
# ==========================================================

print("=" * 65)
print("AULA 06 - CLASSIFICACAO ZERO-SHOT EM PORTUGUES")
print("=" * 65)

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

print("\n1 - CARREGANDO MODELO MULTILINGUE")
print("Modelo:", MODELO)
print("Tarefa: Zero-Shot Classification")
print("Dispositivo: CPU")

modelo = T5ForZeroShotClassification.from_pretrained(
    MODELO
)

tokenizador = T5Tokenizer.from_pretrained(
    MODELO
)

modelo.eval()

classificador = ZeroShotClassificationPipeline(
    model=modelo,
    tokenizer=tokenizador,
    hypothesis_template="{}",
    encoder_decoder=True
)

print("\nMODELO E TOKENIZADOR CARREGADOS!")


# ==========================================================
# 2 - DESCRICOES DA ALURA
# ==========================================================

descricao_gel = """
Este water gel leve e refrescante,
proporciona hidratação imediata que ajuda
a aliviar o repuxamento e aspereza da
pele sensível.
"""

descricao_fritadeira = """
A fritadeira eletrica sem óleo start fry da elgin
possui um design único, capacidade para até 3,5 litros,
potência de 1400w e revestimento antiaderente.
Seu sistema de circulação de ar ultra rápido frita
e economiza energia. Sua grelha de fritura é removível
e super fácil de limpar. Ela conta com uma proteção
contra super aquecimento.
Possui controle de temperatura de 80°c a 200°c
que permite você programar a temperatura de preparo
para cada tipo de alimento, timer para até 60 minutos
com aviso sonoro e desligamento automático,
assim você pode deixar preparando sua refeição
enquanto realiza outras tarefas.
"""


# ==========================================================
# 3 - FUNCAO PARA CLASSIFICAR PRODUTOS
# ==========================================================

def categorizar_produto(nome, descricao, esperado, arquivo):

    print("\n" + "-" * 65)
    print("PRODUTO:", nome)
    print("-" * 65)

    descricao = " ".join(
        descricao.split()
    )

    print("\nDESCRICAO:")
    print(descricao)

    print("\nCATEGORIAS:")
    print(CATEGORIAS)

    print("\nPROCESSANDO COM T5...")

    resultado = classificador(
        descricao,
        CATEGORIAS,
        multi_label=False
    )

    # Mesmo procedimento da professora:
    # converter resultado em DataFrame
    # e remover a coluna sequence.

    df_resultado = (
        pd.DataFrame(resultado)
        .drop(["sequence"], axis=1)
    )

    df_resultado = df_resultado.sort_values(
        "scores",
        ascending=False
    ).reset_index(drop=True)

    print("\nRANKING DE CLASSIFICACAO:")
    print(
        df_resultado.to_string(
            index=False
        )
    )

    labels = df_resultado["labels"].tolist()
    scores = df_resultado["scores"].tolist()

    assert len(labels) == 3
    assert set(labels) == set(CATEGORIAS)

    assert all(
        math.isfinite(float(score))
        and 0 <= score <= 1
        for score in scores
    )

    assert abs(sum(scores) - 1.0) < 0.01

    categoria_prevista = labels[0]
    score_principal = float(scores[0])

    print("\nRESULTADO FINAL:")
    print(
        "Categoria prevista:",
        categoria_prevista
    )

    print(
        "Categoria esperada:",
        esperado
    )

    print(
        "Score principal:",
        f"{score_principal:.6f}"
    )

    print(
        "Correspondencia:",
        "SIM"
        if categoria_prevista == esperado
        else "NAO"
    )

    destino = RESULTADOS / arquivo

    df_resultado.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCSV SALVO:")
    print(destino)

    return {
        "produto": nome,
        "descricao": descricao,
        "esperado": esperado,
        "previsto": categoria_prevista,
        "corresponde": categoria_prevista == esperado,
        "labels": labels,
        "scores": [
            float(score)
            for score in scores
        ]
    }


# ==========================================================
# 4 - CLASSIFICAR GEL HIDRATANTE
# ==========================================================

print("\n2 - CLASSIFICACAO DO GEL FACIAL")

resultado_gel = categorizar_produto(
    nome="Gel hidratante facial",
    descricao=descricao_gel,
    esperado="beleza",
    arquivo="aula06_gel_facial_categorias.csv"
)


# ==========================================================
# 5 - CLASSIFICAR FRITADEIRA ELETRICA
# ==========================================================

print("\n3 - CLASSIFICACAO DA FRITADEIRA")

resultado_fritadeira = categorizar_produto(
    nome="Fritadeira eletrica",
    descricao=descricao_fritadeira,
    esperado="cozinha",
    arquivo="aula06_fritadeira_categorias.csv"
)


# ==========================================================
# 6 - SALVAR TODOS OS RESULTADOS
# ==========================================================

resultados = [
    resultado_gel,
    resultado_fritadeira
]

arquivo_json = (
    RESULTADOS
    / "aula06_categorizacao_portugues.json"
)

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        resultados,
        f,
        ensure_ascii=False,
        indent=4
    )


# ==========================================================
# 7 - RESUMO E VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("4 - RESUMO DA AULA 06")
print("=" * 65)

correspondencias = sum(
    item["corresponde"]
    for item in resultados
)

for item in resultados:
    print(
        f"{item['produto']}: "
        f"{item['previsto']}"
    )

print(
    "\nClassificacoes iguais ao esperado:",
    f"{correspondencias}/2"
)

assert len(resultados) == 2
assert arquivo_json.exists()

print("\nVALIDACAO FINAL")
print("Modelo T5 multilingue: OK")
print("Pipeline LiqFit: OK")
print("Textos em portugues: OK")
print("Tres categorias por produto: OK")
print("Scores normalizados: OK")
print("Dois arquivos CSV: OK")
print("Resultados JSON: OK")

print("\nAULA 06 EXECUTADA COM SUCESSO!")
