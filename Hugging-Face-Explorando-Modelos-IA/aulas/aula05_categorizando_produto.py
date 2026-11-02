# ==========================================================
# HUGGING FACE - AULA 05
# CATEGORIZANDO UM PRODUTO COM ZERO-SHOT CLASSIFICATION
# ==========================================================

from pathlib import Path
import json
import csv

import torch
from transformers import pipeline


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "facebook/bart-large-mnli"

CATEGORIAS = [
    "electronics",
    "food",
    "toys",
    "books"
]

DESCRICAO = (
    "Latest model of smartphone with 5G "
    "connectivity and 128GB internal storage"
)

print("=" * 65)
print("AULA 05 - CATEGORIZANDO UM PRODUTO")
print("ZERO-SHOT CLASSIFICATION COM BART")
print("=" * 65)

print("\n1 - CONFIGURACAO")
print("Modelo:", MODELO)
print("Dispositivo: CPU")
print("Categorias:", CATEGORIAS)

torch.set_num_threads(
    min(4, torch.get_num_threads())
)


# ==========================================================
# 2 - CARREGAR MODELO PRE-TREINADO
# ==========================================================

print("\n2 - CARREGANDO BART LARGE MNLI")
print("A primeira execucao pode baixar os pesos.")
print("Aguarde o download do proprio Hugging Face.")

classificador = pipeline(
    "zero-shot-classification",
    model=MODELO,
    device=-1
)

print("\nMODELO CARREGADO COM SUCESSO!")


# ==========================================================
# 3 - DESCRICAO DO PRODUTO
# ==========================================================

print("\n3 - DESCRICAO DO PRODUTO")
print(DESCRICAO)

print("\nCATEGORIAS CANDIDATAS:")

for categoria in CATEGORIAS:
    print("-", categoria)


# ==========================================================
# 4 - ZERO-SHOT CLASSIFICATION
# ==========================================================

print("\n4 - EXECUTANDO CLASSIFICACAO")

predicao = classificador(
    DESCRICAO,
    candidate_labels=CATEGORIAS,
    multi_label=False
)

print("\nRESULTADO COMPLETO:")

print(predicao)


# ==========================================================
# 5 - EXTRAIR LABELS E SCORES
# ==========================================================

labels = predicao["labels"]
scores = predicao["scores"]

assert len(labels) == len(CATEGORIAS)
assert len(scores) == len(CATEGORIAS)

assert set(labels) == set(CATEGORIAS)

assert all(
    0 <= score <= 1
    for score in scores
)

assert abs(sum(scores) - 1.0) < 0.001


# ==========================================================
# 6 - EXIBIR CLASSIFICACAO ORDENADA
# ==========================================================

print("\n5 - RANKING DE CATEGORIAS")

for posicao, (label, score) in enumerate(
    zip(labels, scores),
    start=1
):

    percentual = score * 100

    print(
        f"{posicao}. {label:<15} "
        f"score={score:.6f} "
        f"({percentual:.2f}%)"
    )


# ==========================================================
# 7 - CATEGORIA PRINCIPAL
# ==========================================================

categoria_prevista = labels[0]
maior_score = float(scores[0])

print("\n6 - CATEGORIA DO PRODUTO")

print("Descricao:", DESCRICAO)
print("Categoria prevista:", categoria_prevista)

print(
    f"Score da categoria: {maior_score:.6f}"
)

print(
    f"Percentual relativo: {maior_score * 100:.2f}%"
)

print(
    "\nObservacao: os scores sao relativos "
    "as categorias candidatas informadas."
)


# ==========================================================
# 8 - SALVAR RESULTADO JSON
# ==========================================================

arquivo_json = (
    RESULTADOS
    / "aula05_categorizacao_produto.json"
)

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        {
            "modelo": MODELO,
            "tarefa": "zero-shot-classification",
            "descricao": DESCRICAO,
            "categorias_candidatas": CATEGORIAS,
            "categoria_prevista": categoria_prevista,
            "maior_score": maior_score,
            "labels": labels,
            "scores": [
                float(score)
                for score in scores
            ]
        },
        arquivo,
        ensure_ascii=False,
        indent=4
    )


# ==========================================================
# 9 - SALVAR RANKING CSV
# ==========================================================

arquivo_csv = (
    RESULTADOS
    / "aula05_ranking_categorias.csv"
)

with arquivo_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as arquivo:

    escritor = csv.writer(arquivo)

    escritor.writerow([
        "Posicao",
        "Categoria",
        "Score",
        "Percentual"
    ])

    for posicao, (label, score) in enumerate(
        zip(labels, scores),
        start=1
    ):

        escritor.writerow([
            posicao,
            label,
            float(score),
            round(score * 100, 4)
        ])


# ==========================================================
# 10 - VALIDACAO FINAL
# ==========================================================

assert arquivo_json.exists()
assert arquivo_csv.exists()

print("\n" + "=" * 65)
print("7 - VALIDACAO FINAL")
print("=" * 65)

print("Pipeline zero-shot: OK")
print("Modelo BART Large MNLI: OK")
print("Quatro categorias avaliadas: OK")
print("Scores normalizados: OK")
print("JSON salvo: OK")
print("CSV salvo: OK")

if categoria_prevista == "electronics":
    print("CATEGORIA ESPERADA: CONFIRMADA")
else:
    print(
        "ATENCAO: categoria diferente da "
        "apresentada na aula."
    )

print("\nArquivos gerados:")
print(arquivo_json)
print(arquivo_csv)

print("\nAULA 05 EXECUTADA COM SUCESSO!")
