# ==========================================================
# HUGGING FACE - AULA 07
# APLICANDO O MODELO T5 AOS PRODUTOS DO E-COMMERCE
# ==========================================================

from pathlib import Path
import json

import pandas as pd
import torch

from liqfit.pipeline import ZeroShotClassificationPipeline
from liqfit.models import T5ForZeroShotClassification
from transformers import T5Tokenizer


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = (
    "knowledgator/"
    "comprehend_it-multilingual-t5-base"
)

ARQUIVO_DADOS = (
    DADOS / "descricoes_produtos.csv"
)

URL = (
    "https://raw.githubusercontent.com/"
    "alura-cursos/hugging_face/main/"
    "Dados/descricoes_produtos.csv"
)

categorias = [
    "eletrodomésticos",
    "eletrônicos",
    "beleza",
    "brinquedos"
]

print("=" * 65)
print("AULA 07 - CATEGORIZACAO AUTOMATICA DE PRODUTOS")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR OS DADOS
# ==========================================================

print("\n1 - CARREGANDO DADOS DO E-COMMERCE")

if ARQUIVO_DADOS.exists():

    dados = pd.read_csv(
        ARQUIVO_DADOS
    )

else:

    print("Baixando CSV oficial da Alura...")

    dados = pd.read_csv(URL)

    dados.to_csv(
        ARQUIVO_DADOS,
        index=False,
        encoding="utf-8-sig"
    )

assert "Descrição" in dados.columns, (
    "Coluna Descrição nao encontrada."
)

assert not dados.empty
assert dados["Descrição"].notna().all()

assert dados["Descrição"].astype(str).str.strip().ne("").all()

print("Total de produtos:", len(dados))

print("\nPRIMEIRAS DESCRICOES:")
print(
    dados.head().to_string(
        index=False
    )
)


# ==========================================================
# 3 - CARREGAR MODELO MULTILINGUE
# ==========================================================

print("\n2 - CARREGANDO MODELO T5")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

modelo = T5ForZeroShotClassification.from_pretrained(
    MODELO
)

tokenizador = T5Tokenizer.from_pretrained(
    MODELO
)

modelo.eval()

classifier = ZeroShotClassificationPipeline(
    model=modelo,
    tokenizer=tokenizador,
    hypothesis_template="{}",
    encoder_decoder=True
)

print("MODELO CARREGADO COM SUCESSO!")


# ==========================================================
# 4 - DEFINIR FUNCAO CATEGORIZAR
# ==========================================================

print("\n3 - DEFININDO FUNCAO CATEGORIZAR")

contador = 0

def categorizar(descricao):

    global contador

    contador += 1

    # Classificacao Zero-Shot
    resultado = classifier(
        str(descricao),
        categorias,
        multi_label=False
    )

    # Obter categoria com maior score
    categoria_max = max(
        zip(
            resultado["labels"],
            resultado["scores"]
        ),
        key=lambda x: x[1]
    )[0]

    print(
        f"[{contador}/{len(dados)}] "
        f"{categoria_max}"
    )

    return categoria_max


# ==========================================================
# 5 - APLICAR A FUNCAO AOS DADOS
# ==========================================================

print("\n4 - CATEGORIZANDO TODOS OS PRODUTOS")

dados["Categoria"] = (
    dados["Descrição"].apply(
        categorizar
    )
)

assert len(dados) == contador

assert dados["Categoria"].notna().all()

assert set(dados["Categoria"]).issubset(
    set(categorias)
)

print("\nCLASSIFICACAO CONCLUIDA!")


# ==========================================================
# 6 - EXIBIR DATAFRAME COMPLETO
# ==========================================================

print("\n" + "=" * 65)
print("5 - DATAFRAME DE PRODUTOS CATEGORIZADOS")
print("=" * 65)

with pd.option_context(
    "display.max_rows", None,
    "display.max_colwidth", None,
    "display.width", 220
):

    print(
        dados[
            ["Descrição", "Categoria"]
        ].to_string(index=True)
    )


# ==========================================================
# 7 - CONTAGEM POR CATEGORIA
# ==========================================================

print("\n6 - CONTAGEM POR CATEGORIA")

contagem = (
    dados.groupby("Categoria")
    .size()
    .reset_index(name="Quantidade")
)

contagem = (
    contagem
    .set_index("Categoria")
    .reindex(
        categorias,
        fill_value=0
    )
    .reset_index()
)

print(
    contagem.to_string(index=False)
)

assert contagem["Quantidade"].sum() == len(dados)


# ==========================================================
# 8 - COMPARAR COM A REFERENCIA DA ALURA
# ==========================================================

print("\n7 - COMPARACAO COM A TRANSCRICAO")

categorias_esperadas = (
    ["eletrodomésticos"] * 5
    + ["eletrônicos"] * 7
    + ["beleza"] * 7
    + ["brinquedos"] * 3
)

if len(dados) == 22:

    correspondencias = sum(
        prevista == esperada
        for prevista, esperada in zip(
            dados["Categoria"],
            categorias_esperadas
        )
    )

    print(
        "Classificacoes iguais a referencia:",
        f"{correspondencias}/22"
    )

    for indice, (prevista, esperada) in enumerate(
        zip(
            dados["Categoria"],
            categorias_esperadas
        )
    ):

        if prevista != esperada:

            print(
                f"Diferenca na linha {indice}: "
                f"esperado={esperada}, "
                f"previsto={prevista}"
            )

else:

    correspondencias = None

    print(
        "Quantidade de produtos diferente "
        "dos 22 exemplos da transcricao."
    )


# ==========================================================
# 9 - SALVAR CSV DOS PRODUTOS
# ==========================================================

arquivo_final = (
    RESULTADOS
    / "aula07_produtos_categorizados.csv"
)

dados.to_csv(
    arquivo_final,
    index=False,
    encoding="utf-8-sig"
)

print("\n8 - CSV DE PRODUTOS SALVO")
print(arquivo_final)


# ==========================================================
# 10 - SALVAR CONTAGEM DAS CATEGORIAS
# ==========================================================

arquivo_contagem = (
    RESULTADOS
    / "aula07_contagem_categorias.csv"
)

contagem.to_csv(
    arquivo_contagem,
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV DE CONTAGEM SALVO")
print(arquivo_contagem)


# ==========================================================
# 11 - SALVAR RESUMO JSON
# ==========================================================

arquivo_json = (
    RESULTADOS
    / "aula07_resumo_categorizacao.json"
)

resumo = {
    "modelo": MODELO,
    "total_produtos": len(dados),
    "categorias": categorias,
    "contagem": {
        str(linha["Categoria"]): int(
            linha["Quantidade"]
        )
        for _, linha in contagem.iterrows()
    },
    "correspondencias_alura": correspondencias
}

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        resumo,
        arquivo,
        indent=4,
        ensure_ascii=False
    )


# ==========================================================
# 12 - VALIDACAO FINAL
# ==========================================================

verificacao = pd.read_csv(
    arquivo_final
)

assert len(verificacao) == len(dados)

assert (
    verificacao["Categoria"].tolist()
    == dados["Categoria"].tolist()
)

assert arquivo_contagem.exists()
assert arquivo_json.exists()

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("Modelo T5 multilingue: OK")
print("Pipeline LiqFit: OK")
print("Pandas apply: OK")
print("Coluna Categoria criada: OK")
print("Total processado:", len(dados))
print("CSV final: OK")
print("CSV contagem: OK")
print("Resumo JSON: OK")

print("\nAULA 07 EXECUTADA COM SUCESSO!")
