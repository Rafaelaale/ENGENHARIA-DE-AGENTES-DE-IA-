# ==========================================================
# HUGGING FACE - AULA 02
# SELECIONANDO MODELO DE SENTIMENTOS EM PORTUGUES
# ==========================================================

from pathlib import Path
import json

from pysentimiento import create_analyzer


BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

MODELO = "pysentimiento/bertweet-pt-sentiment"

CLASSES = {
    "POS": "POSITIVO",
    "NEU": "NEUTRO",
    "NEG": "NEGATIVO"
}


print("=" * 65)
print("AULA 02 - ANALISE DE SENTIMENTOS EM PORTUGUES")
print("=" * 65)

# ==========================================================
# 1 - CARREGAR MODELO PRE-TREINADO
# ==========================================================

print("\n1 - CARREGANDO MODELO")
print("Modelo:", MODELO)
print("Idioma: Portugues")
print("Tarefa: Analise de sentimentos")

modelo_analise_sentimento = create_analyzer(
    task="sentiment",
    lang="pt"
)

print("MODELO CARREGADO COM SUCESSO!")


# ==========================================================
# 2 - RESENHAS UTILIZADAS PELA PROFESSORA
# ==========================================================

avaliacoes = [

    {
        "tipo": "POSITIVO",
        "texto": """
A fritadeira é sensacional, muito além do que imaginava.
É linda, super funcional e muito fácil de manusear.
Fácil de limpar e potente. Super recomendo.
"""
    },

    {
        "tipo": "NEUTRO",
        "texto": """
Após poucos meses de uso a carcaça de aço escovado
começou a oxidar, demonstrando a baixa qualidade de
proteção. Fora esse detalhe, o produto cumpre o prometido.
"""
    },

    {
        "tipo": "NEGATIVO",
        "texto": """
Em menos de 1 ano parou de funcionar, enviei para
assistência técnica por estar na garantia, trocaram
o motor, passou a ficar menos potente e não durou
2 utilizações. Isso se repetiu várias vezes,
até que desisti de ficar levando lá e queimando
de novo em seguida, vi outros clientes com o
mesmo problema. Não comprem!!
"""
    }

]


# ==========================================================
# 3 - EXECUTAR CLASSIFICACOES
# ==========================================================

resultados = []

print("\n2 - CLASSIFICANDO AS RESENHAS")

for indice, avaliacao in enumerate(
    avaliacoes,
    start=1
):

    texto = " ".join(
        avaliacao["texto"].split()
    )

    resposta = modelo_analise_sentimento.predict(
        texto
    )

    classe = resposta.output

    probabilidades = {
        chave: float(valor)
        for chave, valor in resposta.probas.items()
    }

    assert classe in CLASSES

    assert all(
        rotulo in probabilidades
        for rotulo in CLASSES
    )

    assert abs(
        sum(probabilidades.values()) - 1
    ) < 0.01

    sentimento = CLASSES[classe]

    esperado = avaliacao["tipo"]

    print("\n" + "-" * 65)
    print(f"AVALIACAO {indice}")
    print("-" * 65)

    print("RESENHA:")
    print(texto)

    print("\nRESULTADO DO MODELO")
    print("Classificacao:", classe)
    print("Sentimento:", sentimento)

    print("\nPROBABILIDADES:")

    for rotulo in ("POS", "NEU", "NEG"):

        probabilidade = (
            probabilidades[rotulo] * 100
        )

        print(
            f"{rotulo}: {probabilidade:.2f}%"
        )

    print("\nComparacao com a aula:")
    print("Esperado:", esperado)
    print("Previsto:", sentimento)

    corresponde = sentimento == esperado

    print(
        "Corresponde:",
        "SIM" if corresponde else "NAO"
    )

    resultados.append({

        "avaliacao": indice,
        "texto": texto,
        "esperado": esperado,
        "previsto": sentimento,
        "classe": classe,
        "probabilidades": probabilidades,
        "corresponde_ao_esperado": corresponde

    })


# ==========================================================
# 4 - RESUMO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("3 - RESUMO DAS CLASSIFICACOES")
print("=" * 65)

acertos = sum(
    item["corresponde_ao_esperado"]
    for item in resultados
)

for item in resultados:

    print(
        f"Avaliacao {item['avaliacao']}: "
        f"{item['previsto']}"
    )

print(
    f"\nClassificacoes iguais ao esperado: {acertos}/3"
)

# ==========================================================
# 5 - SALVAR RESULTADOS
# ==========================================================

arquivo = (
    RESULTADOS
    / "aula02_sentimentos_portugues.json"
)

with arquivo.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        resultados,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nRESULTADOS SALVOS:")
print(arquivo)

print("\nVALIDACAO DAS PROBABILIDADES: OK")
print("AULA 02 EXECUTADA COM SUCESSO!")


