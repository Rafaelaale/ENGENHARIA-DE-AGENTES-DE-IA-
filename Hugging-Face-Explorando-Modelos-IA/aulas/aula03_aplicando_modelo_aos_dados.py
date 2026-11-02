# ==========================================================
# HUGGING FACE - AULA 03
# APLICANDO O MODELO AOS DADOS DO E-COMMERCE
# ==========================================================

from pathlib import Path
import pandas as pd

from pysentimiento import create_analyzer


BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

ARQUIVO_DADOS = DADOS / "resenhas.csv"

URL = (
    "https://raw.githubusercontent.com/"
    "alura-cursos/hugging_face/main/"
    "Dados/resenhas.csv"
)

print("=" * 65)
print("AULA 03 - APLICANDO O MODELO AOS DADOS")
print("=" * 65)


# ==========================================================
# 1 - CARREGAR O DATAFRAME
# ==========================================================

print("\n1 - CARREGANDO AS RESENHAS")

if not ARQUIVO_DADOS.exists():

    print("Baixando CSV oficial da Alura...")

    dados = pd.read_csv(URL)

    dados.to_csv(
        ARQUIVO_DADOS,
        index=False,
        encoding="utf-8-sig"
    )

else:

    dados = pd.read_csv(
        ARQUIVO_DADOS
    )

assert "ID" in dados.columns
assert "Resenha" in dados.columns

assert not dados.empty

assert dados["Resenha"].notna().all()

assert (
    dados["Resenha"]
    .astype(str)
    .str.strip()
    .ne("")
    .all()
)

print("Arquivo:", ARQUIVO_DADOS)
print("Quantidade de avaliacoes:", len(dados))

print("\nColunas:")
print(dados.columns.tolist())

print("\nPrimeiras avaliacoes:")
print(
    dados[["ID", "Resenha"]]
    .head()
    .to_string(index=False)
)


# ==========================================================
# 2 - CARREGAR MODELO EM PORTUGUES
# ==========================================================

print("\n2 - CARREGANDO MODELO PRE-TREINADO")

modelo_analise_sentimento = create_analyzer(
    task="sentiment",
    lang="pt"
)

print("Modelo carregado com sucesso!")


# ==========================================================
# 3 - CLASSIFICAR TODAS AS RESENHAS
# ==========================================================

print("\n3 - EXECUTANDO CLASSIFICACAO")

textos = (
    dados["Resenha"]
    .astype(str)
    .tolist()
)

print(
    f"Enviando {len(textos)} resenhas "
    "para o modelo..."
)

resultado_previsao = (
    modelo_analise_sentimento.predict(
        textos
    )
)

assert len(resultado_previsao) == len(dados)

print(
    "Classificacoes recebidas:",
    len(resultado_previsao)
)


# ==========================================================
# 4 - EXTRAIR OS SENTIMENTOS
# ==========================================================

sentimento = []

probabilidades = []

for resultado in resultado_previsao:

    sentimento.append(
        resultado.output
    )

    probas = {
        chave: float(valor)
        for chave, valor
        in resultado.probas.items()
    }

    probabilidades.append(probas)


assert len(sentimento) == len(dados)

assert set(sentimento).issubset({
    "POS",
    "NEU",
    "NEG"
})

print("\n4 - LISTA DE SENTIMENTOS")

print(sentimento)


# ==========================================================
# 5 - ADICIONAR COLUNA AO DATAFRAME
# ==========================================================

dados["Sentimento"] = sentimento

assert len(dados) == len(textos)

assert dados["Sentimento"].notna().all()

print("\n5 - DATAFRAME CLASSIFICADO")

print(
    dados.to_string(
        index=False,
        max_colwidth=70
    )
)


# ==========================================================
# 6 - CONFERIR AS PRIMEIRAS CLASSIFICACOES
# ==========================================================

print("\n6 - VERIFICACAO DAS AVALIACOES")

for i in range(min(5, len(dados))):

    linha = dados.iloc[i]

    print("\n" + "-" * 60)

    print("Cliente ID:", linha["ID"])

    print("Resenha:", linha["Resenha"])

    print("Sentimento:", linha["Sentimento"])


# ==========================================================
# 7 - SALVAR TABELA FINAL
# ==========================================================

arquivo_final = (
    RESULTADOS
    / "aula03_resenhas_classificadas.csv"
)

dados.to_csv(
    arquivo_final,
    index=False,
    encoding="utf-8-sig"
)

print("\n7 - RESULTADOS SALVOS")
print(arquivo_final)


# ==========================================================
# 8 - SALVAR PROBABILIDADES
# ==========================================================

dados_detalhados = dados.copy()

for classe in ("POS", "NEU", "NEG"):

    dados_detalhados[
        f"Prob_{classe}"
    ] = [

        item[classe]
        for item in probabilidades

    ]

arquivo_detalhado = (
    RESULTADOS
    / "aula03_resenhas_probabilidades.csv"
)

dados_detalhados.to_csv(
    arquivo_detalhado,
    index=False,
    encoding="utf-8-sig"
)

print(arquivo_detalhado)


# ==========================================================
# 9 - VALIDACAO FINAL
# ==========================================================

verificacao = pd.read_csv(
    arquivo_final
)

assert len(verificacao) == len(dados)

assert list(verificacao.columns) == [
    "ID",
    "Resenha",
    "Sentimento"
]

assert (
    verificacao["Sentimento"].tolist()
    == sentimento
)

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("Total de resenhas:", len(dados))
print("Total classificado:", len(sentimento))
print("Coluna Sentimento: OK")
print("CSV final: OK")
print("Probabilidades: OK")

print("\nAULA 03 EXECUTADA COM SUCESSO!")


if __name__ == "__main__":
    pass
