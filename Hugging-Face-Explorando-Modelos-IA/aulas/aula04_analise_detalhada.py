# ==========================================================
# HUGGING FACE - AULA 04
# ANALISE DETALHADA DOS SENTIMENTOS
# ==========================================================

from pathlib import Path
from collections import Counter
import re

import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# 1 - CAMINHOS DO PROJETO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

ARQUIVO = (
    RESULTADOS
    / "aula03_resenhas_classificadas.csv"
)

print("=" * 65)
print("HUGGING FACE - AULA 04")
print("ANALISE DETALHADA DOS SENTIMENTOS")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR CLASSIFICACOES DA AULA 03
# ==========================================================

print("\n1 - CARREGANDO DADOS DA AULA 03")

if not ARQUIVO.exists():
    raise FileNotFoundError(
        "Arquivo da Aula 03 nao encontrado: "
        + str(ARQUIVO)
    )

dados = pd.read_csv(
    ARQUIVO,
    encoding="utf-8-sig"
)

colunas = {"ID", "Resenha", "Sentimento"}

assert colunas.issubset(dados.columns)
assert not dados.empty
assert dados["Resenha"].notna().all()
assert dados["Sentimento"].notna().all()

assert set(dados["Sentimento"]).issubset({
    "POS", "NEU", "NEG"
})

print("Total de resenhas:", len(dados))
print("Colunas:", dados.columns.tolist())

print("\nPrimeiras linhas:")
print(dados.head().to_string(index=False))


# ==========================================================
# 3 - GROUPBY: CONTAGEM DE SENTIMENTOS
# ==========================================================

print("\n2 - CONTAGEM POR SENTIMENTO")

df_sentimento = (
    dados.groupby("Sentimento")
    .size()
    .reset_index(name="Contagem")
)

# Exibir sempre as tres classes na mesma ordem.
df_sentimento = (
    df_sentimento
    .set_index("Sentimento")
    .reindex(
        ["NEG", "NEU", "POS"],
        fill_value=0
    )
    .reset_index()
)

df_sentimento["Percentual"] = (
    df_sentimento["Contagem"]
    / len(dados)
    * 100
).round(2)

print(df_sentimento.to_string(index=False))

assert (
    df_sentimento["Contagem"].sum()
    == len(dados)
)

arquivo_contagem = (
    RESULTADOS
    / "aula04_contagem_sentimentos.csv"
)

df_sentimento.to_csv(
    arquivo_contagem,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 4 - GRAFICO DE BARRAS COM PLOTLY
# ==========================================================

print("\n3 - CRIANDO GRAFICO INTERATIVO")

fig = px.bar(
    df_sentimento,
    x="Sentimento",
    y="Contagem",
    title="Contagem de Resenhas por Sentimento",
    text="Contagem",
    hover_data=["Percentual"],
    category_orders={
        "Sentimento": ["NEG", "NEU", "POS"]
    }
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Sentimento",
    yaxis_title="Contagem de resenhas",
    yaxis_range=[
        0,
        max(df_sentimento["Contagem"]) * 1.2
    ]
)

arquivo_html = (
    RESULTADOS
    / "aula04_grafico_sentimentos.html"
)

fig.write_html(
    str(arquivo_html),
    include_plotlyjs=True
)

print("Grafico interativo salvo:")
print(arquivo_html)


# ==========================================================
# 5 - VERSAO PNG DO GRAFICO
# ==========================================================

figura, eixo = plt.subplots(
    figsize=(9, 5)
)

eixo.bar(
    df_sentimento["Sentimento"],
    df_sentimento["Contagem"]
)

for i, quantidade in enumerate(
    df_sentimento["Contagem"]
):
    eixo.text(
        i,
        quantidade + 0.2,
        str(quantidade),
        ha="center",
        fontsize=12
    )

eixo.set_title(
    "Contagem de Resenhas por Sentimento"
)

eixo.set_xlabel("Sentimento")
eixo.set_ylabel("Contagem")

eixo.set_ylim(
    0,
    max(df_sentimento["Contagem"]) * 1.2
)

eixo.grid(
    axis="y",
    alpha=0.2
)

figura.tight_layout()

arquivo_png = (
    RESULTADOS
    / "aula04_grafico_sentimentos.png"
)

figura.savefig(
    arquivo_png,
    dpi=150
)

plt.close(figura)

print("Grafico PNG salvo:")
print(arquivo_png)


# ==========================================================
# 6 - STOPWORDS EM PORTUGUES
# ==========================================================

print("\n4 - CARREGANDO STOPWORDS DO NLTK")

try:
    portuguese_stopwords = set(
        stopwords.words("portuguese")
    )

except LookupError:
    print("Baixando corpus de stopwords...")

    if not nltk.download(
        "stopwords",
        quiet=True
    ):
        raise RuntimeError(
            "Nao foi possivel baixar as stopwords."
        )

    portuguese_stopwords = set(
        stopwords.words("portuguese")
    )

print(
    "Quantidade de stopwords:",
    len(portuguese_stopwords)
)

print(
    "Exemplos:",
    sorted(portuguese_stopwords)[:20]
)


# ==========================================================
# 7 - LIMPEZA E TOKENIZACAO
# ==========================================================

def preparar_palavras(textos):
    """
    Converte para minusculas, remove pontuacao,
    numeros e stopwords em portugues.
    """

    texto_unido = " ".join(
        str(texto)
        for texto in textos
    )

    palavras = re.findall(
        r"[^\W\d_]+",
        texto_unido.casefold(),
        flags=re.UNICODE
    )

    palavras_filtradas = [
        palavra
        for palavra in palavras
        if (
            len(palavra) >= 3
            and palavra not in portuguese_stopwords
        )
    ]

    return palavras_filtradas


# ==========================================================
# 8 - FUNCAO DE NUVEM DE PALAVRAS
# ==========================================================

def nuvem_palavras(texto, coluna_texto, sentimento):

    print(
        f"\n5 - NUVEM DE PALAVRAS: {sentimento}"
    )

    # Filtrar as resenhas pelo sentimento
    texto_sentimento = texto.loc[
        texto["Sentimento"] == sentimento,
        coluna_texto
    ]

    print(
        "Resenhas selecionadas:",
        len(texto_sentimento)
    )

    if texto_sentimento.empty:
        print(
            "Nenhuma resenha encontrada "
            "para este sentimento."
        )
        return Counter()

    palavras_filtradas = preparar_palavras(
        texto_sentimento
    )

    if not palavras_filtradas:
        print(
            "Nenhuma palavra restante "
            "apos a limpeza."
        )
        return Counter()

    texto_filtrado = " ".join(
        palavras_filtradas
    )

    frequencias = Counter(
        palavras_filtradas
    )

    print("\n10 PALAVRAS MAIS FREQUENTES:")

    for palavra, quantidade in (
        frequencias.most_common(10)
    ):
        print(
            f"{palavra}: {quantidade}"
        )

    # Criar word cloud
    nuvem = WordCloud(
        width=800,
        height=500,
        max_words=50,
        background_color="black",
        collocations=False,
        random_state=42
    ).generate(
        texto_filtrado
    )

    # Visualizar com matplotlib
    figura, eixo = plt.subplots(
        figsize=(10, 7)
    )

    eixo.imshow(
        nuvem,
        interpolation="bilinear"
    )

    eixo.axis("off")

    eixo.set_title(
        f"Nuvem de palavras - {sentimento}",
        fontsize=16
    )

    figura.tight_layout()

    caminho = (
        RESULTADOS
        / f"aula04_nuvem_{sentimento}.png"
    )

    figura.savefig(
        caminho,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figura)

    print("\nImagem salva:")
    print(caminho)

    return frequencias


# ==========================================================
# 9 - NUVEM DE RESENHAS POSITIVAS
# ==========================================================

freq_positivas = nuvem_palavras(
    dados,
    "Resenha",
    "POS"
)


# ==========================================================
# 10 - NUVEM DE RESENHAS NEGATIVAS
# ==========================================================

freq_negativas = nuvem_palavras(
    dados,
    "Resenha",
    "NEG"
)


# ==========================================================
# 11 - SALVAR PALAVRAS MAIS FREQUENTES
# ==========================================================

linhas = []

for sentimento, frequencias in (
    ("POS", freq_positivas),
    ("NEG", freq_negativas)
):

    for palavra, quantidade in (
        frequencias.most_common(50)
    ):

        linhas.append({
            "Sentimento": sentimento,
            "Palavra": palavra,
            "Frequencia": quantidade
        })

df_palavras = pd.DataFrame(
    linhas,
    columns=[
        "Sentimento",
        "Palavra",
        "Frequencia"
    ]
)

arquivo_palavras = (
    RESULTADOS
    / "aula04_palavras_frequentes.csv"
)

df_palavras.to_csv(
    arquivo_palavras,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 12 - RESUMO GERAL DO E-COMMERCE
# ==========================================================

print("\n" + "=" * 65)
print("6 - RESUMO GERAL DO E-COMMERCE")
print("=" * 65)

contagens = dict(
    zip(
        df_sentimento["Sentimento"],
        df_sentimento["Contagem"]
    )
)

total = len(dados)

for classe, nome in (
    ("POS", "POSITIVAS"),
    ("NEU", "NEUTRAS"),
    ("NEG", "NEGATIVAS")
):

    quantidade = int(
        contagens.get(classe, 0)
    )

    percentual = (
        quantidade / total * 100
    )

    print(
        f"{nome}: {quantidade}"
        f" ({percentual:.2f}%)"
    )

classe_predominante = (
    df_sentimento
    .sort_values(
        "Contagem",
        ascending=False
    )
    .iloc[0]["Sentimento"]
)

print(
    "\nSentimento com maior contagem:",
    classe_predominante
)

print(
    "\nObservacao: as palavras frequentes "
    "ajudam a localizar assuntos recorrentes, "
    "mas nao substituem a leitura das resenhas."
)


# ==========================================================
# 13 - VALIDACAO FINAL
# ==========================================================

assert arquivo_html.exists()
assert arquivo_png.exists()
assert arquivo_contagem.exists()
assert arquivo_palavras.exists()

if contagens.get("POS", 0) > 0:
    assert (
        RESULTADOS / "aula04_nuvem_POS.png"
    ).exists()

if contagens.get("NEG", 0) > 0:
    assert (
        RESULTADOS / "aula04_nuvem_NEG.png"
    ).exists()

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print(
    "Resenhas contabilizadas:",
    int(df_sentimento["Contagem"].sum())
)

print("Agrupamento Pandas: OK")
print("Grafico Plotly: OK")
print("Grafico PNG: OK")
print("Stopwords portugues: OK")
print("Nuvens de palavras: OK")
print("CSV de frequencias: OK")

print("\nAULA 04 EXECUTADA COM SUCESSO!")
