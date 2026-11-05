from pathlib import Path
from datasets import load_dataset

raiz = Path(__file__).resolve().parent.parent
csv = raiz / "dados" / "redacoes.csv"
relatorio = raiz / "resultados" / "aula_04_relatorio_base_redacoes.txt"

print("\n===== CARREGANDO BASE DE REDACOES =====")

dados_redacoes = load_dataset(
    "csv",
    data_files=str(csv)
)

print("\n===== DATASET DICT =====")
print(dados_redacoes)

treino = dados_redacoes["train"]

print("\n===== FEATURES =====")
print(treino.features)

print("\n===== PRIMEIRA REDACAO =====")
print(treino["essay"][0])

print("\n===== PRIMEIRA NOTA =====")
print(treino["score"][0])

df = treino.to_pandas()

print("\n===== PRIMEIRAS CINCO LINHAS =====")
print(df.head().to_string(index=False, max_colwidth=70))

print("\n===== RESUMO =====")
print("Quantidade de redacoes:", len(df))
print("Colunas:", df.columns.tolist())
print("Notas minimas e maximas:", df["score"].min(), df["score"].max())
print("Textos ausentes:", df["essay"].isna().sum())
print("Notas ausentes:", df["score"].isna().sum())

assert len(df) == 4570, "Quantidade diferente da aula."
assert {"essay", "score"}.issubset(df.columns)

conteudo = (
    "HUGGING FACE - AULA 04\n"
    f"Redacoes: {len(df)}\n"
    f"Colunas: {df.columns.tolist()}\n"
    f"Primeira nota: {treino['score'][0]}\n"
    f"Menor nota: {df['score'].min()}\n"
    f"Maior nota: {df['score'].max()}\n"
    f"Notas ausentes: {df['score'].isna().sum()}\n"
)

if relatorio.exists():
    print("\nRelatorio ja existente foi preservado.")
else:
    relatorio.write_text(conteudo, encoding="utf-8")

print("\nAULA 04 - BASE CARREGADA COM SUCESSO")