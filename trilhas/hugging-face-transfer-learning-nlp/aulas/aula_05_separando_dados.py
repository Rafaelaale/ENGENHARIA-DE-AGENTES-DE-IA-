from pathlib import Path

from datasets import load_dataset, DatasetDict

raiz = Path(__file__).resolve().parent.parent

csv_original = raiz / "dados" / "redacoes.csv"

saida_treino = raiz / "dados" / "redacoes_treino.csv"
saida_teste = raiz / "dados" / "redacoes_teste.csv"

relatorio = (
    raiz / "resultados" / "aula_05_separacao_dados.txt"
)

print("\n===== CARREGANDO BASE ORIGINAL =====")

if not csv_original.is_file():
    raise FileNotFoundError(csv_original)

dados_redacoes = load_dataset(
    "csv",
    data_files=str(csv_original)
)

print(dados_redacoes)

# 1. SEPARAR DADOS CONFORME A ALURA

print("\n===== SEPARANDO TREINO E TESTE =====")

treino_teste = dados_redacoes["train"].train_test_split(
    test_size=0.2,
    shuffle=False
)

print(treino_teste)

# 2. RENOMEAR OS CONJUNTOS

dados_redacoes = DatasetDict({
    "treino": treino_teste["train"],
    "teste": treino_teste["test"]
})

print("\n===== DATASET DICT FINAL =====")
print(dados_redacoes)

# 3. VALIDAR QUANTIDADES

total = (
    len(dados_redacoes["treino"])
    + len(dados_redacoes["teste"])
)

assert total == 4570
assert len(dados_redacoes["treino"]) == 3656
assert len(dados_redacoes["teste"]) == 914

assert dados_redacoes["treino"].column_names == [
    "essay", "score"
]

assert dados_redacoes["teste"].column_names == [
    "essay", "score"
]

print("\n===== QUANTIDADES VALIDADAS =====")
print("Total:", total)
print("Treinamento:", len(dados_redacoes["treino"]))
print("Teste:", len(dados_redacoes["teste"]))

# 4. CONVERTER PARA PANDAS

df_treino = dados_redacoes["treino"].to_pandas()
df_teste = dados_redacoes["teste"].to_pandas()

# 5. CONFERIR LIMITES DA DIVISAO

print("\n===== PRIMEIRA REDACAO DE TREINO =====")
print(df_treino.iloc[0]["essay"][:250])
print("Nota:", df_treino.iloc[0]["score"])

print("\n===== PRIMEIRA REDACAO DE TESTE =====")
print(df_teste.iloc[0]["essay"][:250])
print("Nota:", df_teste.iloc[0]["score"])

print("\n===== DISTRIBUICAO DAS NOTAS =====")

print("\nTREINO")
print(df_treino["score"].value_counts().sort_index())

print("\nTESTE")
print(df_teste["score"].value_counts().sort_index())

# 6. SALVAR ARQUIVOS

for arquivo in (saida_treino, saida_teste, relatorio):
    if arquivo.exists():
        raise FileExistsError(arquivo)

df_treino.to_csv(
    saida_treino,
    index=False,
    encoding="utf-8-sig"
)

df_teste.to_csv(
    saida_teste,
    index=False,
    encoding="utf-8-sig"
)

texto_relatorio = [
    "HUGGING FACE - AULA 05",
    "SEPARACAO DOS DADOS",
    "",
    f"Total de redacoes: {total}",
    f"Redacoes de treino: {len(df_treino)}",
    f"Redacoes de teste: {len(df_teste)}",
    "",
    "Metodo: train_test_split",
    "test_size: 0.2",
    "shuffle: False",
    "",
    "FEATURES",
    str(dados_redacoes["treino"].features),
    "",
    "DISTRIBUICAO - TREINO",
    df_treino["score"].value_counts().sort_index().to_string(),
    "",
    "DISTRIBUICAO - TESTE",
    df_teste["score"].value_counts().sort_index().to_string(),
    "",
    "OBSERVACAO",
    "A divisao reproduz a ordem original da Alura.",
    "Nao houve embaralhamento dos registros.",
    "A base de teste deve ser reservada para avaliacao.",
    "Nenhum modelo foi treinado nesta etapa."
]

relatorio.write_text(
    "\n".join(texto_relatorio),
    encoding="utf-8"
)

# 7. VALIDACAO FINAL

print("\n===== ARQUIVOS GERADOS =====")

for arquivo in (saida_treino, saida_teste, relatorio):
    assert arquivo.is_file()
    assert arquivo.stat().st_size > 0

    print(
        f"{arquivo.name}: "
        f"{arquivo.stat().st_size} bytes"
    )

print("\nAULA 05 - SEPARACAO VALIDADA COM SUCESSO")