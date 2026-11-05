from pathlib import Path
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

raiz = Path(__file__).resolve().parent.parent
base_externa = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")

arquivo_treino = raiz / "dados" / "redacoes_treino.csv"
arquivo_teste = raiz / "dados" / "redacoes_teste.csv"

destino = base_externa / "dados" / "redacoes_tokenizadas"
relatorio = raiz / "resultados" / "aula_07_tokenizacao_base.txt"

checkpoint_modelo = "Geotrend/distilbert-base-pt-cased"

print("\n===== AULA 07 - TOKENIZANDO BASE COMPLETA =====")

# 1. CARREGAR CONJUNTOS DA AULA 05

dados_redacoes = load_dataset(
    "csv",
    data_files={
        "treino": str(arquivo_treino),
        "teste": str(arquivo_teste)
    }
)

print("\n===== BASE ORIGINAL =====")
print(dados_redacoes)

assert len(dados_redacoes["treino"]) == 3656
assert len(dados_redacoes["teste"]) == 914

# 2. CARREGAR TOKENIZADOR DA AULA 06

print("\n===== CARREGANDO TOKENIZADOR =====")

tokenizador = AutoTokenizer.from_pretrained(
    checkpoint_modelo,
    local_files_only=True
)

print("Checkpoint:", checkpoint_modelo)
print("Tokenizador:", type(tokenizador).__name__)

# 3. DEFINIR FUNCAO DE TOKENIZACAO

def funcao_tokenizadora(dados_texto):
    return tokenizador(
        dados_texto["essay"],
        truncation=True,
        max_length=512
    )

# 4. APLICAR A TODOS OS CONJUNTOS

print("\n===== TOKENIZANDO TREINO E TESTE =====")

dataset_tokenizado = dados_redacoes.map(
    funcao_tokenizadora,
    batched=True,
    batch_size=100,
    remove_columns=["essay"],
    desc="Tokenizando redacoes"
)

print("\n===== DATASET TOKENIZADO =====")
print(dataset_tokenizado)

# 5. VALIDAR FEATURES

colunas_esperadas = {
    "score",
    "input_ids",
    "attention_mask"
}

for conjunto, quantidade in [
    ("treino", 3656),
    ("teste", 914)
]:
    dados = dataset_tokenizado[conjunto]

    assert len(dados) == quantidade
    assert set(dados.column_names) == colunas_esperadas

    for registro in dados:
        assert len(registro["input_ids"]) > 0
        assert len(registro["input_ids"]) <= 512
        assert len(registro["input_ids"]) == len(
            registro["attention_mask"]
        )

    print(
        f"{conjunto}: {len(dados)} redacoes validadas"
    )

# 6. VISUALIZAR PRIMEIROS REGISTROS

print("\n===== TREINO - PRIMEIRO REGISTRO =====")

primeiro_treino = dataset_tokenizado["treino"][0]

print("Nota:", primeiro_treino["score"])
print("Primeiros 25 IDs:", primeiro_treino["input_ids"][:25])
print(
    "Primeiros 25 valores da mascara:",
    primeiro_treino["attention_mask"][:25]
)

print("\n===== TESTE - PRIMEIRO REGISTRO =====")

primeiro_teste = dataset_tokenizado["teste"][0]

print("Nota:", primeiro_teste["score"])
print("Primeiros 25 IDs:", primeiro_teste["input_ids"][:25])

print("\n===== VISUALIZACAO COM PANDAS =====")

for conjunto in ("treino", "teste"):
    df = dataset_tokenizado[conjunto].select(
        range(3)
    ).to_pandas()

    print(f"\n{conjunto.upper()}")
    print(df.to_string(index=False, max_colwidth=70))

# 7. MEDIR COMPRIMENTOS

comprimentos = {
    conjunto: [
        len(ids)
        for ids in dataset_tokenizado[conjunto]["input_ids"]
    ]
    for conjunto in ("treino", "teste")
}

for conjunto, valores in comprimentos.items():
    print(f"\n===== COMPRIMENTOS - {conjunto.upper()} =====")
    print("Minimo:", min(valores))
    print("Maximo:", max(valores))
    print("Media:", round(sum(valores) / len(valores), 2))
    print("Sequencias com 512 tokens:", valores.count(512))

# 8. SALVAR DATASET NO HD EXTERNO

print("\n===== SALVANDO DATASET NA UNIDADE F =====")

if destino.exists():
    raise FileExistsError(destino)

dataset_tokenizado.save_to_disk(str(destino))

# 9. RECARREGAR E VALIDAR

dataset_recarregado = load_from_disk(str(destino))

assert len(dataset_recarregado["treino"]) == 3656
assert len(dataset_recarregado["teste"]) == 914

assert dataset_recarregado["treino"][0]["score"] == (
    dados_redacoes["treino"][0]["score"]
)

assert dataset_recarregado["teste"][0]["score"] == (
    dados_redacoes["teste"][0]["score"]
)

# 10. SALVAR RELATORIO NO PROJETO

linhas = [
    "HUGGING FACE - AULA 07",
    "TOKENIZANDO A BASE DE DADOS",
    "",
    f"Checkpoint: {checkpoint_modelo}",
    f"Treino: {len(dataset_tokenizado['treino'])}",
    f"Teste: {len(dataset_tokenizado['teste'])}",
    "",
    "FEATURES",
    str(dataset_tokenizado["treino"].features),
    "",
    "COMPRIMENTOS DOS TOKENS"
]

for conjunto, valores in comprimentos.items():
    linhas.extend([
        "",
        conjunto.upper(),
        f"Minimo: {min(valores)}",
        f"Maximo: {max(valores)}",
        f"Media: {sum(valores) / len(valores):.2f}",
        f"Sequencias com 512 tokens: {valores.count(512)}"
    ])

linhas.extend([
    "",
    f"Dataset salvo em: {destino}",
    "",
    "OBSERVACOES",
    "Tokenizacao em lotes com batched=True.",
    "Coluna essay removida somente da versao tokenizada.",
    "Coluna score preservada.",
    "Truncation=True e limite de 512 tokens.",
    "Padding sera tratado na preparacao dos lotes.",
    "Nenhum modelo foi treinado nesta aula."
])

if relatorio.exists():
    raise FileExistsError(relatorio)

relatorio.write_text(
    "\n".join(linhas),
    encoding="utf-8"
)

print("\n===== VALIDACAO FINAL =====")
print("Treino:", len(dataset_recarregado["treino"]))
print("Teste:", len(dataset_recarregado["teste"]))
print("Dataset salvo:", destino)
print("Relatorio:", relatorio)

print("\nAULA 07 - TOKENIZACAO COMPLETA VALIDADA")