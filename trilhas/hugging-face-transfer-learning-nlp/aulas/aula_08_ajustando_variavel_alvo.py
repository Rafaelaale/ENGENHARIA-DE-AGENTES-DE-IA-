from pathlib import Path

from datasets import (
    load_from_disk,
    ClassLabel
)

raiz = Path(__file__).resolve().parent.parent

base_externa = Path(
    r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP"
)

origem = (
    base_externa
    / "dados"
    / "redacoes_tokenizadas"
)

destino = (
    base_externa
    / "dados"
    / "redacoes_prontas_treinamento"
)

relatorio = (
    raiz
    / "resultados"
    / "aula_08_variavel_alvo.txt"
)

print("\n===== HUGGING FACE - AULA 08 =====")
print("AJUSTANDO A VARIAVEL ALVO")

# =====================================
# 1. CARREGAR DATASET TOKENIZADO
# =====================================

print("\n===== CARREGANDO AULA 07 =====")

dataset_tokenizado = load_from_disk(
    str(origem)
)

print(dataset_tokenizado)

assert len(dataset_tokenizado["treino"]) == 3656
assert len(dataset_tokenizado["teste"]) == 914

# =====================================
# 2. VERIFICAR SCORE
# =====================================

print("\n===== VARIAVEL SCORE ORIGINAL =====")

print(
    dataset_tokenizado["treino"].features["score"]
)

notas_unicas = sorted(
    dataset_tokenizado["treino"].unique("score")
)

print("\n===== NOTAS UNICAS =====")
print(notas_unicas)

assert notas_unicas == list(range(11))

# =====================================
# 3. CRIAR CLASSLABEL
# =====================================

scores = ClassLabel(
    names=[str(i) for i in range(11)]
)

print("\n===== CLASSLABEL CRIADO =====")
print(scores)

assert scores.num_classes == 11

# =====================================
# 4. FUNCAO PARA MAPEAR LABELS
# =====================================

def mapear_labels(dados):
    dados["label"] = scores.str2int(
        str(dados["score"])
    )

    return dados

# =====================================
# 5. APLICAR AOS DOIS CONJUNTOS
# =====================================

print("\n===== MAPEANDO LABELS =====")

dataset_tokenizado = dataset_tokenizado.map(
    mapear_labels,
    remove_columns=["score"],
    desc="Convertendo notas para labels"
)

# =====================================
# 6. CONVERTER PARA CLASSLABEL
# =====================================

print("\n===== CONVERTENDO PARA CLASSLABEL =====")

dataset_tokenizado = dataset_tokenizado.cast_column(
    "label",
    scores
)

print("\n===== DATASET FINAL =====")
print(dataset_tokenizado)

# =====================================
# 7. VALIDAR FEATURES
# =====================================

print("\n===== FEATURES DA LABEL =====")

print(
    dataset_tokenizado["treino"].features["label"]
)

colunas_esperadas = {
    "input_ids",
    "attention_mask",
    "label"
}

for conjunto, quantidade in [
    ("treino", 3656),
    ("teste", 914)
]:
    dados = dataset_tokenizado[conjunto]

    assert len(dados) == quantidade

    assert set(dados.column_names) == colunas_esperadas

    assert isinstance(
        dados.features["label"],
        ClassLabel
    )

    assert dados.features["label"].names == [
        str(i) for i in range(11)
    ]

    notas = dados["label"]

    assert all(
        isinstance(nota, int) and 0 <= nota <= 10
        for nota in notas
    )

    print(
        f"{conjunto.upper()}: "
        f"{len(dados)} redacoes validadas"
    )

# =====================================
# 8. EXIBIR PRIMEIROS REGISTROS
# =====================================

print("\n===== PRIMEIRO REGISTRO - TREINO =====")

primeiro_treino = dataset_tokenizado["treino"][0]

print("Label:", primeiro_treino["label"])

print(
    "Primeiros IDs:",
    primeiro_treino["input_ids"][:20]
)

print(
    "Attention mask:",
    primeiro_treino["attention_mask"][:20]
)

print("\n===== PRIMEIRO REGISTRO - TESTE =====")

primeiro_teste = dataset_tokenizado["teste"][0]

print("Label:", primeiro_teste["label"])

# =====================================
# 9. CONFERIR DISTRIBUICAO DAS NOTAS
# =====================================

print("\n===== DISTRIBUICAO DAS LABELS =====")

distribuicoes = {}

for conjunto in ("treino", "teste"):

    contagem = {
        nota: dataset_tokenizado[conjunto][
            "label"
        ].count(nota)
        for nota in range(11)
    }

    distribuicoes[conjunto] = contagem

    print(f"\n{conjunto.upper()}")

    for nota, quantidade in contagem.items():
        print(
            f"Label {nota}: {quantidade}"
        )

# =====================================
# 10. VALIDAR PRESERVACAO DAS NOTAS
# =====================================

print("\n===== VALIDANDO NOTAS ORIGINAIS =====")

dataset_original = load_from_disk(
    str(origem)
)

for conjunto in ("treino", "teste"):

    notas_originais = dataset_original[
        conjunto
    ]["score"]

    labels_convertidas = dataset_tokenizado[
        conjunto
    ]["label"]

    assert (
        notas_originais == labels_convertidas
    ), f"Notas modificadas em {conjunto}"

print("TODAS AS NOTAS FORAM PRESERVADAS")

# =====================================
# 11. SALVAR DATASET
# =====================================

print("\n===== SALVANDO DATASET FINAL =====")

if destino.exists():
    raise FileExistsError(destino)

dataset_tokenizado.save_to_disk(
    str(destino)
)

# =====================================
# 12. RECARREGAR PARA VALIDACAO
# =====================================

dataset_validado = load_from_disk(
    str(destino)
)

assert len(dataset_validado["treino"]) == 3656
assert len(dataset_validado["teste"]) == 914

assert isinstance(
    dataset_validado["treino"].features["label"],
    ClassLabel
)

assert dataset_validado["treino"][0]["label"] == 4
assert dataset_validado["teste"][0]["label"] == 7

print("\n===== VALIDACAO FINAL =====")

print("Treino:", len(dataset_validado["treino"]))
print("Teste:", len(dataset_validado["teste"]))

print(
    "Features:",
    dataset_validado["treino"].column_names
)

print(
    "Primeira label treino:",
    dataset_validado["treino"][0]["label"]
)

print(
    "Primeira label teste:",
    dataset_validado["teste"][0]["label"]
)

# =====================================
# 13. SALVAR RELATORIO
# =====================================

linhas = [
    "HUGGING FACE - AULA 08",
    "AJUSTANDO A VARIAVEL ALVO",
    "",
    "Tipo original: int64",
    "Tipo final: ClassLabel",
    "Classes: 0 a 10",
    "",
    "FEATURES FINAIS",
    str(dataset_validado["treino"].features),
    "",
    "TOTAL DE REDACOES",
    f"Treino: {len(dataset_validado['treino'])}",
    f"Teste: {len(dataset_validado['teste'])}",
    "",
    "PRIMEIRAS LABELS",
    f"Treino: {dataset_validado['treino'][0]['label']}",
    f"Teste: {dataset_validado['teste'][0]['label']}",
    "",
    "VALIDACAO",
    "Todas as notas originais foram preservadas.",
    "",
    f"Dataset salvo em: {destino}",
    "",
    "CONCLUSAO",
    "Dados preparados para classificacao.",
    "Nenhum modelo foi treinado nesta aula."
]

for conjunto, contagem in distribuicoes.items():
    linhas.extend([
        "",
        f"DISTRIBUICAO - {conjunto.upper()}"
    ])

    for nota, quantidade in contagem.items():
        linhas.append(
            f"Label {nota}: {quantidade}"
        )

if relatorio.exists():
    raise FileExistsError(relatorio)

relatorio.write_text(
    "\n".join(linhas),
    encoding="utf-8"
)

assert relatorio.is_file()
assert relatorio.stat().st_size > 0

print("\n===== ARQUIVOS GERADOS =====")
print("Dataset:", destino)
print("Relatorio:", relatorio)

print("\nAULA 08 - VARIAVEL ALVO VALIDADA")