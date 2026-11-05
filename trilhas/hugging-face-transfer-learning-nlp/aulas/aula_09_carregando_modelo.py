from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk, ClassLabel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

raiz = Path(__file__).resolve().parent.parent

base_f = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")

dataset_path = (
    base_f / "dados" / "redacoes_prontas_treinamento"
)

modelo_path = (
    base_f / "modelos" / "distilbert_redacoes_inicial"
)

relatorio_path = (
    raiz / "resultados" / "aula_09_modelo_pre_treinado.txt"
)

checkpoint_modelo = "Geotrend/distilbert-base-pt-cased"

print("\n===== AULA 09 - MODELO PRE-TREINADO =====")

# 1. CARREGAR DATASET DA AULA 08

print("\n===== CARREGANDO DATASET =====")

dataset_tokenizado = load_from_disk(
    str(dataset_path)
)

print(dataset_tokenizado)

assert len(dataset_tokenizado["treino"]) == 3656
assert len(dataset_tokenizado["teste"]) == 914

feature = dataset_tokenizado["treino"].features["label"]

assert isinstance(feature, ClassLabel)

num_labels = feature.num_classes

assert num_labels == 11

# 2. CRIAR MAPEAMENTOS DAS CLASSES

id2label = {
    i: str(i)
    for i in range(num_labels)
}

label2id = {
    valor: chave
    for chave, valor in id2label.items()
}

print("\n===== CONFIGURACAO DAS CLASSES =====")

print("Numero de classes:", num_labels)
print("id2label:", id2label)
print("label2id:", label2id)

# 3. CARREGAR TOKENIZADOR

print("\n===== CARREGANDO TOKENIZADOR =====")

tokenizador = AutoTokenizer.from_pretrained(
    checkpoint_modelo
)

# 4. CARREGAR MODELO PARA CLASSIFICACAO

print("\n===== CARREGANDO DISTILBERT =====")

# Se ja salvamos a configuracao inicial,
# podemos reutiliza-la sem reinicializar a cabeca.

if (modelo_path / "config.json").exists():
    print("Reutilizando modelo inicial salvo.")

    modelo = AutoModelForSequenceClassification.from_pretrained(
        str(modelo_path)
    )
else:
    print("Carregando pesos pre-treinados.")

    torch.manual_seed(42)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_modelo,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

modelo.to("cpu")

assert modelo.config.num_labels == 11
assert modelo.config.id2label == id2label
assert modelo.config.label2id == label2id

print("Arquitetura:", modelo.config.model_type)
print("Classes:", modelo.config.num_labels)
print("Dispositivo:", next(modelo.parameters()).device)

parametros = sum(
    p.numel()
    for p in modelo.parameters()
)

print("Parametros totais:", parametros)

# 5. PREPARAR LOTES COM PADDING DINAMICO

print("\n===== PREPARANDO DATA COLLATOR =====")

collator = DataCollatorWithPadding(
    tokenizer=tokenizador,
    padding=True,
    return_tensors="pt"
)

# Adaptacao da aula TensorFlow para PyTorch

dados_treino = DataLoader(
    dataset_tokenizado["treino"],
    shuffle=True,
    batch_size=4,
    collate_fn=collator,
    num_workers=0
)

dados_validacao = DataLoader(
    dataset_tokenizado["teste"],
    shuffle=False,
    batch_size=4,
    collate_fn=collator,
    num_workers=0
)

print("Lotes de treino:", len(dados_treino))
print("Lotes de validacao:", len(dados_validacao))

# 6. VALIDAR PRIMEIRO LOTE

print("\n===== VALIDANDO PRIMEIRO LOTE =====")

lote = next(iter(dados_treino))

print("Colunas:", list(lote.keys()))

for nome, tensor in lote.items():
    print(nome, tuple(tensor.shape))

assert "input_ids" in lote
assert "attention_mask" in lote
assert "labels" in lote

assert lote["input_ids"].shape[0] == 4
assert lote["input_ids"].shape[1] <= 512

assert lote["attention_mask"].shape == (
    lote["input_ids"].shape
)

assert lote["labels"].shape == (4,)

print("Labels do primeiro lote:", lote["labels"].tolist())

# 7. SALVAR MODELO INICIAL

print("\n===== SALVANDO MODELO INICIAL =====")

if not (modelo_path / "config.json").exists():
    modelo_path.mkdir(parents=True, exist_ok=True)

    modelo.save_pretrained(
        str(modelo_path),
        safe_serialization=True
    )

    tokenizador.save_pretrained(
        str(modelo_path)
    )

# 8. RECARREGAR CONFIGURACAO

from transformers import AutoConfig

configuracao = AutoConfig.from_pretrained(
    str(modelo_path)
)

assert configuracao.num_labels == 11

# 9. SALVAR RELATORIO

linhas = [
    "HUGGING FACE - AULA 09",
    "CARREGANDO O MODELO PRE-TREINADO",
    "",
    f"Checkpoint: {checkpoint_modelo}",
    f"Arquitetura: {modelo.config.model_type}",
    f"Numero de classes: {num_labels}",
    f"Parametros totais: {parametros}",
    "",
    f"id2label: {id2label}",
    f"label2id: {label2id}",
    "",
    f"Redacoes treino: {len(dataset_tokenizado['treino'])}",
    f"Redacoes teste: {len(dataset_tokenizado['teste'])}",
    "",
    "PYTORCH",
    "Dispositivo: CPU",
    "Batch size: 4",
    f"Lotes de treino: {len(dados_treino)}",
    f"Lotes de validacao: {len(dados_validacao)}",
    "",
    f"Modelo inicial salvo em: {modelo_path}",
    "",
    "OBSERVACOES",
    "Modelo base carregado.",
    "Camada classificadora ainda nao foi treinada.",
    "Lotes preparados com padding dinamico.",
    "Nao foi executado treinamento nesta aula."
]

if not relatorio_path.exists():
    relatorio_path.write_text(
        "\n".join(linhas),
        encoding="utf-8"
    )

print("\n===== VALIDACAO FINAL =====")

print("Modelo:", checkpoint_modelo)
print("Classes:", modelo.config.num_labels)
print("Treino:", len(dataset_tokenizado["treino"]))
print("Teste:", len(dataset_tokenizado["teste"]))
print("Modelo salvo:", modelo_path)

print("\nAULA 09 - MODELO CARREGADO E VALIDADO")