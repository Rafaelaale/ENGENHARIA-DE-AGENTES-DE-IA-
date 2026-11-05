from pathlib import Path
import json
import sys

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

BASE_F = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")
PROJETO = Path(r"C:\Users\rafal\Desktop\HUGGING-FACE-TRANSFER-LEARNING-NLP")

MODELO = BASE_F / "modelos" / "distilbert_redacoes_treinado"
RELATORIO_TREINO = BASE_F / "resultados" / "aula_10_treinamento.json"

CSV = PROJETO / "dados" / "redacoes_teste.csv"

SAIDA = BASE_F / "resultados" / "aula_12_previsoes.json"

torch.set_num_threads(1)

print("\n===== AULA 12 - PREVISAO DE REDACOES =====", flush=True)

# 1. SELECIONAR AS REDACOES DA AULA

dados_redacoes = pd.read_csv(CSV)

assert len(dados_redacoes) == 914

indices = [2, 909]

textos = [
    str(dados_redacoes.iloc[i]["essay"])
    for i in indices
]

notas_reais = [
    int(dados_redacoes.iloc[i]["score"])
    for i in indices
]

print("\n===== REDACOES SELECIONADAS =====")

for i, nota in zip(indices, notas_reais):
    print(f"Indice {i} | Nota real: {nota}")

# 2. VERIFICAR TREINAMENTO

print("\n===== VERIFICANDO MODELO TREINADO =====")

arquivos = [
    MODELO / "config.json",
    MODELO / "model.safetensors",
    MODELO / "tokenizer_config.json",
    RELATORIO_TREINO,
]

faltando = [p for p in arquivos if not p.is_file()]

if faltando:
    print("AULA 12 PREPARADA, PREVISAO AINDA BLOQUEADA.")
    print("O treinamento completo da Aula 10 ainda nao foi confirmado.")
    print("\nArquivos pendentes:")

    for arquivo in faltando:
        print("-", arquivo.name)

    print("\nO modelo inicial nao sera usado para atribuir notas.")
    sys.exit(0)

# Validar se o relatorio confirma as duas epocas

relatorio = json.loads(
    RELATORIO_TREINO.read_text(encoding="utf-8")
)

historico = relatorio.get("historico", [])

epocas_concluidas = {
    item.get("epoca")
    for item in historico
}

if not {1, 2}.issubset(epocas_concluidas):
    print("Treinamento de duas epocas nao confirmado.")
    print("Previsao bloqueada.")
    sys.exit(0)

# 3. CARREGAR MODELO E TOKENIZADOR

print("\n===== CARREGANDO MODELO TREINADO =====")

tokenizador = AutoTokenizer.from_pretrained(
    str(MODELO),
    local_files_only=True
)

modelo = AutoModelForSequenceClassification.from_pretrained(
    str(MODELO),
    local_files_only=True
)

assert modelo.config.num_labels == 11

modelo.eval()

# 4. TOKENIZAR OS TEXTOS

textos_tokenizados = tokenizador(
    textos,
    return_tensors="pt",
    padding="longest",
    truncation=True,
    max_length=512
)

print("Input IDs:", tuple(textos_tokenizados["input_ids"].shape))
print("Attention mask:", tuple(textos_tokenizados["attention_mask"].shape))

# 5. REALIZAR PREVISOES

print("\n===== PREVISOES COM LOGITS =====")

with torch.no_grad():
    resultados = modelo(**textos_tokenizados).logits

    probabilidades = torch.softmax(
        resultados,
        dim=1
    )

    classificacoes = torch.argmax(
        resultados,
        dim=1
    )

assert resultados.shape == (2, 11)

previsoes = []

for posicao, indice in enumerate(indices):

    nota_prevista = int(
        classificacoes[posicao].item()
    )

    confianca = float(
        probabilidades[posicao, nota_prevista].item()
    )

    nota_real = notas_reais[posicao]

    print(
        f"Redacao {indice} | "
        f"Real: {nota_real} | "
        f"Prevista: {nota_prevista} | "
        f"Score: {confianca:.4f}"
    )

    previsoes.append({
        "indice": indice,
        "nota_real": nota_real,
        "nota_prevista": nota_prevista,
        "score_modelo": confianca,
        "acertou": nota_real == nota_prevista
    })

# 6. TESTAR PIPELINE

print("\n===== TESTANDO PIPELINE =====")

classificador = pipeline(
    "text-classification",
    model=modelo,
    tokenizer=tokenizador,
    framework="pt",
    device=-1
)

resultados_pipeline = classificador(
    textos,
    truncation=True,
    max_length=512,
    batch_size=2
)

for indice, resultado in zip(indices, resultados_pipeline):
    print(f"Redacao {indice}: {resultado}")

# 7. SALVAR RESULTADOS

saida = {
    "aula": 12,
    "modelo": str(MODELO),
    "previsoes": previsoes,
    "pipeline": resultados_pipeline
}

if SAIDA.exists():
    print("\nRelatorio anterior preservado:", SAIDA)
else:
    SAIDA.write_text(
        json.dumps(
            saida,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

print("\nAULA 12 - PREVISOES EXECUTADAS")