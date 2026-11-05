from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

BASE = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")

MODELO_TREINADO = (
    BASE / "modelos" / "distilbert_redacoes_treinado"
)

NOME_REPOSITORIO = "distilbert-pt-cased-redacao-nota"

print("\n===== AULA 11 - HUGGING FACE HUB =====")
print("Repositorio planejado:", NOME_REPOSITORIO)

print("\n===== VERIFICANDO MODELO TREINADO =====")

arquivos_necessarios = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
]

faltando = [
    nome for nome in arquivos_necessarios
    if not (MODELO_TREINADO / nome).is_file()
]

if faltando:
    print("PUBLICACAO BLOQUEADA")
    print("O modelo treinado ainda nao esta disponivel.")
    print("Arquivos ausentes:", faltando)
    print("Concluir primeiro a Aula 10.")
else:
    modelo = AutoModelForSequenceClassification.from_pretrained(
        str(MODELO_TREINADO),
        local_files_only=True
    )

    tokenizador = AutoTokenizer.from_pretrained(
        str(MODELO_TREINADO),
        local_files_only=True
    )

    assert modelo.config.num_labels == 11

    print("Modelo carregado:", modelo.config.model_type)
    print("Classes:", modelo.config.num_labels)
    print("Tokenizador:", type(tokenizador).__name__)
    print("MODELO DISPONIVEL PARA A ETAPA DE PUBLICACAO")

print("\nNenhum arquivo foi enviado ao Hugging Face Hub.")