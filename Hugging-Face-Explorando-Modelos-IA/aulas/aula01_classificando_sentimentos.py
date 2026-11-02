from transformers import pipeline
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)

MODELO = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

print("=" * 60)
print("AULA 01 - HUGGING FACE")
print("CLASSIFICANDO SENTIMENTOS COM DISTILBERT")
print("=" * 60)

print("\nCarregando modelo pre-treinado...")
print("Modelo:", MODELO)

# O modelo e o tokenizador serao baixados
# automaticamente na primeira execucao.
classificador_sentimento = pipeline(
    "sentiment-analysis",
    model=MODELO,
    device="cpu"
)

avaliacoes = [
    "I love this product",
    "I am totally disappointed in this product"
]

resultados = []

for numero, texto in enumerate(avaliacoes, start=1):
    resposta = classificador_sentimento(texto)[0]

    classe = resposta["label"]
    score = float(resposta["score"])

    print(f"\nAVALIACAO {numero}")
    print("Texto:", texto)
    print("Classe:", classe)
    print(f"Score: {score:.6f}")
    print(f"Confianca do modelo: {score * 100:.2f}%")

    resultados.append({
        "texto": texto,
        "sentimento": classe,
        "score": score
    })

# Verificar classificacoes esperadas
assert resultados[0]["sentimento"] == "POSITIVE"
assert resultados[1]["sentimento"] == "NEGATIVE"

# Salvar resultados para consulta
arquivo = RESULTADOS / "aula01_sentimentos.json"

with arquivo.open("w", encoding="utf-8") as f:
    json.dump(
        resultados,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nVALIDACAO DAS CLASSIFICACOES: OK")
print("Resultados salvos:", arquivo)
print("\nAULA 01 EXECUTADA COM SUCESSO!")
