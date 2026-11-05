from pathlib import Path
import sys
import json
import math

import mlflow

PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJETO / "aula_09"))

from tracking_genai_huggingface import GenAIModel

BANCO = PROJETO / "mlflow.db"

mlflow.set_tracking_uri(
    "sqlite:///" + BANCO.as_posix()
)

mlflow.set_registry_uri(
    mlflow.get_tracking_uri()
)

mlflow.set_experiment(
    "genai-mlflow-3.0-demo"
)

print("===== ALURA - AULA 10 =====")
print("QA ASSIST COM TRACKING COMPLETO")

print("\n===== INICIALIZANDO MODELO =====")

modelo = GenAIModel(model_type="gpt2")
modelo._initialize_model()

print("\n===== EXECUTANDO QA ASSIST =====")

resultado_qa = modelo.generate_with_tracking(
    prompt_type="qa_assistant",
    domain="MLOps e Machine Learning",
    context=(
        "MLflow 3.0 introduz recursos nativos "
        "para modelos generativos"
    ),
    question="Quais sao as principais novidades?",
    temperature=0.7,
    max_tokens=100
)

print("\n===== RUN REGISTRADA =====")

print("Run ID:", resultado_qa["run_id"])

print(
    "Tempo de geracao:",
    f'{resultado_qa["generation_time"]:.2f}s'
)

print(
    "Modo:",
    "MOCK" if resultado_qa["mock_mode"]
    else "GPT-2 REAL"
)

print("\n===== PROMPT UTILIZADO =====")

print("-" * 50)
print(resultado_qa["prompt"][:200] + "...")
print("-" * 50)

print("\n===== RESPOSTA GERADA =====")

print(resultado_qa["response"])

print("\n===== METRICAS =====")

for metrica, valor in resultado_qa["metrics"].items():
    print(f"{metrica}: {valor:.3f}")

print("\n===== VALIDANDO TRACKING =====")

run = mlflow.get_run(
    resultado_qa["run_id"]
)

assert run.info.status == "FINISHED"

assert run.data.params["prompt_type"] == "qa_assistant"

assert run.data.params["max_tokens"] == "100"

assert resultado_qa["response"]

artefatos = mlflow.artifacts.list_artifacts(
    run_id=resultado_qa["run_id"]
)

nomes_artefatos = [
    item.path for item in artefatos
]

assert "prompts" in nomes_artefatos
assert "outputs" in nomes_artefatos

print("Run finalizada: OK")
print("Parametros registrados: OK")
print("Metricas registradas: OK")
print("Artefatos registrados: OK")

print("\nTracking URI:", mlflow.get_tracking_uri())

print("\n===== AULA 10 CONCLUIDA =====")
