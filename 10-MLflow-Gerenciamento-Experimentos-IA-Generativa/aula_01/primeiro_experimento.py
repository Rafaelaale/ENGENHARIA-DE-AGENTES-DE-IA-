from pathlib import Path
import os
import mlflow

BASE_DIR = Path(__file__).resolve().parent
PROJETO = BASE_DIR.parent

BANCO = PROJETO / "mlflow.db"

TRACKING_URI = "sqlite:///" + BANCO.as_posix()

os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)

print("MLflow:", mlflow.__version__)
print("Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("aula_01_primeiro_experimento")

artefato = BASE_DIR / "resultado.txt"

artefato.write_text(
    "Primeiro experimento MLflow\n"
    "Accuracy demonstrativa: 0.95\n",
    encoding="utf-8"
)

with mlflow.start_run(run_name="primeira_execucao") as run:

    mlflow.log_param("learning_rate", 0.01)

    mlflow.log_metric("accuracy", 0.95)

    mlflow.log_artifact(str(artefato))

    print("Experimento registrado com sucesso!")
    print("Run ID:", run.info.run_id)
    print("Accuracy demonstrativa:", 0.95)
    print("Artefato:", artefato.name)

print("Execucao finalizada.")
