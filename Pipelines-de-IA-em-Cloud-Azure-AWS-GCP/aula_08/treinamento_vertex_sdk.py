import argparse
import json
from pathlib import Path

CONFIG = {
    "project_id": "CONFIGURAR_PROJECT_ID",
    "location": "us-central1",
    "dataset_name": "ChurnData",
    "gcs_source": "gs://alura-469701-bucket/Churn_Modelling.csv",
    "job_name": "Treinamento_Churn_SDK",
    "target_column": "Exited",
    "prediction_type": "classification",
    "optimization_objective": "maximize-au-prc",
    "training_fraction": 0.8,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "budget_milli_node_hours": 1000,
}


def preparar():
    destino = Path(__file__).resolve().parent
    arquivo = destino / "configuracao_automl.json"

    arquivo.write_text(
        json.dumps(CONFIG, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=== AULA 08 - VERTEX AI SDK ===")
    print("Projeto:", CONFIG["project_id"])
    print("Regiao:", CONFIG["location"])
    print("Dataset:", CONFIG["dataset_name"])
    print("Origem:", CONFIG["gcs_source"])
    print("Alvo:", CONFIG["target_column"])
    print("Objetivo:", CONFIG["optimization_objective"])
    print("Orcamento:", CONFIG["budget_milli_node_hours"], "milli-node-hours")
    print("Configuracao salva:", arquivo)
    print("Nenhum recurso Cloud foi criado.")


def executar():
    if CONFIG["project_id"] == "CONFIGURAR_PROJECT_ID":
        raise ValueError("Configure um Project ID real antes de executar.")

    from google.cloud import aiplatform

    aiplatform.init(
        project=CONFIG["project_id"],
        location=CONFIG["location"]
    )

    dataset = aiplatform.TabularDataset.create(
        display_name=CONFIG["dataset_name"],
        gcs_source=[CONFIG["gcs_source"]]
    )

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=CONFIG["job_name"],
        optimization_prediction_type=CONFIG["prediction_type"],
        optimization_objective=CONFIG["optimization_objective"]
    )

    model = job.run(
        dataset=dataset,
        target_column=CONFIG["target_column"],
        training_fraction_split=CONFIG["training_fraction"],
        validation_fraction_split=CONFIG["validation_fraction"],
        test_fraction_split=CONFIG["test_fraction"],
        budget_milli_node_hours=CONFIG["budget_milli_node_hours"],
        sync=True
    )

    print("Treinamento finalizado.")
    print("Modelo:", model.resource_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executar-cloud",
        action="store_true",
        help="Cria recursos e executa treinamento real na Google Cloud"
    )

    args = parser.parse_args()

    preparar()

    if args.executar_cloud:
        executar()
    else:
        print("Modo de preparacao: execucao Cloud desativada.")


if __name__ == "__main__":
    main()