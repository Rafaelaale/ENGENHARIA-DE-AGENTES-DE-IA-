from pathlib import Path
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")

PROJETO = Path(__file__).resolve().parent.parent
BANCO = PROJETO / "mlflow.db"

mlflow.set_tracking_uri("sqlite:///" + BANCO.as_posix())
mlflow.set_registry_uri(mlflow.get_tracking_uri())

EXPERIMENT_NAME = "genai-mlflow-3.0-demo"

print("===== ALURA - AULA 7 =====")
print("CONFIGURANDO PROJETO GENAI")

print("\n===== BIBLIOTECAS =====")
print("MLflow:", mlflow.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo disponivel:", device)

print("\n===== CONFIGURACAO MLFLOW =====")

mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError("Experimento nao encontrado.")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experimento:", experiment.name)
print("Experiment ID:", experiment.experiment_id)

print("\n===== PROMPT REGISTRY =====")

prompts = [
    {
        "name": "demo_qa_assistant",
        "template": """Você é um assistente especializado em {{domain}}.

Contexto: {{context}}

Pergunta: {{question}}

Por favor, forneça uma resposta detalhada e precisa.""",
        "commit_message": "Q&A Assistant para demonstracao",
        "tags": {
            "task": "question_answering",
            "demo": "mlflow-3.0"
        }
    },
    {
        "name": "demo_summarization",
        "template": """Resuma o seguinte texto em {{style}} estilo:

Texto: {{text}}

Limite: {{max_words}} palavras.""",
        "commit_message": "Prompt de sumarizacao para demonstracao",
        "tags": {
            "task": "summarization",
            "demo": "mlflow-3.0"
        }
    },
    {
        "name": "demo_creative_writing",
        "template": """Escreva um {{genre}} sobre {{topic}}.

Tom: {{tone}}
Audiência: {{audience}}

Seja criativo e engajante.""",
        "commit_message": "Prompt de escrita criativa para demonstracao",
        "tags": {
            "task": "creative_writing",
            "demo": "mlflow-3.0"
        }
    }
]

registrados = []

for item in prompts:
    print("\nRegistrando:", item["name"])

    registro = mlflow.genai.register_prompt(
        name=item["name"],
        template=item["template"],
        commit_message=item["commit_message"],
        tags=item["tags"]
    )

    print("Versao:", registro.version)
    registrados.append((item["name"], registro.version))

print("\n===== VALIDACAO DOS PROMPTS =====")

for nome, versao in registrados:
    prompt = mlflow.genai.load_prompt(
        f"prompts:/{nome}/{versao}"
    )

    print("Prompt:", nome)
    print("Versao:", prompt.version)
    print("Template validado:", bool(prompt.template))

assert len(registrados) == 3

print("\nTotal de prompts registrados:", len(registrados))
print("Dispositivo:", device)
print("Banco SQLite:", BANCO.name)

print("\n===== AULA 7 CONCLUIDA =====")
