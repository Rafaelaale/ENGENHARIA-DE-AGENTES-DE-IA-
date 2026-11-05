from pathlib import Path
from typing import Dict, Any
import sys
import time
import math

import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJETO / "aula_08"))

from avaliador_metricas_genai import PromptManager, GenAIEvaluator

BANCO = PROJETO / "mlflow.db"
mlflow.set_tracking_uri("sqlite:///" + BANCO.as_posix())
mlflow.set_registry_uri(mlflow.get_tracking_uri())
mlflow.set_experiment("genai-mlflow-3.0-demo")


class GenAIModel:

    def __init__(self, model_type="gpt2", device=None):
        self.model_type = model_type
        self.prompt_manager = PromptManager()
        self.evaluator = GenAIEvaluator()

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.tokenizer = None
        self.mock_mode = False

        print("GenAIModel inicializado")
        print("Modelo:", self.model_type)
        print("Dispositivo:", self.device)

    def _initialize_model(self):
        try:
            if self.model_type != "gpt2":
                raise ValueError("Modelo nao suportado nesta aula.")

            print("Carregando GPT-2...")

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = self.model.to(self.device)
            self.model.eval()

            print("GPT-2 carregado com sucesso.")
            print("Vocabulario:", self.tokenizer.vocab_size)

        except Exception as erro:
            print("Falha ao carregar GPT-2:", erro)
            print("Modo demonstrativo ativado.")

            self.model = None
            self.tokenizer = None
            self.mock_mode = True

    def generate(self, prompt, max_tokens=40, temperature=0.7):

        if self.model is None:
            return (
                "MLflow permite acompanhar experimentos, "
                "modelos e prompts de inteligencia artificial."
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            chave: valor.to(self.device)
            for chave, valor in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        quantidade_entrada = inputs["input_ids"].shape[1]

        return self.tokenizer.decode(
            outputs[0][quantidade_entrada:],
            skip_special_tokens=True
        ).strip()

    def generate_with_tracking(self, prompt_type, **kwargs):

        max_tokens = kwargs.pop("max_tokens", 40)
        temperature = kwargs.pop("temperature", 0.7)
        reference = kwargs.pop("reference", None)

        with mlflow.start_run(
            run_name=f"tracking_{prompt_type}"
        ) as run:

            mlflow.log_params({
                "model_type": self.model_type,
                "prompt_type": prompt_type,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "mock_mode": self.mock_mode
            })

            prompt = self.prompt_manager.get_prompt(
                prompt_type, **kwargs
            )

            self.prompt_manager.log_prompt_to_mlflow(
                prompt_type, prompt, run.info.run_id
            )

            mlflow.log_param("prompt_length", len(prompt))

            inicio = time.perf_counter()

            response = self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            generation_time = time.perf_counter() - inicio

            metrics = self.evaluator.evaluate_response(
                response,
                reference=reference
            )

            for nome, valor in metrics.items():
                if math.isfinite(float(valor)):
                    mlflow.log_metric(nome, float(valor))

            mlflow.log_metric(
                "generation_time_seconds",
                generation_time
            )

            mlflow.log_dict(
                {
                    "prompt": prompt,
                    "response": response,
                    "mock_mode": self.mock_mode
                },
                "outputs/generation.json"
            )

            mlflow.set_tags({
                "task": prompt_type,
                "model_family": "generative",
                "generation_mode": (
                    "mock" if self.mock_mode else "real"
                )
            })

            return {
                "run_id": run.info.run_id,
                "prompt": prompt,
                "response": response,
                "metrics": metrics,
                "generation_time": generation_time,
                "mock_mode": self.mock_mode
            }

    def generate_with_tracking_nested(self, prompt_type, **kwargs):

        if mlflow.active_run() is None:
            raise RuntimeError("Nenhuma run principal ativa.")

        max_tokens = kwargs.pop("max_tokens", 40)
        temperature = kwargs.pop("temperature", 0.7)
        reference = kwargs.pop("reference", None)

        with mlflow.start_run(
            run_name=f"nested_{prompt_type}",
            nested=True
        ) as run:

            mlflow.log_params({
                "model_type": self.model_type,
                "prompt_type": prompt_type,
                "max_tokens": max_tokens,
                "temperature": temperature
            })

            prompt = self.prompt_manager.get_prompt(
                prompt_type, **kwargs
            )

            inicio = time.perf_counter()
            response = self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            tempo = time.perf_counter() - inicio

            metrics = self.evaluator.evaluate_response(
                response, reference=reference
            )

            for nome, valor in metrics.items():
                if math.isfinite(float(valor)):
                    mlflow.log_metric(nome, float(valor))

            mlflow.log_metric("generation_time_seconds", tempo)

            self.prompt_manager.log_prompt_to_mlflow(
                prompt_type, prompt, run.info.run_id
            )

            mlflow.log_dict(
                {"prompt": prompt, "response": response},
                "outputs/generation.json"
            )

            return {
                "run_id": run.info.run_id,
                "response": response,
                "metrics": metrics,
                "generation_time": tempo
            }


if __name__ == "__main__":
    print("===== ALURA - AULA 9 =====")
    modelo = GenAIModel()
    modelo._initialize_model()

    print("Modo:", "MOCK" if modelo.mock_mode else "GPT-2 REAL")

    resposta = modelo.generate(
        "MLflow 3.0 is",
        max_tokens=20
    )

    print("Resposta:", resposta)
    print("===== AULA 9 CONCLUIDA =====")
