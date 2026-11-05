from pathlib import Path
from datetime import datetime
from typing import Dict
from collections import Counter
import json
import math
import re

import mlflow
import numpy as np

PROJETO = Path(__file__).resolve().parent.parent
BANCO = PROJETO / "mlflow.db"

mlflow.set_tracking_uri("sqlite:///" + BANCO.as_posix())
mlflow.set_registry_uri(mlflow.get_tracking_uri())
mlflow.set_experiment("genai-mlflow-3.0-demo")


class PromptManager:
    """Gerencia prompts registrados no MLflow."""

    def __init__(self):
        self.prompt_names = {
            "qa_assistant": "demo_qa_assistant",
            "summarization": "demo_summarization",
            "creative_writing": "demo_creative_writing"
        }
        print("PromptManager inicializado.")

    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        if prompt_type not in self.prompt_names:
            raise ValueError(
                f"Tipo de prompt nao encontrado: {prompt_type}"
            )

        prompt_name = self.prompt_names[prompt_type]

        try:
            prompt = mlflow.genai.load_prompt(
                f"prompts:/{prompt_name}@latest"
            )
            template = prompt.to_single_brace_format()

        except Exception as erro:
            print("Falha ao carregar prompt registrado:", erro)

            templates = {
                "qa_assistant": (
                    "Voce e um assistente especializado em {domain}.\n"
                    "Contexto: {context}\n"
                    "Pergunta: {question}\n"
                    "Forneca uma resposta detalhada e precisa."
                ),
                "summarization": (
                    "Resuma o seguinte texto em {style} estilo:\n"
                    "Texto: {text}\n"
                    "Limite: {max_words} palavras."
                ),
                "creative_writing": (
                    "Escreva um {genre} sobre {topic}.\n"
                    "Tom: {tone}\n"
                    "Audiencia: {audience}"
                )
            }

            template = templates[prompt_type]

        try:
            return template.format(**kwargs)
        except KeyError as erro:
            raise ValueError(
                f"Variavel obrigatoria ausente: {erro}"
            ) from erro

    def log_prompt_to_mlflow(
        self,
        prompt_type: str,
        filled_prompt: str,
        run_id: str
    ):
        """Registra o prompt como artefato JSON na run ativa."""

        run = mlflow.active_run()

        if run is None or run.info.run_id != run_id:
            raise RuntimeError(
                "A run informada precisa estar ativa."
            )

        prompt_name = self.prompt_names.get(
            prompt_type, prompt_type
        )

        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_type", prompt_type)

        prompt_data = {
            "type": prompt_type,
            "prompt_name": prompt_name,
            "filled_prompt": filled_prompt,
            "timestamp": datetime.now().isoformat()
        }

        mlflow.log_dict(
            prompt_data,
            f"prompts/prompt_{prompt_type}_{run_id}.json"
        )

        print("Prompt registrado como artefato JSON.")


class GenAIEvaluator:
    """Avaliador educativo de metricas de texto."""

    def __init__(self):
        self.metrics_calculated = 0
        print("GenAIEvaluator inicializado.")

    @staticmethod
    def tokenize(text: str):
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def calculate_bleu_score(reference: str, candidate: str) -> float:
        """Precisao unigram com penalidade de tamanho (nao BLEU completo)."""

        ref_words = GenAIEvaluator.tokenize(reference)
        cand_words = GenAIEvaluator.tokenize(candidate)

        if not ref_words or not cand_words:
            return 0.0

        ref_counts = Counter(ref_words)
        cand_counts = Counter(cand_words)

        matches = sum(
            min(count, ref_counts[word])
            for word, count in cand_counts.items()
        )

        precision = matches / len(cand_words)

        brevity_penalty = min(
            1.0, len(cand_words) / len(ref_words)
        )

        return float(precision * brevity_penalty)

    @staticmethod
    def calculate_rouge_l(reference: str, candidate: str) -> float:
        """ROUGE-L simplificado com LCS real."""

        ref_words = GenAIEvaluator.tokenize(reference)
        cand_words = GenAIEvaluator.tokenize(candidate)

        if not ref_words or not cand_words:
            return 0.0

        previous = [0] * (len(cand_words) + 1)

        for ref_word in ref_words:
            current = [0] * (len(cand_words) + 1)

            for j, cand_word in enumerate(cand_words, start=1):
                if ref_word == cand_word:
                    current[j] = previous[j - 1] + 1
                else:
                    current[j] = max(
                        previous[j], current[j - 1]
                    )

            previous = current

        lcs = previous[-1]

        precision = lcs / len(cand_words)
        recall = lcs / len(ref_words)

        if precision + recall == 0:
            return 0.0

        return float(
            2 * precision * recall / (precision + recall)
        )

    @staticmethod
    def calculate_perplexity(text: str) -> float:
        """Indicador simulado de diversidade, nao perplexidade real."""

        words = GenAIEvaluator.tokenize(text)

        if not words:
            return float("inf")

        diversity_ratio = len(set(words)) / len(words)

        return float(
            np.exp(-np.log(diversity_ratio + 0.1))
        )

    @staticmethod
    def check_toxicity(text: str) -> float:
        """Heuristica educativa limitada a palavras-chave."""

        palavras = set(GenAIEvaluator.tokenize(text))

        termos = {
            "idiota",
            "imbecil",
            "burro",
            "estupido",
            "estúpido"
        }

        ocorrencias = len(palavras.intersection(termos))

        return float(min(1.0, ocorrencias / 3))

    def evaluate_response(
        self,
        response: str,
        reference: str = None
    ) -> Dict[str, float]:

        words = self.tokenize(response)

        metrics = {
            "perplexity_simulated":
                self.calculate_perplexity(response),
            "toxicity_heuristic":
                self.check_toxicity(response),
            "response_length": float(len(words)),
            "unique_words": float(len(set(words)))
        }

        if reference is not None:
            metrics["bleu_simplified"] = (
                self.calculate_bleu_score(reference, response)
            )
            metrics["rouge_l_simplified"] = (
                self.calculate_rouge_l(reference, response)
            )

        self.metrics_calculated += len(metrics)

        return metrics


print("===== ALURA - AULA 8 =====")
print("CRIANDO O AVALIADOR DE METRICAS")

print("\n===== PROMPT MANAGER =====")

prompt_manager = PromptManager()

example_prompt = prompt_manager.get_prompt(
    "qa_assistant",
    domain="MLOps",
    context="MLflow 3.0 tem recursos para GenAI",
    question="Quais sao as novidades?"
)

print("\nPrompt gerado:")
print("-" * 50)
print(example_prompt)
print("-" * 50)

print("\n===== GENAI EVALUATOR =====")

evaluator = GenAIEvaluator()

texto_teste = (
    "MLflow 3.0 oferece recursos avancados para GenAI"
)

referencia = (
    "MLflow 3.0 tem features para modelos generativos"
)

metricas = evaluator.evaluate_response(
    texto_teste,
    referencia
)

for metrica, valor in metricas.items():
    print(f"{metrica}: {valor:.3f}")

print(
    "Total de metricas calculadas:",
    evaluator.metrics_calculated
)

print("\n===== REGISTRANDO NO MLFLOW =====")

with mlflow.start_run(
    run_name="aula_08_prompt_manager_evaluator"
) as run:

    prompt_manager.log_prompt_to_mlflow(
        "qa_assistant",
        example_prompt,
        run.info.run_id
    )

    for nome, valor in metricas.items():
        if math.isfinite(valor):
            mlflow.log_metric(nome, valor)

    mlflow.set_tag("lesson", "aula_08")
    mlflow.set_tag("evaluation_type", "educational_heuristics")

    print("Run ID:", run.info.run_id)
    print("Metricas registradas no MLflow.")

print("\n===== VALIDACAO =====")

assert len(metricas) == 6
assert evaluator.metrics_calculated == 6
assert 0 <= metricas["bleu_simplified"] <= 1
assert 0 <= metricas["rouge_l_simplified"] <= 1

print("PromptManager: OK")
print("GenAIEvaluator: OK")
print("Registro de metricas: OK")
print("Artefato JSON: OK")

print("\n===== AULA 8 CONCLUIDA =====")
