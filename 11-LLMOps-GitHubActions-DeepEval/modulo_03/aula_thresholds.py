import os
import sys
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["DEEPEVAL_DISABLE_UPLOADS"] = "true"

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from modulo_03.custom_ollama_model import CustomOllamaModel

CONTEXTO = (ROOT / "data" / "context.txt").read_text(encoding="utf-8")

CASOS = [
    {
        "nome": "preco_consulta",
        "pergunta": "Qual é o preço da consulta inicial?",
        "resposta": "A consulta inicial custa R$ 850,00."
    },
    {
        "nome": "direito_familia",
        "pergunta": "Vocês trabalham com direito de família ou divórcio?",
        "resposta": "Sim, trabalhamos com direito de família e divórcio."
    }
]

THRESHOLDS = [0.25, 0.50, 0.75]
REPETICOES = 3
resultados = []

modelo = CustomOllamaModel("llama3.2")

for caso in CASOS:
    teste = LLMTestCase(
        input=caso["pergunta"],
        actual_output=caso["resposta"],
        retrieval_context=[CONTEXTO]
    )

    for nome, classe in [
        ("AnswerRelevancy", AnswerRelevancyMetric),
        ("Faithfulness", FaithfulnessMetric)
    ]:
        scores = []

        for repeticao in range(1, REPETICOES + 1):
            metrica = classe(
                model=modelo,
                threshold=0.25,
                async_mode=False,
                include_reason=True
            )

            try:
                metrica.measure(teste)
                score = float(metrica.score)
                scores.append(score)

                print(
                    f"{caso['nome']} | {nome} | "
                    f"execucao {repeticao} | score={score:.2f}"
                )

            except Exception as erro:
                print(
                    f"{caso['nome']} | {nome} | "
                    f"ERRO: {type(erro).__name__}: {erro}"
                )

        for threshold in THRESHOLDS:
            aprovados = sum(s >= threshold for s in scores)

            resultados.append({
                "caso": caso["nome"],
                "metrica": nome,
                "threshold": threshold,
                "scores": scores,
                "media": statistics.mean(scores) if scores else None,
                "desvio_padrao": statistics.pstdev(scores) if scores else None,
                "aprovacoes": aprovados,
                "execucoes_validas": len(scores)
            })

            print(
                f"THRESHOLD {threshold:.2f} | "
                f"APROVACOES {aprovados}/{len(scores)}"
            )

saida = ROOT / "modulo_03" / "resultados_thresholds.json"
saida.write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nRESULTADOS SALVOS:", saida)
