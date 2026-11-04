import json
import urllib.request
from pathlib import Path
from time import perf_counter

import boto3
import botocore

BASE = Path(__file__).resolve().parent

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODELO_LOCAL = "llama3.2:latest"

MODELOS_BEDROCK = [
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0"
]


def simular_converse(model_id, messages, inference_config):
    """
    Adaptador didatico da estrutura Converse para Ollama.

    Nao utiliza Amazon Bedrock.
    Nao executa modelos Amazon Nova.
    """

    if model_id not in MODELOS_BEDROCK:
        raise ValueError("ID de modelo nao reconhecido.")

    mensagens_ollama = []

    for mensagem in messages:
        partes = mensagem.get("content", [])

        texto = "\n".join(
            parte["text"]
            for parte in partes
            if "text" in parte
        )

        mensagens_ollama.append({
            "role": mensagem["role"],
            "content": texto
        })

    payload = {
        "model": MODELO_LOCAL,
        "stream": False,
        "messages": mensagens_ollama,
        "options": {
            "temperature": inference_config.get(
                "temperature", 0.5
            ),
            "num_predict": inference_config.get(
                "maxTokens", 512
            )
        }
    }

    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        requisicao,
        timeout=180
    ) as resposta:
        dados = json.loads(
            resposta.read().decode("utf-8")
        )

    texto = dados["message"]["content"]

    return {
        "modelIdSolicitado": model_id,
        "modeloExecutado": MODELO_LOCAL,
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": texto}
                ]
            }
        },
        "usage": {
            "inputTokens": dados.get("prompt_eval_count"),
            "outputTokens": dados.get("eval_count")
        }
    }


def executar_teste(model_id, pergunta):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": pergunta}
            ]
        }
    ]

    inference_config = {
        "maxTokens": 512,
        "temperature": 0.5
    }

    inicio = perf_counter()

    response = simular_converse(
        model_id=model_id,
        messages=messages,
        inference_config=inference_config
    )

    duracao = round(
        perf_counter() - inicio,
        3
    )

    resposta_texto = (
        response["output"]["message"]["content"][0]["text"]
    )

    print("\nModelo solicitado:", model_id)
    print("Modelo realmente executado:", MODELO_LOCAL)
    print("Pergunta:", pergunta)
    print("Resposta:", resposta_texto)
    print("Tempo:", duracao, "segundos")

    return {
        "modelo_solicitado": model_id,
        "modelo_executado": MODELO_LOCAL,
        "pergunta": pergunta,
        "resposta": resposta_texto,
        "duracao_segundos": duracao,
        "usage": response["usage"]
    }


def main():
    print("=== AULA 24 - AWS BEDROCK ===")

    print("\n=== SDK ===")
    print("Boto3:", boto3.__version__)
    print("Botocore:", botocore.__version__)

    print("\n=== CONFIGURACAO DIDATICA ===")
    print("Servico:", "bedrock-runtime")
    print("Regiao de exemplo:", "us-east-1")
    print("Operacao estudada:", "converse")
    print("Chamadas reais AWS:", "NAO")

    testes = [
        (
            "amazon.nova-lite-v1:0",
            "Como voce pode me ajudar?"
        ),
        (
            "amazon.nova-lite-v1:0",
            "Explique em tres topicos por que prompts "
            "menores podem ser vantajosos para LLMs."
        ),
        (
            "amazon.nova-micro-v1:0",
            "Explique em tres topicos por que prompts "
            "menores podem ser vantajosos para LLMs."
        )
    ]

    resultados = []

    for model_id, pergunta in testes:
        resultados.append(
            executar_teste(model_id, pergunta)
        )

    relatorio = {
        "aula": 24,
        "servico_estudado": "Amazon Bedrock",
        "operacao": "Converse",
        "tipo_execucao": "Adaptador local Ollama",
        "modelo_real": MODELO_LOCAL,
        "modelos_aws_apenas_referencia": MODELOS_BEDROCK,
        "total_testes": len(resultados),
        "resultados": resultados,
        "aws_invocada": False
    }

    arquivo = BASE / "resultado_bedrock_local.json"

    arquivo.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== VALIDACAO FINAL ===")
    print("Testes executados:", len(resultados))
    print("Modelo real:", MODELO_LOCAL)
    print("AWS invocada: NAO")
    print("Relatorio:", arquivo)


if __name__ == "__main__":
    main()