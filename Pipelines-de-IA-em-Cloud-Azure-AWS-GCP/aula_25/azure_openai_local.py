import json
import urllib.request
from pathlib import Path
from time import perf_counter

import openai
from openai import AzureOpenAI

BASE = Path(__file__).resolve().parent

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODELO_LOCAL = "llama3.2:latest"
DEPLOYMENT_REFERENCIA = "gpt-4.1-mini"


def completar_chat(messages, temperature=0.7, max_tokens=512):
    """
    Adaptador local para estudar a estrutura de chat.

    Nao utiliza AzureOpenAI.
    Nao invoca o modelo GPT-4.1 Mini.
    """

    payload = {
        "model": MODELO_LOCAL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    requisicao = urllib.request.Request(
        OLLAMA,
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

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": dados["message"]["content"]
                }
            }
        ],
        "usage": {
            "prompt_tokens": dados.get("prompt_eval_count"),
            "completion_tokens": dados.get("eval_count")
        },
        "modelo_executado": MODELO_LOCAL
    }


def main():
    print("=== AULA 25 - AZURE OPENAI ===")

    print("\n=== CONFIGURACAO ===")
    print("SDK OpenAI:", openai.__version__)
    print("Classe estudada:", AzureOpenAI.__name__)
    print("Deployment de referencia:", DEPLOYMENT_REFERENCIA)
    print("Modelo realmente executado:", MODELO_LOCAL)
    print("Azure invocada: NAO")

    historico = [
        {
            "role": "system",
            "content": (
                "Voce e um assistente de inteligencia artificial. "
                "Responda em portugues brasileiro, com clareza "
                "e precisao tecnica."
            )
        }
    ]

    perguntas = [
        "Como voce pode me ajudar?",
        "O que e um LLM?",
        "Qual a diferenca entre um LLM e um modelo tradicional de classificacao?",
        "Resuma essa diferenca em tres topicos."
    ]

    resultados = []

    for numero, pergunta in enumerate(perguntas, start=1):
        print(f"\n=== TESTE {numero} ===")
        print("Pergunta:", pergunta)

        historico.append({
            "role": "user",
            "content": pergunta
        })

        inicio = perf_counter()

        resposta = completar_chat(
            messages=historico,
            temperature=0.7,
            max_tokens=512
        )

        duracao = round(
            perf_counter() - inicio,
            3
        )

        texto = resposta["choices"][0]["message"]["content"]

        print("Resposta:", texto)
        print("Tempo:", duracao, "segundos")

        historico.append({
            "role": "assistant",
            "content": texto
        })

        resultados.append({
            "teste": numero,
            "pergunta": pergunta,
            "resposta": texto,
            "duracao_segundos": duracao,
            "usage": resposta["usage"]
        })

    relatorio = {
        "aula": 25,
        "plataforma_estudada": "Azure OpenAI",
        "classe_sdk_estudada": "AzureOpenAI",
        "deployment_referencia": DEPLOYMENT_REFERENCIA,
        "modelo_executado": MODELO_LOCAL,
        "execucao": "Ollama local",
        "total_testes": len(resultados),
        "resultados": resultados,
        "historico_final": historico,
        "azure_invocada": False
    }

    arquivo = BASE / "resultado_azure_openai_local.json"

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
    print("Mensagens no historico:", len(historico))
    print("Modelo real:", MODELO_LOCAL)
    print("Azure invocada: NAO")
    print("Relatorio:", arquivo)


if __name__ == "__main__":
    main()