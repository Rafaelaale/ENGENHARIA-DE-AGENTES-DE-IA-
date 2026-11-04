from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from time import perf_counter
import json
import urllib.request
import urllib.error

app = FastAPI(
    title="Aula 15 - LLM API",
    version="1.0.0"
)

MODELO = "llama3.2:latest"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

SYSTEM_PROMPT = (
    "Voce e um assistente tecnico especializado "
    "em Python, APIs e integracao de LLMs. "
    "Responda de forma concisa, com passos "
    "acionaveis. Forneca exemplos em Python "
    "quando forem uteis."
)


class ChatRequest(BaseModel):
    pergunta: str = Field(min_length=3, max_length=2000)
    temperatura: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0
    )


@app.get("/status")
def status():
    return {
        "status": "online",
        "modelo": MODELO,
        "provedor": "ollama-local"
    }


@app.post("/chat")
def chat(dados: ChatRequest):
    payload = {
        "model": MODELO,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": dados.pergunta
            }
        ],
        "options": {
            "temperature": dados.temperatura,
            "top_p": 0.95,
            "num_predict": 250
        }
    }

    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    inicio = perf_counter()

    try:
        with urllib.request.urlopen(
            requisicao,
            timeout=180
        ) as resposta:
            resultado = json.loads(
                resposta.read().decode("utf-8")
            )

    except (
        urllib.error.URLError,
        TimeoutError
    ) as erro:
        raise HTTPException(
            status_code=502,
            detail=f"Falha no Ollama: {erro}"
        )

    duracao = perf_counter() - inicio

    texto = resultado.get(
        "message", {}
    ).get("content", "").strip()

    return {
        "modelo": MODELO,
        "resposta": texto,
        "tempo_segundos": round(duracao, 3)
    }