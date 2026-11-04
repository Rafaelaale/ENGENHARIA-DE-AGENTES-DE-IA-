from pathlib import Path
import json
import urllib.request

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8015"

TESTES = [
    {
        "pergunta": (
            "Explique em 3 passos como configurar "
            "autenticacao para usar LLMs no Vertex AI."
        ),
        "temperatura": 0.3
    },
    {
        "pergunta": (
            "Crie um checklist curto para investigar "
            "erro HTTP 404 em uma API de LLM."
        ),
        "temperatura": 0.3
    },
    {
        "pergunta": (
            "Como voce pode me ajudar com integracoes "
            "de APIs em Python?"
        ),
        "temperatura": 0.3
    }
]


def requisitar(caminho, payload=None):
    dados = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )

    req = urllib.request.Request(
        URL + caminho,
        data=dados,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET"
    )

    with urllib.request.urlopen(req, timeout=200) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=== AULA 15 - CLIENTE HTTP ===")

    status = requisitar("/status")

    print("Status:", status)

    resultados = []

    for indice, teste in enumerate(TESTES, 1):
        resposta = requisitar("/chat", teste)

        print(f"\n=== TESTE {indice} ===")
        print("Pergunta:", teste["pergunta"])
        print("Resposta:", resposta["resposta"])
        print(
            "Tempo:",
            resposta["tempo_segundos"],
            "segundos"
        )

        assert resposta["modelo"] == "llama3.2:latest"
        assert resposta["resposta"].strip()

        resultados.append({
            "pergunta": teste["pergunta"],
            **resposta
        })

    relatorio = {
        "status_api": status,
        "testes_executados": len(resultados),
        "resultados": resultados
    }

    destino = BASE / "resultado_integracao_api.json"

    destino.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n=== VALIDACAO ===")
    print("Testes executados:", len(resultados))
    print("Relatorio:", destino)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    main()