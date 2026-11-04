from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json
import urllib.request
import urllib.error

BASE = Path(__file__).resolve().parent
MODELO = "llama3.2:latest"
ENDPOINT = "http://127.0.0.1:11434/api/generate"

TAREFAS = [
    {
        "nome": "classificacao",
        "prompt": (
            "Classifique a mensagem como RECLAMACAO, "
            "ELOGIO ou SOLICITACAO. Responda apenas "
            "com a categoria.\n\n"
            "Mensagem: Meu pedido chegou atrasado "
            "e nao consegui contato com o suporte."
        )
    },
    {
        "nome": "resumo",
        "prompt": (
            "Resuma o texto em no maximo duas frases, "
            "sem inventar informacoes.\n\n"
            "Texto: Uma empresa recebe diariamente "
            "centenas de mensagens de clientes. "
            "Atualmente, a equipe classifica cada "
            "mensagem manualmente e encaminha os "
            "casos para os setores responsaveis. "
            "O processo consome tempo e pode gerar "
            "atrasos no atendimento."
        )
    },
    {
        "nome": "extracao_json",
        "prompt": (
            "Extraia as informacoes abaixo e responda "
            "somente com um objeto JSON valido, com "
            "as chaves cliente, produto e quantidade.\n\n"
            "Texto: Maria comprou 3 notebooks."
        )
    }
]


def gerar(prompt):
    payload = json.dumps({
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 180
        }
    }).encode("utf-8")

    requisicao = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    inicio = perf_counter()

    with urllib.request.urlopen(
        requisicao,
        timeout=180
    ) as resposta:
        dados = json.loads(
            resposta.read().decode("utf-8")
        )

    duracao = perf_counter() - inicio

    return dados.get("response", "").strip(), duracao


def main():
    print("=== AULA 13 - FOUNDATION MODELS ===")
    print("Modelo:", MODELO)

    resultados = []

    for tarefa in TAREFAS:
        print("\nTarefa:", tarefa["nome"])

        try:
            resposta, duracao = gerar(tarefa["prompt"])
        except (urllib.error.URLError, TimeoutError) as erro:
            raise RuntimeError(
                "Falha na comunicacao com Ollama. "
                "Verifique se o servico esta ativo."
            ) from erro

        print("Resposta:", resposta)
        print(f"Tempo: {duracao:.2f} segundos")

        resultado = {
            "tarefa": tarefa["nome"],
            "resposta": resposta,
            "duracao_segundos": round(duracao, 3)
        }

        if tarefa["nome"] == "extracao_json":
            try:
                resultado["json_valido"] = isinstance(
                    json.loads(resposta), dict
                )
            except json.JSONDecodeError:
                resultado["json_valido"] = False

            print(
                "JSON valido:",
                resultado["json_valido"]
            )

        resultados.append(resultado)

    relatorio = {
        "modelo": MODELO,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tarefas_executadas": len(resultados),
        "resultados": resultados
    }

    destino = BASE / "resultados_llm.json"

    destino.write_text(
        json.dumps(
            relatorio,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== RELATORIO ===")
    print("Tarefas executadas:", len(resultados))
    print("Arquivo:", destino)
    print("Nenhum treinamento foi realizado.")


if __name__ == "__main__":
    main()