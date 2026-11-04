from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json
import re
import urllib.request

BASE = Path(__file__).resolve().parent
MODELO = "llama3.2:latest"
URL = "http://127.0.0.1:11434/api/generate"

DOCUMENTO = [
    {
        "pagina": 1,
        "texto": (
            "O Brasil esta localizado na America do Sul. "
            "Seu territorio apresenta extensas areas "
            "de planaltos, planicies e depressoes."
        )
    },
    {
        "pagina": 2,
        "texto": (
            "O clima brasileiro apresenta diversidade. "
            "Ha predominio de climas tropicais em grande "
            "parte do territorio, enquanto a Regiao Sul "
            "possui areas de clima subtropical."
        )
    },
    {
        "pagina": 3,
        "texto": (
            "A Amazonia abriga uma extensa floresta "
            "tropical. O Cerrado possui vegetacao "
            "adaptada a periodos de seca. A Caatinga "
            "ocorre em areas semiaridas."
        )
    }
]

INSTRUCOES = (
    "Voce e um assistente de perguntas e respostas. "
    "Responda exclusivamente com informacoes do "
    "documento fornecido. "
    "Nao utilize conhecimentos externos. "
    "Se a pergunta nao puder ser respondida com "
    "o documento, diga: "
    "'Nao estou autorizado a responder esse assunto. "
    "Posso responder perguntas sobre a geografia "
    "do Brasil presentes no documento.' "
    "Ao responder, indique a pagina da fonte "
    "no formato [Pagina N]. "
    "Seja objetivo e nao invente informacoes."
)

TESTES = [
    {
        "nome": "relevo",
        "pergunta": "Como e o relevo do Brasil?",
        "paginas_esperadas": [1],
        "temperatura": 0.1
    },
    {
        "nome": "clima",
        "pergunta": "Como e o clima do Brasil?",
        "paginas_esperadas": [2],
        "temperatura": 0.1
    },
    {
        "nome": "fora_escopo",
        "pergunta": "Quero uma receita de bolo.",
        "paginas_esperadas": [],
        "temperatura": 0.1
    },
    {
        "nome": "clima_temperatura_alta",
        "pergunta": "Como e o clima do Brasil?",
        "paginas_esperadas": [2],
        "temperatura": 0.9
    }
]


def gerar(pergunta, temperatura):
    contexto = "\n\n".join(
        f"[Pagina {item['pagina']}]\n{item['texto']}"
        for item in DOCUMENTO
    )

    prompt = (
        f"{INSTRUCOES}\n\n"
        f"DOCUMENTO:\n{contexto}\n\n"
        f"PERGUNTA:\n{pergunta}\n\n"
        "RESPOSTA:"
    )

    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperatura,
            "top_p": 0.9,
            "num_predict": 200
        }
    }

    requisicao = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
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

    tempo = perf_counter() - inicio

    return dados.get("response", "").strip(), tempo


def main():
    print("=== AULA 14 - VERTEX AI STUDIO LOCAL ===")
    print("Modelo:", MODELO)

    resultados = []

    for teste in TESTES:
        resposta, tempo = gerar(
            teste["pergunta"],
            teste["temperatura"]
        )

        referencias = sorted({
            int(numero)
            for numero in re.findall(
                r"\[P[aá]gina\s+(\d+)\]",
                resposta,
                flags=re.IGNORECASE
            )
        })

        referencias_validas = all(
            pagina in [1, 2, 3]
            for pagina in referencias
        )

        citacao_esperada = (
            set(teste["paginas_esperadas"]).issubset(
                set(referencias)
            )
            if teste["paginas_esperadas"]
            else not referencias
        )

        recusa_detectada = (
            "nao estou autorizado" in
            resposta.lower()
        )

        resultado = {
            "teste": teste["nome"],
            "pergunta": teste["pergunta"],
            "temperatura": teste["temperatura"],
            "resposta": resposta,
            "tempo_segundos": round(tempo, 3),
            "paginas_citadas": referencias,
            "referencias_validas": referencias_validas,
            "citacao_esperada": citacao_esperada,
            "recusa_detectada": recusa_detectada
        }

        resultados.append(resultado)

        print("\nTeste:", teste["nome"])
        print("Temperatura:", teste["temperatura"])
        print("Resposta:", resposta)
        print("Paginas citadas:", referencias)
        print("Citacao esperada:", citacao_esperada)
        print("Recusa detectada:", recusa_detectada)
        print(f"Tempo: {tempo:.2f} segundos")

    relatorio = {
        "modelo": MODELO,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "quantidade_testes": len(resultados),
        "resultados": resultados
    }

    destino = BASE / "resultados_vertex_studio.json"

    destino.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n=== RESUMO ===")
    print("Testes executados:", len(resultados))
    print("Relatorio:", destino)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    main()