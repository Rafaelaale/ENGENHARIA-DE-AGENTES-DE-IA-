import json
import urllib.request
from pathlib import Path
from time import perf_counter

import numpy as np

BASE = Path(__file__).resolve().parent
PROJETO = BASE.parent

CORPUS = PROJETO / "aula_21" / "corpus_completo.json"
RELATORIO = BASE / "resultado_retrieval_generation.json"

OLLAMA = "http://127.0.0.1:11434"
MODELO_EMBEDDING = "nomic-embed-text:latest"
MODELO_GERADOR = "llama3.2:latest"

PERGUNTAS = [
    "O que e inteligencia artificial?",
    "O que e o LLM SmartDesk?",
    "Quais funcionalidades possui o LLM SmartDesk?",
    "Quem criou a inteligencia artificial?",
    "Quero uma receita de bolo."
]

VALORES_K = [1, 3, 10]


def chamar_api(endpoint, payload):
    req = urllib.request.Request(
        OLLAMA + endpoint,
        data=json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gerar_embedding(pergunta):
    resposta = chamar_api(
        "/api/embed",
        {
            "model": MODELO_EMBEDDING,
            "input": f"query: {pergunta}"
        }
    )

    vetor = np.asarray(
        resposta["embeddings"][0],
        dtype=np.float64
    )

    if vetor.size == 0 or not np.isfinite(vetor).all():
        raise ValueError("Embedding da pergunta invalido.")

    return vetor


def recuperar(pergunta, indice, top_k):
    vetor_pergunta = gerar_embedding(pergunta)

    resultados = []

    for item in indice:
        vetor_documento = np.asarray(
            item["embedding"],
            dtype=np.float64
        )

        if vetor_pergunta.shape != vetor_documento.shape:
            raise ValueError("Dimensoes dos embeddings diferentes.")

        denominador = (
            np.linalg.norm(vetor_pergunta)
            * np.linalg.norm(vetor_documento)
        )

        score = (
            float(
                np.dot(vetor_pergunta, vetor_documento)
                / denominador
            )
            if denominador > 0
            else 0.0
        )

        resultados.append({
            "documento": item["documento"],
            "chunk": item["chunk"],
            "score": round(score, 6),
            "texto": item["texto"]
        })

    resultados.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return resultados[:top_k]


def gerar_resposta(pergunta, trechos):
    contexto = "\n\n".join(
        f"FONTE: {item['documento']} "
        f"| TRECHO: {item['chunk']}\n"
        f"{item['texto']}"
        for item in trechos
    )

    resposta = chamar_api(
        "/api/chat",
        {
            "model": MODELO_GERADOR,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Voce e um assistente de perguntas e respostas. "
                        "Responda somente com informacoes presentes "
                        "nos trechos fornecidos. "
                        "Nao utilize conhecimento externo. "
                        "Nao invente fatos. "
                        "Se os trechos nao contiverem a resposta, "
                        "informe claramente que a informacao "
                        "nao foi encontrada nos documentos. "
                        "Quando houver informacao suficiente, "
                        "responda de forma completa e cite "
                        "o nome do documento."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"PERGUNTA:\n{pergunta}\n\n"
                        f"TRECHOS RECUPERADOS:\n{contexto}"
                    )
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 350
            }
        }
    )

    return resposta["message"]["content"]


def avaliar_funcionalidades(pergunta, trechos, resposta):
    if "funcionalidades" not in pergunta.lower():
        return None

    contexto = " ".join(
        item["texto"].lower()
        for item in trechos
    )

    texto_resposta = resposta.lower()

    verificacoes = {
        "ajuste_eletrico_no_contexto": (
            "ajuste eletrico" in contexto
        ),
        "memoria_no_contexto": (
            "duas alturas" in contexto
        ),
        "botoes_no_contexto": (
            "botoes" in contexto
        ),
        "ajuste_mencionado_na_resposta": (
            "ajuste" in texto_resposta
        ),
        "memoria_mencionada_na_resposta": (
            "memoria" in texto_resposta
            or "duas alturas" in texto_resposta
            or "posicoes" in texto_resposta
        ),
        "botoes_mencionados_na_resposta": (
            "botoes" in texto_resposta
        )
    }

    return verificacoes


def main():
    inicio = perf_counter()

    print("=== AULA 22 - RECUPERACAO E GERACAO ===")

    indice = json.loads(
        CORPUS.read_text(encoding="utf-8")
    )

    print("\n=== CORPUS ===")
    print("Chunks:", len(indice))
    print("Modelo embedding:", MODELO_EMBEDDING)
    print("Modelo gerador:", MODELO_GERADOR)

    resultados = []

    for pergunta in PERGUNTAS:
        print("\n" + "=" * 60)
        print("PERGUNTA:", pergunta)

        for k in VALORES_K:
            print(f"\n--- RETRIEVAL TOP-K={k} ---")

            trechos = recuperar(pergunta, indice, k)

            for item in trechos:
                print(
                    f"{item['documento']} "
                    f"| chunk={item['chunk']} "
                    f"| score={item['score']}"
                )
                print("Trecho:", item["texto"][:200])

            print("\n--- GENERATION ---")

            resposta = gerar_resposta(pergunta, trechos)

            print("Resposta:", resposta)

            avaliacao = avaliar_funcionalidades(
                pergunta,
                trechos,
                resposta
            )

            if avaliacao is not None:
                print("\n--- ANALISE DE COMPLETUDE ---")

                for chave, valor in avaliacao.items():
                    print(f"{chave}: {valor}")

            resultados.append({
                "pergunta": pergunta,
                "top_k": k,
                "recuperacao": trechos,
                "resposta": resposta,
                "avaliacao_funcionalidades": avaliacao
            })

    relatorio = {
        "aula": 22,
        "tipo": "Retrieval e Generation",
        "corpus": str(CORPUS),
        "chunks": len(indice),
        "perguntas": len(PERGUNTAS),
        "valores_top_k": VALORES_K,
        "total_testes": len(resultados),
        "resultados": resultados,
        "tempo_segundos": round(
            perf_counter() - inicio,
            3
        ),
        "cloud_deployed": False
    }

    RELATORIO.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== VALIDACAO FINAL ===")
    print("Perguntas:", len(PERGUNTAS))
    print("Top-K testados:", VALORES_K)
    print("Total de testes:", len(resultados))
    print("Tempo:", relatorio["tempo_segundos"])
    print("Relatorio:", RELATORIO)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    main()