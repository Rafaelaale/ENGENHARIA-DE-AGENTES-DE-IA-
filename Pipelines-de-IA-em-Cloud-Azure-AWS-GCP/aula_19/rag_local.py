import json
import urllib.request
from pathlib import Path
from time import perf_counter

import numpy as np

BASE = Path(__file__).resolve().parent
ARQUIVO_INDICE = BASE / "indice_vetorial.json"
ARQUIVO_RESULTADOS = BASE / "resultados_rag.json"

OLLAMA = "http://127.0.0.1:11434"
MODELO_EMBEDDING = "nomic-embed-text:latest"
MODELO_GERADOR = "llama3.2:latest"

DOCUMENTOS = [
    {
        "id": "POL-001",
        "titulo": "Politica de Reembolso",
        "texto": (
            "Clientes podem solicitar reembolso em ate 7 dias "
            "corridos apos a compra. A solicitacao deve ser "
            "registrada no portal de atendimento. "
            "O prazo de processamento e de ate 5 dias uteis "
            "apos a aprovacao."
        )
    },
    {
        "id": "SUP-001",
        "titulo": "Suporte Tecnico",
        "texto": (
            "O suporte tecnico funciona de segunda a sexta-feira, "
            "das 8h as 18h, horario de Brasilia. "
            "Os chamados devem ser abertos pelo portal. "
            "Chamados criticos recebem prioridade de atendimento."
        )
    },
    {
        "id": "ENT-001",
        "titulo": "Politica de Entrega",
        "texto": (
            "As entregas sao realizadas em ate 10 dias uteis "
            "apos a confirmacao do pagamento. "
            "O cliente pode acompanhar o pedido pelo portal. "
            "Em caso de atraso, deve abrir uma solicitacao "
            "na central de atendimento."
        )
    },
    {
        "id": "SEG-001",
        "titulo": "Seguranca da Informacao",
        "texto": (
            "Os colaboradores devem utilizar autenticacao "
            "multifator para acessar sistemas internos. "
            "Senhas nao devem ser compartilhadas. "
            "Incidentes de seguranca devem ser comunicados "
            "imediatamente a equipe responsavel."
        )
    }
]


def chamar_api(endpoint, payload):
    requisicao = urllib.request.Request(
        OLLAMA + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        requisicao,
        timeout=180
    ) as resposta:
        return json.loads(
            resposta.read().decode("utf-8")
        )


def embedding(texto):
    resposta = chamar_api(
        "/api/embed",
        {
            "model": MODELO_EMBEDDING,
            "input": texto
        }
    )

    vetores = resposta.get("embeddings", [])

    if not vetores or not vetores[0]:
        raise ValueError("Embedding nao retornado.")

    return np.asarray(vetores[0], dtype=np.float64)


def similaridade_cosseno(a, b):
    denominador = np.linalg.norm(a) * np.linalg.norm(b)

    if denominador == 0:
        return 0.0

    return float(np.dot(a, b) / denominador)


def criar_indice():
    indice = []

    for documento in DOCUMENTOS:
        vetor = embedding(
            documento["titulo"] + "\n" + documento["texto"]
        )

        indice.append({
            **documento,
            "embedding": vetor.tolist()
        })

        print(
            "Indexado:",
            documento["id"],
            "| dimensoes:",
            len(vetor)
        )

    ARQUIVO_INDICE.write_text(
        json.dumps(indice, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return indice


def recuperar(pergunta, indice, top_k=2):
    vetor_pergunta = embedding(pergunta)

    candidatos = []

    for documento in indice:
        vetor_documento = np.asarray(
            documento["embedding"],
            dtype=np.float64
        )

        score = similaridade_cosseno(
            vetor_pergunta,
            vetor_documento
        )

        candidatos.append({
            "id": documento["id"],
            "titulo": documento["titulo"],
            "texto": documento["texto"],
            "similaridade": round(score, 6)
        })

    candidatos.sort(
        key=lambda item: item["similaridade"],
        reverse=True
    )

    return candidatos[:top_k]


def gerar_resposta(pergunta, trechos):
    contexto = "\n\n".join(
        f"Fonte [{item['id']}]: {item['titulo']}\n"
        f"{item['texto']}"
        for item in trechos
    )

    mensagens = [
        {
            "role": "system",
            "content": (
                "Voce e um assistente de uma empresa ficticia. "
                "Responda apenas com base nas fontes fornecidas. "
                "Se a resposta nao estiver nas fontes, diga: "
                "'Nao encontrei essa informacao nos documentos.' "
                "Nao invente politicas, prazos ou procedimentos. "
                "Inclua o ID da fonte utilizada."
            )
        },
        {
            "role": "user",
            "content": (
                f"Contexto:\n{contexto}\n\n"
                f"Pergunta: {pergunta}"
            )
        }
    ]

    resposta = chamar_api(
        "/api/chat",
        {
            "model": MODELO_GERADOR,
            "messages": mensagens,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 250
            }
        }
    )

    return resposta["message"]["content"]


def main():
    inicio = perf_counter()

    print("=== AULA 19 - RAG LOCAL ===")
    print("\n=== INDEXACAO ===")

    indice = criar_indice()

    perguntas = [
        "Qual e o prazo para solicitar reembolso?",
        "Qual e o horario de funcionamento do suporte tecnico?",
        "Como devo comunicar um incidente de seguranca?",
        "Qual e o nome do diretor financeiro da empresa?"
    ]

    resultados = []

    for pergunta in perguntas:
        print("\nPergunta:", pergunta)

        trechos = recuperar(
            pergunta,
            indice,
            top_k=2
        )

        print("Documentos recuperados:")

        for trecho in trechos:
            print(
                f"  {trecho['id']} "
                f"| similaridade={trecho['similaridade']}"
            )

        resposta = gerar_resposta(
            pergunta,
            trechos
        )

        print("Resposta:", resposta)

        resultados.append({
            "pergunta": pergunta,
            "documentos_recuperados": trechos,
            "resposta": resposta
        })

    relatorio = {
        "arquitetura": "RAG",
        "modelo_embedding": MODELO_EMBEDDING,
        "modelo_gerador": MODELO_GERADOR,
        "documentos_indexados": len(indice),
        "top_k": 2,
        "resultados": resultados,
        "tempo_segundos": round(
            perf_counter() - inicio,
            3
        )
    }

    ARQUIVO_RESULTADOS.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== VALIDACAO ===")
    print("Documentos indexados:", len(indice))
    print("Perguntas executadas:", len(resultados))
    print("Modelo de embeddings:", MODELO_EMBEDDING)
    print("Modelo gerador:", MODELO_GERADOR)
    print("Tempo total:", relatorio["tempo_segundos"])
    print("Indice:", ARQUIVO_INDICE)
    print("Relatorio:", ARQUIVO_RESULTADOS)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    main()