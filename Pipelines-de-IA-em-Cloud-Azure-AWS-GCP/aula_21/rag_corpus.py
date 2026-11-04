import json
import re
import urllib.request
from pathlib import Path
from time import perf_counter

import numpy as np

BASE = Path(__file__).resolve().parent
DOCS = BASE / "documentos"
DOCS.mkdir(exist_ok=True)

OLLAMA = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
GENERATION_MODEL = "llama3.2:latest"

DOCUMENTOS = {
    "smartdesk": {
        "arquivo": "smartdesk_faq.txt",
        "conteudo": """LLM SmartDesk - Perguntas Frequentes

O que e o LLM SmartDesk?
O LLM SmartDesk e uma mesa de escritorio com ajuste eletrico de altura.

Como ajustar a altura?
O usuario pode utilizar os botoes de controle para subir ou descer a mesa.

O produto possui memoria de posicoes?
Sim. O painel permite salvar duas alturas preferidas.

Como solicitar suporte?
O suporte do LLM SmartDesk e solicitado pelo portal de atendimento.

O produto possui aplicativo para celular?
A documentacao disponivel nao informa suporte a aplicativo para celular.
"""
    },
    "ia": {
        "arquivo": "inteligencia_artificial.txt",
        "conteudo": """Introducao a Inteligencia Artificial

Inteligencia Artificial e uma area da computacao que desenvolve
sistemas capazes de executar tarefas associadas a capacidades
como aprendizado, reconhecimento de padroes e tomada de decisao.

Aplicacoes incluem sistemas de recomendacao, assistentes virtuais,
classificacao de imagens e processamento de linguagem natural.

Machine Learning e uma abordagem na qual modelos aprendem
padroes a partir de dados.

Modelos de linguagem podem gerar texto, responder perguntas
e resumir documentos, mas podem produzir respostas incorretas.
"""
    }
}


def api(endpoint, payload):
    req = urllib.request.Request(
        OLLAMA + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embedding(texto, prefixo="passage"):
    resposta = api(
        "/api/embed",
        {
            "model": EMBEDDING_MODEL,
            "input": f"{prefixo}: {texto}"
        }
    )

    vetor = np.asarray(
        resposta["embeddings"][0],
        dtype=np.float64
    )

    if vetor.size == 0 or not np.isfinite(vetor).all():
        raise ValueError("Embedding invalido.")

    return vetor


def dividir_documento(texto):
    blocos = re.split(r"\n\s*\n", texto.strip())

    return [
        bloco.strip()
        for bloco in blocos
        if bloco.strip()
    ]


def criar_corpus(nomes, arquivo_indice):
    indice = []

    for nome in nomes:
        doc = DOCUMENTOS[nome]

        caminho = DOCS / doc["arquivo"]

        caminho.write_text(
            doc["conteudo"],
            encoding="utf-8"
        )

        texto = caminho.read_text(encoding="utf-8")

        for numero, trecho in enumerate(
            dividir_documento(texto),
            start=1
        ):
            vetor = embedding(trecho)

            indice.append({
                "documento": doc["arquivo"],
                "chunk": numero,
                "texto": trecho,
                "embedding": vetor.tolist()
            })

    arquivo_indice.write_text(
        json.dumps(
            indice,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return indice


def recuperar(pergunta, indice, top_k):
    pergunta_vetor = embedding(
        pergunta,
        prefixo="query"
    )

    resultados = []

    for item in indice:
        vetor = np.asarray(
            item["embedding"],
            dtype=np.float64
        )

        denominador = (
            np.linalg.norm(pergunta_vetor)
            * np.linalg.norm(vetor)
        )

        score = (
            float(np.dot(pergunta_vetor, vetor) / denominador)
            if denominador > 0
            else 0.0
        )

        resultados.append({
            "documento": item["documento"],
            "chunk": item["chunk"],
            "texto": item["texto"],
            "score": round(score, 6)
        })

    resultados.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return resultados[:top_k]


def responder(pergunta, trechos):
    contexto = "\n\n".join(
        f"[{t['documento']} - trecho {t['chunk']}]\n"
        f"{t['texto']}"
        for t in trechos
    )

    resposta = api(
        "/api/chat",
        {
            "model": GENERATION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Responda em portugues brasileiro. "
                        "Utilize apenas o contexto recuperado. "
                        "Nao invente informacoes. "
                        "Se a resposta nao estiver no contexto, "
                        "diga que nao encontrou a informacao. "
                        "Identifique o documento utilizado."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXTO:\n{contexto}\n\n"
                        f"PERGUNTA:\n{pergunta}"
                    )
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 250
            }
        }
    )

    return resposta["message"]["content"]


def testar(pergunta, indice, top_k):
    trechos = recuperar(pergunta, indice, top_k)
    resposta = responder(pergunta, trechos)

    print(f"\nPergunta: {pergunta}")
    print(f"Top-K: {top_k}")

    for trecho in trechos:
        print(
            f"  {trecho['documento']} "
            f"| chunk={trecho['chunk']} "
            f"| score={trecho['score']}"
        )

    print("Resposta:", resposta)

    return {
        "pergunta": pergunta,
        "top_k": top_k,
        "recuperados": trechos,
        "resposta": resposta
    }


def main():
    inicio = perf_counter()

    print("=== AULA 21 - RAG CORPUS LOCAL ===")

    print("\n=== FASE 1: SMARTDESK ===")

    indice_1 = criar_corpus(
        ["smartdesk"],
        BASE / "corpus_smartdesk.json"
    )

    resultados_1 = [
        testar(
            "O que e o LLM SmartDesk?",
            indice_1,
            2
        ),
        testar(
            "Como ajustar a altura da mesa?",
            indice_1,
            2
        )
    ]

    print("\n=== FASE 2: SMARTDESK + IA ===")

    indice_2 = criar_corpus(
        ["smartdesk", "ia"],
        BASE / "corpus_completo.json"
    )

    resultados_2 = [
        testar(
            "O que e inteligencia artificial?",
            indice_2,
            2
        ),
        testar(
            "O LLM SmartDesk possui memoria de posicoes?",
            indice_2,
            2
        ),
        testar(
            "Qual e a receita de bolo de chocolate?",
            indice_2,
            2
        )
    ]

    print("\n=== COMPARACAO TOP-K ===")

    pergunta = "Quais funcionalidades possui o LLM SmartDesk?"

    comparacoes = []

    for k in (1, 2, 4):
        comparacoes.append(
            testar(pergunta, indice_2, k)
        )

    relatorio = {
        "aula": 21,
        "arquitetura": "RAG Corpus local",
        "modelo_embedding": EMBEDDING_MODEL,
        "modelo_gerador": GENERATION_MODEL,
        "fase_1": {
            "chunks": len(indice_1),
            "resultados": resultados_1
        },
        "fase_2": {
            "chunks": len(indice_2),
            "resultados": resultados_2
        },
        "comparacao_top_k": comparacoes,
        "tempo_segundos": round(
            perf_counter() - inicio,
            3
        ),
        "cloud_deployed": False
    }

    caminho = BASE / "resultado_rag_corpus.json"

    caminho.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== VALIDACAO ===")
    print("Chunks SmartDesk:", len(indice_1))
    print("Chunks corpus completo:", len(indice_2))
    print("Perguntas totais:", len(resultados_1) + len(resultados_2))
    print("Comparacoes Top-K:", len(comparacoes))
    print("Tempo total:", relatorio["tempo_segundos"])
    print("Relatorio:", caminho)
    print("Nenhum recurso Google Cloud foi criado.")


if __name__ == "__main__":
    main()