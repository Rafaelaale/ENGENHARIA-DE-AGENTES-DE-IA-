from pathlib import Path
import subprocess

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS


PASTA_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_REGRAS = PASTA_PROJETO / "documentos" / "regras_futebol.txt"
PASTA_INDICE = PASTA_PROJETO / "indice" / "regras_futebol"

MODELO_EMBEDDING = "nomic-embed-text:latest"
MODELO_LLM = "gemma3:1b"


def parar(modelo):
    subprocess.run(["ollama", "stop", modelo], check=False)


def obter_indice():
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    if (PASTA_INDICE / "index.faiss").exists():
        print("Carregando indice salvo...")
        return FAISS.load_local(
            str(PASTA_INDICE),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Criando indice...")
    documentos = TextLoader(str(ARQUIVO_REGRAS), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    ).split_documents(documentos)
    indice = FAISS.from_documents(chunks, embeddings)
    PASTA_INDICE.parent.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(PASTA_INDICE))
    return indice


def responder(indice, pergunta, modelo):
    documentos = indice.similarity_search(pergunta, k=3)
    contexto = "\n\n".join(doc.page_content[:400] for doc in documentos)
    parar(MODELO_EMBEDDING)

    prompt = (
        "Responda em portugues, em uma frase, usando somente o contexto. "
        "Se nao houver resposta, diga que nao encontrou.\n\n"
        f"CONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}\n\nRESPOSTA:"
    )

    print("\n" + "=" * 70)
    print("PERGUNTA:", pergunta)
    print("=" * 70)
    print(modelo.invoke(prompt))
    parar(MODELO_LLM)


def main():
    indice = obter_indice()
    modelo = OllamaLLM(
        model=MODELO_LLM,
        num_ctx=1024,
        num_gpu=0,
        num_predict=80,
    )
    for pergunta in [
        "Um jogador pode usar a mao para marcar um gol?",
        "O que caracteriza impedimento no futebol?",
    ]:
        responder(indice, pergunta, modelo)


if __name__ == "__main__":
    main()
