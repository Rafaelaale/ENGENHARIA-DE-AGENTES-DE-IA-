"""Chat local de regras de futebol. Resumo didatico, nao e regulamento oficial."""
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).resolve().parents[1]
DOC = BASE / "documentos" / "regras_futebol_didatico.txt"
CHROMA = BASE / "chroma_futebol_local"

TEXTO = """
Resumo didatico para estudo. Nao substitui o regulamento oficial.
Uma partida tem duas equipes. Cada equipe coloca 11 jogadores em campo, incluindo o goleiro.
A partida tem dois tempos de 45 minutos. O arbitro pode acrescentar acrescimos.
O gol vale quando a bola inteira cruza a linha de gol.
Impedimento: no momento do passe, o atacante esta mais perto da linha de gol adversaria do que a bola e do que o penultimo adversario, e participa da jogada.
Nao ha impedimento em tiro de meta, arremesso lateral ou tiro de canto.
Cartao amarelo e advertencia. Dois amarelos na mesma partida geram vermelho. O vermelho expulsa o jogador.
Penalti e cobrado da marca penal apos falta dentro da area.
O goleiro so pode usar as maos dentro da propria area.
"""

def main():
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(TEXTO.strip(), encoding="utf-8")
    docs = TextLoader(str(DOC), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
    base = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
        persist_directory=str(CHROMA),
    )
    cadeia = (
        ChatPromptTemplate.from_template(
            "Use apenas o contexto. Se nao estiver no contexto, diga que nao encontrou.\nContexto:\n{context}\n\nPergunta: {question}"
        )
        | ChatOllama(model="gemma3:1b", temperature=0)
        | StrOutputParser()
    )
    for pergunta in [
        "Quantos jogadores cada equipe coloca em campo?",
        "Quando existe impedimento?",
        "O que acontece com dois cartoes amarelos?",
    ]:
        contexto = "\n".join(d.page_content for d in base.similarity_search(pergunta, k=3))
        print("\nPergunta:", pergunta)
        print(cadeia.invoke({"context": contexto, "question": pergunta}))

if __name__ == "__main__":
    main()
