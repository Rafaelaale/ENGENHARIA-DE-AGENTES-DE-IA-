from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

modelo = ChatOpenAI(
    model="llama3.2:1b",
    temperature=0.5,
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

caminho = Path("documentos/GTB_gold_Nov23.txt")
try:
    texto = caminho.read_text(encoding="utf-8")
except UnicodeDecodeError:
    texto = caminho.read_text(encoding="cp1252")

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
).split_text(texto)

print(f"Pedaços: {len(pedacos)}", flush=True)

store = None
lote = 4
for inicio in range(0, len(pedacos), lote):
    fim = min(inicio + lote, len(pedacos))
    print(f"Embeddings {inicio + 1}-{fim}", flush=True)
    documentos = [Document(page_content=p) for p in pedacos[inicio:fim]]
    if store is None:
        store = FAISS.from_documents(documentos, embeddings)
    else:
        store.add_documents(documentos)

dados_recuperados = store.as_retriever(search_kwargs={"k": 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages([
    ("system", "Responda em portugues usando exclusivamente o conteudo fornecido."),
    ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:"),
])

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({"query": pergunta, "contexto": contexto})

print(responder("Como devo proceder caso tenha um item roubado?"), flush=True)
