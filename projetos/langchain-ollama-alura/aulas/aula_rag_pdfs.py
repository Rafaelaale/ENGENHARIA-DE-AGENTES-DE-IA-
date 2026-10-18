from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf",
]

documentos = sum(
    [PyPDFLoader(arquivo).load() for arquivo in arquivos],
    [],
)
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
).split_documents(documentos)
print(f"Paginas: {len(documentos)} | Pedaços: {len(pedacos)}", flush=True)

store = None
lote = 8
for inicio in range(0, len(pedacos), lote):
    fim = min(inicio + lote, len(pedacos))
    print(f"Embeddings {inicio + 1}-{fim}", flush=True)
    bloco = pedacos[inicio:fim]
    if store is None:
        store = FAISS.from_documents(bloco, embeddings)
    else:
        store.add_documents(bloco)

dados_recuperados = store.as_retriever(search_kwargs={"k": 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages([
    ("system", "Responda em portugues usando exclusivamente o conteudo fornecido."),
    ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:"),
])

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    print("Arquivos usados:", flush=True)
    for um_trecho in trechos:
        print("-", um_trecho.metadata.get("source", "sem origem"), flush=True)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({"query": pergunta, "contexto": contexto})

perguntas = [
    "Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold",
    "Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão platinum",
]

for pergunta in perguntas:
    print("\nVocê:", pergunta, flush=True)
    print(responder(pergunta), flush=True)
