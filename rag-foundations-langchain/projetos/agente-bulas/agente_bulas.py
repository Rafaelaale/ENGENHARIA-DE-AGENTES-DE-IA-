import json
import random
import shutil
import urllib.request
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

raiz = Path(__file__).resolve().parents[1]
caminhos_bulas = [raiz / "dipirona.pdf", raiz / "paracetamol.pdf"]
faltando = [p.name for p in caminhos_bulas if not p.exists()]
if faltando:
    raise SystemExit("Faltam os PDFs na pasta do projeto: " + ", ".join(faltando))

def modelo_chat():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as resp:
            modelos = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except Exception:
        raise SystemExit("Ollama não está respondendo em http://127.0.0.1:11434.")
    for prefixo in ("gemma3", "llama3.2", "llama3"):
        for nome in modelos:
            if nome.startswith(prefixo) and "embed" not in nome:
                return nome
    raise SystemExit("Nenhum modelo de chat encontrado no Ollama.")

documentos = []
for caminho in caminhos_bulas:
    docs = PyPDFLoader(str(caminho)).load()
    for doc in docs:
        doc.metadata["medicamento"] = caminho.stem
    documentos.extend(docs)
print(f"Páginas carregadas: {len(documentos)}")

chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150).split_documents(documentos)
for chunk in chunks:
    texto = chunk.page_content.lower()
    if "identificação do medicamento" in texto or "composição" in texto:
        chunk.metadata["categoria"] = "identificacao"
    elif "indicação" in texto or "para que este medicamento é indicado" in texto:
        chunk.metadata["categoria"] = "indicacao"
    elif "como este medicamento funciona" in texto or "ação" in texto:
        chunk.metadata["categoria"] = "como_funciona"
    elif "contraindicação" in texto or "quando não devo usar" in texto:
        chunk.metadata["categoria"] = "contraindicacao"
    elif "advertência" in texto or "precaução" in texto or "o que devo saber antes de usar" in texto:
        chunk.metadata["categoria"] = "advertencias_precaucoes"
    elif "interação" in texto or "interações medicamentosas" in texto:
        chunk.metadata["categoria"] = "interacoes"
    elif "dose" in texto or "posologia" in texto or "como devo usar" in texto:
        chunk.metadata["categoria"] = "posologia_modo_uso"
    elif "reações adversas" in texto or "quais os males" in texto:
        chunk.metadata["categoria"] = "reacoes_adversas"
    elif "onde, como e por quanto tempo posso guardar" in texto or "armazenar" in texto:
        chunk.metadata["categoria"] = "armazenamento"
    elif "quantidade maior do que a indicada" in texto or "superdosagem" in texto:
        chunk.metadata["categoria"] = "superdosagem"
    else:
        chunk.metadata["categoria"] = "geral"
print(f"Chunks gerados: {len(chunks)}")

for i, chunk in enumerate(random.sample(chunks, min(2, len(chunks))), start=1):
    print(f"\n--- Chunk Aleatório {i} ---")
    print(f"Metadados: {chunk.metadata}")
    print("\nConteúdo (início):")
    print(chunk.page_content[:300])

destino = raiz / "chroma_bulas_local"
if destino.exists():
    shutil.rmtree(destino)
print("Gerando embeddings locais...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
    persist_directory=str(destino)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
nome_modelo = modelo_chat()
print(f"Modelo local: {nome_modelo}")
llm = ChatOllama(model=nome_modelo, temperature=0)
cadeia = ChatPromptTemplate.from_messages([
    ("system", "Responda somente com o contexto das bulas. Não invente informações. Se o contexto não contiver a resposta, diga que a bula carregada não traz essa informação."),
    ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}")
]) | llm

def mostrar(pergunta):
    docs = retriever.invoke(pergunta)
    contexto = "\n\n".join(d.page_content for d in docs)
    resposta = cadeia.invoke({"contexto": contexto, "pergunta": pergunta})
    print(f"\nPergunta: {pergunta}")
    print(f"\nResposta do Agente: \n{resposta.content}")
    print("\nTrechos utilizados como contexto:\n")
    for i, doc in enumerate(docs, start=1):
        print(f"--- Trecho {i} ---")
        print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
        print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
        print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print("\nConteúdo do chunk:")
        print(doc.page_content)
        print("\n" + "=" * 50)

mostrar("Quais são as contraindicações da dipirona?")
mostrar("Qual é a posologia recomendada do paracetamol para adultos?")
