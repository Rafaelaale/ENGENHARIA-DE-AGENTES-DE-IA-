import os
import random
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "sk-..."

caminhos_bulas = ["dipirona.pdf", "paracetamol.pdf"]
documentos = []

for caminho in caminhos_bulas:
    loader = PyPDFLoader(caminho)
    docs = loader.load()
    for doc in docs:
        doc.metadata["medicamento"] = caminho.split('/')[0].replace(".pdf", "")
    documentos.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
chunks = text_splitter.split_documents(documentos)

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

chunks_aleatorios = random.sample(chunks, 2)
for i, chunk in enumerate(chunks_aleatorios, start=1):
    print(f"\n--- Chunk Aleatório {i} ---")
    print(f"Metadados: {chunk.metadata}")
    print("\nConteúdo (início):")
    print(chunk.page_content[:300])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_bulas")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

pergunta = "Quais são as contraindicações da dipirona?"
resposta = qa_chain.invoke(pergunta)
print(f"Pergunta: {pergunta}")
print(f"\nResposta do Agente: \n{resposta['result']}")
print("\nTrechos utilizados como contexto:\n")
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print("\n" + "="*50)

pergunta2 = "Qual é a posologia recomendada do paracetamol para adultos?"
resposta = qa_chain.invoke(pergunta2)
print(f"Pergunta: {pergunta2}")
print(f"\nResposta do Agente: \n{resposta['result']}")
print("\nTrechos utilizados como contexto:\n")
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print("\n" + "="*50)
