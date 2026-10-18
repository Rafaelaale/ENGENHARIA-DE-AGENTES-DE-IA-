import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "sk-..."

PERSIST_DIRECTORY = "./chroma_rh"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

@st.cache_data
def carregar_documentos():
    caminhos = ["politica_ferias.pdf", "politica_home_office.pdf", "codigo_conduta.pdf"]
    documentos = []
    for caminho in caminhos:
        loader = PyPDFLoader(caminho)
        docs = loader.load()
        for doc in docs:
            doc.metadata["documento"] = caminho
        documentos.extend(docs)
    return documentos

def gerar_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documentos)

def enriquecer_chunks(chunks):
    for chunk in chunks:
        texto = chunk.page_content.lower()
        if "ferias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "etica" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"
    return chunks

@st.cache_resource
def criar_vectorstore(_chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=_chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    return vectorstore

def rerank_documentos(pergunta, documentos, llm):
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="Avalie a relevância do texto para a pergunta de 0 a 10. Responda APENAS com o número.\nPergunta: {pergunta}\nTexto: {texto}"
    )
    documentos_com_score = []
    for doc in documentos:
        score = llm.invoke(prompt_rerank.format(pergunta=pergunta, texto=doc.page_content)).content
        try:
            score = float(score)
        except:
            score = 0
        documentos_com_score.append((score, doc))
    documentos_ordenados = sorted(documentos_com_score, key=lambda x: x[0], reverse=True)
    return [doc for _, doc in documentos_ordenados]

def responder_pergunta(pergunta, vectorstore):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    documentos_recuperados = vectorstore.similarity_search(pergunta, k=8)
    documentos_rerankeados = rerank_documentos(pergunta, documentos_recuperados, llm)
    contexto_final = documentos_rerankeados[:4]
    contexto_texto = "\n\n".join([doc.page_content for doc in contexto_final])
    prompt_final = f"Responda à pergunta com base no contexto.\nContexto: {contexto_texto}\nPergunta: {pergunta}"
    resposta = llm.invoke(prompt_final)
    return resposta.content, contexto_final

st.title("Chat de RH - Políticas Internas")
documentos = carregar_documentos()
chunks = gerar_chunks(documentos)
chunks_enriquecidos = enriquecer_chunks(chunks)
vectorstore = criar_vectorstore(chunks_enriquecidos)

pergunta = st.text_input("Faça uma pergunta sobre as políticas de RH:")
if pergunta:
    with st.spinner("Buscando..."):
        resposta, contexto = responder_pergunta(pergunta, vectorstore)
        st.write("### Resposta")
        st.write(resposta)
        st.write("### Contexto Utilizado")
        for i, doc in enumerate(contexto, 1):
            st.info(f"Trecho {i} | Arquivo: {doc.metadata.get('documento')} | Categoria: {doc.metadata.get('categoria')}\n\n{doc.page_content}")
