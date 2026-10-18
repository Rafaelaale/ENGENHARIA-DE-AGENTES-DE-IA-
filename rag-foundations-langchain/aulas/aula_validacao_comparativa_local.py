from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.evaluation.qa import QAEvalChain

PDF = "documentos/politica_ferias.pdf"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:1b"

exemplos = [
    {
        "query": "Quantos dias de férias o funcionário tem por ano?",
        "answer": "O funcionário tem direito a 30 dias de férias por ano.",
    },
    {
        "query": "Com quanto tempo de antecedência as férias devem ser solicitadas?",
        "answer": "A solicitação deve ser feita com 30 dias de aviso prévio.",
    },
]

documentos = PyPDFLoader(PDF).load()
divisor = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
pedacos = divisor.split_documents(documentos)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(pedacos)

llm = ChatOllama(model=LLM_MODEL, temperature=0)

def gerar_respostas(com_rag):
    previsoes = []

    for exemplo in exemplos:
        if com_rag:
            docs = vectorstore.similarity_search(exemplo["query"], k=3)
            contexto = "\n\n".join(doc.page_content for doc in docs)
            prompt = (
                "Responda usando apenas o contexto. Se a informação não estiver "
                "nele, diga que não consta no documento.\n\n"
                f"Contexto:\n{contexto}\n\nPergunta: {exemplo['query']}"
            )
        else:
            prompt = (
                "Responda à pergunta usando seu conhecimento. "
                "Se não souber, diga que não sabe.\n\n"
                f"Pergunta: {exemplo['query']}"
            )

        resposta = llm.invoke(prompt).content
        previsoes.append({"query": exemplo["query"], "result": resposta})

    return previsoes

eval_chain = QAEvalChain.from_llm(llm=llm)

for nome, com_rag in [("SEM RAG", False), ("COM RAG", True)]:
    previsoes = gerar_respostas(com_rag)
    resultados = eval_chain.evaluate(exemplos, previsoes)

    print(f"\n===== {nome} =====")
    for i, resultado in enumerate(resultados):
        print(f"\nPergunta: {exemplos[i]['query']}")
        print(f"Esperado: {exemplos[i]['answer']}")
        print(f"Gerado:   {previsoes[i]['result']}")
        print(f"Avaliação: {resultado}")
