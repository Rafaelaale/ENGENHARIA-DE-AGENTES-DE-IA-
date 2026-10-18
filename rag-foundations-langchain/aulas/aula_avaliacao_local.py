from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.evaluation.qa import QAEvalChain

documentos = PyPDFLoader("documentos/politica_ferias.pdf").load()
divisor = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
pedacos = divisor.split_documents(documentos)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(pedacos)

llm = ChatOllama(model="gemma3:1b", temperature=0)

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

predictions = []
for exemplo in exemplos:
    documentos_relevantes = vectorstore.similarity_search(exemplo["query"], k=3)
    contexto = "\n\n".join(doc.page_content for doc in documentos_relevantes)

    prompt = (
        "Responda usando apenas o contexto. Se a informação não estiver nele, "
        "diga que não consta no documento.\n\n"
        f"Contexto:\n{contexto}\n\nPergunta: {exemplo['query']}"
    )
    resposta = llm.invoke(prompt).content
    predictions.append({"query": exemplo["query"], "result": resposta})

eval_chain = QAEvalChain.from_llm(llm=llm)
resultados = eval_chain.evaluate(exemplos, predictions)

for i, resultado in enumerate(resultados, start=1):
    print(f"\nPergunta {i}: {exemplos[i - 1]['query']}")
    print(f"Esperado: {exemplos[i - 1]['answer']}")
    print(f"Gerado:   {predictions[i - 1]['result']}")
    print(f"Avaliação: {resultado}")
