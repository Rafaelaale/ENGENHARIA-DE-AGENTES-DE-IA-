from langchain_ollama import OllamaEmbeddings

def get_embedding(text, model="nomic-embed-text:latest"):
    text = text.replace("\n", " ")
    return OllamaEmbeddings(model=model).embed_query(text)

vetor = get_embedding("contrato de trabalho")
print(len(vetor))
