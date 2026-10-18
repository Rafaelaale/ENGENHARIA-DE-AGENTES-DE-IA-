from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

modelo = ChatOpenAI(
    model="llama3.2:1b",
    temperature=0.5,
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sra Praia. Você é uma especialista em viagens para praia. Responda em português e de forma curta."),
    ("human", "{query}"),
])

prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sr Montanha. Você é um especialista em viagens para montanhas e atividades radicais. Responda em português e de forma curta."),
    ("human", "{query}"),
])

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com praia ou montanha."),
    ("human", "{query}"),
])

roteador = prompt_roteador | modelo | StrOutputParser()

def escolher_destino(texto: str) -> str:
    texto = texto.lower()
    if "montanha" in texto:
        return "montanha"
    return "praia"

def responda(pergunta: str) -> str:
    rota = escolher_destino(roteador.invoke({"query": pergunta}))
    print("Rota:", rota)
    if rota == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    return cadeia_montanha.invoke({"query": pergunta})

for pergunta in ["Quero surfar em um lugar quente", "Quero escalar uma montanha"]:
    print("Você:", pergunta)
    print(responda(pergunta))
    print()
