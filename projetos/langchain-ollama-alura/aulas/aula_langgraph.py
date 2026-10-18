from typing import Literal, TypedDict
import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

modelo = ChatOpenAI(
    model="llama3.2:1b",
    temperature=0.5,
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia. Responda em português e de forma curta."),
    ("human", "{query}"),
])

prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sr Montanha. Você é um especialista em viagens com destinos para montanhas e atividades radicais. Responda em português e de forma curta."),
    ("human", "{query}"),
])

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com praia ou montanha."),
    ("human", "{query}"),
])

roteador = prompt_roteador | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

class Estado(TypedDict, total=False):
    query: str
    destino: Rota
    resposta: str

def escolher_destino(texto: str) -> str:
    if "montanha" in texto.lower():
        return "montanha"
    return "praia"

async def no_roteador(estado: Estado, config: RunnableConfig):
    texto = await roteador.ainvoke({"query": estado["query"]}, config)
    return {"destino": {"destino": escolher_destino(texto)}}

async def no_praia(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)
grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no, {"praia": "praia", "montanha": "montanha"})
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    perguntas = [
        "Quero visitar um lugar no Brasil famoso por praias e cultura",
        "Quero escalar uma montanha no Brasil",
    ]
    for pergunta in perguntas:
        resultado = await app.ainvoke({"query": pergunta})
        print("Você:", pergunta)
        print("Rota:", resultado["destino"]["destino"])
        print(resultado["resposta"])
        print()

asyncio.run(main())
