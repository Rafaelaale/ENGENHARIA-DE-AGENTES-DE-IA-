import logging
import operator
import requests
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.agents import classificar_intencao

logger = logging.getLogger(__name__)

class SupervisorState(TypedDict):
    query: str
    respostas: Annotated[List[str], operator.add]

def chamar_agente(url: str, mensagem: str) -> str:
    try:
        response = requests.post(url, json={"message": mensagem}, timeout=30)
        if response.status_code == 200:
            return response.json().get("resposta", "Resposta não encontrada.")
        return d"Erro ao consultar o agente ({response.status_code})"
    except Exception as e:
        logger.error(f"Falha de conexão com {url}: {e}")
        return "Erro ao consultar o agente."

def node_abrir_conta(state: dict) -> dict:
    resposta = chamar_agente("http://abrir_conta_agent:8000/send", state["query"])
    return {"respostas": [resposta]}

def node_cartao_credito(state: dict) -> dict:
    resposta = chamar_agente("http://cartao_credito_agent:8000/send", state["query"])
    return {"respostas": [resposta]}

def roteador(state: SupervisorState):
    agentes = classificar_intencao(state["query"])
    destinos = []
    for agente in agentes:
        if agente == "abrir_conta":
            destinos.append(Send("abrir_conta", {"query": state["query"]}))
        elif agente == "cartao_credito":
            destinos.append(Send("cartao_credito", {"query": state["query"]}))
    return destinos if destinos else END

builder = StateGraph(SupervisorState)
builder.add_node("abrir_conta", node_abrir_conta)
builder.add_node("cartao_credito", node_cartao_credito)

builder.add_conditional_edges(START, roteador, ["abrir_conta", "cartao_credito", END])
builder.add_edge("abrir_conta", END)
builder.add_edge("cartao_credito", END)

grafo = builder.compile()

def executar_supervisor(query: str) -> str:
    resultado = grafo.invoke({"query": query, "respostas": []})
    respostas = resultado.get("respostas", [])
    if not respostas:
        return "Não foi possível atender sua solicitação no momento."
    return "\n\n".join(respostas)
