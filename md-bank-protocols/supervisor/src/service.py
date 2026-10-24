import logging
import operator
import uuid
import httpx
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.agents import classificar_intencao

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Role, Part, SendMessageRequest

logger = logging.getLogger(__name__)

AGENTS = {
    "cartao_credito": "http://cartao_credito_agent:8000",
    "abrir_conta": "http://abrir_conta_agent:8000",
}

HTTPX_CLIENT = httpx.AsyncClient(timeout=30.0)
CLIENT_CACHE = {}

class SupervisorState(TypedDict):
    query: str
    respostas: Annotated[List[str], operator.add]

def extrair_texto_evento(event) -> str:
    # Se o evento contiver 'parts' diretamente
    parts = getattr(event, "parts", None)
    if not parts and hasattr(event, "message"):
        parts = getattr(event.message, "parts", None)

    if parts:
        textos = []
        for p in parts:
            if hasattr(p, "text") and p.text:
                textos.append(p.text)
        if textos:
            return "\n".join(textos)

    # Fallback se vier como dicionario ou string
    if isinstance(event, dict):
        return event.get("text", "")
    return ""

async def chamar_agente_a2a(agent_url: str, message: str) -> str:
    try:
        if agent_url not in CLIENT_CACHE:
            logger.info(f"Descobrindo AgentCard em {agent_url}")
            resolver = A2ACardResolver(
                httpx_client=HTTPX_CLIENT,
                base_url=agent_url,
            )
            agent_card = await resolver.get_agent_card()
            logger.info(f"Agent encontrado: {agent_card.name}")

            config = ClientConfig(
                httpx_client=HTTPX_CLIENT,
                streaming=False
            )
            factory = ClientFactory(config)
            CLIENT_CACHE[agent_url] = factory.create(agent_card)

        client = CLIENT_CACHE[agent_url]

        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.ROLE_USER,
            parts=[Part(text=message)],
        )

        logger.info(f"Enviando mensagem para agente: {message}")

        request = SendMessageRequest(message=msg)
        async for event in client.send_message(request):
            texto = extrair_texto_evento(event)
            if texto:
                return texto

        return "Sem resposta do agente."
    except Exception as e:
        logger.error(f"Erro ao consultar agente em {agent_url}: {e}")
        return "Erro ao consultar o agente."

async def rota_cartao(state: SupervisorState):
    resp = await chamar_agente_a2a(AGENTS["cartao_credito"], state["query"])
    return {"respostas": [resp]}

async def rota_abrir_conta(state: SupervisorState):
    resp = await chamar_agente_a2a(AGENTS["abrir_conta"], state["query"])
    return {"respostas": [resp]}

def roteador(state: SupervisorState):
    intencao = classificar_intencao(state["query"])
    logger.info(f"Intencao detectada: {intencao}")
    rotas = []
    if "cartao_credito" in intencao:
        rotas.append(Send("rota_cartao", state))
    if "abrir_conta" in intencao:
        rotas.append(Send("rota_abrir_conta", state))
    if not rotas:
        return END
    return rotas

builder = StateGraph(SupervisorState)
builder.add_node("rota_cartao", rota_cartao)
builder.add_node("rota_abrir_conta", rota_abrir_conta)
builder.add_conditional_edges(START, roteador, ["rota_cartao", "rota_abrir_conta", END])
builder.add_edge("rota_cartao", END)
builder.add_edge("rota_abrir_conta", END)

grafo = builder.compile()

async def executar_supervisor(query: str) -> str:
    resultado = await grafo.ainvoke({"query": query, "respostas": []})
    respostas = resultado.get("respostas", [])
    if respostas:
        return "\n\n".join(respostas)
    return "Nao foi possivel atender sua solicitacao no momento."
