import os
import sqlite3
from uuid import uuid4
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama

load_dotenv()

# Conexão SQLite para persistência
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Função personalizada de redução de mensagens para Human-in-the-Loop
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    for message in right:
        if not message.id:
            message.id = str(uuid4())

    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged

# AgentState utilizando a função reduce_messages em vez de operator.add
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

if __name__ == "__main__":
    print("--- Testando função reduce_messages (HITL) ---")

    # Mensagem inicial
    msg_id = str(uuid4())
    msg_original = HumanMessage(content="Mensagem original do usuário", id=msg_id)
    estado_inicial = [msg_original]
    print(f"Estado inicial: 1 mensagem (ID: {msg_id}) -> Conteúdo: '{msg_original.content}'")

    # Caso 1: Substituição de mensagem com o mesmo ID (intervenção humana / correção)
    msg_corrigida = HumanMessage(content="Mensagem corrigida pelo humano", id=msg_id)
    resultado_substituicao = reduce_messages(estado_inicial, [msg_corrigida])
    print(f"\nApós intervenção (mesmo ID): Total {len(resultado_substituicao)} mensagem -> Conteúdo: '{resultado_substituicao[0].content}'")

    # Caso 2: Anexação de nova mensagem com ID diferente
    nova_msg = AIMessage(content="Resposta gerada pelo agente", id=str(uuid4()))
    resultado_anexo = reduce_messages(resultado_substituicao, [nova_msg])
    print(f"\nApós anexar nova mensagem: Total {len(resultado_anexo)} mensagens")
    for idx, m in enumerate(resultado_anexo, start=1):
        print(f"  {idx}. [{m.__class__.__name__}] ID: {m.id} | Conteúdo: {m.content}")

    print("\nConfiguração de AgentState e reduce_messages pronta para a próxima etapa!")
