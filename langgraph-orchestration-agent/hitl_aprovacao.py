import os
import sqlite3
import uuid
from datetime import date
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

# Conexão SQLite para persistência
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Função personalizada de redução de mensagens
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    for message in right:
        if not message.id:
            message.id = str(uuid.uuid4())
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

# Ferramenta de clima simulada
@tool
def buscar_clima(cidade: str = "São Paulo") -> str:
    """Busca a previsão do tempo para uma cidade específica."""
    cidade_normalizada = (cidade or "São Paulo").strip().lower()
    if "rio" in cidade_normalizada:
        return "Rio de Janeiro: Ensolarado, 29°C, minima de 22°C, sem chuva."
    return "São Paulo: Parcialmente nublado, 31°C, minima de 19°C, umidade moderada."

class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {"True": "action", "False": END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        # Compila com persistência e interrupção ANTES de executar a ação
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )

    def call_llm(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        if self.system and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "True"
        return "False"

    def take_action(self, state: AgentState) -> Dict[str, Any]:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            nome_tool = t["name"]
            tool_obj = self.tools.get(nome_tool)
            args = t.get("args") or {}

            # Normalização de argumentos caso venha formato aninhado
            if not isinstance(args, dict):
                args = {"cidade": str(args)}
            elif "cidade" not in args:
                if "properties" in args:
                    args = {"cidade": str(args["properties"])}
                elif len(args) == 1:
                    args = {"cidade": str(next(iter(args.values())))}
                else:
                    args = {"cidade": "São Paulo"}

            if tool_obj:
                try:
                    resultado = tool_obj.invoke(args)
                except Exception:
                    resultado = tool_obj.invoke({"cidade": "São Paulo"})
            else:
                resultado = f"Ferramenta {nome_tool} não encontrada."

            results.append(ToolMessage(tool_call_id=t["id"], name=nome_tool, content=str(resultado)))
        return {"messages": results}

if __name__ == "__main__":
    current_date = date.today().strftime("%d/%m/%Y")
    prompt = f"""Você é um assistente de pesquisa inteligente.
A data atual é {current_date}.
Ao buscar sobre o tempo em uma cidade, use a ferramenta buscar_clima informando a cidade.
"""
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    agent = Agent(model=llm, tools=[buscar_clima], system=prompt, checkpointer=memory)

    dynamic_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": dynamic_thread_id}}
    print(f"Meu novo Thread ID dinâmico é: {dynamic_thread_id}")

    pergunta = "Como está o tempo em São Paulo hoje?"
    print(f"\nPergunta do usuário: {pergunta}\n")

    # ETAPA 1: Executa até o ponto de interrupção (interrupt_before=['action'])
    print("--- [ETAPA 1] Executando até o ponto de interrupção ---")
    for event in agent.graph.stream({"messages": [HumanMessage(content=pergunta)]}, config):
        for node_name, node_output in event.items():
            print(f">> Nó executado: {node_name}")

    # Verifica o estado atual pausado
    estado_atual = agent.graph.get_state(config)
    print(f"\nPróximo nó agendado: {estado_atual.next}")

    if "action" in estado_atual.next:
        ultima_msg = estado_atual.values["messages"][-1]
        print("\n[AGENTE PAUSADO: INTERVENÇÃO HUMANA NECESSÁRIA]")
        print(f"Ação solicitada pelo agente: {ultima_msg.tool_calls}")

        resposta_usuario = input("\nDeseja autorizar a execução da ferramenta? (sim/nao): ").strip().lower()

        if resposta_usuario in ["s", "sim", "y", "yes"]:
            print("\n--- [ETAPA 2] Ação APROVADA: Retomando execução do grafo ---")
            for event in agent.graph.stream(None, config):
                for node_name, node_output in event.items():
                    print(f">> Nó executado: {node_name}")

            estado_final = agent.graph.get_state(config)
            print(f"\nResposta final do agente:\n{estado_final.values['messages'][-1].content}")
        else:
            print("\n[AÇÃO CANCELADA PELO USUÁRIO] O grafo foi encerrado sem executar a ferramenta.")
    else:
        print("\nO agente respondeu diretamente sem requerer ação de ferramenta.")
