import sqlite3
import operator
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

@tool
def buscar_clima(cidade: str = "") -> str:
    """Busca a previsão do tempo para uma cidade."""
    cidade_lower = str(cidade).lower()
    if "são paulo" in cidade_lower or "sao paulo" in cidade_lower:
        return "São Paulo: Ensolarado, mínima de 17°C e máxima de 31°C."
    elif "rio de janeiro" in cidade_lower or "rio" in cidade_lower:
        return "Rio de Janeiro: Sol com algumas nuvens, mínima de 20°C e máxima de 29°C."
    return f"Previsão para {cidade}: Ensolarado com máxima de 28°C."

tools = [buscar_clima]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)

        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_llm(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            args = t.get("args", {})
            if isinstance(args, dict) and "cidade" not in args:
                for v in args.values():
                    if isinstance(v, str) and v not in ("string", ""):
                        args = {"cidade": v}
                        break
                else:
                    args = {"cidade": "São Paulo"}

            print(f"-> Chamando ferramenta: {t['name']} com args: {args}")
            tool_obj = self.tools[t["name"]]
            result = tool_obj.invoke(args)
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        return {"messages": results}

prompt_system = (
    "Você é um assistente meteorológico objetivo. Use a ferramenta buscar_clima para consultar previsões. "
    "Ao comparar cidades, consulte os dados obtidos e responda diretamente."
)

model = ChatOllama(model="llama3.2:1b", temperature=0)
abot = Agent(model, tools, checkpointer=memory, system=prompt_system)

# Demonstração 1: Pergunta inicial com Thread ID 1
print("\n--- Pergunta 1: Tempo em São Paulo (Thread 1) ---")
thread1 = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": [HumanMessage(content="Como está o tempo em São Paulo?")]}, thread1):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"[{k}]: {v['messages'][-1].content}")

# Demonstração 2: Contexto mantido na Thread 1
print("\n--- Pergunta 2: E no Rio de Janeiro? (Thread 1) ---")
for event in abot.graph.stream({"messages": [HumanMessage(content="E no Rio de Janeiro?")]}, thread1):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"[{k}]: {v['messages'][-1].content}")

# Demonstração 3: Comparação com histórico SQLite (Thread 1)
print("\n--- Pergunta 3: Qual cidade estava mais quente? (Thread 1) ---")
for event in abot.graph.stream({"messages": [HumanMessage(content="Qual cidade estava mais quente?")]}, thread1):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"[{k}]: {v['messages'][-1].content}")

# Demonstração 4: Nova Thread ID 2 (sem memória prévia)
print("\n--- Pergunta 4: Qual cidade estava mais quente? (Thread 2 - Sem contexto) ---")
thread2 = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": [HumanMessage(content="Qual cidade estava mais quente?")]}, thread2):
    for k, v in event.items():
        if k in ("llm", "action"):
            print(f"[{k}]: {v['messages'][-1].content}")
