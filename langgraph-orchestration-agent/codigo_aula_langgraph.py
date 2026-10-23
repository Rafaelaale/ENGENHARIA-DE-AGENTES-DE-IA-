import os
import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError(
        "TAVILY_API_KEY não encontrada. Configure essa chave no arquivo .env."
    )


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END},
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def exists_action(self, state: AgentState):
        ultima_mensagem = state["messages"][-1]
        return len(ultima_mensagem.tool_calls) > 0

    def call_gemini(self, state: AgentState):
        messages = state["messages"]

        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for chamada in tool_calls:
            nome = chamada["name"]
            argumentos = chamada["args"]
            print(f"Calling: {chamada}")

            if nome not in self.tools:
                resultado = "Nome de ferramenta inválido. Tente novamente."
            else:
                resultado = self.tools[nome].invoke(argumentos)

            results.append(
                ToolMessage(
                    tool_call_id=chamada["id"],
                    name=nome,
                    content=str(resultado),
                )
            )

        print("Back to the model!")
        return {"messages": results}


tool = TavilySearchResults(max_results=4)
print(type(tool))
print(tool.name)

model = ChatOllama(model="gemma3:1b", temperature=0)
agent = Agent(model=model, tools=[tool])

print("Agente LangGraph criado.")
print(agent.graph.get_graph().draw_mermaid())
