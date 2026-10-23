import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

# Ferramenta de busca da aula adaptada localmente
@tool("tavily_search_results_json")
def tool_instance(query: str) -> str:
    """Mecanismo de busca na web para consultar informacoes atualizadas."""
    q = query.lower()
    if "ia" in q or "inteligência artificial" in q or "inteligencia artificial" in q:
        return (
            "IA Generativa cria conteudo original como texto e imagens. "
            "LangGraph e um framework para criacao de agentes em grafos com estado, "
            "enquanto LangChain e um conjunto mais amplo de ferramentas e chains."
        )
    if "1998" in q or "copa" in q:
        return (
            "Copa 1998: Franca campea (venceu o Brasil por 3x0). "
            "Capital: Paris. Moeda: Euro. PIB em 1998: 1.5 tri USD. PIB 2023: 3.0 tri USD."
        )
    return f"Resultados para '{query}': informacoes obtidas com sucesso."

tool = tool_instance

# 1. Estrutura do Estado
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# 2. Classe do Agente LangGraph
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:
                result = "bad tool name, retry"
            else:
                args = t['args']
                if isinstance(args, dict) and "properties" in args and "query" in args["properties"]:
                    args = {"query": args["properties"]["query"]}
                result = self.tools[t['name']].invoke(args)
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# 3. Prompt e Instancia do Modelo
prompt = """Você é um assistente de pesquisa inteligente. Use o mecanismo de busca para procurar informações. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

model = ChatOllama(model="llama3.2:1b", temperature=0)
abot = Agent(model, [tool], system=prompt)

# 4. Loop Interativo da Aula
print("\n--- Agente de Pesquisa Interativo ---")
print("Digite sua pergunta ou 'sair' para encerrar.")

while True:
    user_input = input("\nVocê: ")
    if user_input.lower() == "sair":
        print("Agente: Encerrando a conversa. Até logo!")
        break

    messages = [HumanMessage(content=user_input)]

    print("Agente: Pensando e buscando...")
    try:
        current_state = {}
        for s in abot.graph.stream({"messages": messages}):
            current_state.update(s)

        print("\nAgente:")
        if 'llm' in current_state and 'messages' in current_state['llm'] and current_state['llm']['messages']:
            final_message = current_state['llm']['messages'][-1]
            if hasattr(final_message, 'content'):
                print(final_message.content)
            else:
                print("Não foi possível extrair o conteúdo da resposta final do LLM.")
        else:
            print("Não foi possível obter uma resposta do agente para esta pergunta.")

    except Exception as e:
        print(f"Agente: Ocorreu um erro durante a execução: {e}")
        print("Tente novamente ou digite 'sair'.")

print("\n--- Conversa Encerrada ---")
