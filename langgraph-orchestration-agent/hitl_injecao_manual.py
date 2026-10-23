import sqlite3
import uuid
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

# Conexão de persistência SQLite
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

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

@tool
def buscar_distancia(origem: str, destino: str) -> str:
    """Calcula a distância aproximada entre duas cidades."""
    return f"Distância real calculada entre {origem} e {destino}: aproximadamente 18.500 km."

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
            resultado = tool_obj.invoke(args) if tool_obj else f"Ferramenta {nome_tool} não encontrada."
            results.append(ToolMessage(tool_call_id=t["id"], name=nome_tool, content=str(resultado)))
        return {"messages": results}

if __name__ == "__main__":
    prompt = "Você é um assistente de viagens. Sempre use a ferramenta buscar_distancia para responder sobre distâncias entre cidades."
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    abot = Agent(model=llm, tools=[buscar_distancia], system=prompt, checkpointer=memory)

    new_session_id = str(uuid.uuid4())
    new_thread_config = {"configurable": {"thread_id": new_session_id}}
    print(f"Meu novo Thread ID dinâmico é: {new_session_id}")

    pergunta = "Qual a distância entre Rio de Janeiro e Tóquio?"
    print(f"Pergunta: {pergunta}\n")

    print("--- Executando agente até a interrupção ---")
    for event in abot.graph.stream({"messages": [HumanMessage(content=pergunta)]}, new_thread_config):
        for node_name in event.keys():
            print(f">> Nó executado: {node_name}")

    # Captura do snapshot no momento da pausa
    current_state_snapshot = abot.graph.get_state(new_thread_config)

    if current_state_snapshot:
        print(f"\nDEBUG: Estado atual obtido para NOVA thread ID: {new_session_id}")
        snapshot_thread_id = None
        snapshot_thread_ts = None

        if hasattr(current_state_snapshot, 'config') and isinstance(current_state_snapshot.config, dict):
            cfg = current_state_snapshot.config.get('configurable', {})
            snapshot_thread_id = cfg.get('thread_id') or cfg.get('__run_id')
            snapshot_thread_ts = cfg.get('thread_ts')

        if snapshot_thread_id is None:
            snapshot_thread_id = new_session_id

        print(f"DEBUG: ID da Thread (do snapshot): {snapshot_thread_id}")
        print(f"DEBUG: Timestamp do snapshot (thread_ts): {snapshot_thread_ts}")
        print(f"DEBUG: Mensagens no snapshot (no momento da pausa): {current_state_snapshot.values.get('messages')}")

        if current_state_snapshot.values and 'messages' in current_state_snapshot.values:
            last_msg_in_snapshot = current_state_snapshot.values['messages'][-1]
            print(f"DEBUG: Tipo da última mensagem no snapshot para injeção: {type(last_msg_in_snapshot)}")
            if hasattr(last_msg_in_snapshot, 'tool_calls') and last_msg_in_snapshot.tool_calls:
                print("DEBUG: Última mensagem no snapshot TEM tool_calls. PRONTO PARA INJEÇÃO!")
            else:
                print("DEBUG: Última mensagem no snapshot NÃO TEM tool_calls.")

            if current_state_snapshot.next:
                print("\n--- Agente está PAUSADO e pronto para intervenção. ---")
            else:
                print("\n--- ATENÇÃO: O agente NÃO está pausado onde esperávamos. ---")

        # Injeção manual da resposta alterada (conforme código da aula)
        final_injected_message = AIMessage(
            content="A distância entre o Rio de Janeiro e Tóquio é de aproximadamente 450 km. (Dados fornecidos MANUALMENTE por você!)"
        )

        modified_state_values = current_state_snapshot.values.copy()
        ai_message_found = False
        for i, msg in enumerate(modified_state_values['messages']):
            if isinstance(msg, AIMessage):
                modified_state_values['messages'][i] = final_injected_message
                ai_message_found = True
                break

        if not ai_message_found:
            modified_state_values['messages'].append(final_injected_message)

        print("\n--- Estado sendo MODIFICADO MANUALMENTE (Injetando AIMessage Final) ---")
        print(f"DEBUG: Conteúdo da AIMessage falsa injetada: {final_injected_message.content}")

        print("\n--- Finalizando o estado com a resposta injetada ---")
        abot.graph.update_state(new_thread_config, modified_state_values)

        final_state_after_injection_obj = abot.graph.get_state(new_thread_config)
        print("\n--- Saída final do agente após intervenção ---")
        final_messages = final_state_after_injection_obj.values.get('messages', [])
        if final_messages and isinstance(final_messages[-1], AIMessage):
            print(f"\nAgente: {final_messages[-1].content}")
        else:
            print("\nAgente: Resposta final não encontrada ou não é um AIMessage.")

        print("\n--- Fluxo de Human-in-the-Loop concluído ---")
