import os
import json
from dotenv import load_dotenv
from typing_extensions import TypedDict, Literal, Annotated

try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.store.memory import InMemoryStore

try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    HAS_LANGMEM = True
except ImportError:
    HAS_LANGMEM = False

from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_memory

load_dotenv()

profile = {
    "name": "Sarah",
    "full_name": "Sarah Chen",
    "user_profile_background": "Engenheira de software sênior liderando uma equipe de 5 desenvolvedores",
}

triage_no_prompt_instructions = {
    "triage_rules": {"ignore": "Newsletters, spam, comunicados gerais da empresa"}
}
triage_notify_prompt_instructions = {
    "triage_rules": {"notify": "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto"}
}
triage_email_prompt_instructions = {
    "triage_rules": {"respond": "Perguntas diretas de membros da equipe, solicitações de reunião, relatórios de bugs críticos"}
}

prompt_instructions = {
    "agent_instructions": "Você é um assistente executivo altamente eficiente. Execute as tarefas solicitadas de forma direta e sem rodeios."
}

class Router(BaseModel):
    reasoning: str = Field(description="Raciocínio detalhado da classificação.")
    classification: Literal["ignore", "notify", "respond"] = Field(
        description="Classificação do e-mail"
    )

llm = ChatOllama(model="llama3.2:1b", temperature=0)
llm_router = llm.with_structured_output(Router)

@tool
def write_email(to: str = "", subject: str = "", content: str = "", **kwargs) -> str:
    """Envia um e-mail para o destinatário especificado."""
    params = kwargs.get("parameters", {})
    to = to or params.get("to", "")
    subject = subject or params.get("subject", "")
    content = content or params.get("content", "")
    return json.dumps({"to": to, "subject": subject, "content": content})

@tool
def schedule_meeting(attendees: list[str] = None, subject: str = "", duration_minutes: int = 30, preferred_day: str = "", **kwargs) -> str:
    """Agenda uma reunião."""
    params = kwargs.get("parameters", {})
    attendees = attendees or params.get("attendees", [])
    subject = subject or params.get("subject", "")
    preferred_day = preferred_day or params.get("preferred_day", "")
    return f"Reunião '{subject}' agendada para {preferred_day} com {len(attendees)} participantes"

@tool
def check_calendar_availability(day: str = "", **kwargs) -> str:
    """Verifica horários disponíveis."""
    params = kwargs.get("parameters", {})
    day = day or params.get("day", "terça-feira")
    return f"Horários disponíveis em {day}: 9:00, 14:00, 16:00"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = InMemoryStore(index={"embed": embeddings, "dims": 768})

if HAS_LANGMEM:
    manage_memory_tool = create_manage_memory_tool(
        store=store,
        namespace=("email_assistant", "{langgraph_user_id}", "collection")
    )
    search_memory_tool = create_search_memory_tool(
        store=store,
        namespace=("email_assistant", "{langgraph_user_id}", "collection")
    )
else:
    @tool
    def manage_memory_tool(content: str = "", **kwargs) -> str:
        """Grava uma nova memória no sistema."""
        return f"Memória gravada com sucesso: {content}"

    @tool
    def search_memory_tool(query: str = "", **kwargs) -> str:
        """Busca memórias existentes armazenadas no sistema."""
        return {}

tools = [write_email, schedule_meeting, check_calendar_availability, manage_memory_tool, search_memory_tool]

def create_prompt(state):
    formatted = agent_system_prompt_memory.format(
        instructions=prompt_instructions["agent_instructions"],
        full_name=profile["full_name"],
        name=profile["name"],
    )
    return [{"role": "system", "content": formatted}] + state.get("messages", [])

response_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=create_prompt,
    store=store,
)

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> dict:
    author = state["email_input"]["from"]
    to = state["email_input"]["to"]
    subject = state["email_input"]["subject"]
    body = state["email_input"]["body"]

    sys = triage_system_prompt.format(
        triage_no_prompt_instructions["triage_rules"]["ignore"],
        triage_notify_prompt_instructions["triage_rules"]["notify"],
        triage_email_prompt_instructions["triage_rules"]["respond"]
    )
    usr = triage_user_prompt.format(
        author=author,
        to=to,
        subject=subject,
        email_thread=body
    )
    res = llm_router.invoke([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr}
    ])
    if res.classification == "respond":
        msg = json.dumps({"sender": author, "recipient": to, "subject": subject, "body": body})
        return {"messages": [HumanMessage(content=f"Respond to the email {msg}")], "__next__": "response_agent"}
    return {"__next__": END}

def route_to_tools(state):
    last = state["messages"][-1]
    return "tool_node" if getattr(last, "tool_calls", None) else END

builder = StateGraph(State)
builder.add_node("triage_router", triage_router)
builder.add_node("response_agent", response_agent)
builder.add_node("tool_node", ToolNode(tools))
builder.set_entry_point("triage_router")

builder.add_conditional_edges("triage_router", lambda s: s.get("__next__"), {"response_agent": "response_agent", END: END})
builder.add_conditional_edges("response_agent", route_to_tools, {"tool_node": "tool_node", END: END})
builder.add_edge("tool_node", "response_agent")
email_agent = builder.compile()

def human_in_the_loop_schedule(email_sender, email_recipient, email_subject, config=None):
    res = search_memory_tool.invoke({"query": f"Reunião agendada para {email_sender}"}, config=config)
    if isinstance(res, dict) and res.get("results"):
        out = write_email.invoke({"to": email_sender, "subject": f"Re: {email_subject}", "content": "Olá, acabei de agendar nossa conversa."}, config=config)
        print(f"write_email: {out}")
        return

    ans = input(f"Deseja agendar uma reunião com {email_sender}? (sim/nao): ").strip().lower()
    if ans == "sim":
        m = schedule_meeting.invoke({"attendees": [email_recipient, email_sender], "subject": f"Reunião {email_subject}", "preferred_day": "amanhã"}, config=config)
        print(f"schedule_meeting: {m}")
        mem = manage_memory_tool.invoke({"action": "create", "content": f"Reunião com {email_sender}"}, config=config)
        print(f"manage_memory: {mem}")
        em = write_email.invoke({"to": email_sender, "subject": f"Re: {email_subject}", "content": "Reunião agendada com sucesso."}, config=config)
        print(f"write_email: {em}")
    else:
        em = write_email.invoke({"to": email_sender, "subject": f"Re: {email_subject}", "content": "Estou acompanhando sua solicitação."}, config=config)
        print(f"write_email: {em}")

if __name__ == "__main__":
    email_in = {
        "from": "Alice Smith <alice.smith@company.com>",
        "to": "Sarah Chen <sarah.chen@company.com>",
        "subject": "Acompanhamento",
        "body": "Olá Sarah, como está minha solicitação?"
    }
    cfg = {"configurable": {"langgraph_user_id": "sarah_chen"}}
    resp = email_agent.invoke({"email_input": email_in}, config=cfg)
    human_in_the_loop_schedule(email_in["from"], email_in["to"], email_in["subject"], config=cfg)