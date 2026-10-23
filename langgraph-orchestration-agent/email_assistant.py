import os
from typing import List
from typing_extensions import TypedDict, Annotated, Literal
from dotenv import load_dotenv

try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt

load_dotenv()

profile = {
    "name": "Sarah",
    "full_name": "Sarah Chen",
    "user_profile_background": "Engenheira de software sênior liderando uma equipe de 5 desenvolvedores",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Newsletters de marketing, e-mails de spam, comunicados gerais da empresa",
        "notify": "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto",
        "respond": "Perguntas diretas de membros da equipe, dúvidas sobre código/API, solicitações de reunião, relatórios de bugs críticos",
    },
    "agent_instructions": "Use estas ferramentas quando apropriado para ajudar a gerenciar as tarefas de Sarah de forma eficiente."
}

class Router(BaseModel):
    reasoning: str = Field(description="Raciocínio detalhado da classificação.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="Classifique como: 'ignore' (para spam/ofertas), 'respond' (para dúvidas/perguntas diretas) ou 'notify' (para avisos informativos)"
    )

llm = ChatOllama(model="llama3.2:1b", temperature=0)
llm_router = llm.with_structured_output(Router)

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Escreve e envia um e-mail."""
    return f"E-mail enviado para {to} com o assunto '{subject}'"

@tool
def schedule_meeting(attendees: List[str], subject: str, duration_minutes: int, preferred_day: str) -> str:
    """Agenda uma reunião no calendário."""
    return f"Reunião '{subject}' agendada para {preferred_day} com {len(attendees)} participantes"

@tool
def check_calendar_availability(day: str) -> str:
    """Verifica a disponibilidade do calendário para um determinado dia."""
    return f"Horários disponíveis em {day}: 9:00 AM, 2:00 PM, 4:00 PM"

tools = [write_email, schedule_meeting, check_calendar_availability]

def create_prompt(state):
    return [
        {
            "role": "system",
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
            ),
        }
    ] + state.get("messages", [])

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=create_prompt,
)

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    system_prompt = (
        f"Você é assistente de Sarah Chen. Classifique o e-mail em estritamente UMA destas opções:\n"
        f"- ignore: se for propaganda, marketing, promoções ou spam.\n"
        f"- respond: se for colega fazendo perguntas técnicas, dúvidas sobre documentação ou pedidos de reunião.\n"
        f"- notify: se for aviso do sistema ou colega ausente."
    )

    user_prompt = (
        f"De: {author}\nPara: {to}\nAssunto: {subject}\nMensagem:\n{email_thread}\n\n"
        f"Qual é a classificação exata (ignore, respond ou notify)?"
    )

    result = llm_router.invoke([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    if result.classification == "respond":
        print(f"\n[TRIAGEM] Classificação: RESPONDER - Encaminhando para o agente de resposta.")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Responda ao e-mail de {author} sobre '{subject}'. Conteúdo: {email_thread}",
                }
            ]
        }
    elif result.classification == "ignore":
        print(f"\n[TRIAGEM] Classificação: IGNORAR - E-mail descartado com sucesso.")
        update = None
        goto = END
    elif result.classification == "notify":
        print(f"\n[TRIAGEM] Classificação: NOTIFICAR - Apenas aviso registrado.")
        update = None
        goto = END
    else:
        raise ValueError(f"Classificação inválida: {result.classification}")

    return Command(goto=goto, update=update)

builder = StateGraph(State)
builder.add_node("triage_router", triage_router)
builder.add_node("response_agent", agent)
builder.add_edge(START, "triage_router")
email_agent = builder.compile()

if __name__ == "__main__":
    print("--- Teste 1: E-mail de Marketing (Deve ignorar) ---")
    email_spam = {
        "author": "Equipe de Marketing <marketing@amazingdeals.com>",
        "to": "Sarah Chen <sarah.chen@company.com>",
        "subject": "🔥 OFERTA EXCLUSIVA: Desconto por Tempo Limitado! 🔥",
        "email_thread": "Obtenha 80% de desconto hoje mesmo em cursos e ferramentas!",
    }
    email_agent.invoke({"email_input": email_spam})

    print("\n--- Teste 2: E-mail da Alice (Dúvida técnica - Deve responder) ---")
    email_work = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "Sarah Chen <sarah.chen@company.com>",
        "subject": "Dúvida rápida sobre a documentação da API",
        "email_thread": "Olá Sarah, notei que faltam endpoints na documentação (/auth/refresh e /auth/validate). Você pode me esclarecer se devemos atualizar?",
    }
    response = email_agent.invoke({"email_input": email_work})

    if response and "messages" in response:
        print("\n--- Histórico de Mensagens do Agente ---")
        for m in response["messages"]:
            if hasattr(m, "pretty_print"):
                m.pretty_print()
            else:
                print(m)
