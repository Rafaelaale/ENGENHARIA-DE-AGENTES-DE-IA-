import os
from typing import TypedDict, List
from dotenv import load_dotenv

try:
    from langchain_core.pydantic_v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from prompts import (
    PLAN_PROMPT,
    WRITER_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_PLAN_PROMPT,
    RESEARCH_CRITIQUE_PROMPT
)

load_dotenv()

# Estrutura do Estado dos Multiagentes
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Modelo de saída estruturada para consultas de pesquisa
class Queries(BaseModel):
    queries: List[str]

# Configuração do modelo LLM local (Ollama)
model = ChatOllama(model="llama3.2:1b", temperature=0)

# Função auxiliar de busca com proteção contra erro de API
def search_web(query: str, max_results: int = 2) -> List[str]:
    try:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            client = TavilyClient(api_key=api_key)
            res = client.search(query=query, max_results=max_results)
            return [r["content"] for r in res.get("results", [])]
    except Exception:
        pass
    return [f"Informações contextuais sobre: {query}"]

# --- Nós do Grafo ---

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state.get('content') or []
    for q in queries.queries:
        results = search_web(query=q, max_results=2)
        content.extend(results)
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state.get('content') or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state.get('plan', '')}"
    )
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state.get('draft', ''))
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state.get('critique', ''))
    ])
    content = state.get('content') or []
    for q in queries.queries:
        results = search_web(query=q, max_results=2)
        content.extend(results)
    return {"content": content}

def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# Construção do Grafo
builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

# Aresta condicional
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)

# Arestas sequenciais
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

# Compilação com memória
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("\n--- Executando o Fluxo Multiagentes via Stream ---")
    thread = {"configurable": {"thread_id": "1"}}
    for s in graph.stream({
        "task": "Qual é a diferença entre o langchain e langsmith",
        "max_revisions": 2,
        "revision_number": 1,
        "content": [],
    }, thread):
        print(s)
