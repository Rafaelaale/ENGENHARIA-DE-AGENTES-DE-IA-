from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

app = FastAPI(title="Agente Cartão de Crédito")

__llm = ChatOpenAI(
    base_url="http://host.docker.internal:11434/v1",
    api_key="ollama",
    model="gemma3:1b",
    temperature=0.7
)

agente = create_agent(
    __llm,
    tools=[],
    system_prompt=(
        "Você é um especialista em cartão de crédito do banco MDBank. "
        "Ajude o cliente com dúvidas, solicitação e limites."
    )
)

class RequestMessage(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/send")
async def send_message(payload: RequestMessage):
    resultado = agente.invoke(
        {"messages": [HumanMessage(content=payload.message)]}
    )
    mensagem_ia = resultado["messages"][-1]
    return {"resposta": str(mensagem_ia.content)}
