from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Agente Abrir Conta")

__llm = init_chat_model(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

agente = create_agent(
    __llm,
    tools=[],
    system_prompt=(
        "Você é um especialista em abertura de contas do banco MDBank. "
        "Ajude o cliente a abrir uma conta. "
        "Explique os tipos de contas disponíveis: conta simples e conta completa."
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
