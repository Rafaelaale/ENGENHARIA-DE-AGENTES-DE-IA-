from fastapi import FastAPI
from pydantic import BaseModel
from src.service import executar_supervisor

app = FastAPI(title="Supervisor Agents")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "123"
    client_id: str = "123"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(payload: ChatRequest):
    resposta = executar_supervisor(payload.message)
    return {"resposta": resposta}
