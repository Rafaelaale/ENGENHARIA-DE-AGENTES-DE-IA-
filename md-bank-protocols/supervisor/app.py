import json
from fastapi import FastAPI, Response
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
async def chat(payload: ChatRequest):
    resposta = await executar_supervisor(payload.message)
    conteudo = json.dumps({"resposta": resposta}, ensure_ascii=False)
    return Response(content=conteudo, media_type="application/json; charset=utf-8")
