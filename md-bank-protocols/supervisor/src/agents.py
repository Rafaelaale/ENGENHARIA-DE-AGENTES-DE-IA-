import os
import re
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

openai_api_key = os.getenv("OPENAI_API_KEY", "ollama")
openai_api_base = os.getenv("OPENAI_API_BASE", "http://host.docker.internal:11434/v1")
model_name = os.getenv("MODEL_NAME", "gemma3:1b")

llm = ChatOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model=model_name,
    temperature=0.0,
    max_tokens=30,
)

class RouteResponse(BaseModel):
    agents: List[str] = Field(
        description="Lista com os agentes: 'cartao_credito', 'abrir_conta' ou ambos."
    )

def classificar_intencao(mensagem: str) -> List[str]:
    msg = mensagem.lower()
    tem_cartao = bool(re.search(r"\b(cartao|cartoes|credito|limite|fatura|platinum|goldsilver|mdzone)\b", msg))
    tem_conta = bool(re.search(r"\b(conta|abrir|abertura|corrente|poupanca|documentos|documento)\b", msg))

    if tem_cartao and tem_conta:
        return ["cartao_credito", "abrir_conta"]
    if tem_cartao:
        return ["cartao_credito"]
    if tem_conta:
        return ["abrir_conta"]

    structured_llm = llm.with_structured_output(RouteResponse)
    system_prompt = (
        "Classifique a mensagem escolhendo: 'cartao_credito' e/ou 'abrir_conta'."
    )
    try:
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=mensagem)
        ])
        return response.agents
    except Exception:
        return ["cartao_credito"]

route_agent = classificar_intencao