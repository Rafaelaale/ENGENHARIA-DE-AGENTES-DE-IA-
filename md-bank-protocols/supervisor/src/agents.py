import os
import logging
from typing import List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
logger = logging.getLogger(__name__)

__llm = init_chat_model(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0
)

class AgentSelection(BaseModel):
    agentes: List[Literal["abrir_conta", "cartao_credito"]] = Field(
        description="Lista com os nomes dos agentes necessários para atender à mensagem do usuário."
    )

structured_llm = __llm.with_structured_output(AgentSelection)

def classificar_intencao(mensagem: str) -> List[str]:
    prompt = (
        "Você é um classificador de intenções para o banco MDBank. "
        "Analise a mensagem do cliente e determine quais agentes devem ser acionados: "
        "'abrir_conta' (se o assunto for abertura de conta) e/ou "
        "'cartao_credito' (se o assunto for cartão de crédito, solicitação de cartão, limites). "
        "Se o cliente mencionar ambos, selecione os dois."
    )
    resposta: AgentSelection = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=mensagem)
    ])
    logger.info(f"Agentes selecionados: {resposta.agentes}")
    return resposta.agentes
