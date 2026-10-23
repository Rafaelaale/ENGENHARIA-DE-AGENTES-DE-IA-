from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

__llm = init_chat_model(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

agente_cartao_credito = create_agent(
    __llm,
    tools=[],
    system_prompt=(
        "Você é um especialista em cartão de crédito do banco MDBank. "
        "Ajude o cliente com dúvidas, solicitação e limites."
    )
)

agente_abertura_conta = create_agent(
    __llm,
    tools=[],
    system_prompt=(
        "Você é um especialista em abertura de contas do banco MDBank. "
        "Ajude o cliente a abrir uma conta. "
        "Explique os tipos de contas disponíveis: conta simples e conta completa."
    ),
)

def classificar_pergunta(pergunta: str) -> str:
    prompt = f\"\"\"
    Classifique a intenção do usuário.

    Possíveis agentes:
    cartao_credito
    abrir_conta

    Pergunta: {pergunta}

    Responda apenas com o nome do agente.
    \"\"\"
    resposta = __llm.invoke(prompt)
    return str(resposta.content).strip()

async def executar_supervisor(texto_usuario: str) -> str:
    agente = classificar_pergunta(texto_usuario)

    if agente == "cartao_credito":
        resultado = agente_cartao_credito.invoke(
            {"messages": [HumanMessage(content=texto_usuario)]}
        )
    elif agente == "abrir_conta":
        resultado = agente_abertura_conta.invoke(
            {"messages": [HumanMessage(content=texto_usuario)]}
        )
    else:
        resultado = {"messages": [HumanMessage(content="Não consegui entender sua solicitação")]}

    mensagem_ia = resultado["messages"][-1]
    return str(mensagem_ia.content)
