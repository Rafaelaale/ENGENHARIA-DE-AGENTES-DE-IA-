import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt="", checkpointer=None):
        return create_react_agent(model, tools=tools, prompt=system_prompt, checkpointer=checkpointer)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "ollama")
openai_api_base = os.getenv("OPENAI_API_BASE", "http://host.docker.internal:11434/v1")
model_name = os.getenv("MODEL_NAME", "llama3.2:1b")

llm = ChatOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model=model_name,
    temperature=0.3,
)

client = MultiServerMCPClient(
    {
        "conta": {
            "transport": "http",
            "url": "http://recursos:8000/mcp_gateway",
        }  # type: ignore
    }
)

memory = InMemorySaver()

async def build_agent():
    tools = await client.get_tools()

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=(
            "Voce e o assistente de abertura de contas do MDBank.\n\n"
            "Sua funcao e abrir contas para clientes.\n\n"
            "Fluxo para abrir conta:\n"
            "1. Sempre execute a tool 'criar_ou_buscar_conta' passando o cpf e o nome fornecidos.\n"
            "2. Apos executar a tool com sucesso, confirme ao cliente que a conta foi criada e informe os dados retornados.\n\n"
            "Regras:\n"
            "- Nunca invente dados.\n"
            "- Sempre use as tools para executar acoes reais.\n"
            "- Seja direto e objetivo."
        ),
        checkpointer=memory,
    )
    return agent

async def run_agent(mensagem: str, thread_id: str = "1"):
    agent = await build_agent()

    resultado = await agent.ainvoke(
        {"messages": [HumanMessage(content=mensagem)]},
        {"configurable": {"thread_id": thread_id}},
    )

    return resultado["messages"][-1].content
