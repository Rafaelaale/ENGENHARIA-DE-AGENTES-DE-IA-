from starlette.applications import Starlette
from starlette.routing import Mount
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from executor import CartaoCreditoExecutor

skill = AgentSkill(
    id="cartao_credito",
    name="Cartao de Credito MDBank",
    description="Especialista em informacoes, solicitacao e limites de cartoes de credito (MDZone, GoldSilver e Platinum).",
    tags=[
        "cartao",
        "cartao de credito",
        "limite",
        "fatura",
        "platinum",
        "goldsilver",
        "mdzone",
    ],
)

agent_interface = AgentInterface(
    url="http://cartao_credito_agent:8000/",
    protocol_version="1.0",
    protocol_binding="JSONRPC",
)

agent_card = AgentCard(
    name="Agente de Cartao de Credito MDBank",
    description="Especialista em cartoes de credito do MDBank.",
    supported_interfaces=[agent_interface],
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[skill],
    version="1.0.0",
    capabilities=AgentCapabilities(),
)

handler = DefaultRequestHandler(
    agent_executor=CartaoCreditoExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

routes = []
routes.extend(create_agent_card_routes(agent_card=agent_card))
routes.extend(create_jsonrpc_routes(request_handler=handler, rpc_url="/"))

app = Starlette(routes=routes)
