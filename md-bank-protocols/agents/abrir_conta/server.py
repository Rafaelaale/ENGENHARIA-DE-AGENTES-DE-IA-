from starlette.applications import Starlette
from starlette.routing import Mount
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from executor import AbrirContaExecutor

skill = AgentSkill(
    id="abrir_conta",
    name="Abertura de Conta MDBank",
    description="Especialista em abertura de contas correntes e documentos necessarios.",
    tags=[
        "conta",
        "abrir conta",
        "abertura",
        "corrente",
        "documentos",
    ],
)

agent_interface = AgentInterface(
    url="http://abrir_conta_agent:8000/",
    protocol_version="1.0",
    protocol_binding="JSONRPC",
)

agent_card = AgentCard(
    name="Agente de Abertura de Conta MDBank",
    description="Especialista em abertura de contas do MDBank.",
    supported_interfaces=[agent_interface],
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[skill],
    version="1.0.0",
    capabilities=AgentCapabilities(),
)

handler = DefaultRequestHandler(
    agent_executor=AbrirContaExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

routes = []
routes.extend(create_agent_card_routes(agent_card=agent_card))
routes.extend(create_jsonrpc_routes(request_handler=handler, rpc_url="/"))

app = Starlette(routes=routes)