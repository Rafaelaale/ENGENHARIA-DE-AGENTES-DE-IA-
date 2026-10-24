import logging
import uuid

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message, Part, Role
from agent.abrir_conta import run_agent

logger = logging.getLogger("a2a")

class AbrirContaExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        logger.info("agent.execute.abrir_conta")

        user_text = context.get_user_input()

        response_agent = await run_agent(mensagem=user_text)

        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.ROLE_AGENT,
            parts=[Part(text=str(response_agent))],
        )

        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.info("agent.cancel.abrir_conta")