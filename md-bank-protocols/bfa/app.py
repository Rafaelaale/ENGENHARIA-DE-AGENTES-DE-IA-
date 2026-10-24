from fastapi import FastAPI
from contextlib import asynccontextmanager

from discovery import discover_agents
from mcp_discovery import discover_tools

from registry import (
    AGENT_REGISTRY,
    build_index,
    resolve_agent,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    agents = await discover_agents()
    tools = await discover_tools()

    AGENT_REGISTRY.update(agents)
    AGENT_REGISTRY.update(tools)

    build_index()

    print("Unified registry:", AGENT_REGISTRY)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/agents")
async def list_agents():
    return {
        skill_id: data
        for skill_id, data in AGENT_REGISTRY.items()
        if data.get("type") == "agent"
    }


@app.get("/tools")
async def list_tools():
    return {
        skill_id: data
        for skill_id, data in AGENT_REGISTRY.items()
        if data.get("type") == "tool"
    }


@app.get("/resolve")
async def resolve(query: str, filter_type: str | None = None):
    return resolve_agent(query, filter_type=filter_type)


@app.get("/health")
async def health():
    return {"status": "ok"}
