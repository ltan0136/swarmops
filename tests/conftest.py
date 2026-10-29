from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from swarmops.api.app import create_app
from swarmops.config import Settings
from swarmops.swarm.graph import compile_swarm


@pytest.fixture
def settings() -> Settings:
    """Offline settings: scripted model, in-memory checkpointer, no .env file."""
    return Settings(_env_file=None, llm_provider="fake", database_url=None)


@pytest.fixture
def graph(settings: Settings) -> CompiledStateGraph:
    return compile_swarm(settings, InMemorySaver())


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    # ASGITransport does not run lifespan events, so enter the lifespan explicitly.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest.fixture
def postgres_url() -> str:
    url = os.environ.get("SWARMOPS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set SWARMOPS_TEST_DATABASE_URL to run Postgres integration tests")
    return url
