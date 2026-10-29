"""FastAPI application factory.

Run locally with:  uvicorn swarmops.api.app:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from swarmops import __version__
from swarmops.api.routes import ThreadLocks, router
from swarmops.checkpoint import open_checkpointer
from swarmops.config import Settings, get_settings
from swarmops.swarm.graph import compile_swarm

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass `settings` in tests; otherwise they come from env / .env."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        configure_logging(resolved.log_level)
        # The checkpointer (and its DB pool) lives exactly as long as the process.
        async with open_checkpointer(resolved) as checkpointer:
            app.state.settings = resolved
            app.state.graph = compile_swarm(resolved, checkpointer)
            app.state.thread_locks = ThreadLocks()
            logger.info(
                "SwarmOps %s started (llm_provider=%s, checkpointer=%s)",
                __version__,
                resolved.llm_provider,
                type(checkpointer).__name__,
            )
            yield

    app = FastAPI(
        title="SwarmOps",
        version=__version__,
        description="A LangGraph agent swarm served by an async API.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
