"""Pick and open the LangGraph checkpointer (Postgres in deployments, memory in dev/tests)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from swarmops.config import Settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """Yield a ready-to-use checkpointer and close its resources on exit."""
    if not settings.database_url:
        logger.warning("DATABASE_URL is not set: using in-memory checkpointer (lost on restart).")
        yield InMemorySaver()
        return

    # autocommit + prepare_threshold=0 + dict_row are what AsyncPostgresSaver expects.
    async with AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=settings.db_pool_max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    ) as pool:
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()  # idempotent: creates or migrates checkpoint tables
        logger.info("Postgres checkpointer ready (pool max_size=%d).", settings.db_pool_max_size)
        yield saver
