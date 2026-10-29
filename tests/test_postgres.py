"""Integration test: thread state survives a full "process restart" when backed by Postgres.

Skipped unless SWARMOPS_TEST_DATABASE_URL is set (CI sets it via a Postgres service).
"""

from __future__ import annotations

import uuid

from langchain_core.messages import ToolMessage

from swarmops.checkpoint import open_checkpointer
from swarmops.config import Settings
from swarmops.messages import final_answer
from swarmops.swarm.graph import compile_swarm


async def test_state_survives_restart(postgres_url: str) -> None:
    settings = Settings(_env_file=None, llm_provider="fake", database_url=postgres_url)
    thread_id = f"pg-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    # First "process": run one turn, then close the pool.
    async with open_checkpointer(settings) as checkpointer:
        graph = compile_swarm(settings, checkpointer)
        await graph.ainvoke({"messages": [{"role": "user", "content": "What is 8 * 8?"}]}, config)

    # Second "process": a fresh pool and graph continue the same thread.
    async with open_checkpointer(settings) as checkpointer:
        graph = compile_swarm(settings, checkpointer)
        snapshot = await graph.aget_state(config)
        assert snapshot.values["active_agent"] == "reviewer"

        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "And 64 + 1?"}]}, config
        )

    tool_outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "64" in tool_outputs and "65" in tool_outputs
    assert "65" in final_answer(result["messages"])
