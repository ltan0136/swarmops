"""Command-line entry point: run the swarm once without starting the API server.

Usage:
    swarmops "What is (12 + 3) * 4?"
    swarmops --thread-id demo "and 2 ** 10?"   # reuses state when DATABASE_URL is set
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from langchain_core.messages import HumanMessage

from swarmops.checkpoint import open_checkpointer
from swarmops.config import get_settings
from swarmops.messages import current_turn, final_answer, message_text
from swarmops.swarm.graph import compile_swarm


async def run_once(message: str, thread_id: str) -> int:
    settings = get_settings()
    async with open_checkpointer(settings) as checkpointer:
        graph = compile_swarm(settings, checkpointer)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.recursion_limit,
        }
        result = await graph.ainvoke({"messages": [HumanMessage(content=message)]}, config)

    _, turn = current_turn(result["messages"])
    for step in turn:
        calls = [call["name"] for call in getattr(step, "tool_calls", None) or []]
        label = f"{step.type}:{step.name}" if step.name else step.type
        text = message_text(step).replace("\n", " ")
        suffix = f"  -> {', '.join(calls)}" if calls else ""
        print(f"[{label}] {text}{suffix}")
    print(f"\nthread_id: {thread_id}\nanswer: {final_answer(result['messages'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swarmops", description="Run the SwarmOps swarm once.")
    parser.add_argument("message", help="User message to send to the swarm.")
    parser.add_argument("--thread-id", default=None, help="Conversation id (default: random).")
    args = parser.parse_args(argv)
    return asyncio.run(run_once(args.message, args.thread_id or uuid.uuid4().hex))


if __name__ == "__main__":
    sys.exit(main())
