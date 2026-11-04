"""Run episodes through the swarm: live (to record them) or replayed (to compare variants)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from swarmops.config import Settings
from swarmops.evals.episodes import Episode, Expected, extract_tool_returns
from swarmops.evals.replay import MissPolicy, ReplayToolbox
from swarmops.llm import get_chat_model
from swarmops.messages import current_turn, final_answer
from swarmops.swarm.graph import ModelFactory, build_swarm
from swarmops.swarm.prompts import DEFAULT_PROMPTS, PromptSet
from swarmops.swarm.tools import WORK_TOOLS

# Cell labels follow the 2x2 grid: rows are prompts, columns are traffic.
#              old traffic   new traffic
#  old prompt       A             B
#  new prompt       C             D
CELLS: dict[str, tuple[str, str]] = {
    "A": ("old", "old"),
    "B": ("old", "new"),
    "C": ("new", "old"),
    "D": ("new", "new"),
}


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    answer: str
    scored: bool  # False when the episode has no expert-defined expectation
    success: bool
    steps: int  # AI messages in the turn
    handoffs: int
    replay_hits: int
    replay_reused: int
    replay_misses: int
    error: str = ""


async def _invoke(
    episode: Episode,
    settings: Settings,
    prompts: PromptSet,
    tools: Sequence[BaseTool],
    model_factory: ModelFactory,
) -> tuple[list[BaseMessage], str]:
    """Run one fresh, single-turn thread (no checkpointer: nothing outlives the episode).

    Returns (messages produced after the user's message, error label or "").
    """
    graph = build_swarm(
        settings, prompts=prompts, work_tools=tools, model_factory=model_factory
    ).compile()
    config = {"recursion_limit": settings.recursion_limit}
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(content=episode.input)]}, config)
    except GraphRecursionError:
        return [], "recursion_limit"
    _, turn = current_turn(result["messages"])
    return turn, ""


async def run_replayed(
    episode: Episode,
    settings: Settings,
    prompts: PromptSet,
    *,
    live_tools: Sequence[BaseTool] = WORK_TOOLS,
    model_factory: ModelFactory = get_chat_model,
    on_miss: MissPolicy = "live",
) -> EpisodeResult:
    """Run one episode with its tools replayed from the recording, and score the answer."""
    toolbox = ReplayToolbox(episode.tool_returns, live_tools, on_miss)
    turn, error = await _invoke(episode, settings, prompts, toolbox.tools(), model_factory)
    answer = final_answer([HumanMessage(content=episode.input), *turn])
    if not answer and not error:
        error = "no_answer"
    ai_messages = [m for m in turn if isinstance(m, AIMessage)]
    return EpisodeResult(
        episode_id=episode.id,
        answer=answer,
        scored=episode.expected.defined,
        success=episode.expected.defined and not error and episode.expected.check(answer),
        steps=len(ai_messages),
        handoffs=sum(
            call["name"].startswith("transfer_to_") for m in ai_messages for call in m.tool_calls
        ),
        replay_hits=toolbox.stats.hits,
        replay_reused=toolbox.stats.reused,
        replay_misses=len(toolbox.stats.misses),
        error=error,
    )


async def run_cell(
    episodes: Sequence[Episode],
    settings: Settings,
    prompts: PromptSet,
    *,
    concurrency: int = 4,
    **kwargs,
) -> list[EpisodeResult]:
    """Run every episode of one traffic set under one prompt set, `concurrency` at a time."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(episode: Episode) -> EpisodeResult:
        async with semaphore:
            return await run_replayed(episode, settings, prompts, **kwargs)

    return list(await asyncio.gather(*(one(e) for e in episodes)))


async def run_grid(
    traffic: dict[str, Sequence[Episode]],
    prompts: dict[str, PromptSet],
    settings: Settings,
    **kwargs,
) -> dict[str, list[EpisodeResult]]:
    """Run the four cells A-D. `traffic` and `prompts` are keyed "old" and "new"."""
    results: dict[str, list[EpisodeResult]] = {}
    for cell, (prompt_key, traffic_key) in CELLS.items():
        results[cell] = await run_cell(
            traffic[traffic_key], settings, prompts[prompt_key], **kwargs
        )
    return results


async def record(
    requests: Sequence[tuple[str, str, Expected, str]],
    settings: Settings,
    *,
    prompts: PromptSet = DEFAULT_PROMPTS,
    live_tools: Sequence[BaseTool] = WORK_TOOLS,
    model_factory: ModelFactory = get_chat_model,
) -> list[Episode]:
    """Run (id, input, expected, note) requests with live tools and keep what the tools returned."""
    names = {tool.name for tool in live_tools}
    episodes = []
    for episode_id, text, expected, note in requests:
        probe = Episode(id=episode_id, input=text)
        turn, error = await _invoke(probe, settings, prompts, live_tools, model_factory)
        if error == "recursion_limit":
            raise RuntimeError(f"{episode_id}: the swarm hit RECURSION_LIMIT while recording")
        episodes.append(
            Episode(
                id=episode_id,
                input=text,
                tool_returns=extract_tool_returns(turn, names),
                expected=expected,
                note=note,
            )
        )
    return episodes
