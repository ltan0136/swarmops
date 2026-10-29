from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from swarmops.messages import current_turn, final_answer


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}


async def _ask(graph: CompiledStateGraph, thread_id: str, text: str) -> dict:
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": text}]}, _config(thread_id)
    )


async def test_swarm_runs_planner_executor_reviewer(graph: CompiledStateGraph) -> None:
    result = await _ask(graph, "t-basic", "What is (12 + 3) * 4?")

    _, turn = current_turn(result["messages"])
    speakers = [m.name for m in turn if isinstance(m, AIMessage)]
    assert speakers == ["planner", "executor", "executor", "reviewer"]

    tool_results = {m.name: m.content for m in turn if isinstance(m, ToolMessage)}
    assert tool_results["calculator"] == "60"
    assert "60" in final_answer(result["messages"])
    assert result["active_agent"] == "reviewer"


async def test_second_turn_routes_back_through_planner(graph: CompiledStateGraph) -> None:
    await _ask(graph, "t-multi", "What is 6 * 7?")
    result = await _ask(graph, "t-multi", "And 2 ** 10?")

    _, turn = current_turn(result["messages"])
    speakers = [m.name for m in turn if isinstance(m, AIMessage)]
    # The reviewer was active at the end of turn 1, so it receives turn 2 and hands off.
    assert speakers[:2] == ["reviewer", "planner"]
    assert "1024" in final_answer(result["messages"])
    # History from turn 1 is still in the thread.
    assert any(isinstance(m, ToolMessage) and m.content == "42" for m in result["messages"])


async def test_threads_are_isolated(graph: CompiledStateGraph) -> None:
    await _ask(graph, "t-a", "What is 1 + 1?")
    result = await _ask(graph, "t-b", "What is 2 + 2?")
    assert all(m.content != "2" for m in result["messages"] if isinstance(m, ToolMessage))


async def test_tool_error_is_reported_not_raised(graph: CompiledStateGraph) -> None:
    result = await _ask(graph, "t-error", "What is 1 / 0?")
    answer = final_answer(result["messages"])
    assert "division by zero" in answer
