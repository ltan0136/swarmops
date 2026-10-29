"""Agent names in the swarm. Node names and handoff tool names derive from these."""

from __future__ import annotations

from typing import Literal

AgentName = Literal["planner", "executor", "reviewer"]

PLANNER: AgentName = "planner"
EXECUTOR: AgentName = "executor"
REVIEWER: AgentName = "reviewer"

AGENT_NAMES: tuple[AgentName, ...] = (PLANNER, EXECUTOR, REVIEWER)


def handoff_tool_name(agent: AgentName) -> str:
    """Tool name that langgraph-swarm generates for a handoff to `agent`."""
    return f"transfer_to_{agent}"
