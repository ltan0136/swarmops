"""Build the planner -> executor -> reviewer swarm with LangGraph."""

from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_swarm import create_handoff_tool, create_swarm

from swarmops.config import Settings
from swarmops.llm import get_chat_model
from swarmops.roles import EXECUTOR, PLANNER, REVIEWER
from swarmops.swarm.prompts import EXECUTOR_PROMPT, PLANNER_PROMPT, REVIEWER_PROMPT
from swarmops.swarm.tools import WORK_TOOLS


def build_swarm(settings: Settings) -> StateGraph:
    """Return the uncompiled swarm graph. Each agent can only hand off along allowed edges."""
    planner = create_agent(
        get_chat_model(PLANNER, settings),
        tools=[
            create_handoff_tool(
                agent_name=EXECUTOR, description="Hand the plan to the executor to carry out."
            ),
        ],
        system_prompt=PLANNER_PROMPT,
        name=PLANNER,
    )
    executor = create_agent(
        get_chat_model(EXECUTOR, settings),
        tools=[
            *WORK_TOOLS,
            create_handoff_tool(
                agent_name=REVIEWER, description="Send the finished results to the reviewer."
            ),
        ],
        system_prompt=EXECUTOR_PROMPT,
        name=EXECUTOR,
    )
    reviewer = create_agent(
        get_chat_model(REVIEWER, settings),
        tools=[
            create_handoff_tool(
                agent_name=PLANNER, description="Ask the planner to plan a new user request."
            ),
            create_handoff_tool(
                agent_name=EXECUTOR, description="Send the work back to the executor to fix."
            ),
        ],
        system_prompt=REVIEWER_PROMPT,
        name=REVIEWER,
    )
    return create_swarm([planner, executor, reviewer], default_active_agent=PLANNER)


def compile_swarm(settings: Settings, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Compile the swarm with a checkpointer so every thread's state is persisted."""
    return build_swarm(settings).compile(checkpointer=checkpointer)
