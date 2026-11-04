"""Build the planner -> executor -> reviewer swarm with LangGraph."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_swarm import create_handoff_tool, create_swarm

from swarmops.config import Settings
from swarmops.llm import get_chat_model
from swarmops.roles import EXECUTOR, PLANNER, REVIEWER, AgentName
from swarmops.swarm.prompts import DEFAULT_PROMPTS, PromptSet
from swarmops.swarm.tools import WORK_TOOLS

ModelFactory = Callable[[AgentName, Settings], BaseChatModel]


def build_swarm(
    settings: Settings,
    *,
    prompts: PromptSet = DEFAULT_PROMPTS,
    work_tools: Sequence[BaseTool] = WORK_TOOLS,
    model_factory: ModelFactory = get_chat_model,
) -> StateGraph:
    """Return the uncompiled swarm graph. Each agent can only hand off along allowed edges.

    The keyword arguments are seams for experiments (see docs/adr/0004): evals swap in a
    prompt variant, replayed tools or a test model. Production code uses the defaults.
    """
    planner = create_agent(
        model_factory(PLANNER, settings),
        tools=[
            create_handoff_tool(
                agent_name=EXECUTOR, description="Hand the plan to the executor to carry out."
            ),
        ],
        system_prompt=prompts.planner,
        name=PLANNER,
    )
    executor = create_agent(
        model_factory(EXECUTOR, settings),
        tools=[
            *work_tools,
            create_handoff_tool(
                agent_name=REVIEWER, description="Send the finished results to the reviewer."
            ),
        ],
        system_prompt=prompts.executor,
        name=EXECUTOR,
    )
    reviewer = create_agent(
        model_factory(REVIEWER, settings),
        tools=[
            create_handoff_tool(
                agent_name=PLANNER, description="Ask the planner to plan a new user request."
            ),
            create_handoff_tool(
                agent_name=EXECUTOR, description="Send the work back to the executor to fix."
            ),
        ],
        system_prompt=prompts.reviewer,
        name=REVIEWER,
    )
    return create_swarm([planner, executor, reviewer], default_active_agent=PLANNER)


def compile_swarm(settings: Settings, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Compile the swarm with a checkpointer so every thread's state is persisted."""
    return build_swarm(settings).compile(checkpointer=checkpointer)
