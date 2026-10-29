"""A deterministic, offline chat model that plays one role of the swarm.

It reads the conversation since the latest human message and emits the next step of a
planner -> executor -> reviewer workflow, including real tool calls and handoffs. The
LangGraph runtime, tools, checkpointer and API all run for real; only the "thinking" is
scripted. This keeps demos, tests and CI free of API keys and fully reproducible.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from swarmops.messages import current_turn, message_text
from swarmops.roles import EXECUTOR, PLANNER, REVIEWER, AgentName, handoff_tool_name
from swarmops.swarm.tools import WORK_TOOL_NAMES

_EXPRESSION_RE = re.compile(r"[\d.\s+\-*/%()]+")


def extract_expression(text: str) -> str | None:
    """Pick the longest arithmetic-looking span in `text`, e.g. '(12 + 3) * 4'."""
    candidates = [match.group().strip() for match in _EXPRESSION_RE.finditer(text)]
    candidates = [c for c in candidates if any(ch.isdigit() for ch in c)]
    return max(candidates, key=len) if candidates else None


def _tool_call(name: str, args: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(
        name=name, args=args or {}, id=f"call_{uuid.uuid4().hex[:12]}", type="tool_call"
    )


class ScriptedSwarmModel(BaseChatModel):
    """Offline stand-in for an LLM, specialised to one swarm role."""

    role: AgentName

    @property
    def _llm_type(self) -> str:
        return "scripted-swarm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"role": self.role}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The script already knows the tool names it may call, so binding is a no-op.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        human, turn = current_turn(messages)
        request = message_text(human) if human else ""
        step = {PLANNER: self._plan, EXECUTOR: self._execute, REVIEWER: self._review}[self.role]
        return ChatResult(generations=[ChatGeneration(message=step(request, turn))])

    @staticmethod
    def _plan(request: str, turn: list[BaseMessage]) -> AIMessage:
        expression = extract_expression(request)
        if expression:
            first = f"1. executor: evaluate `{expression}` with the calculator tool."
        else:
            first = "1. executor: look up the current UTC time with get_current_time."
        plan = f"Plan:\n{first}\n2. reviewer: check the result and give the final answer."
        return AIMessage(content=plan, tool_calls=[_tool_call(handoff_tool_name(EXECUTOR))])

    @staticmethod
    def _execute(request: str, turn: list[BaseMessage]) -> AIMessage:
        last = turn[-1] if turn else None
        if isinstance(last, ToolMessage) and last.name in WORK_TOOL_NAMES:
            return AIMessage(
                content=f"{last.name} returned: {message_text(last)}",
                tool_calls=[_tool_call(handoff_tool_name(REVIEWER))],
            )
        expression = extract_expression(request)
        if expression:
            return AIMessage(
                content="", tool_calls=[_tool_call("calculator", {"expression": expression})]
            )
        return AIMessage(
            content="", tool_calls=[_tool_call("get_current_time", {"timezone": "UTC"})]
        )

    @staticmethod
    def _review(request: str, turn: list[BaseMessage]) -> AIMessage:
        results = [m for m in turn if isinstance(m, ToolMessage) and m.name in WORK_TOOL_NAMES]
        if not results:
            # A new user request arrived while the reviewer was the active agent.
            return AIMessage(content="", tool_calls=[_tool_call(handoff_tool_name(PLANNER))])
        result = results[-1]
        text = message_text(result)
        if text.startswith("Error"):
            return AIMessage(content=f"I could not complete the task. {result.name} said: {text}")
        return AIMessage(
            content=f"Final answer: {text}\n(Checked the executor's {result.name} result.)"
        )
