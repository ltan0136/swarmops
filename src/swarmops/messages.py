"""Helpers for reading LangChain messages."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def message_text(message: BaseMessage) -> str:
    """Return the plain-text content of a message, flattening multimodal content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def current_turn(messages: Sequence[BaseMessage]) -> tuple[HumanMessage | None, list[BaseMessage]]:
    """Split history into (latest human message, messages produced after it)."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return message, list(messages[index + 1 :])
    return None, []


def final_answer(messages: Sequence[BaseMessage]) -> str:
    """Return the last AI reply in the current turn that is addressed to the user."""
    _, turn = current_turn(messages)
    for message in reversed(turn):
        if isinstance(message, AIMessage) and not message.tool_calls and message_text(message):
            return message_text(message)
    return ""
