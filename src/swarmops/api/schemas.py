"""Request and response models for the HTTP API."""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from swarmops.messages import message_text

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000, description="The user's message.")


class MessageOut(BaseModel):
    type: str = Field(description="human | ai | tool")
    name: str | None = Field(default=None, description="Agent name for ai, tool name for tool.")
    content: str
    tool_calls: list[str] = Field(default_factory=list, description="Names of tools called.")

    @classmethod
    def from_message(cls, message: BaseMessage) -> MessageOut:
        return cls(
            type=message.type,
            name=message.name,
            content=message_text(message),
            tool_calls=[call["name"] for call in getattr(message, "tool_calls", None) or []],
        )


class ThreadCreated(BaseModel):
    thread_id: str


class RunResponse(BaseModel):
    thread_id: str
    answer: str
    active_agent: str | None
    steps: list[MessageOut] = Field(description="Messages produced during this run.")


class ThreadState(BaseModel):
    thread_id: str
    active_agent: str | None
    messages: list[MessageOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    checkpointer: str
