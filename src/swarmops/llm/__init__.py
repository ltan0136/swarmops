"""Chat model factory. Swap providers through configuration, not code."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from swarmops.config import Settings
from swarmops.llm.fake import ScriptedSwarmModel
from swarmops.roles import AgentName

__all__ = ["ScriptedSwarmModel", "get_chat_model"]


def get_chat_model(role: AgentName, settings: Settings) -> BaseChatModel:
    """Return the chat model one swarm agent should use."""
    if settings.llm_provider == "fake":
        return ScriptedSwarmModel(role=role)

    # Local servers such as Ollama or vLLM accept any non-empty key.
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
