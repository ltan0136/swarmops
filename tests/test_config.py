from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from swarmops.config import Settings
from swarmops.llm import ScriptedSwarmModel, get_chat_model
from swarmops.swarm.graph import compile_swarm


def test_defaults_are_offline() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "fake"
    assert settings.database_url is None
    assert isinstance(get_chat_model("planner", settings), ScriptedSwarmModel)


def test_empty_strings_mean_unset() -> None:
    settings = Settings(_env_file=None, database_url="  ", llm_base_url="", llm_api_key="")
    assert settings.database_url is None
    assert settings.llm_base_url is None
    assert settings.llm_api_key is None


def test_openai_compatible_requires_endpoint_or_key() -> None:
    with pytest.raises(ValidationError, match="LLM_BASE_URL"):
        Settings(_env_file=None, llm_provider="openai_compatible")


def test_openai_compatible_builds_real_model_and_graph() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen2.5:7b",
    )
    model = get_chat_model("executor", settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen2.5:7b"
    # Compiling binds the tools to the model without making any network call.
    assert compile_swarm(settings, InMemorySaver()) is not None
