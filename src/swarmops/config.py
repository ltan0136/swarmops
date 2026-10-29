"""Application settings, loaded from environment variables or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["fake", "openai_compatible"]


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to an upper-case env var (e.g. LLM_MODEL)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    # "fake" runs a deterministic offline model so the whole stack works without an API key.
    # "openai_compatible" covers Ollama, vLLM, DeepSeek, DashScope (Qwen), OpenAI, etc.
    llm_provider: LLMProvider = "fake"
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    # --- Persistence ---------------------------------------------------------
    # Postgres URL for the LangGraph checkpointer. Empty means in-memory (lost on restart).
    database_url: str | None = None
    db_pool_max_size: int = Field(default=10, ge=1, le=100)

    # --- Swarm runtime -------------------------------------------------------
    # Upper bound on graph steps per run; stops runaway agent loops.
    recursion_limit: int = Field(default=25, ge=5, le=200)

    log_level: str = "INFO"

    @field_validator("database_url", "llm_base_url", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _empty_key_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_llm_endpoint(self) -> Settings:
        if self.llm_provider == "openai_compatible" and not (self.llm_base_url or self.llm_api_key):
            raise ValueError(
                "LLM_PROVIDER=openai_compatible needs LLM_BASE_URL (e.g. a local Ollama server) "
                "or LLM_API_KEY (for the default OpenAI endpoint)."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return process-wide settings (cached after the first call)."""
    return Settings()
