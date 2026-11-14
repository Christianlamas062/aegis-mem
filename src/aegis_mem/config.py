"""Typed configuration loaded from environment variables via Pydantic Settings."""

from __future__ import annotations

from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AegisSettings(BaseSettings):
    """Centralized settings for Qdrant, LLM providers, and SQLite telemetry."""

    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vector Store (Qdrant)
    qdrant_host: str = Field(default="localhost", description="Qdrant server hostname")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_in_memory: bool = Field(
        default=False,
        description="Use local in-memory Qdrant (:memory:) for zero-dependency testing",
    )
    collection_name: str = Field(
        default="aegis_procedural_rules",
        description="Default Qdrant collection name for procedural memories",
    )

    # Primary LLM Provider (OpenAI-compatible or local endpoints)
    llm_provider: str = Field(default="openai", description="Mem0 LLM provider identifier")
    llm_model: str = Field(default="gpt-4o-mini", description="Model identifier for distillation")
    llm_api_key: str = Field(default="", description="API key for LLM provider")
    llm_base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL (e.g. Ollama, vLLM, OpenRouter)",
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Embedder Provider
    embedder_provider: str = Field(default="openai", description="Mem0 embedder provider")
    embedder_model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )

    # Telemetry & Audit
    telemetry_db_path: str = Field(
        default=":memory:",
        description="Path to SQLite database for latency, recall utility, and reflection audit logs",
    )

    # Cloud API Server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)

    def build_mem0_config(self) -> dict[str, Any]:
        """Construct the dictionary expected by `mem0.Memory.from_config`."""
        qdrant_cfg: dict[str, Any] = {
            "collection_name": self.collection_name,
        }
        if self.qdrant_in_memory:
            qdrant_cfg["location"] = ":memory:"
        else:
            qdrant_cfg["host"] = self.qdrant_host
            qdrant_cfg["port"] = self.qdrant_port

        llm_cfg: dict[str, Any] = {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "api_key": self.llm_api_key,
        }
        if self.llm_base_url:
            llm_cfg["openai_base_url"] = self.llm_base_url

        embedder_cfg: dict[str, Any] = {
            "model": self.embedder_model,
            "api_key": self.llm_api_key,
        }
        if self.llm_base_url:
            embedder_cfg["openai_base_url"] = self.llm_base_url

        return {
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_cfg,
            },
            "llm": {
                "provider": self.llm_provider,
                "config": llm_cfg,
            },
            "embedder": {
                "provider": self.embedder_provider,
                "config": embedder_cfg,
            },
        }


settings = AegisSettings()

