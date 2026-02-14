"""
Centralized configuration using Pydantic Settings.
All environment variables are validated and typed.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────
    # Groq Configuration (LLM Provider)
    # ─────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # ─────────────────────────────────────────────────────────────
    # Voyage AI Configuration (Embedding Provider)
    # ─────────────────────────────────────────────────────────────
    voyage_api_key: str
    voyage_embedding_model: str = "voyage-4-large"
    voyage_embedding_dim: int = 1024

    # ─────────────────────────────────────────────────────────────
    # Qdrant Configuration
    # ─────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "ideas"

    # ─────────────────────────────────────────────────────────────
    # RAG Configuration
    # ─────────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    @property
    def active_embedding_dim(self) -> int:
        """Return embedding dimension for Voyage AI."""
        return self.voyage_embedding_dim


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance for performance."""
    return Settings()
