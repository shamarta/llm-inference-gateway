"""Application settings loaded from environment variables (prefix GATEWAY_)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, validated configuration for the gateway."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    # Semantic cache
    similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0)
    qdrant_url: str = ":memory:"  # use "http://qdrant:6333" in Docker
    qdrant_collection: str = "semantic_cache"

    # Dynamic batching
    batch_timeout_ms: int = Field(default=20, gt=0)
    max_batch_size: int = Field(default=8, gt=0)

    # Mock upstream backend
    backend_base_latency_ms: int = Field(default=150, ge=0)
    backend_per_item_latency_ms: int = Field(default=10, ge=0)

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()