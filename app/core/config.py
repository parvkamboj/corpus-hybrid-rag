from functools import lru_cache

from arq.connections import RedisSettings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    api_key: str = Field(default="dev-secret-key")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    environment: str = Field(default="development")
    log_level: str = Field(default="DEBUG")

    # PostgreSQL
    database_url: str = Field(
        default="postgresql+asyncpg://corpus:corpus@localhost:5432/corpus_rag"
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="corpus")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    arq_redis_url: str = Field(default="redis://localhost:6379/1")

    # OpenAI / LiteLLM
    openai_api_key: str = Field(default="")
    llm_model: str = Field(default="gpt-4o-mini")

    # Embeddings — BAAI/bge-m3
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_batch_size: int = Field(default=32)
    embedding_dim: int = Field(default=1024)

    # Reranking
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Chunking
    chunk_size: int = Field(default=512)   # max tokens per chunk (recursive fallback)
    chunk_overlap: int = Field(default=50)

    # File storage
    uploads_dir: str = Field(default="uploads")

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    def arq_redis_settings(self) -> RedisSettings:
        from urllib.parse import urlparse

        parsed = urlparse(self.arq_redis_url)
        return RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            database=int(parsed.path.lstrip("/")) if parsed.path.lstrip("/").isdigit() else 0,
            password=parsed.password,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
