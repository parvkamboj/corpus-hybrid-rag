from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from arq import create_pool
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.router import router
from app.core.config import get_settings
from app.core.database import DatabaseManager
from app.core.limiter import limiter
from app.core.logging import configure_logging
from app.core.qdrant import ensure_collection
from app.embeddings.service import EmbeddingService
from app.query.llm import LLMClient
from app.retrieval.reranked import CrossEncoderService

logger = structlog.get_logger(__name__)

_OPENAPI_TAGS = [
    {"name": "system", "description": "Health check."},
    {"name": "documents", "description": "Upload and manage PDFs. Ingestion is async."},
    {"name": "search", "description": "Retrieve chunks. Supports dense, sparse, hybrid, auto."},
    {"name": "query", "description": "Retrieve + generate a grounded answer. Streaming supported."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, development=settings.is_development)

    logger.info("starting corpus-rag", environment=settings.environment)

    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)

    app.state.db = DatabaseManager(settings)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.arq = await create_pool(settings.arq_redis_settings())
    app.state.embedder = EmbeddingService(settings)
    app.state.reranker = CrossEncoderService(settings)
    app.state.llm = LLMClient(settings)

    await ensure_collection(app.state.qdrant, settings)

    logger.info("connections established")

    yield

    await app.state.db.dispose()
    await app.state.redis.aclose()
    await app.state.qdrant.close()
    await app.state.arq.aclose()

    logger.info("connections closed, shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Corpus RAG",
        description="PDF knowledge assistant with hybrid retrieval and streaming generation.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=_OPENAPI_TAGS,
    )

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,  # type: ignore[arg-type]
    )

    app.include_router(router)
    return app


app = create_app()
