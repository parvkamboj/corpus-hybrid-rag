import asyncio

import structlog

from app.core.config import Settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """bge-m3 embeddings. Model is lazy-loaded on first call.

    Using a lock here because sentence-transformers isn't thread-safe during
    the initial model load — ran into issues without it.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: object = None
        self._lock = asyncio.Lock()

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model", model=self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
            logger.info("embedding model loaded")
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async with self._lock:
            model = self._load()

        def _encode() -> list[list[float]]:
            from sentence_transformers import SentenceTransformer

            m: SentenceTransformer = model  # type: ignore[assignment]
            return m.encode(
                texts,
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

        return await asyncio.get_event_loop().run_in_executor(None, _encode)
