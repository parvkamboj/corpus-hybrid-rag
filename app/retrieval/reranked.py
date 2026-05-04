import asyncio
from typing import Any

import structlog

from app.core.config import Settings
from app.retrieval.base import MetadataFilter, SearchOutput, SearchResult

logger = structlog.get_logger(__name__)

_RERANK_MULTIPLIER = 3


class CrossEncoderService:
    """Wraps the cross-encoder model. Lazy-loads on first use, runs inference in a thread pool."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.reranker_model
        self._model: object = None
        self._lock = asyncio.Lock()

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("cross-encoder loaded")
        return self._model

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        async with self._lock:
            model = self._load()

        pairs = [(query, p) for p in passages]

        def _predict() -> list[float]:
            from sentence_transformers import CrossEncoder

            m: CrossEncoder = model  # type: ignore[assignment]
            return m.predict(pairs).tolist()  # type: ignore[arg-type]

        return await asyncio.get_event_loop().run_in_executor(None, _predict)


class RerankedRetriever:
    def __init__(self, base: Any, reranker: CrossEncoderService) -> None:
        self._base = base
        self._reranker = reranker

    async def search(
        self,
        query: str,
        top_k: int,
        filters: MetadataFilter | None,
        debug: bool,
    ) -> SearchOutput:
        fetch_k = top_k * _RERANK_MULTIPLIER
        base_out: SearchOutput = await self._base.search(query, fetch_k, filters, debug)

        candidates = base_out.results
        if not candidates:
            return SearchOutput(results=[], debug=base_out.debug)

        scores = await self._reranker.rerank(query, [c.content for c in candidates])

        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for i, (result, score) in enumerate(reranked):
            results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    content=result.content,
                    score=float(score),
                    page_numbers=result.page_numbers,
                    section_header=result.section_header,
                    filename=result.filename,
                    rank=i,
                )
            )

        dbg = dict(base_out.debug)
        if debug:
            dbg["pre_rerank_order"] = [
                {"chunk_id": c.chunk_id, "score": round(c.score, 8)} for c in candidates
            ]
            dbg["reranker_scores"] = [
                {"chunk_id": c.chunk_id, "cross_encoder_score": round(float(s), 6)}
                for c, s in zip(candidates, scores, strict=True)
            ]

        return SearchOutput(results=results, debug=dbg)
