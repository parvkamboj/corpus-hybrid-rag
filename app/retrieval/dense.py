from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.retrieval.base import MetadataFilter, SearchOutput, SearchResult

from ._filters import build_qdrant_filter


class DenseRetriever:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: EmbeddingService,
        settings: Settings,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._settings = settings

    async def search(
        self,
        query: str,
        top_k: int,
        filters: MetadataFilter | None,
        debug: bool,
    ) -> SearchOutput:
        vector = (await self._embedder.embed([query]))[0]
        hits = await self._qdrant.search(  # type: ignore[attr-defined]
            collection_name=self._settings.qdrant_collection,
            query_vector=NamedVector(name="dense", vector=vector),
            query_filter=build_qdrant_filter(filters),
            limit=top_k,
            with_payload=True,
        )

        results = [_to_result(h, i) for i, h in enumerate(hits)]

        dbg: dict[str, Any] = {}
        if debug:
            dbg["vector_scores"] = [
                {"chunk_id": r.chunk_id, "score": round(r.score, 6)} for r in results
            ]

        return SearchOutput(results=results, debug=dbg)


def _to_result(hit: Any, rank: int) -> SearchResult:
    p = hit.payload or {}
    return SearchResult(
        chunk_id=str(hit.id),
        doc_id=p.get("doc_id", ""),
        content=p.get("content", ""),
        score=float(hit.score),
        page_numbers=p.get("page_numbers", []),
        section_header=p.get("section_header"),
        filename=p.get("filename", ""),
        rank=rank,
    )
