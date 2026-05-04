from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedSparseVector, SparseVector

from app.core.config import Settings
from app.retrieval.base import MetadataFilter, SearchOutput
from app.retrieval.bm25 import compute_query_sparse

from ._filters import build_qdrant_filter
from .dense import _to_result


class SparseRetriever:
    def __init__(self, qdrant: AsyncQdrantClient, settings: Settings) -> None:
        self._qdrant = qdrant
        self._settings = settings

    async def search(
        self,
        query: str,
        top_k: int,
        filters: MetadataFilter | None,
        debug: bool,
    ) -> SearchOutput:
        indices, values = compute_query_sparse(query)
        if not indices:
            return SearchOutput(results=[])

        hits = await self._qdrant.search(  # type: ignore[attr-defined]
            collection_name=self._settings.qdrant_collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=indices, values=values),
            ),
            query_filter=build_qdrant_filter(filters),
            limit=top_k,
            with_payload=True,
        )

        results = [_to_result(h, i) for i, h in enumerate(hits)]

        dbg: dict[str, Any] = {}
        if debug:
            dbg["sparse_scores"] = [
                {"chunk_id": r.chunk_id, "score": round(r.score, 6)} for r in results
            ]

        return SearchOutput(results=results, debug=dbg)
