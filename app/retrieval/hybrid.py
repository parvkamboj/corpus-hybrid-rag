import asyncio
from typing import Any

from app.retrieval.base import MetadataFilter, SearchOutput, SearchResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever

# k=60 is the standard RRF default. tried k=30 and results felt worse for short queries.
_RRF_K = 60

# fetch 3x candidates before fusion so there's enough overlap to rerank from
_OVERSAMPLE = 3


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever) -> None:
        self._dense = dense
        self._sparse = sparse

    async def search(
        self,
        query: str,
        top_k: int,
        filters: MetadataFilter | None,
        debug: bool,
    ) -> SearchOutput:
        fetch_k = top_k * _OVERSAMPLE

        dense_out, sparse_out = await asyncio.gather(
            self._dense.search(query, fetch_k, filters, debug),
            self._sparse.search(query, fetch_k, filters, debug),
        )

        results, rrf_scores = _rrf(dense_out.results, sparse_out.results, top_k)

        dbg: dict[str, Any] = {}
        if debug:
            dbg.update(
                {
                    "dense_candidates": [
                        {"chunk_id": r.chunk_id, "score": round(r.score, 6), "rank": r.rank}
                        for r in dense_out.results
                    ],
                    "sparse_candidates": [
                        {"chunk_id": r.chunk_id, "score": round(r.score, 6), "rank": r.rank}
                        for r in sparse_out.results
                    ],
                    "rrf_scores": [
                        {"chunk_id": cid, "rrf_score": round(s, 8)}
                        for cid, s in rrf_scores
                    ],
                }
            )

        return SearchOutput(results=results, debug=dbg)


def _rrf(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    top_k: int,
) -> tuple[list[SearchResult], list[tuple[str, float]]]:
    scores: dict[str, float] = {}
    chunks: dict[str, SearchResult] = {}

    for rank, result in enumerate(dense, 1):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        chunks[result.chunk_id] = result

    for rank, result in enumerate(sparse, 1):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        chunks.setdefault(result.chunk_id, result)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[SearchResult] = []
    for i, (chunk_id, rrf_score) in enumerate(ranked):
        r = chunks[chunk_id]
        results.append(
            SearchResult(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                content=r.content,
                score=rrf_score,
                page_numbers=r.page_numbers,
                section_header=r.section_header,
                filename=r.filename,
                rank=i,
            )
        )

    return results, ranked
