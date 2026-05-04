import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.core.deps import verify_api_key
from app.core.limiter import limiter
from app.models.schemas import (
    MetadataFilter,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.query.pipeline import run_pipeline
from app.retrieval.base import MetadataFilter as RetrieverFilter
from app.retrieval.base import SearchResult

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

AuthDep = Annotated[None, Depends(verify_api_key)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/", response_model=SearchResponse)
@limiter.limit("60/minute")
async def search(
    body: SearchRequest,
    request: Request,
    settings: SettingsDep,
    _: AuthDep,
) -> SearchResponse:
    """Retrieve relevant chunks. Set debug=true to see intermediate scores."""
    t0 = time.perf_counter()

    result = await run_pipeline(
        query=body.query,
        strategy=body.strategy,
        top_k=body.top_k,
        filters=_to_filter(body.filters),
        use_hyde=body.use_hyde,
        decompose=body.decompose,
        history=body.history,
        debug=body.debug,
        llm=request.app.state.llm,
        qdrant=request.app.state.qdrant,
        embedder=request.app.state.embedder,
        reranker=request.app.state.reranker,
        settings=settings,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "search complete",
        strategy=result.effective_strategy,
        result_count=len(result.output.results),
        latency_ms=round(latency_ms, 1),
    )

    combined_debug = {**result.pipeline_debug, **result.output.debug} if body.debug else None

    return SearchResponse(
        query=body.query,
        strategy=result.effective_strategy,
        result_count=len(result.output.results),
        latency_ms=round(latency_ms, 2),
        results=[_to_item(r) for r in result.output.results],
        debug=combined_debug,
    )


def _to_filter(f: MetadataFilter | None) -> RetrieverFilter | None:
    if f is None:
        return None
    return RetrieverFilter(doc_id=f.doc_id)


def _to_item(r: SearchResult) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=r.chunk_id,
        doc_id=r.doc_id,
        content=r.content,
        score=r.score,
        page_numbers=r.page_numbers,
        section_header=r.section_header,
        filename=r.filename,
        rank=r.rank,
    )
