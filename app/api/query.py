import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.deps import verify_api_key
from app.core.limiter import limiter
from app.generation.synthesizer import AnswerSynthesizer
from app.models.schemas import (
    MetadataFilter,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.query.pipeline import run_pipeline
from app.retrieval.base import MetadataFilter as RetrieverFilter
from app.retrieval.base import SearchResult

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

AuthDep = Annotated[None, Depends(verify_api_key)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(
    body: QueryRequest,
    request: Request,
    settings: SettingsDep,
    _: AuthDep,
) -> QueryResponse:
    """Retrieve chunks and generate a grounded answer with source citations."""
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

    synthesizer = AnswerSynthesizer(request.app.state.llm)
    answer = await synthesizer.synthesize(result.effective_query, result.output.results)

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "query complete",
        strategy=result.effective_strategy,
        result_count=len(result.output.results),
        latency_ms=round(latency_ms, 1),
    )

    combined_debug = {**result.pipeline_debug, **result.output.debug} if body.debug else None

    return QueryResponse(
        query=body.query,
        answer=answer,
        sources=[_to_source(r) for r in result.output.results],
        strategy=result.effective_strategy,
        result_count=len(result.output.results),
        latency_ms=round(latency_ms, 2),
        debug=combined_debug,
    )


@router.post("/stream")
@limiter.limit("20/minute")
async def query_stream(
    body: QueryRequest,
    request: Request,
    settings: SettingsDep,
    _: AuthDep,
) -> StreamingResponse:
    """Same as /query but streams tokens as SSE. Final event includes sources and latency."""

    async def _generate() -> AsyncGenerator[str, None]:
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

        synthesizer = AnswerSynthesizer(request.app.state.llm)
        async for token in synthesizer.stream(result.effective_query, result.output.results):
            yield f"data: {json.dumps({'token': token})}\n\n"

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        sources = [_to_source(r).model_dump() for r in result.output.results]
        final: dict[str, object] = {
            "done": True,
            "sources": sources,
            "strategy": result.effective_strategy,
            "result_count": len(result.output.results),
            "latency_ms": latency_ms,
        }
        if body.debug:
            final["debug"] = {**result.pipeline_debug, **result.output.debug}
        yield f"data: {json.dumps(final)}\n\n"

        logger.info(
            "stream complete",
            strategy=result.effective_strategy,
            result_count=len(result.output.results),
            latency_ms=latency_ms,
        )

    return StreamingResponse(_generate(), media_type="text/event-stream")


def _to_filter(f: MetadataFilter | None) -> RetrieverFilter | None:
    if f is None:
        return None
    return RetrieverFilter(doc_id=f.doc_id)


def _to_source(r: SearchResult) -> SourceItem:
    return SourceItem(
        chunk_id=r.chunk_id,
        doc_id=r.doc_id,
        filename=r.filename,
        page_numbers=r.page_numbers,
        section_header=r.section_header,
        score=r.score,
        rank=r.rank,
    )
