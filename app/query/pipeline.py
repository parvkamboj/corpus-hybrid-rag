from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.models.schemas import ChatMessage
from app.query.classify import QueryRouter
from app.query.decompose import QueryDecomposer
from app.query.hyde import HyDETransformer
from app.query.llm import LLMClient
from app.query.rewrite import ConversationQueryRewriter
from app.retrieval.base import MetadataFilter, SearchOutput
from app.retrieval.factory import Strategy, get_retriever
from app.retrieval.reranked import CrossEncoderService

_query_router = QueryRouter()


@dataclass
class PipelineResult:
    effective_query: str
    effective_strategy: Strategy
    effective_top_k: int
    output: SearchOutput
    pipeline_debug: dict[str, Any] = field(default_factory=dict)


async def run_pipeline(
    *,
    query: str,
    strategy: str,
    top_k: int,
    filters: MetadataFilter | None,
    use_hyde: bool,
    decompose: bool,
    history: list[ChatMessage] | None,
    debug: bool,
    llm: LLMClient,
    qdrant: object,
    embedder: EmbeddingService,
    reranker: CrossEncoderService,
    settings: Settings,
) -> PipelineResult:
    effective_strategy: Strategy
    effective_top_k = top_k
    dbg: dict[str, Any] = {}

    if history:
        rewriter = ConversationQueryRewriter(llm)
        history_dicts = [{"role": m.role, "content": m.content} for m in history]
        original = query
        query = await rewriter.rewrite(query, history_dicts)
        if debug and query != original:
            dbg["rewritten_query"] = query

    if strategy == "auto":
        intent, route = _query_router.route(query)
        effective_strategy = route.strategy
        effective_top_k = route.top_k
        if debug:
            dbg["auto_intent"] = intent
            dbg["auto_strategy"] = effective_strategy
            dbg["auto_top_k"] = effective_top_k
    else:
        effective_strategy = strategy  # type: ignore[assignment]

    if use_hyde:
        hyde = HyDETransformer(llm)
        hyde_query = await hyde.transform(query)
        if debug:
            dbg["hyde_passage"] = hyde_query
        query = hyde_query

    retriever = get_retriever(
        strategy=effective_strategy,
        qdrant=qdrant,  # type: ignore[arg-type]
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )

    if decompose:
        decomposer = QueryDecomposer(llm)
        output: SearchOutput = await decomposer.retrieve_and_merge(
            query=query,
            retriever=retriever,
            top_k=effective_top_k,
            filters=filters,
            debug=debug,
        )
    else:
        output = await retriever.search(
            query=query,
            top_k=effective_top_k,
            filters=filters,
            debug=debug,
        )

    return PipelineResult(
        effective_query=query,
        effective_strategy=effective_strategy,
        effective_top_k=effective_top_k,
        output=output,
        pipeline_debug=dbg,
    )
