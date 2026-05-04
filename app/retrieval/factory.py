from typing import Literal

from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.retrieval.base import Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranked import CrossEncoderService, RerankedRetriever
from app.retrieval.sparse import SparseRetriever

Strategy = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


def get_retriever(
    strategy: Strategy,
    qdrant: AsyncQdrantClient,
    embedder: EmbeddingService,
    reranker: CrossEncoderService,
    settings: Settings,
) -> Retriever:
    dense = DenseRetriever(qdrant, embedder, settings)
    sparse = SparseRetriever(qdrant, settings)

    if strategy == "dense":
        return dense
    if strategy == "sparse":
        return sparse

    hybrid = HybridRetriever(dense, sparse)

    if strategy == "hybrid":
        return hybrid
    if strategy == "hybrid_rerank":
        return RerankedRetriever(hybrid, reranker)

    raise ValueError(f"unknown strategy: {strategy!r}")
