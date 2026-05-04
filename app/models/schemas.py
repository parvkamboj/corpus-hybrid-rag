from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    doc_id: UUID
    job_id: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: str
    chunk_count: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    chunk_count: int | None = None
    error_message: str | None = None


# ── Search ────────────────────────────────────────────────────────────────────

class MetadataFilter(BaseModel):
    doc_id: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "auto"] = "hybrid_rerank"
    top_k: int = Field(default=10, ge=1, le=50)
    filters: MetadataFilter | None = None
    debug: bool = False
    use_hyde: bool = False
    decompose: bool = False
    history: list[ChatMessage] | None = None


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    page_numbers: list[int]
    section_header: str | None
    filename: str
    rank: int


class SearchResponse(BaseModel):
    query: str
    strategy: str
    result_count: int
    latency_ms: float
    results: list[SearchResultItem]
    debug: dict[str, Any] | None = None


# ── Query (generation) ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "auto"] = "hybrid_rerank"
    top_k: int = Field(default=10, ge=1, le=50)
    filters: MetadataFilter | None = None
    debug: bool = False
    use_hyde: bool = False
    decompose: bool = False
    history: list[ChatMessage] | None = None


class SourceItem(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_numbers: list[int]
    section_header: str | None
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    strategy: str
    result_count: int
    latency_ms: float
    debug: dict[str, Any] | None = None
