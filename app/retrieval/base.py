from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    content: str
    score: float
    page_numbers: list[int]
    section_header: str | None
    filename: str
    rank: int = 0


@dataclass
class SearchOutput:
    results: list[SearchResult]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataFilter:
    doc_id: str | None = None


class Retriever(Protocol):
    """Common interface for all retrieval strategies."""

    async def search(
        self,
        query: str,
        top_k: int,
        filters: MetadataFilter | None,
        debug: bool,
    ) -> SearchOutput: ...
