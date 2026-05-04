from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.connector import ParsedElement


@dataclass
class RawChunk:
    content: str
    page_numbers: list[int]
    section_header: str | None
    token_count: int
    chunk_index: int = field(default=0)


class Chunker(Protocol):
    """Splits parsed elements into retrievable chunks. Synchronous — no I/O."""

    def chunk(self, elements: list[ParsedElement]) -> list[RawChunk]: ...
