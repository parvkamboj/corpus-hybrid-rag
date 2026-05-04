from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ParsedElement:
    text: str
    page_number: int
    element_type: str            # "NarrativeText", "Table", "Title", …
    section_header: str | None = None


class SourceConnector(Protocol):
    """Parses a source file into a flat list of text elements with metadata."""

    async def parse(self, file_path: Path) -> list[ParsedElement]: ...
