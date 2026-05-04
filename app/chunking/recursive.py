from app.chunking.base import RawChunk
from app.core.config import Settings
from app.ingestion.connector import ParsedElement

_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveCharacterChunker:
    """Recursive text splitter with overlap."""

    def __init__(self, settings: Settings) -> None:
        self._max_tokens = settings.chunk_size
        self._overlap = settings.chunk_overlap

    def chunk(self, elements: list[ParsedElement]) -> list[RawChunk]:
        result: list[RawChunk] = []
        for elem in elements:
            for _i, text in enumerate(self._split(elem.text)):
                text = text.strip()
                if text:
                    result.append(
                        RawChunk(
                            content=text,
                            page_numbers=[elem.page_number],
                            section_header=elem.section_header,
                            token_count=len(text.split()),
                            chunk_index=len(result),
                        )
                    )
        return result

    def _split(self, text: str) -> list[str]:
        return self._recursive_split(text, _SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text.split()) <= self._max_tokens:
            return [text]

        sep = separators[0]
        rest = separators[1:]

        parts = text.split(sep) if sep else list(text)

        merged: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = len(part.split())
            if current_tokens + part_tokens > self._max_tokens and current:
                merged.append(sep.join(current))
                # retain overlap from the tail
                overlap: list[str] = []
                overlap_tokens = 0
                for p in reversed(current):
                    w = len(p.split())
                    if overlap_tokens + w <= self._overlap:
                        overlap.insert(0, p)
                        overlap_tokens += w
                    else:
                        break
                current = overlap
                current_tokens = overlap_tokens

            current.append(part)
            current_tokens += part_tokens

        if current:
            merged.append(sep.join(current))

        # recursively split anything still over the limit
        result: list[str] = []
        for chunk in merged:
            if len(chunk.split()) > self._max_tokens and rest:
                result.extend(self._recursive_split(chunk, rest))
            else:
                result.append(chunk)
        return result
