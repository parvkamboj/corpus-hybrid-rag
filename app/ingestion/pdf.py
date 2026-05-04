import asyncio
from pathlib import Path

import structlog

from app.ingestion.connector import ParsedElement

logger = structlog.get_logger(__name__)


class PDFConnector:
    """Parses PDFs with unstructured; falls back to pypdf on failure."""

    async def parse(self, file_path: Path) -> list[ParsedElement]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._parse_unstructured, file_path)
        except Exception as exc:
            logger.warning(
                "unstructured parse failed, falling back to pypdf",
                path=str(file_path),
                error=str(exc),
            )
            return await loop.run_in_executor(None, self._parse_pypdf, file_path)

    def _parse_unstructured(self, file_path: Path) -> list[ParsedElement]:
        from unstructured.partition.pdf import partition_pdf

        raw = partition_pdf(str(file_path))
        elements: list[ParsedElement] = []
        current_header: str | None = None

        for elem in raw:
            text = str(elem).strip()
            if not text:
                continue

            element_type = type(elem).__name__
            page_number = int(getattr(elem.metadata, "page_number", None) or 1)

            if element_type in ("Title", "Header"):
                current_header = text

            elements.append(
                ParsedElement(
                    text=text,
                    page_number=page_number,
                    element_type=element_type,
                    section_header=current_header,
                )
            )

        return elements

    def _parse_pypdf(self, file_path: Path) -> list[ParsedElement]:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        elements: list[ParsedElement] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for paragraph in text.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    elements.append(
                        ParsedElement(
                            text=paragraph,
                            page_number=page_num,
                            element_type="NarrativeText",
                        )
                    )

        return elements
