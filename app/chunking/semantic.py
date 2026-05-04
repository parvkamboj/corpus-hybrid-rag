import re
import statistics

import numpy as np
import structlog

from app.chunking.base import RawChunk
from app.core.config import Settings
from app.ingestion.connector import ParsedElement

logger = structlog.get_logger(__name__)

# (sentence_text, page_number, section_header)
_Sentence = tuple[str, int, str | None]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = np.array(a), np.array(b)
    denom = float(np.linalg.norm(na) * np.linalg.norm(nb))
    return float(np.dot(na, nb) / denom) if denom > 1e-8 else 0.0


class SemanticChunker:
    """Splits text at sentence-similarity drops.

    Uses mean - 1.5*std as the split threshold. The 1.5 multiplier was tuned
    by hand on a few test docs — might need adjusting for different content types.

    Falls back to per-element chunks if embedding fails.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: object = None  # lazy-loaded SentenceTransformer

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading model for semantic chunking", model=self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
        return self._model

    def chunk(self, elements: list[ParsedElement]) -> list[RawChunk]:
        sentences: list[_Sentence] = []
        for elem in elements:
            for sent in _split_sentences(elem.text):
                sentences.append((sent, elem.page_number, elem.section_header))

        if not sentences:
            return []

        texts = [s[0] for s in sentences]

        try:
            from sentence_transformers import SentenceTransformer

            model: SentenceTransformer = self._get_model()  # type: ignore[assignment]
            vecs: list[list[float]] = model.encode(
                texts,
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        except Exception:
            logger.exception("semantic chunker failed, falling back to element chunks")
            return self._element_fallback(elements)

        split_indices = self._find_splits(vecs)
        return self._build_chunks(sentences, split_indices)

    def _find_splits(self, vecs: list[list[float]]) -> list[int]:
        if len(vecs) < 2:
            return [0, len(vecs)]

        sims = [_cosine_similarity(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        mean = statistics.mean(sims)
        std = statistics.stdev(sims) if len(sims) > 1 else 0.0

        # TODO: expose this multiplier as a config param, 1.5 works ok for now
        threshold = mean - 1.5 * std

        splits = [0]
        for i, sim in enumerate(sims):
            if sim < threshold:
                splits.append(i + 1)
        splits.append(len(vecs))
        return splits

    def _build_chunks(self, sentences: list[_Sentence], splits: list[int]) -> list[RawChunk]:
        chunks: list[RawChunk] = []
        for idx, (start, end) in enumerate(zip(splits, splits[1:], strict=False)):
            group = sentences[start:end]
            if not group:
                continue
            content = " ".join(s[0] for s in group)
            chunks.append(
                RawChunk(
                    content=content,
                    page_numbers=sorted({s[1] for s in group}),
                    section_header=next((s[2] for s in group if s[2]), None),
                    token_count=len(content.split()),
                    chunk_index=idx,
                )
            )
        return chunks

    def _element_fallback(self, elements: list[ParsedElement]) -> list[RawChunk]:
        return [
            RawChunk(
                content=elem.text,
                page_numbers=[elem.page_number],
                section_header=elem.section_header,
                token_count=len(elem.text.split()),
                chunk_index=i,
            )
            for i, elem in enumerate(elements)
            if elem.text.strip()
        ]
