from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector
from sqlalchemy import select

from app.chunking.recursive import RecursiveCharacterChunker
from app.chunking.semantic import SemanticChunker
from app.core.config import Settings, get_settings
from app.core.database import DatabaseManager
from app.embeddings.service import EmbeddingService
from app.ingestion.pdf import PDFConnector
from app.models.orm.document import Document
from app.retrieval.bm25 import compute_document_sparse

logger = structlog.get_logger(__name__)


async def startup(ctx: dict) -> None:  # type: ignore[type-arg]
    settings = get_settings()
    ctx["settings"] = settings
    ctx["db"] = DatabaseManager(settings)
    ctx["qdrant"] = AsyncQdrantClient(url=settings.qdrant_url)
    ctx["embedder"] = EmbeddingService(settings)
    logger.info("worker started")


async def shutdown(ctx: dict) -> None:  # type: ignore[type-arg]
    await ctx["db"].dispose()
    await ctx["qdrant"].close()
    logger.info("worker shutdown")


async def ingest_document(ctx: dict, doc_id: str) -> None:  # type: ignore[type-arg]
    settings: Settings = ctx["settings"]
    db: DatabaseManager = ctx["db"]
    qdrant: AsyncQdrantClient = ctx["qdrant"]
    embedder: EmbeddingService = ctx["embedder"]

    log = logger.bind(doc_id=doc_id)

    async with db.session_factory() as session:
        result = await session.execute(select(Document).where(Document.id == UUID(doc_id)))
        doc = result.scalar_one()
        doc.status = "running"
        doc.updated_at = datetime.now(UTC)
        await session.commit()
        filename: str = doc.filename
        file_path_str: str = doc.file_path

    log.info("ingestion started", filename=filename)

    try:
        elements = await PDFConnector().parse(Path(file_path_str))
        log.info("pdf parsed", element_count=len(elements))

        try:
            raw_chunks = SemanticChunker(settings).chunk(elements)
            log.info("semantic chunking done", chunk_count=len(raw_chunks))
        except Exception:
            log.warning("semantic chunking failed, falling back to recursive")
            raw_chunks = RecursiveCharacterChunker(settings).chunk(elements)

        if not raw_chunks:
            raise ValueError("document produced zero chunks")

        texts = [c.content for c in raw_chunks]
        vectors = await embedder.embed(texts)
        log.info("embeddings done", count=len(vectors))

        # compute sparse vectors for BM25 — done in-process since it's fast
        sparse = [compute_document_sparse(t) for t in texts]

        points = [
            PointStruct(
                id=str(uuid4()),
                vector={
                    "dense": vectors[i],
                    "sparse": SparseVector(indices=sparse[i][0], values=sparse[i][1]),
                },
                payload={
                    "doc_id": doc_id,
                    "content": raw_chunks[i].content,
                    "page_numbers": raw_chunks[i].page_numbers,
                    "section_header": raw_chunks[i].section_header,
                    "filename": filename,
                    "chunk_index": raw_chunks[i].chunk_index,
                    "token_count": raw_chunks[i].token_count,
                },
            )
            for i in range(len(raw_chunks))
        ]

        await qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
        log.info("vectors stored", point_count=len(points))

        async with db.session_factory() as session:
            result = await session.execute(select(Document).where(Document.id == UUID(doc_id)))
            doc = result.scalar_one()
            doc.status = "done"
            doc.chunk_count = len(points)
            doc.updated_at = datetime.now(UTC)
            await session.commit()

        log.info("ingestion complete", chunk_count=len(points))

    except Exception as exc:
        log.exception("ingestion failed", error=str(exc))
        async with db.session_factory() as session:
            result = await session.execute(select(Document).where(Document.id == UUID(doc_id)))
            doc = result.scalar_one()
            doc.status = "failed"
            doc.error_message = str(exc)
            doc.updated_at = datetime.now(UTC)
            await session.commit()
        raise


class WorkerSettings:
    functions = [ingest_document]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_settings().arq_redis_settings()
