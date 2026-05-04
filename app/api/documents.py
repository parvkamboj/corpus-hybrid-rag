import contextlib
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.deps import get_db, verify_api_key
from app.models.orm.document import Document
from app.models.schemas import DocumentResponse, DocumentStatusResponse, DocumentUploadResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

AuthDep = Annotated[None, Depends(verify_api_key)]
DbDep = Annotated[AsyncSession, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile,
    request: Request,
    db: DbDep,
    settings: SettingsDep,
    _: AuthDep,
) -> DocumentUploadResponse:
    """Upload a PDF for ingestion. Returns immediately; processing runs in background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only PDF files are accepted"
        )

    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    existing = (
        await db.execute(select(Document).where(Document.content_hash == content_hash))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Document already exists with id {existing.id}",
        )

    doc_id = uuid4()
    upload_dir = Path(settings.uploads_dir) / str(doc_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / (file.filename or "document.pdf")
    file_path.write_bytes(content)

    doc = Document(
        id=doc_id,
        filename=file.filename,
        file_path=str(file_path),
        content_hash=content_hash,
        status="pending",
    )
    db.add(doc)
    await db.commit()

    arq = request.app.state.arq
    job = await arq.enqueue_job("ingest_document", str(doc_id))
    job_id: str = job.job_id if job else "unknown"

    logger.info("document queued", doc_id=str(doc_id), filename=file.filename)
    return DocumentUploadResponse(doc_id=doc_id, job_id=job_id)


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    db: DbDep,
    _: AuthDep,
) -> list[DocumentResponse]:
    """List all ingested documents ordered by upload time (newest first)."""
    rows = (
        await db.execute(select(Document).order_by(Document.created_at.desc()))
    ).scalars().all()
    return [DocumentResponse.model_validate(r) for r in rows]


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: UUID,
    db: DbDep,
    _: AuthDep,
) -> DocumentResponse:
    """Get document metadata and chunk count."""
    doc = (
        await db.execute(select(Document).where(Document.id == doc_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: UUID,
    db: DbDep,
    _: AuthDep,
) -> DocumentStatusResponse:
    """Poll ingestion job status: pending | running | done | failed."""
    doc = (
        await db.execute(select(Document).where(Document.id == doc_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentStatusResponse.model_validate(doc)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: UUID,
    request: Request,
    db: DbDep,
    settings: SettingsDep,
    _: AuthDep,
) -> None:
    """Remove a document, its Qdrant vectors, and its uploaded file."""
    doc = (
        await db.execute(select(Document).where(Document.id == doc_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    qdrant: AsyncQdrantClient = request.app.state.qdrant
    await qdrant.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id)))]
            )
        ),
    )

    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()
    with contextlib.suppress(OSError):
        file_path.parent.rmdir()

    doc.updated_at = datetime.now(UTC)
    await db.delete(doc)
    await db.commit()
    logger.info("document deleted", doc_id=str(doc_id))
