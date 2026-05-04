import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

from app.core.config import Settings

logger = structlog.get_logger(__name__)


async def ensure_collection(client: AsyncQdrantClient, settings: Settings) -> None:
    """Creates the collection if it doesn't exist. Both vector fields declared here
    so we don't have to recreate the collection later when sparse indexing is added."""
    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}

    if settings.qdrant_collection in existing:
        logger.info("qdrant collection ready", collection=settings.qdrant_collection)
        return

    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    logger.info("qdrant collection created", collection=settings.qdrant_collection)
