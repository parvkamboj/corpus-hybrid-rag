"""Shared Qdrant filter builder used by all retrievers."""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.retrieval.base import MetadataFilter


def build_qdrant_filter(filters: MetadataFilter | None) -> Filter | None:
    if not filters or not filters.doc_id:
        return None
    return Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=filters.doc_id))]
    )
