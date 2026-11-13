"""Qdrant retrieval service for the agent backend."""
from __future__ import annotations

import logging
from typing import Any

from config import settings
from src.ingestion.qdrant_vectordb import QdrantVectorDB
from src.observability import span
from src.pipeline_factory import get_qdrant_client
from src.services.retrieval.embedding import embed_query

logger = logging.getLogger(__name__)


def _collection_for(user_id: str | None, collection_name: str | None) -> str:
    if collection_name:
        return collection_name
    if user_id:
        return f"{settings.COLLECTION_NAME}__{user_id}"
    return settings.QDRANT_COLLECTION


def search_enterprise_knowledge(
    query: str,
    limit: int = 8,
    *,
    user_id: str | None = None,
    collection_name: str | None = None,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search one tenant collection and return agent-ready rows.

    Pass ``user_id`` (normal) or ``collection_name`` (explicit). With neither,
    the base collection is used — it usually does not exist, and Qdrant's error
    is surfaced instead of being hidden behind an empty list.
    """
    if not query or not query.strip():
        return []

    store = QdrantVectorDB(
        collection_name=_collection_for(user_id, collection_name),
        client=get_qdrant_client(),
    )
    where = {"document_id": document_id} if document_id else None

    with span("agent.qdrant.search", k=limit, filtered=bool(where)):
        hits = store.search(embed_query(query), k=limit, where=where)

    results: list[dict[str, Any]] = []
    for doc in hits:
        meta = doc.metadata or {}
        results.append(
            {
                "content": doc.page_content,
                "source": meta.get("filename") or meta.get("source") or "Unknown",
                "document_id": meta.get("document_id"),
                "page": meta.get("page_number"),
                "score": meta.get("similarity"),
            }
        )
    return results