"""Vector store factory.

Qdrant Cloud is the only vector backend. :class:`VectorDB` is an alias for
:class:`src.ingestion.qdrant_vectordb.QdrantVectorDB`, kept so existing type
annotations and imports keep working.

Interface used elsewhere:
  * add_chunks(documents, embeddings)
  * search(query_embedding, k, where=None) -> list[Document]
  * delete_document(document_id)
  * reset()
  * get_all_documents() -> list[Document]
  * document_ids() -> list[str]
"""
from __future__ import annotations

from typing import Optional

from src.ingestion.qdrant_vectordb import (
    EmbeddingDimensionMismatch,
    QdrantVectorDB,
)

VectorDB = QdrantVectorDB

__all__ = [
    "EmbeddingDimensionMismatch",
    "QdrantVectorDB",
    "VectorDB",
    "build_vector_db",
]


def build_vector_db(
    *,
    collection_name: Optional[str] = None,
    qdrant_client=None,
    dim: Optional[int] = None,
) -> QdrantVectorDB:
    """Return the Qdrant vector store for a collection.

    ``qdrant_client`` lets callers (and tests) supply a client; otherwise the
    store opens its own connection from the configured Qdrant endpoint.
    """
    from config import settings

    name = collection_name or settings.COLLECTION_NAME
    return QdrantVectorDB(collection_name=name, client=qdrant_client, dim=dim)
