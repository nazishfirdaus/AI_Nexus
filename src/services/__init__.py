"""Services package.

Only the embedding service is exported eagerly: importing this package must not
pull in optional vector-store backends (Qdrant/FlashRank) or require settings
that may be absent in lightweight environments. Import those modules directly
when needed.
"""

from .retrieval.embedding import (
    embed_documents,
    embed_query,
    embed_texts,
    get_active_model_info,
    get_embedding_dim,
    get_embedding_model,
    token_length,
)

__all__ = [
    "embed_documents",
    "embed_query",
    "embed_texts",
    "get_active_model_info",
    "get_embedding_dim",
    "get_embedding_model",
    "token_length",
]
