"""Backwards-compatible embedding shim.

The canonical embedder lives in :mod:`src.services.retrieval.embedding`. This
module previously hosted a second copy of the model logic; delegating keeps the
old import surface working without loading a second model instance.
"""
from .embedding import (
    EMBEDDING_MODEL_NAME,
    QUERY_INSTRUCTION,
    embed_documents,
    embed_query,
    embed_texts,
    get_active_model_info,
    get_embedding_dim,
    get_embedding_model,
    token_length,
)

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "QUERY_INSTRUCTION",
    "embed_documents",
    "embed_query",
    "embed_texts",
    "get_active_model_info",
    "get_embedding_dim",
    "get_embedding_model",
    "token_length",
]
