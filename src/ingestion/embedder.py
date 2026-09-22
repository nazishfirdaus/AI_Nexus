
# src/ingestion/embedder.py
"""Local embedding model (BAAI/bge-large-en). Loaded once per process and reused.

1024-dim, CPU inference, batched cautiously for a 4 GB RAM machine.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL_NAME  # noqa: F401  (re-exported below)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DEFAULT_BATCH_SIZE = 8  # small batches: keeps peak RAM low on 4 GB machines

# BGE v1 retrieves better when the QUERY (not the passages) carries this instruction.
# Passages are embedded as-is; only embed_query() adds it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load BGE-large once per process on CPU. Thread-safe; later calls reuse the same
    instance (so a second copy is never loaded into memory)."""
    with _LOCK:
        logger.info("Loading embedding model %s on CPU", EMBEDDING_MODEL_NAME)
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def embed_documents(
    documents: List[Document],
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> List[List[float]]:
    """Embed a list of chunk Documents. Blank/whitespace-only chunks are dropped, so the
    output can be shorter than the input -- pair Documents with vectors by re-filtering
    the same way, not by zipping blindly."""
    texts = [d.page_content for d in documents if d.page_content and d.page_content.strip()]
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # unit-length vectors: cosine similarity == dot product
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]  # .tolist() -> plain Python floats, not numpy.float32


def embed_query(text: str) -> List[float]:
    """Embed a single query for retrieval. Adds the BGE query instruction prefix."""
    if not text or not text.strip():
        return []
    model = get_embedding_model()
    vector = model.encode(
        QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()