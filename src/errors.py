"""Lightweight shared exceptions.

Kept out of ``rag_pipeline.py`` so the HTTP layer can import them without
pulling in the heavy pipeline (Presidio/spaCy, reranker, LLM providers).
"""


class DuplicateDocumentError(ValueError):
    """Raised when content already exists under another document name and the
    configured DEDUP_ACTION is "skip". The UI reports it as information, not as an
    ingestion failure."""
