# src/retrieval/retriever.py
"""Hybrid retrieval : dense from ChromaDB + BM25, fused with RRF."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from config import settings
from config.settings import BASE_K_RETRIEVAL

if TYPE_CHECKING:  # avoids importing Chroma/the embedder just to import this module
    from src.ingestion.vectordb import VectorDB

logger = logging.getLogger(__name__)

RRF_K = 60
# BGE v1 retrieves better when the *query* (not the passages) has this instruction.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# keeps numbers such as 7.25 and 4,50,000 intact, strips trailing punctuation
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


class HybridRetriever:
    """
    Hybrid retrieval using:
      1. Dense retrieval from ChromaDB
      2. Sparse retrieval from BM25
    Results are fused with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        vector_db: "VectorDB",
        documents: Optional[List[Document]] = None,
        k: int = BASE_K_RETRIEVAL,
        embedding_model=None,
    ):
        self.vector_db = vector_db
        self.k = k
        self._embedding_model = embedding_model  # injectable for tests
        self.query_prefix = getattr(settings, "QUERY_INSTRUCTION", _BGE_QUERY_INSTRUCTION)

        self.documents: List[Document] = []
        self.bm25: Optional[BM25Plus] = None
        self._token_sets: List[set] = []
        self._rebuild_attempted = False
        if documents:
            self._build_bm25(documents)

    # ------------------------------------------------------------------ setup
    @property
    def embedding_model(self):
        """Loaded lazily, so importing/constructing the retriever never loads BGE."""
        if self._embedding_model is None:
            from src.ingestion.embedder import get_embedding_model

            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def _build_bm25(self, documents: List[Document]) -> None:
        self.documents = list(documents)
        tokenized = [self._tokenize(d.page_content) for d in self.documents]
        self._token_sets = [set(t) for t in tokenized]
        # BM25Plus (not Okapi): idf stays positive on small corpora
        self.bm25 = BM25Plus(tokenized) if any(tokenized) else None
        self._rebuild_attempted = True

    def reset(self) -> None:
        """Call on 'Upload New / Reset'."""
        self.documents, self.bm25, self._token_sets = [], None, []
        self._rebuild_attempted = False

    def _ensure_bm25(self) -> None:
        """BM25 is in memory, Chroma is on disk: after an app restart BM25 would be empty
        and search silently dense-only. Rebuild once from the vector DB if possible."""
        if self.bm25 is not None or self._rebuild_attempted:
            return
        self._rebuild_attempted = True
        getter = getattr(self.vector_db, "get_all_documents", None)
        if getter is None:
            logger.warning("BM25 index empty and vector_db has no get_all_documents(); dense-only search")
            return
        try:
            self._build_bm25(list(getter()))
            logger.info("BM25 index rebuilt from vector DB (%d chunks)", len(self.documents))
        except Exception:
            logger.exception("Could not rebuild BM25 index from vector DB")

    # -------------------------------------------------------------- retrieval
    def dense_retrieve(self, query: str) -> List[Document]:
        """Semantic retrieval using ChromaDB."""
        if not query.strip():
            return []
        try:
            query_embedding = self.embedding_model.encode(
                self.query_prefix + query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()
            return list(self.vector_db.search(query_embedding=query_embedding, k=self.k))
        except Exception:  # dense failure must not kill BM25 results
            logger.exception("Dense retrieval failed; continuing with BM25 only")
            return []

    def bm25_retrieve(self, query: str) -> List[Document]:
        """Keyword retrieval using BM25 (only chunks that share a word with the query)."""
        self._ensure_bm25()
        if not query.strip() or self.bm25 is None:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

        q_set = set(tokens)
        # Without this filter BM25 returns k arbitrary chunks even when nothing matches.
        return [self.documents[i] for i in ranked if q_set & self._token_sets[i]][: self.k]

    @staticmethod
    def _document_id(document: Document):
        """Stable ID for a chunk. Never collapses chunks that lack metadata."""
        m = document.metadata or {}
        if m.get("chunk_id") is not None:
            return (m.get("document_id"), m.get("page_number"), m.get("chunk_id"))
        digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
        return (m.get("document_id"), m.get("page_number"), digest)

    def hybrid_retrieve(self, query: str) -> List[Document]:
        """Dense + BM25, fused by Reciprocal Rank Fusion. Returns up to 2*k candidates."""
        scores: dict = {}
        docs: dict = {}
        for results in (self.dense_retrieve(query), self.bm25_retrieve(query)):
            for rank, doc in enumerate(results, start=1):
                key = self._document_id(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
                docs.setdefault(key, doc)
        return [docs[key] for key in sorted(scores, key=scores.__getitem__, reverse=True)]