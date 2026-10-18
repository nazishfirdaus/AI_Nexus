# src/retrieval/retriever.py
"""Hybrid retrieval : dense from ChromaDB + BM25, fused with RRF."""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from config import settings
from config.settings import BASE_K_RETRIEVAL

if TYPE_CHECKING:  # avoids importing Chroma/the embedder just to import this module
    from src.ingestion.vectordb import VectorDB

logger = logging.getLogger(__name__)

# BGE v1 retrieves better when the *query* (not the passages) has this instruction.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# keeps numbers such as 7.25 and 4,50,000 intact, strips trailing punctuation.
# Unicode-aware ([^\\W_] = letters and digits, not underscore) so accented and
# non-Latin script text tokenizes instead of being dropped. Python's \\w already
# includes Unicode letters/digits; the explicit class just excludes '_'.
_TOKEN_RE = re.compile(r"[^\W_]+(?:[.,][0-9]+)*")


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
        self._tokenized: List[List[str]] = []
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
        self._tokenized = [self._tokenize(d.page_content) for d in self.documents]
        self._token_sets = [set(t) for t in self._tokenized]
        # BM25Plus (not Okapi): idf stays positive on small corpora
        self.bm25 = BM25Plus(self._tokenized) if any(self._tokenized) else None
        self._rebuild_attempted = True

    def add_documents(self, documents: List[Document]) -> None:
        """Append chunks to the BM25 index without re-tokenizing the old corpus.

        Used on the append path of ingestion: instead of reading *every* chunk
        back from Chroma and re-tokenizing the whole corpus, only the new chunks
        are tokenized and ``BM25Plus`` is re-instantiated over the cached
        tokenized lists (BM25's idf depends on the whole document set, so the
        scorer object itself still has to be rebuilt, but tokenization -- the
        expensive part -- is not repeated).

        After a delete/reset/restore a full ``rebuild_bm25`` must be used
        instead, because scrubbing the vector DB is the only reliable way to
        know what actually remains.
        """
        new = [d for d in list(documents) if d.page_content and d.page_content.strip()]
        if not new:
            return
        if not self._tokenized:
            self._build_bm25(new)
            return
        self.documents.extend(new)
        tokenized = [self._tokenize(d.page_content) for d in new]
        self._tokenized.extend(tokenized)
        self._token_sets.extend(set(t) for t in tokenized)
        self.bm25 = BM25Plus(self._tokenized)
        self._rebuild_attempted = True
        logger.info("BM25 index extended by %d chunk(s)", len(new))

    def reset(self) -> None:
        """Call on 'Upload New / Reset'."""
        self.documents, self.bm25 = [], None
        self._tokenized, self._token_sets = [], []
        self._rebuild_attempted = False

    def rebuild_bm25(self, documents: List[Document]) -> None:
        """Rebuild the BM25 index from the chunk corpus (used after ingestion)."""
        self._build_bm25(documents)
        self._rebuild_attempted = True
        logger.info("BM25 index rebuilt from %d chunk(s)", len(documents))

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
    @staticmethod
    def _cosine_similarity(document: Document) -> Optional[float]:
        """Recover cosine similarity from Chroma's stored distance.

        Embeddings are unit-length and the collection uses Chroma's default L2
        space, which stores the *squared* L2 distance ``d = 2 - 2*cos``; therefore
        ``cos = 1 - d/2`` (clamped for numerical safety). Returns ``None`` when no
        distance is present, so a threshold never discards such a hit.
        """
        distance = (document.metadata or {}).get("distance")
        if distance is None:
            return None
        return max(-1.0, min(1.0, 1.0 - float(distance) / 2.0))

    def dense_retrieve(self, query: str, document_id: Optional[str] = None) -> List[Document]:
        """Semantic retrieval using ChromaDB.

        When ``document_id`` is given, the search is restricted (via a Chroma
        metadata filter) to that document's chunks. When the optional
        ``RETRIEVAL_MIN_SIMILARITY`` threshold is set, hits below that cosine
        similarity are dropped (hits without a distance are always kept).
        """
        if not query.strip():
            return []
        where = {"document_id": document_id} if document_id is not None else None
        try:
            query_embedding = self.embedding_model.encode(
                self.query_prefix + query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()
            hits = list(
                self.vector_db.search(
                    query_embedding=query_embedding, k=self.k, where=where
                )
            )
        except Exception:  # dense failure must not kill BM25 results
            logger.exception("Dense retrieval failed; continuing with BM25 only")
            return []

        threshold = getattr(settings, "RETRIEVAL_MIN_SIMILARITY", None)
        if threshold is None:
            return hits
        return [
            doc
            for doc in hits
            if (sim := self._cosine_similarity(doc)) is None or sim >= threshold
        ]

    def bm25_retrieve(self, query: str, document_id: Optional[str] = None) -> List[Document]:
        """Keyword retrieval using BM25 (only chunks that share a word with the query).

        When ``document_id`` is given, only that document's chunks are considered.
        BM25 keeps its full-corpus idf; the scope is applied to the ranked
        candidates (BM25 has no native metadata filter).
        """
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
        def in_scope(i: int) -> bool:
            return document_id is None or (
                self.documents[i].metadata or {}
            ).get("document_id") == document_id

        return [
            self.documents[i]
            for i in ranked
            if q_set & self._token_sets[i] and in_scope(i)
        ][: self.k]

    @staticmethod
    def _document_id(document: Document):
        """Stable ID for a chunk. Never collapses chunks that lack metadata."""
        m = document.metadata or {}
        if m.get("chunk_id") is not None:
            return (m.get("document_id"), m.get("page_number"), m.get("chunk_id"))
        digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
        return (m.get("document_id"), m.get("page_number"), digest)

    @staticmethod
    def _rrf_fuse(sources, rrf_k: int) -> List[Document]:
        """Weighted Reciprocal Rank Fusion of ``(results, weight)`` sources."""
        scores: dict = {}
        docs: dict = {}
        for results, weight in sources:
            for rank, doc in enumerate(results, start=1):
                key = HybridRetriever._document_id(doc)
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
                docs.setdefault(key, doc)
        return [docs[key] for key in sorted(scores, key=scores.__getitem__, reverse=True)]

    def hybrid_retrieve(self, query: str, document_id: Optional[str] = None) -> List[Document]:
        """Dense + BM25, fused by weighted Reciprocal Rank Fusion.

        Each source contributes ``weight / (rrf_k + rank)`` per candidate, so
        ``DENSE_WEIGHT``/``BM25_WEIGHT`` bias the fusion toward semantic or keyword
        matching. Equal weights with ``rrf_k = 60`` reproduce classic RRF; all three
        are env-tunable without code changes. Returns up to 2*k candidates.

        When ``document_id`` is given the scope is applied to *both* channels, so
        only chunks of that document can appear in the fused result.
        """
        sources = (
            (self.dense_retrieve(query, document_id), getattr(settings, "DENSE_WEIGHT", 1.0)),
            (self.bm25_retrieve(query, document_id), getattr(settings, "BM25_WEIGHT", 1.0)),
        )
        return self._rrf_fuse(sources, settings.RRF_K)

    def multi_query_retrieve(
        self, queries: list[str] | Sequence[str], document_id: Optional[str] = None
    ) -> List[Document]:
        """Run hybrid retrieval for multiple query variants, RRF-fuse them all."""
        if not queries:
            return []
        # Deduplicate queries early to avoid duplicate calls.
        unique: list[str] = []
        seen: set[str] = set()
        for q in queries:
            if not q or not q.strip():
                continue
            key = q.strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(q.strip())
        if not unique:
            return []

        sources: list[tuple[List[Document], float]] = []
        for q in unique:
            results = self.hybrid_retrieve(q, document_id)
            sources.append((results, 1.0))
        fused = self._rrf_fuse(sources, settings.RRF_K)
        # Cap to the same size as hybrid candidate pool for stability.
        cap = getattr(settings, "HYBRID_CANDIDATES", self.k * 2)
        if cap > 0 and len(fused) > cap:
            fused = fused[:cap]
        return fused