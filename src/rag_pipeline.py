"""The ANexus RAG pipeline: end-to-end orchestration in one place.

answer_query() follows the architecture exactly:

  validate -> retrieve (dense+BM25, RRF) -> rerank -> redact (Presidio)
  -> route (semantic router) -> generate w/ fallback -> aggregate + cite

Only scrubbed text (ScrubbedText) is ever passed to the LLM handler.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document

from config import settings
from src.generation.aggregator import Aggregator, FinalAnswer
from src.generation.llm import LLMHandler
from src.ingestion import embedder as embedder_module
from src.ingestion.chunker import Chunker
# parse_pdf imported lazily in ingest_document to reduce startup cost
from src.ingestion.vectordb import VectorDB
from src.redaction.presidio_scrubber import PresidioScrubber, build_llm_inputs
from src.retrieval.reranker import CrossEncoderReranker, RankedChunk
from src.retrieval.retriever import HybridRetriever
from src.routing.quota_tracker import QuotaTracker
from src.routing.semantic_router import RouteDecision, SemanticRouter

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    pages: int
    chunks: int
    document_id: str
    filename: str
    duration_s: float
    vectors_added: int = 0


@dataclass
class Quote:
    page_number: int
    text: str


class RAGPipeline:
    def __init__(
        self,
        vector_db: VectorDB,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        scrubber: PresidioScrubber,
        router: SemanticRouter,
        llm_handler: LLMHandler,
        aggregator: Optional[Aggregator] = None,
        chunker: Optional[Chunker] = None,
        embedder=None,
    ) -> None:
        self.vector_db = vector_db
        self.retriever = retriever
        self.reranker = reranker
        self.scrubber = scrubber
        self.router = router
        self.llm_handler = llm_handler
        self.aggregator = aggregator or Aggregator()
        self.chunker = chunker or Chunker()
        self.embedder = embedder or embedder_module
        self.active_document_id: Optional[str] = None
        self.active_filename: Optional[str] = None

    # ------------------------------------------------------------------- state
    @property
    def has_document(self) -> bool:
        return self.active_document_id is not None and self.vector_db.count() > 0

    def reset(self) -> None:
        """New-document reset: drop vectors/BM25 and forget the active document."""
        self.vector_db.reset()
        self.retriever.reset()
        self.active_document_id = None
        self.active_filename = None
        logger.info("RAG pipeline state reset")

    # ---------------------------------------------------------------- ingestion
    def ingest_document(self, pdf_path: str | Path) -> IngestionResult:
        """Parse -> chunk -> embed -> store -> rebuild BM25. Resets any previous doc."""
        pdf_path = Path(pdf_path)
        start = time.perf_counter()
        from src.ingestion.parser import parse_pdf

        page_documents = parse_pdf(pdf_path)
        if not page_documents:
            raise ValueError(f"No text could be extracted from {pdf_path.name}")

        chunks = self.chunker.split(page_documents)
        if not chunks:
            raise ValueError(f"No usable chunks were produced from {pdf_path.name}")

        vectors = self.embedder.embed_documents(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        # A brand-new PDF replaces any previous active document.
        self.vector_db.reset()
        self.retriever.reset()
        self.vector_db.add_chunks(chunks, vectors)
        self.retriever.rebuild_bm25(chunks)

        self.active_document_id = str(pdf_path.stem)
        self.active_filename = pdf_path.name
        duration = time.perf_counter() - start
        logger.info(
            "Ingested %s: %d page(s) -> %d chunk(s) in %.2fs",
            pdf_path.name,
            len(page_documents),
            len(chunks),
            duration,
        )
        return IngestionResult(
            pages=len(page_documents),
            chunks=len(chunks),
            document_id=self.active_document_id,
            filename=self.active_filename,
            duration_s=duration,
            vectors_added=len(vectors),
        )

    # ----------------------------------------------------------------- queries
    def answer_query(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
    ) -> FinalAnswer:
        """Answer a question against the active document."""
        del chat_history  # multi-turn memory lives in the UI; each turn re-retrieves.
        validation = self._validate_query(query)
        if validation is not None:
            return self.aggregator.no_context(routing_reason=validation)

        if not self.has_document:
            return self.aggregator.no_context(
                routing_reason="No document is active yet."
            )

        start = time.perf_counter()

        # 1) Hybrid retrieval (dense + BM25, RRF fused)
        retrieved = self.retriever.hybrid_retrieve(query)
        if not retrieved:
            return self.aggregator.no_context(routing_reason="No context matched the query.")

        # 2) Local cross-encoder reranking
        reranked = self.reranker.rerank(query, retrieved, top_k=settings.TOP_K_RERANK)
        if not reranked:
            return self.aggregator.no_context(routing_reason="Reranker found no relevant context.")

        # 3) PII security gate: only scrubbed text may reach a provider
        scrubbed_query, scrubbed_context = build_llm_inputs(query, reranked, self.scrubber)

        # 4) Semantic routing (query only, no document content)
        route: RouteDecision = self.router.route(
            str(scrubbed_query),
            available_providers=self.llm_handler.available_providers,
        )

        # 5) Generate with automatic fallback
        try:
            llm_response = self.llm_handler.generate(
                scrubbed_query,
                scrubbed_context,
                provider_order=route.provider_order,
            )
        except Exception as e:
            logger.exception("All LLM providers failed")
            return self.aggregator.graceful_failure(
                f"All configured language models are unavailable right now. "
                f"Please try again shortly. ({str(e)[:200]})",
                intent=route.intent,
                routing_reason=route.reason,
            )

        # 6) Aggregate + cite
        return self.aggregator.aggregate(
            llm_response,
            reranked,
            intent=route.intent,
            routing_reason=route.reason,
        )

    @staticmethod
    def _validate_query(query: str) -> Optional[str]:
        if query is None or not query.strip():
            return "Please enter a question."
        cleaned = query.strip()
        if len(cleaned) > 5000:
            return "Question is too long (max 5000 characters)."
        return None