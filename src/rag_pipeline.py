"""The ANexus RAG pipeline: end-to-end orchestration in one place.

answer_query() follows the architecture exactly:

  validate -> [history window + follow-up rewrite] -> retrieve (dense+BM25, RRF)
  -> rerank -> redact (Presidio) -> route (semantic router)
  -> generate w/ fallback -> aggregate + cite

Only scrubbed text (ScrubbedText) is ever passed to the LLM handler - including
conversation history, summaries and rewrite prompts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document

from config import settings
from config.prompts import SUMMARY_PROMPT
from src.generation.aggregator import Aggregator, FinalAnswer
from src.generation.llm import LLMHandler, ScrubbedText
from src.ingestion import embedder as embedder_module
from src.ingestion.chunker import Chunker
# parse_pdf imported lazily in ingest_document to reduce startup cost
from src.ingestion.vectordb import VectorDB
from src.memory.history import (
    build_history,
    history_transcript,
    looks_like_followup,
)
from src.memory.rewriter import QueryRewriter
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
        self.aggregator = aggregator or Aggregator(scrubber=scrubber)
        self.chunker = chunker or Chunker()
        self.embedder = embedder or embedder_module
        self.rewriter = QueryRewriter(llm_handler, scrubber=self.scrubber)
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
        conversation_summary: Optional[str] = None,
    ) -> FinalAnswer:
        """Answer a question against the active document.

        `chat_history` is the prior conversation (oldest first, current prompt
        excluded) and `conversation_summary` is the rolling summary of turns
        that fell outside the window. Both are scrubbed before use.
        """
        validation = self._validate_query(query)
        if validation is not None:
            return self.aggregator.no_context(routing_reason=validation)

        if not self.has_document:
            return self.aggregator.no_context(
                routing_reason="No document is active yet."
            )

        # 0) Memory: window the history and scrub it (PII gate applies to the
        #    conversation too - raw user text must never reach a provider).
        summary, turns = build_history(chat_history, conversation_summary)
        summary = self._scrub_optional(summary)
        turns = self._scrub_turns(turns)

        # 0b) Follow-up resolution: heuristic-gated rewrite into a standalone
        #     query, used for retrieval, routing and the prompt question.
        search_query = query
        if turns and looks_like_followup(query):
            rewritten = self.rewriter.rewrite(query, turns, summary)
            if rewritten:
                search_query = rewritten

        # 1) Hybrid retrieval (dense + BM25, RRF fused)
        retrieved = self.retriever.hybrid_retrieve(search_query)
        if not retrieved and search_query != query:
            # The rewrite went nowhere: fall back to the user's raw query.
            retrieved = self.retriever.hybrid_retrieve(query)
            if retrieved:
                search_query = query
        if not retrieved:
            return self.aggregator.no_context(routing_reason="No context matched the query.")

        # 2) Local cross-encoder reranking
        reranked = self.reranker.rerank(search_query, retrieved, top_k=settings.TOP_K_RERANK)
        if not reranked:
            return self.aggregator.no_context(routing_reason="Reranker found no relevant context.")

        # 3) PII security gate: only scrubbed text may reach a provider
        scrubbed_query, scrubbed_context, scrubbed_by_page = build_llm_inputs(
            search_query, reranked, self.scrubber
        )

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
                history=turns,
                summary=summary,
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
            scrubbed_by_page=scrubbed_by_page,
        )

    def summarize_history(
        self,
        chat_history: Optional[list[dict]],
        existing_summary: Optional[str] = None,
    ) -> Optional[str]:
        """Maintain the rolling summary of turns that fall outside the window.

        Returns the new summary text, or None when no update should happen
        (below the trigger, nothing to fold in, or the LLM call failed - the
        caller then keeps whatever summary it already has).
        """
        _, turns = build_history(chat_history, None, max_turns=-1, max_chars=5000)
        if len(turns) <= settings.HISTORY_SUMMARY_TRIGGER:
            return None
        overflow = turns[: len(turns) - settings.HISTORY_MAX_TURNS]
        if not overflow:
            return None
        overflow = self._scrub_turns(overflow)
        if not overflow:
            return None

        parts = []
        if existing_summary:
            scrubbed_existing = self._scrub_optional(existing_summary)
            if scrubbed_existing:
                parts.append(f"Existing summary:\n{scrubbed_existing}")
        parts.append("Conversation transcript:\n" + history_transcript(overflow))

        try:
            messages = [
                {"role": "system", "content": ScrubbedText(self.scrubber.scrub(SUMMARY_PROMPT))},
                {"role": "user", "content": ScrubbedText(self.scrubber.scrub("\n\n".join(parts)))},
            ]
            response = self.llm_handler.generate_messages(
                messages, self.llm_handler.available_providers
            )
        except Exception:
            logger.warning("Conversation summarization failed; keeping the old summary",
                           exc_info=True)
            return None
        text = (response.text or "").strip()
        if not text or text.startswith("[SIMULATED]"):
            return None
        return text

    def restore_active_document(self) -> bool:
        """Re-attach the active document after a restart, if its vectors remain.

        Chroma persists on disk but active_document_id/filename are process
        state. Chunks carry document_id + filename metadata, so a single
        stored document can be identified without re-uploading. BM25 rebuilds
        lazily on first retrieval.
        """
        if self.active_document_id is not None:
            return False
        try:
            docs = self.vector_db.get_all_documents()
        except Exception:
            logger.exception("Could not read stored documents for restore")
            return False
        if not docs:
            return False
        ids = {(d.metadata or {}).get("document_id") for d in docs}
        names = {(d.metadata or {}).get("filename") for d in docs}
        if len(ids) != 1 or len(names) != 1 or None in ids or None in names:
            return False  # mixed or incomplete metadata: do not guess
        self.active_document_id = ids.pop()
        self.active_filename = names.pop()
        logger.info("Restored active document %s (%s)", self.active_document_id,
                    self.active_filename)
        return True

    # ----------------------------------------------------------------- memory
    def _scrub_optional(self, text: Optional[str]) -> Optional[str]:
        """Scrub free text destined for a provider; None stays None.

        A scrub failure drops the text rather than passing raw PII through -
        degrading to less context is safe, sending unscrubbed text is not.
        """
        if not text:
            return None
        try:
            return self.scrubber.scrub(text)
        except Exception:
            logger.warning("Scrub of conversation memory failed; dropping it",
                           exc_info=True)
            return None

    def _scrub_turns(self, turns: list[dict]) -> list[dict]:
        """Scrub every history turn's content; drop turns that fail."""
        scrubbed = []
        for turn in turns:
            content = self._scrub_optional(turn.get("content"))
            if content:
                scrubbed.append({"role": turn["role"], "content": content})
        return scrubbed

    @staticmethod
    def _validate_query(query: str) -> Optional[str]:
        if query is None or not query.strip():
            return "Please enter a question."
        cleaned = query.strip()
        if len(cleaned) > 5000:
            return "Question is too long (max 5000 characters)."
        return None