"""The ANexus RAG pipeline: end-to-end orchestration in one place.

answer_query() follows the architecture exactly:

   validate -> [history window + follow-up rewrite] -> [query expansion] ->
   multi-query retrieve (dense+BM25, RRF-fused) or hybrid retrieve
  -> rerank -> redact (Presidio) -> route (semantic router)
  -> generate w/ fallback -> aggregate + cite

Only scrubbed text (ScrubbedText) is ever passed to the LLM handler - including
conversation history, summaries and rewrite prompts.
"""
from __future__ import annotations

import hashlib
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from langchain_core.documents import Document

from config import settings
from config.prompts import SUMMARY_PROMPT
from src.generation.aggregator import Aggregator, FinalAnswer
from src.generation.llm import (
    AllProvidersFailed,
    LLMHandler,
    LLMResponse,
    ScrubbedText,
)
from src.ingestion import embedder as embedder_module
from src.ingestion.chunker import Chunker
from src.ingestion.document_store import DocumentStore
# parse_document imported lazily in ingest_document to reduce startup cost
from src.ingestion.parsers.common import (
    drop_repeated_boilerplate,
    normalize_text,
)
from src.ingestion.vectordb import VectorDB
from src.memory.history import (
    build_history,
    history_transcript,
    looks_like_followup,
)
from src.memory.rewriter import QueryRewriter
from src.retrieval.query_expander import QueryExpander
from src.redaction.presidio_scrubber import PresidioScrubber, build_llm_inputs
from src.retrieval.reranker import CrossEncoderReranker, RankedChunk
from src.retrieval.retriever import HybridRetriever
from src.routing.quota_tracker import QuotaTracker
from src.routing.semantic_router import RouteDecision, SemanticRouter

logger = logging.getLogger(__name__)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_key(text: str) -> str:
    """Normalized, lowercased digest used to spot exact-duplicate chunks."""
    return _sha256_hex(normalize_text(text or "").lower())


def _pages_digest(page_documents: Iterable[Document]) -> str:
    """Deterministic content digest for non-byte sources (web/cloud crawls)."""
    return _sha256_hex(
        "\n\n".join(normalize_text(p.page_content or "") for p in page_documents)
    )


def _chunk_char_stats(chunks: Iterable[Document]) -> tuple[float, float, int]:
    """Char-length summary of a chunk batch: (mean, median, max)."""

    lengths = [len(c.page_content or "") for c in chunks]
    if not lengths:
        return 0.0, 0.0, 0
    return (
        sum(lengths) / len(lengths),
        float(statistics.median(lengths)),
        max(lengths),
    )


def _build_default_chunker() -> Chunker:
    """Build the ingest chunker from settings.

    Token-aware sizing is opt-in: it measures pieces with the embedder's own
    tokenizer so chunks are capped by the model's real window. The tokenizer is
    resolved lazily (on first measure) by ``embedder.token_length``, so building
    the chunker never loads a model.
    """

    if settings.CHUNK_TOKEN_AWARE:
        from src.ingestion.embedder import token_length

        return Chunker(
            chunk_size=settings.CHUNK_MAX_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
            length_fn=token_length,
            target_size=settings.CHUNK_TARGET_TOKENS,
            sentence_aware=settings.CHUNK_SENTENCE_AWARE,
        )
    return Chunker(sentence_aware=settings.CHUNK_SENTENCE_AWARE)


def _context_prefix(chunk: Document) -> str:
    """Short "title · page N" header describing where a chunk came from.

    Uses the document title when the parser found one, else the filename stem.
    Returns "" when there is nothing meaningful to add.
    """
    meta = chunk.metadata or {}
    title = meta.get("doc_title")
    if not title:
        filename = meta.get("filename") or meta.get("source") or ""
        title = Path(filename).stem if filename else ""
    title = str(title).strip()
    page = meta.get("page_number")
    parts = [p for p in (title, f"page {page}" if page is not None else "") if p]
    return " · ".join(parts)


def _chunk_embed_texts(chunks: list[Document]) -> list[str]:
    """Build the per-chunk strings that get embedded (header + clean text).

    The header is also recorded in each chunk's metadata as ``context_prefix`` so
    the UI can display it without it polluting the stored chunk text.
    """
    texts: list[str] = []
    for chunk in chunks:
        prefix = _context_prefix(chunk)
        if prefix:
            chunk.metadata = {**(chunk.metadata or {}), "context_prefix": prefix}
            texts.append(f"{prefix}\n{chunk.page_content}")
        else:
            texts.append(chunk.page_content)
    return texts


def _embed_chunk_vectors(embedder, texts: list[str]):
    """Embed raw strings, falling back to ``embed_documents`` for test doubles."""
    fn = getattr(embedder, "embed_texts", None)
    if fn is not None:
        return fn(texts)
    return embedder.embed_documents([Document(page_content=t) for t in texts])


def _page_overcleaned(page: Document) -> bool:
    """True if cleaning removed most of a page's content.

    Pages that never recorded ``raw_chars`` (e.g. pre-parsed web pages) are not
    judged, and very short pages are ignored as too small to be meaningful.
    """

    meta = page.metadata or {}
    raw = meta.get("raw_chars")
    clean = meta.get("clean_chars")
    if not isinstance(raw, int) or not isinstance(clean, int):
        return False
    if raw < settings.MIN_TEXT_LENGTH_FOR_OCR:
        return False
    return raw > 0 and (clean / raw) < settings.MIN_PAGE_CONTENT_RATIO


class DuplicateDocumentError(ValueError):
    """Raised when content already exists under another document name and the
    configured DEDUP_ACTION is "skip". The UI reports it as information, not as an
    ingestion failure."""


@dataclass
class IngestionResult:
    pages: int
    chunks: int
    document_id: str
    filename: str
    duration_s: float
    vectors_added: int = 0
    content_hash: Optional[str] = None
    pages_empty_dropped: int = 0
    pages_ocr: int = 0
    pages_overcleaned: int = 0
    chunks_dropped_short: int = 0
    boilerplate_lines_removed: int = 0
    chunks_mean_chars: float = 0.0
    chunks_median_chars: float = 0.0
    chunks_max_chars: int = 0


@dataclass
class Quote:
    page_number: int
    text: str


@dataclass
class _Prepared:
    """Everything generation needs, resolved by _prepare before any LLM call."""

    search_query: str
    reranked: list
    scrubbed_query: "ScrubbedText"
    scrubbed_context: "ScrubbedText"
    scrubbed_by_page: dict
    turns: list
    summary: Optional[str]
    route: RouteDecision


class _StreamingScrubber:
    """Buffer streamed text and only release an already-scrubbed *stable* prefix.

    The persisted answer is always scrubbed in full before it is saved, so this
    gate exists purely to keep the live preview safe. Naively releasing
    ``raw[:-tail]`` is not enough: a PII value can straddle the release boundary
    and its first half would go out unrecognised. Instead each step scrubs the
    whole accumulated text and releases the longest prefix ``p`` (bounded by the
    tail) for which ``scrub(raw[:p])`` is already a prefix of ``scrub(raw)`` -
    i.e. a prefix the full-text scrub will not later revise. A value split
    across ``p`` fails that check (the partial digits differ from the full
    text's placeholder), so it is held back until it can be redacted in full.

    A scrub failure is logged and the text passes through unsanitised, matching
    the aggregator's post-scrub: nothing leaves the machine at this point, and
    killing the UI would be worse than the (local) leak.
    """

    def __init__(self, scrubber, tail=None) -> None:
        self.scrubber = scrubber
        self.tail = tail if tail is not None else settings.STREAM_SCRUB_TAIL_CHARS
        self.raw = ""
        self._emitted = ""      # scrubbed text already released
        self._raw_consumed = 0  # raw characters covered by _emitted

    def feed(self, chunk: str) -> str:
        """Consume one raw chunk; return the newly-safe (scrubbed) text, if any."""
        if not chunk:
            return ""
        self.raw += chunk
        limit = max(0, len(self.raw) - self.tail)
        stable = self._stable_prefix(limit)
        if stable <= self._raw_consumed:
            return ""
        scrubbed = self._scrub(self.raw[:stable])
        if not scrubbed.startswith(self._emitted):
            # The full-text scrub revised what we already showed (rare, e.g. a
            # late redaction). Hold everything back rather than desync.
            return ""
        delta = scrubbed[len(self._emitted):]
        self._emitted = scrubbed
        self._raw_consumed = stable
        return delta

    def flush(self) -> str:
        """Scrub and release everything still held back; call once at the end."""
        scrubbed = self._scrub(self.raw)
        if not scrubbed.startswith(self._emitted):
            # Desynced: fall back to showing the full scrubbed text once.
            self._emitted = scrubbed
            self._raw_consumed = len(self.raw)
            return scrubbed
        delta = scrubbed[len(self._emitted):]
        self._emitted = scrubbed
        self._raw_consumed = len(self.raw)
        return delta

    def _stable_prefix(self, limit: int) -> int:
        """Largest p in [consumed, limit] with scrub(raw[:p]) a prefix of scrub(raw)."""
        full = self._scrub(self.raw)
        lo, hi = self._raw_consumed, limit
        best = self._raw_consumed
        while lo <= hi:
            mid = (lo + hi) // 2
            if full.startswith(self._scrub(self.raw[:mid])):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _scrub(self, text: str) -> str:
        if self.scrubber is None or not text:
            return text
        try:
            return str(self.scrubber.scrub(text))
        except Exception:
            logger.warning(
                "Streaming scrub failed; passing the chunk through unsanitised.",
                exc_info=True,
            )
            return text


class _StreamHandle:
    """Iterate scrubbed text deltas and materialise the FinalAnswer at the end."""

    def __init__(self, chunks: Iterator[str], build_final) -> None:
        self._chunks = chunks
        self._build_final = build_final
        self._parts: list[str] = []
        self.final: Optional[FinalAnswer] = None

    def __iter__(self):
        for chunk in self._chunks:
            self._parts.append(chunk)
            yield chunk
        if self.final is None:
            self.final = self._build_final("".join(self._parts))


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
        document_store: Optional[DocumentStore] = None,
    ) -> None:
        self.vector_db = vector_db
        self.retriever = retriever
        self.reranker = reranker
        self.scrubber = scrubber
        self.router = router
        self.llm_handler = llm_handler
        self.aggregator = aggregator or Aggregator(scrubber=scrubber)
        self.chunker = chunker or _build_default_chunker()
        self.embedder = embedder or embedder_module
        # In-memory default: unit tests construct RAGPipeline directly and must
        # never write to the real chat database. The factory injects the shared store.
        self.document_store = document_store or DocumentStore(":memory:")
        self.rewriter = QueryRewriter(llm_handler, scrubber=self.scrubber)
        self.expander = QueryExpander(llm_handler, scrubber=self.scrubber)
        # Most recently ingested document (kept for display/back-compat).
        self.active_document_id: Optional[str] = None
        self.active_filename: Optional[str] = None

    # ------------------------------------------------------------------- state
    @property
    def has_document(self) -> bool:
        return bool(self.list_documents()) and self.vector_db.count() > 0

    def list_documents(self) -> list[dict]:
        """Every ingested document, oldest first."""
        return self.document_store.list()

    def reset(self) -> None:
        """Clear-all: drop vectors/BM25, the registry and the active document."""
        self.vector_db.reset()
        self.retriever.reset()
        self.document_store.clear()
        self.active_document_id = None
        self.active_filename = None
        logger.info("RAG pipeline state reset")

    # ---------------------------------------------------------------- ingestion
    def ingest_document(
        self, file_path: str | Path, source_name: Optional[str] = None
    ) -> IngestionResult:
        """Parse -> chunk -> embed -> store -> rebuild BM25.

        Appends to the existing corpus: previously uploaded documents stay
        searchable. Re-ingesting a document id that is already stored replaces
        it instead of duplicating its chunks.

        Accepts any format the parser layer supports (PDF, DOCX, CSV, XLSX,
        PPTX, TXT/MD, images).

        `source_name` is the user-facing filename. The UI saves uploads under a
        temp name, so without it the document identity would be the temp stem
        (`anexus_upload_...`) and two different files could alias each other.
        """
        file_path = Path(file_path)
        filename = source_name or file_path.name
        # Full filename (stem + extension): `notes.txt` and `notes.pdf` must be
        # two corpus entries, not one.
        document_id = Path(filename).name
        from src.ingestion.parsers import parse_document

        page_documents = parse_document(file_path)
        if not page_documents:
            raise ValueError(f"No text could be extracted from {filename}")
        content_hash = _sha256_bytes(file_path.read_bytes())
        return self.ingest_parsed_documents(
            page_documents,
            filename=filename,
            document_id=document_id,
            source=filename,
            content_hash=content_hash,
        )

    def ingest_parsed_documents(
        self,
        page_documents: list[Document],
        *,
        filename: str,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> IngestionResult:
        """Chunk -> embed -> store -> rebuild BM25 for already-parsed pages.

        Shared by file uploads and non-file sources (e.g. a crawled website that
        yields several page Documents under a single document id). `source`, when
        given, overrides the per-page source metadata (file uploads set it to the
        filename; web pages keep their own URL as source).

        `content_hash` enables content-level dedup: re-ingesting the same bytes
        under a different name either skips (``DEDUP_ACTION=skip``, raising
        DuplicateDocumentError) or replaces the earlier document (``replace``).
        """
        if not page_documents:
            raise ValueError(f"No text could be extracted from {filename}")
        document_id = document_id or Path(filename).name
        start = time.perf_counter()

        # Re-bind identity (parsers only see the temp path / own metadata).
        for page in page_documents:
            meta = dict(page.metadata or {})
            meta["document_id"] = document_id
            meta["filename"] = filename
            if source is not None:
                meta["source"] = source
            page.metadata = meta

        # Quality stats for the ingest report (computed on the raw parser pages).
        pages_empty_dropped = sum(
            1 for p in page_documents if not (p.page_content or "").strip()
        )
        pages_ocr = sum(
            1
            for p in page_documents
            if (p.metadata or {}).get("ocr_used") is True
        )
        pages_overcleaned = sum(1 for p in page_documents if _page_overcleaned(p))

        content_hash = content_hash or _pages_digest(page_documents)

        # Content-level dedup: same content under a different name.
        existing = self.document_store.find_by_content_hash(content_hash)
        if existing and existing["id"] != document_id:
            if settings.DEDUP_ACTION == "replace":
                self.remove_document(existing["id"])
            else:
                raise DuplicateDocumentError(
                    f"'{filename}' is a duplicate of '{existing['filename']}' "
                    "(same content) and was not ingested."
                )

        # Running headers/footers and page stamps repeat on every page; drop them
        # before chunking so they don't pollute the corpus. Conservative: only
        # multi-page documents, and never at the cost of a page's real content.
        boilerplate_removed = 0
        if settings.BOILERPLATE_REMOVAL in ("auto", "true", "on", "1"):
            page_documents, boilerplate_removed = drop_repeated_boilerplate(
                page_documents
            )

        chunks = self.chunker.split(page_documents)
        if not chunks:
            raise ValueError(f"No usable chunks were produced from {filename}")

        # Drop exact-duplicate chunks within this batch (keep the first) so
        # repeated boilerplate never bloats the vector store.
        chunks, dup_dropped = self._dedup_chunks(chunks)

        # Over-cleaning safeguard: noise-only chunks (and sub-floor chunks when
        # opted in) are dropped and reported, rather than silently embedded.
        chunks, short_dropped = self._drop_short_chunks(chunks)
        if not chunks:
            raise ValueError(f"No usable chunks were produced from {filename}")

        chunks_mean_chars, chunks_median_chars, chunks_max_chars = _chunk_char_stats(chunks)

        # Contextual headers (opt-in): embed "title · page N" + text while the
        # stored chunk text stays clean for citations and the LLM.
        if settings.CHUNK_CONTEXT_HEADER == "meta":
            vectors = _embed_chunk_vectors(self.embedder, _chunk_embed_texts(chunks))
        else:
            vectors = self.embedder.embed_documents(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        # Append mode: only this document's previous chunks are dropped, then the
        # BM25 index is rebuilt from the *whole* corpus (all documents). Records
        # stored under a legacy id (stem-only, from before ids included the
        # extension) are reconciled here so a re-ingest replaces instead of
        # duplicating.
        replaced = self.document_store.get(document_id) is not None
        for record in self.document_store.list():
            if record["filename"] == filename and record["id"] != document_id:
                self.vector_db.delete_document(record["id"])
                self.document_store.remove(record["id"])
                replaced = True
        if replaced:
            self.vector_db.delete_document(document_id)
        self.vector_db.add_chunks(chunks, vectors)
        if replaced:
            # Old chunks of a replaced document may still be in the in-memory
            # BM25 index; only a full rebuild from the vector DB is correct here.
            self.retriever.rebuild_bm25(self.vector_db.get_all_documents())
        else:
            # New document: extend the index with just these chunks.
            self.retriever.add_documents(chunks)

        self.active_document_id = document_id
        self.active_filename = filename
        duration = time.perf_counter() - start
        self.document_store.add(
            document_id,
            filename,
            pages=len(page_documents),
            chunks=len(chunks),
            vectors=len(vectors),
            duration_s=duration,
            content_hash=content_hash,
        )
        logger.info(
            "Ingested %s: %d page(s) -> %d chunk(s) in %.2fs "
            "(mean %.0f chars, max %d; +%d duplicate chunk(s) dropped, "
            "%d short chunk(s) dropped, %d boilerplate line(s) removed)",
            filename,
            len(page_documents),
            len(chunks),
            duration,
            chunks_mean_chars,
            chunks_max_chars,
            dup_dropped,
            short_dropped,
            boilerplate_removed,
        )
        if pages_overcleaned:
            logger.warning(
                "%d page(s) of %s lost most of their text to cleaning "
                "(check extraction quality / cleaning settings)",
                pages_overcleaned,
                filename,
            )
        return IngestionResult(
            pages=len(page_documents),
            chunks=len(chunks),
            document_id=document_id,
            filename=filename,
            duration_s=duration,
            vectors_added=len(vectors),
            content_hash=content_hash,
            pages_empty_dropped=pages_empty_dropped,
            pages_ocr=pages_ocr,
            pages_overcleaned=pages_overcleaned,
            chunks_dropped_short=short_dropped,
            boilerplate_lines_removed=boilerplate_removed,
            chunks_mean_chars=chunks_mean_chars,
            chunks_median_chars=chunks_median_chars,
            chunks_max_chars=chunks_max_chars,
        )

    @staticmethod
    def _dedup_chunks(chunks: list[Document]) -> tuple[list[Document], int]:
        """Drop exact-duplicate chunks within one batch (keep the first occurrence).

        Uses the normalized, lowercased content digest, so chunks that differ only
        in casing/whitespace collapse to one. Returns (kept_chunks, dropped_count).
        """
        seen: set[str] = set()
        kept: list[Document] = []
        for chunk in chunks:
            key = _chunk_key(chunk.page_content)
            if key in seen:
                continue
            seen.add(key)
            kept.append(chunk)
        return kept, len(chunks) - len(kept)

    @staticmethod
    def _drop_short_chunks(chunks: list[Document]) -> tuple[list[Document], int]:
        """Drop noise-only chunks, plus sub-floor chunks when opted in.

        A chunk with no alphanumeric character (``...``, bullets, whitespace) can
        never match a query, so it is always dropped. The length floor
        (``MIN_CHUNK_CHARS``) is disabled by default because chunker tail
        fragments can be legitimately short; set it to opt in.
        """

        floor = settings.MIN_CHUNK_CHARS
        kept: list[Document] = []
        for chunk in chunks:
            text = (chunk.page_content or "").strip()
            if not any(ch.isalnum() for ch in text):
                continue
            if floor > 0 and len(text) < floor:
                continue
            kept.append(chunk)
        return kept, len(chunks) - len(kept)

    def remove_document(self, document_id: str) -> bool:
        """Delete one document's chunks and registry entry. Others stay intact."""
        self.vector_db.delete_document(document_id)
        self.document_store.remove(document_id)
        self.retriever.rebuild_bm25(self.vector_db.get_all_documents())
        if self.active_document_id == document_id:
            remaining = self.document_store.list()
            last = remaining[-1] if remaining else None
            self.active_document_id = last["id"] if last else None
            self.active_filename = last["filename"] if last else None
        return True

    # ----------------------------------------------------------------- queries
    @staticmethod
    def _fit_context_budget(reranked: list[RankedChunk]) -> list[RankedChunk]:
        """Trim the reranked chunks to the configured character budget.

        Keeps the highest-scored chunks in order until the budget is reached,
        always including at least the top chunk (so context is never empty). The
        truncated list is what both the LLM prompt and the citations are built
        from, keeping the two consistent.
        """
        budget = getattr(settings, "CONTEXT_MAX_CHARS", 0)
        if not budget or budget <= 0:
            return list(reranked)
        kept: list[RankedChunk] = []
        used = 0
        for chunk in reranked:
            size = len(chunk.content or "")
            if kept and used + size > budget:
                break
            kept.append(chunk)
            used += size
        return kept

    def answer_query(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
        conversation_summary: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> FinalAnswer:
        """Answer a question against every uploaded document.

        `chat_history` is the prior conversation (oldest first, current prompt
        excluded) and `conversation_summary` is the rolling summary of turns
        that fell outside the window. Both are scrubbed before use.

        When `document_id` is given, retrieval is restricted to that single
        document (both dense and BM25). An unknown id yields a no-context answer
        rather than silently leaking a cross-document result.
        """
        prepared = self._prepare(query, chat_history, conversation_summary, document_id)
        if isinstance(prepared, FinalAnswer):
            return prepared

        try:
            llm_response = self.llm_handler.generate(
                prepared.scrubbed_query,
                prepared.scrubbed_context,
                provider_order=prepared.route.provider_order,
                history=prepared.turns,
                summary=prepared.summary,
                temperature=settings.LLM_TEMPERATURE,
            )
        except Exception as e:
            logger.exception("All LLM providers failed")
            return self.aggregator.graceful_failure(
                f"All configured language models are unavailable right now. "
                f"Please try again shortly. ({str(e)[:200]})",
                intent=prepared.route.intent,
                routing_reason=prepared.route.reason,
            )

        return self.aggregator.aggregate(
            llm_response,
            prepared.reranked,
            intent=prepared.route.intent,
            routing_reason=prepared.route.reason,
            scrubbed_by_page=prepared.scrubbed_by_page,
        )

    def answer_query_stream(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
        conversation_summary: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> "_StreamHandle":
        """Streaming twin of answer_query: yields scrubbed text deltas.

        Every emitted chunk has already passed the PII scrubber (see
        _StreamingScrubber), so the live preview never shows unredacted text.
        After the iterator is exhausted, `.final` holds the same FinalAnswer
        the non-streaming path would have produced (scrubbed text + citations).
        """
        prepared = self._prepare(query, chat_history, conversation_summary, document_id)
        if isinstance(prepared, FinalAnswer):
            return _StreamHandle(iter([prepared.text]), lambda _text: prepared)

        raw_parts: list[str] = []
        stream_meta: dict = {}
        failure_msg = (
            "All configured language models are unavailable right now. "
            "Please try again shortly."
        )

        def generate() -> Iterator[str]:
            stream = None
            scrubber = _StreamingScrubber(self.scrubber)
            raw_parts.clear()
            stream_meta.clear()
            try:
                stream = self.llm_handler.generate_stream(
                    prepared.scrubbed_query,
                    prepared.scrubbed_context,
                    provider_order=prepared.route.provider_order,
                    history=prepared.turns,
                    summary=prepared.summary,
                    temperature=settings.LLM_TEMPERATURE,
                )
                for raw_chunk in stream:
                    raw_parts.append(raw_chunk)
                    delta = scrubber.feed(raw_chunk)
                    if delta:
                        yield delta
                tail = scrubber.flush()
                if tail:
                    yield tail
            except Exception as e:
                logger.exception("Streaming generation failed")
                stream_meta["error"] = str(e)
                tail = scrubber.flush()
                if tail:
                    yield tail
                if not raw_parts:
                    # Nothing was shown yet; surface a safe, user-facing message.
                    yield failure_msg
            else:
                stream_meta["text"] = "".join(raw_parts)
                stream_meta["provider"] = stream.provider
                stream_meta["model"] = stream.model
                stream_meta["latency_s"] = stream.latency_s
                stream_meta["fallbacks_tried"] = list(stream.fallbacks_tried)

        def build_final(_displayed: str) -> FinalAnswer:
            if "error" in stream_meta:
                # Mirrors answer_query()'s handler for a total provider failure.
                return self.aggregator.graceful_failure(
                    f"{failure_msg} ({stream_meta['error'][:200]})",
                    intent=prepared.route.intent,
                    routing_reason=prepared.route.reason,
                )
            if "text" not in stream_meta:
                # The generator never completed (e.g. a hard stream failure).
                return self.aggregator.graceful_failure(
                    "The response stream was interrupted before it finished.",
                    intent=prepared.route.intent,
                    routing_reason=prepared.route.reason,
                )
            llm_response = LLMResponse(
                text=stream_meta["text"],
                provider=stream_meta["provider"],
                model=stream_meta["model"],
                latency_s=stream_meta["latency_s"],
                fallbacks_tried=stream_meta["fallbacks_tried"],
            )
            return self.aggregator.aggregate(
                llm_response,
                prepared.reranked,
                intent=prepared.route.intent,
                routing_reason=prepared.route.reason,
                scrubbed_by_page=prepared.scrubbed_by_page,
            )

        return _StreamHandle(generate(), build_final)

    def _prepare(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
        conversation_summary: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Union[FinalAnswer, "_Prepared"]:
        """Shared pre-generation work for answer_query / answer_query_stream.

        Returns either an early FinalAnswer (validation failure, no document, no
        context) or a _Prepared bundle of everything the generation step needs.
        """
        validation = self._validate_query(query)
        if validation is not None:
            return self.aggregator.no_context(routing_reason=validation)

        if not self.has_document:
            return self.aggregator.no_context(
                routing_reason="No documents are uploaded yet."
            )

        if document_id is not None:
            known = {d["id"] for d in self.list_documents()}
            if document_id not in known:
                return self.aggregator.no_context(
                    routing_reason="The selected document is no longer available."
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

        # 1) Hybrid retrieval (dense + BM25, RRF fused), optionally doc-scoped
        queries: list[str] = [search_query]
        # Multi-query expansion (paraphrases), RRF-fused with original.
        expanded = self.expander.expand(search_query)
        if expanded:
            queries = [search_query] + expanded

        if len(queries) == 1:
            retrieved = self.retriever.hybrid_retrieve(search_query, document_id)
        else:
            retrieved = self.retriever.multi_query_retrieve(queries, document_id)

        if not retrieved and search_query != query:
            # The rewrite went nowhere: fall back to the user's raw query.
            retrieved = self.retriever.hybrid_retrieve(query, document_id)
            if retrieved:
                search_query = query
        if not retrieved:
            if document_id is not None:
                return self.aggregator.no_context(
                    routing_reason="No matching context in the selected document."
                )
            return self.aggregator.no_context(routing_reason="No context matched the query.")

        # 2) Local cross-encoder reranking
        reranked = self.reranker.rerank(search_query, retrieved, top_k=settings.TOP_K_RERANK)
        if not reranked:
            return self.aggregator.no_context(routing_reason="Reranker found no relevant context.")

        # 2b) Character budget: trim to the configured context window; the same
        #     list feeds both the prompt and the citations.
        reranked = self._fit_context_budget(reranked)

        # 3) PII security gate: only scrubbed text may reach a provider
        scrubbed_query, scrubbed_context, scrubbed_by_page = build_llm_inputs(
            search_query, reranked, self.scrubber
        )

        # 4) Semantic routing (query only, no document content)
        route: RouteDecision = self.router.route(
            str(scrubbed_query),
            available_providers=self.llm_handler.available_providers,
        )

        return _Prepared(
            search_query=search_query,
            reranked=reranked,
            scrubbed_query=scrubbed_query,
            scrubbed_context=scrubbed_context,
            scrubbed_by_page=scrubbed_by_page,
            turns=turns,
            summary=summary,
            route=route,
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
                messages,
                self.llm_handler.available_providers,
                temperature=settings.LLM_TEMPERATURE_SUMMARY,
            )
        except Exception:
            logger.warning("Conversation summarization failed; keeping the old summary",
                           exc_info=True)
            return None
        text = (response.text or "").strip()
        if not text or text.startswith("[SIMULATED]"):
            return None
        return text

    def restore_documents(self) -> list[dict]:
        """Re-attach stored documents after a restart, if their vectors remain.

        Chroma persists on disk but the registry and BM25 are rebuilt. The
        registry is the source of truth when it has rows; otherwise (upgrade
        from the single-document version, or a wiped registry) the document
        list is reconstructed from chunk metadata and backfilled. Registry rows
        whose chunks are gone are dropped.

        Returns the restored document records (oldest first); BM25 rebuilds
        lazily on first retrieval.
        """
        try:
            stored_chunks = self.vector_db.get_all_documents()
        except Exception:
            logger.exception("Could not read stored documents for restore")
            return []
        if not stored_chunks:
            if self.document_store.list():
                self.document_store.clear()
            return []

        grouped: dict[str, dict] = {}
        for doc in stored_chunks:
            meta = doc.metadata or {}
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            group = grouped.setdefault(
                str(doc_id),
                {"filename": meta.get("filename") or str(doc_id), "pages": set(), "chunks": 0},
            )
            group["chunks"] += 1
            if meta.get("page_number") is not None:
                group["pages"].add(meta["page_number"])

        existing = {r["id"]: r for r in self.document_store.list()}
        restored: list[dict] = []
        for doc_id, group in grouped.items():
            record = existing.get(doc_id)
            if record is None:
                record = self.document_store.add(
                    doc_id,
                    group["filename"],
                    pages=len(group["pages"]),
                    chunks=group["chunks"],
                    vectors=group["chunks"],
                )
            restored.append(record)
            self.active_document_id = record["id"]
            self.active_filename = record["filename"]

        # Drop registry rows whose chunks no longer exist (drift).
        for doc_id in existing:
            if doc_id not in grouped:
                self.document_store.remove(doc_id)

        if restored:
            logger.info(
                "Restored %d document(s): %s",
                len(restored),
                ", ".join(r["filename"] for r in restored),
            )
        return restored

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