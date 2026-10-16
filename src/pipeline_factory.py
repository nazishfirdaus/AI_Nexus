"""Pipeline construction.

`get_pipeline()` returns one lazily-shared RAGPipeline for the whole process (Streamlit
only imports app.py once; this keeps embedding/reranking/spaCy models loaded once).
`build_pipeline(...)` accepts test doubles for every heavyweight dependency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import settings

# Imports deferred into build_pipeline to reduce startup cost
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generation.aggregator import Aggregator
    from src.generation.llm import LLMHandler
    from src.ingestion.chunker import Chunker
    from src.ingestion.document_store import DocumentStore
    from src.ingestion.vectordb import VectorDB
    from src.rag_pipeline import RAGPipeline
    from src.redaction.presidio_scrubber import PresidioScrubber
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import HybridRetriever
    from src.routing.quota_tracker import QuotaTracker
    from src.routing.semantic_router import SemanticRouter

logger = logging.getLogger(__name__)

_PIPELINE: Optional[RAGPipeline] = None
_PIPELINE_KEY: tuple = ()
_CHROMA_CLIENT = None


def get_chroma_client():
    """One process-wide Chroma client; every tenant collection shares it.

    Chroma namespaces tenants by collection name, so a single on-disk client can
    back many isolated collections without reloading anything per user.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        from chromadb import PersistentClient

        _CHROMA_CLIENT = PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        logger.info("Opened shared Chroma client at %s", settings.CHROMA_PERSIST_DIRECTORY)
    return _CHROMA_CLIENT


@dataclass
class PipelineBundle:
    """Everything a stage (UI, tests, evaluation) may want to touch."""

    pipeline: RAGPipeline
    vector_db: VectorDB
    retriever: HybridRetriever
    reranker: CrossEncoderReranker
    scrubber: PresidioScrubber
    router: SemanticRouter
    quota_tracker: QuotaTracker
    llm_handler: LLMHandler
    aggregator: Aggregator
    chunker: Chunker
    document_store: DocumentStore


def build_pipeline(
    *,
    vector_db: Optional[VectorDB] = None,
    retriever: Optional[HybridRetriever] = None,
    reranker: Optional[CrossEncoderReranker] = None,
    scrubber: Optional[PresidioScrubber] = None,
    router: Optional[SemanticRouter] = None,
    quota_tracker: Optional[QuotaTracker] = None,
    llm_handler: Optional[LLMHandler] = None,
    aggregator: Optional[Aggregator] = None,
    chunker: Optional[Chunker] = None,
    document_store: Optional[DocumentStore] = None,
    embedder=None,
    collection_name: Optional[str] = None,
    db_path: Optional[str] = None,
    chroma_client=None,
) -> PipelineBundle:
    """Wire a complete pipeline, defaulting every part to the production component.

    `collection_name` / `db_path` / `chroma_client` select a tenant namespace;
    the defaults reproduce the original single-tenant behaviour.
    """
    from src.generation.aggregator import Aggregator
    from src.generation.llm import build_default_handler
    from src.ingestion.document_store import DocumentStore
    from src.ingestion.vectordb import VectorDB
    from src.rag_pipeline import RAGPipeline, _build_default_chunker
    from src.redaction.presidio_scrubber import get_scrubber
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import HybridRetriever
    from src.routing.quota_tracker import QuotaTracker
    from src.routing.semantic_router import SemanticRouter

    tracker = quota_tracker or QuotaTracker()
    vec_db = vector_db or VectorDB(
        collection_name=collection_name or settings.COLLECTION_NAME,
        client=chroma_client,
    )
    rerank = reranker or CrossEncoderReranker()
    scrub = scrubber or get_scrubber()
    rout = router or SemanticRouter(quota_tracker=tracker)
    handler = llm_handler or build_default_handler(tracker)
    pipe_retriever = retriever or HybridRetriever(
        vector_db=vec_db,
        documents=[],
        k=10,
        embedding_model=None,
    )
    chunk = chunker or _build_default_chunker()
    agg = aggregator or Aggregator(scrubber=scrub)
    store = document_store or DocumentStore(db_path)
    pipeline = RAGPipeline(
        vector_db=vec_db,
        retriever=pipe_retriever,
        reranker=rerank,
        scrubber=scrub,
        router=rout,
        llm_handler=handler,
        aggregator=agg,
        chunker=chunk,
        embedder=embedder,
        document_store=store,
    )
    return PipelineBundle(
        pipeline=pipeline,
        vector_db=vec_db,
        retriever=pipe_retriever,
        reranker=rerank,
        scrubber=scrub,
        router=rout,
        quota_tracker=tracker,
        llm_handler=handler,
        aggregator=agg,
        chunker=chunk,
        document_store=store,
    )


def build_user_pipeline(
    user_id: str,
    *,
    credentials: Optional[dict] = None,
    quota_tracker: Optional[QuotaTracker] = None,
    llm_handler: Optional[LLMHandler] = None,
    **build_kwargs,
) -> PipelineBundle:
    """Build an isolated pipeline for one tenant.

    Heavy models (BGE embedder, cross-encoder reranker, Presidio/spaCy) are
    process-cached inside their own modules and therefore shared automatically.
    Only the namespace-bound pieces (Chroma collection, SQLite registry, BM25
    index, LLM handler, quota state) are per user.
    """
    from src.auth.tenancy import user_collection_name, user_db_path
    from src.generation.llm import build_handler_from_credentials
    from src.routing.quota_tracker import QuotaTracker

    tracker = quota_tracker if quota_tracker is not None else QuotaTracker(state_path=None)
    handler = llm_handler or build_handler_from_credentials(credentials, tracker)
    return build_pipeline(
        collection_name=user_collection_name(user_id),
        db_path=user_db_path(user_id),
        chroma_client=get_chroma_client(),
        quota_tracker=tracker,
        llm_handler=handler,
        **build_kwargs,
    )


def get_pipeline() -> PipelineBundle:
    """Process-wide singleton. Reused across Streamlit reruns; models load once."""
    global _PIPELINE, _PIPELINE_KEY
    if _PIPELINE is not None:
        return _PIPELINE
    bundle = build_pipeline()
    _PIPELINE = bundle
    _PIPELINE_KEY = "default"
    logger.info("Built shared RAG pipeline")
    return bundle


def reset_pipeline_cache() -> None:
    global _PIPELINE, _PIPELINE_KEY
    _PIPELINE = None
    _PIPELINE_KEY = ()