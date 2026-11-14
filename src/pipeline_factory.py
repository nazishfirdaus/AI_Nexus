"""Pipeline construction.

`get_pipeline()` returns one lazily-shared RAGPipeline for the whole process, so
the embedding, reranking and spaCy models load once per worker.
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
_QDRANT_CLIENT = None


def get_qdrant_client():
    """One process-wide Qdrant client; every tenant collection shares it.

    Qdrant namespaces tenants by collection name, so a single client can back
    many isolated collections without reconnecting per user.
    """
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        from qdrant_client import QdrantClient

        endpoint = (settings.QDRANT_URL or "").strip()
        if not endpoint:
            raise RuntimeError(
                "Qdrant is the vector backend but no endpoint is set. "
                "Set ANEXUS_QDRANT_URL (or QDRANT_URL / QDRANT_CLUSTER_ENDPOINT) and "
                "QDRANT_API_KEY in .env."
            )
        _QDRANT_CLIENT = QdrantClient(
            url=endpoint,
            api_key=settings.QDRANT_API_KEY or None,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            timeout=settings.QDRANT_TIMEOUT_S,
        )
        logger.info("Opened shared Qdrant client at %s", endpoint)
    return _QDRANT_CLIENT


def reset_clients() -> None:
    """Drop the cached Qdrant client (tests / reconnect)."""
    global _QDRANT_CLIENT
    _QDRANT_CLIENT = None


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
    qdrant_client=None,
) -> PipelineBundle:
    """Wire a complete pipeline, defaulting every part to the production component.

    ``collection_name`` / ``db_path`` / ``qdrant_client`` select a tenant
    namespace; the defaults reproduce single-tenant behaviour.
    """
    from src.generation.aggregator import Aggregator
    from src.generation.llm import build_default_handler
    from src.ingestion.document_store import DocumentStore
    from src.ingestion.vectordb import build_vector_db
    from src.rag_pipeline import RAGPipeline, _build_default_chunker
    from src.redaction.presidio_scrubber import get_scrubber
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import HybridRetriever
    from src.routing.quota_tracker import QuotaTracker
    from src.routing.semantic_router import SemanticRouter

    tracker = quota_tracker or QuotaTracker()
    vec_db = vector_db or build_vector_db(
        collection_name=collection_name or settings.COLLECTION_NAME,
        qdrant_client=qdrant_client,
    )
    rerank = reranker or CrossEncoderReranker()
    scrub = scrubber or get_scrubber()
    rout = router or SemanticRouter(quota_tracker=tracker)
    handler = llm_handler or build_default_handler(tracker)
    pipe_retriever = retriever or HybridRetriever(
        vector_db=vec_db,
        documents=[],
        k=settings.BASE_K_RETRIEVAL,
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

    Heavy models (embedder, cross-encoder reranker, Presidio/spaCy) are
    process-cached inside their own modules and therefore shared automatically.
    Only the namespace-bound pieces (vector-store collection, SQLite registry,
    BM25 index, LLM handler, quota state) are per user.
    """
    from src.auth.tenancy import user_collection_name, user_db_path
    from src.generation.llm import build_handler_from_credentials
    from src.routing.quota_tracker import QuotaTracker

    tracker = quota_tracker if quota_tracker is not None else QuotaTracker(state_path=None)
    handler = llm_handler or build_handler_from_credentials(credentials, tracker)
    return build_pipeline(
        collection_name=user_collection_name(user_id),
        db_path=user_db_path(user_id),
        quota_tracker=tracker,
        llm_handler=handler,
        qdrant_client=get_qdrant_client(),
        **build_kwargs,
    )


def get_pipeline() -> PipelineBundle:
    """Process-wide singleton; models load once per worker."""
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