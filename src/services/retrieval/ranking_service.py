"""Optional FlashRank cross-encoder reranking (local ONNX model)."""
import time

from flashrank import Ranker, RerankRequest

from src.observability import error, info, span

# Lazy initialization: the ONNX model is loaded on first use, not at import.
_ranker = None


def _get_ranker() -> Ranker:
    """Initialise FlashRank lazily (local ONNX model)."""
    global _ranker
    if _ranker is None:
        info("Initializing FlashRank model locally")
        try:
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """Re-score ``documents`` against ``query`` with a local cross-encoder."""
    if not documents:
        return []

    with span(
        "services.ranking_service.rerank_documents",
        query_length=len(query),
        document_count=len(documents),
        top_n=top_n,
    ):
        start_time = time.time()
        info("Sending documents to FlashRank cross-encoder", document_count=len(documents))

        try:
            ranker = _get_ranker()
            passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
            results = ranker.rerank(RerankRequest(query=query, passages=passages))

            reranked_docs = [res["text"] for res in results[:top_n]]
            duration = time.time() - start_time
            top_score = results[0]["score"] if results else "N/A"
            info(
                "Reranking complete",
                duration_s=round(duration, 3),
                top_score=top_score,
                returned_count=len(reranked_docs),
            )
            return reranked_docs
        except Exception as exc:
            error("Semantic reranking failed", error_type=type(exc).__name__)
            # Fallback to the original order so the user still gets an answer.
            return documents[:top_n]
