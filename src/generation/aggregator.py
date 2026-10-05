"""Response + citation aggregation.

Combines an LLMResponse with the reranked source chunks into a FinalAnswer whose
citations are derived *only* from the reranked (and therefore actually-retrieved)
chunks. Page numbers are deduplicated and sorted by rerank score.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.generation.llm import LLMResponse
    from src.retrieval.reranker import RankedChunk

NO_CONTEXT_ANSWER = (
    "The answer is not available in the uploaded document. "
    "Please ask about information contained in the document."
)


@dataclass
class Citation:
    page_number: int
    score: float
    snippet: str

    def to_dict(self) -> dict:
        return {"page_number": self.page_number, "score": self.score, "snippet": self.snippet}


@dataclass
class FinalAnswer:
    text: str
    provider: str
    model: str
    latency_s: float
    fallbacks_tried: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    intent: str = ""
    routing_reason: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citations"] = [c.to_dict() for c in self.citations]
        return data


class Aggregator:
    @staticmethod
    def _page_of(chunk: RankedChunk) -> Optional[int]:
        raw = chunk.metadata.get("page_number")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def aggregate(
        self,
        llm_response: LLMResponse,
        reranked: Iterable[RankedChunk],
        intent: str = "",
        routing_reason: str = "",
        snippet_chars: int = 220,
    ) -> FinalAnswer:
        """Attach deduplicated page citations from the reranked chunks only."""
        best_by_page: dict[int, RankedChunk] = {}
        for chunk in reranked:
            page = self._page_of(chunk)
            if page is None:
                continue
            if page not in best_by_page or chunk.score > best_by_page[page].score:
                best_by_page[page] = chunk

        citations = [
            Citation(
                page_number=page,
                score=best_by_page[page].score,
                snippet=best_by_page[page].content[:snippet_chars],
            )
            for page in sorted(best_by_page, key=lambda p: best_by_page[p].score, reverse=True)
        ]
        return FinalAnswer(
            text=llm_response.text,
            provider=llm_response.provider,
            model=llm_response.model,
            latency_s=llm_response.latency_s,
            fallbacks_tried=list(llm_response.fallbacks_tried),
            citations=citations,
            intent=intent,
            routing_reason=routing_reason,
        )

    @staticmethod
    def no_context(intent: str = "", routing_reason: str = "") -> FinalAnswer:
        """Safe response when there is no retrieved context; carries no citations."""
        return FinalAnswer(
            text=NO_CONTEXT_ANSWER,
            provider="",
            model="",
            latency_s=0.0,
            intent=intent,
            routing_reason=routing_reason,
        )

    @staticmethod
    def graceful_failure(
        message: str,
        intent: str = "",
        routing_reason: str = "",
    ) -> FinalAnswer:
        """Friendly response when every provider failed; no invented citations."""
        return FinalAnswer(
            text=message,
            provider="",
            model="",
            latency_s=0.0,
            intent=intent,
            routing_reason=routing_reason,
        )