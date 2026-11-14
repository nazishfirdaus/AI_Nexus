"""Response + citation aggregation.

Combines an LLMResponse with the reranked source chunks into a FinalAnswer whose
citations are derived *only* from the reranked (and therefore actually-retrieved)
chunks. Sources are deduplicated per (document, page) and sorted by rerank score,
so the same page number coming from two different documents stays distinct.

Both the answer text and every citation snippet are scrubbed on the way out. The
chunks these snippets are built from are the raw stored text, so without this step
the UI would render full names, addresses and account numbers even though the LLM
call itself was protected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, TYPE_CHECKING

from src.redaction.placeholders import display_text

if TYPE_CHECKING:
    from src.generation.llm import LLMResponse
    from src.retrieval.reranker import RankedChunk
    from src.redaction.presidio_scrubber import PresidioScrubber

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "The answer is not available in the uploaded documents. "
    "Please ask about information contained in the uploaded documents."
)


@dataclass
class Citation:
    page_number: int
    score: float
    snippet: str
    document_id: str = ""
    filename: str = ""

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "score": self.score,
            "snippet": self.snippet,
            "document_id": self.document_id,
            "filename": self.filename,
        }


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
    pii_redactions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citations"] = [c.to_dict() for c in self.citations]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FinalAnswer":
        """Rebuild a cached answer; unknown fields from older entries are ignored."""
        data = data or {}
        citations = []
        for item in data.get("citations") or []:
            if not isinstance(item, dict):
                continue
            citations.append(
                Citation(
                    page_number=int(item.get("page_number") or 0),
                    score=float(item.get("score") or 0.0),
                    snippet=str(item.get("snippet") or ""),
                    document_id=str(item.get("document_id") or ""),
                    filename=str(item.get("filename") or ""),
                )
            )
        return cls(
            text=str(data.get("text") or ""),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            latency_s=float(data.get("latency_s") or 0.0),
            fallbacks_tried=list(data.get("fallbacks_tried") or []),
            citations=citations,
            intent=str(data.get("intent") or ""),
            routing_reason=str(data.get("routing_reason") or ""),
            pii_redactions=dict(data.get("pii_redactions") or {}),
        )


class Aggregator:
    def __init__(self, scrubber: Optional["PresidioScrubber"] = None) -> None:
        self.scrubber = scrubber

    def _scrub(self, text: str) -> str:
        """Best-effort redaction of text that is about to be displayed and persisted.

        Unlike the pre-LLM gate this deliberately does NOT fail closed. Nothing is
        leaving the machine at this point, and raising here would take down the UI
        instead of merely degrading it, so a failure is logged and the text passes
        through unchanged.
        """
        return self._scrub_with_report(text)[0]

    def _scrub_with_report(self, text: str) -> tuple[str, dict[str, int]]:
        """``_scrub`` plus the per-entity counts it redacted.

        Placeholders are always neutralised on the way out, even when no
        scrubber is wired: the model can echo a token from the context it was
        given, and the user must never see ``<IN_LOAN_NO>`` in an answer.
        """
        if not text:
            return text, {}
        cleaned, report = text, {}
        if self.scrubber is not None:
            try:
                scrub_with_report = getattr(self.scrubber, "scrub_with_report", None)
                if scrub_with_report is None:
                    cleaned = str(self.scrubber.scrub(text))
                else:
                    cleaned, report = scrub_with_report(text)
                    cleaned = str(cleaned)
            except Exception:
                logger.warning(
                    "Post-scrub of display text failed; passing it through unsanitised.",
                    exc_info=True,
                )
                cleaned, report = text, {}
        return display_text(cleaned), report

    @staticmethod
    def _source_key(chunk: RankedChunk) -> tuple:
        """(document_id, page_number): page numbers only collide across documents."""
        meta = chunk.metadata or {}
        raw_page = meta.get("page_number")
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            return None
        return (str(meta.get("document_id") or ""), page)

    def aggregate(
        self,
        llm_response: LLMResponse,
        reranked: Iterable[RankedChunk],
        intent: str = "",
        routing_reason: str = "",
        snippet_chars: int = 220,
        scrubbed_by_page: Optional[dict] = None,
    ) -> FinalAnswer:
        """Attach deduplicated source citations from the reranked chunks only.

        Citations are keyed by (document_id, page) so the same page number in
        two different documents stays distinct. `scrubbed_by_page` uses the same
        key and is the page -> already-scrubbed-text map produced by
        build_llm_inputs. Preferred over the raw chunk for snippets, because the
        raw chunk can begin mid-sentence with the label of a value already lost -
        a snippet sliced out of it on its own has nothing for a label-anchored
        pattern to match.
        """
        best_by_key: dict[tuple, RankedChunk] = {}
        for chunk in reranked:
            key = self._source_key(chunk)
            if key is None:
                continue
            if key not in best_by_key or chunk.score > best_by_key[key].score:
                best_by_key[key] = chunk

        citations = []
        for key in sorted(best_by_key, key=lambda k: best_by_key[k].score, reverse=True):
            best = best_by_key[key]
            scrubbed = (scrubbed_by_page or {}).get(key)
            if scrubbed is None:
                scrubbed = self._scrub(best.content)
            # `scrubbed_by_page` is the model-facing text, which still contains
            # placeholders. Neutralise before slicing so a token is never cut in
            # half and leaked to the UI.
            snippet = display_text(str(scrubbed))[:snippet_chars]
            citations.append(Citation(
                page_number=key[1],
                score=best.score,
                snippet=snippet,
                document_id=key[0],
                filename=str((best.metadata or {}).get("filename") or ""),
            ))
        answer_text, pii_report = self._scrub_with_report(llm_response.text)
        return FinalAnswer(
            text=answer_text,
            provider=llm_response.provider,
            model=llm_response.model,
            latency_s=llm_response.latency_s,
            fallbacks_tried=list(llm_response.fallbacks_tried),
            citations=citations,
            intent=intent,
            routing_reason=routing_reason,
            pii_redactions=pii_report,
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