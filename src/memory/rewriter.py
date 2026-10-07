"""Follow-up resolution: rewrite an underspecified question into a standalone one.

A query like "What about the second one?" cannot retrieve anything useful on
its own. When the heuristic gate says the query depends on prior turns, the
rewriter asks the LLM for a standalone version, which is then used for
retrieval, reranking, routing and as the prompt question.

Everything sent to the provider goes through the Presidio scrubber first - the
PII gate applies to this call like any other. Failures return None so the
pipeline falls back to the raw query.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from config import settings
from config.prompts import REWRITE_PROMPT
from src.generation.llm import ScrubbedText
from src.memory.history import history_transcript

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, llm_handler, scrubber=None) -> None:
        self.llm_handler = llm_handler
        self.scrubber = scrubber

    def _scrub(self, text: str) -> str:
        if self.scrubber is None:
            return text
        return str(self.scrubber.scrub(text))

    def rewrite(
        self,
        query: str,
        turns: Sequence[dict],
        summary: Optional[str] = None,
        provider_order: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Return a standalone version of `query`, or None to use it as-is.

        None means "no rewrite happened" (disabled, failed, or empty) and the
        caller must keep the original query.
        """
        if not settings.QUERY_REWRITE_ENABLED or not turns:
            return None

        order = provider_order or self.llm_handler.available_providers
        if not order:
            return None

        try:
            transcript_parts = []
            if summary:
                transcript_parts.append(f"Conversation summary:\n{self._scrub(summary)}")
            transcript_parts.append("Recent conversation:\n" + history_transcript(turns))
            transcript_parts.append(f"Follow-up question:\n{self._scrub(query)}")
            messages = [
                {"role": "system", "content": ScrubbedText(self._scrub(REWRITE_PROMPT))},
                {"role": "user", "content": ScrubbedText("\n\n".join(transcript_parts))},
            ]
            response = self.llm_handler.generate_messages(messages, order)
        except Exception:
            logger.warning("Query rewrite failed; using the raw query", exc_info=True)
            return None

        standalone = (response.text or "").strip().strip('"').strip()
        if not standalone or standalone.startswith("[SIMULATED]"):
            # The offline adapter cannot condense questions; keep the raw query.
            return None
        return standalone
