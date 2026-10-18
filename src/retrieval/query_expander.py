"""Query expansion: generate paraphrases of a query and fuse them via RRF.

This module mirrors the structure of `src/memory/rewriter.py`. If expansion is
disabled, the query is too short, or the LLM fails to return usable paraphrases,
it returns an empty list so the pipeline keeps the single-query path.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from config import settings
from config.prompts import EXPAND_PROMPT
from src.generation.llm import ScrubbedText

logger = logging.getLogger(__name__)

# Very strict parsing: one alternative per line, ignore numbering/bulleting.
_BULLET_RE = re.compile(r"^\s*(?:[-*+\u2022]\s*|\d+\.\s*)?")
_EMPTY_LINE_RE = re.compile(r"^\s*$")

# Patterns that typically indicate the LLM is refusing/instructing, not paraphrasing.
_SIMULATED_PREFIX = "[SIMULATED]"


def should_expand(query: str) -> bool:
    """Heuristic gate: expand when enabled and the query is substantial."""
    if not settings.QUERY_EXPANSION_ENABLED:
        return False
    cleaned = (query or "").strip()
    if not cleaned:
        return False
    words = cleaned.split()
    if len(words) < settings.QUERY_EXPANSION_MIN_WORDS:
        return False
    return True


class QueryExpander:
    def __init__(self, llm_handler, scrubber=None) -> None:
        self.llm_handler = llm_handler
        self.scrubber = scrubber

    def _scrub(self, text: str) -> str:
        if self.scrubber is None:
            return text
        return str(self.scrubber.scrub(text))

    def expand(
        self,
        query: str,
        provider_order: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Return a list of alternative query strings (paraphrases), in order."""
        if not should_expand(query):
            return []

        order = list(provider_order or self.llm_handler.available_providers)
        if not order:
            return []

        max_variants = max(0, settings.QUERY_EXPANSION_VARIANTS)
        if max_variants == 0:
            return []

        prompt = EXPAND_PROMPT.format(variant_count=max_variants)
        try:
            messages = [
                {
                    "role": "system",
                    "content": ScrubbedText(self._scrub(prompt)),
                },
                {
                    "role": "user",
                    "content": ScrubbedText(self._scrub(query)),
                },
            ]
            response = self.llm_handler.generate_messages(
                messages,
                order,
                temperature=settings.LLM_TEMPERATURE_EXPANSION,
            )
        except Exception:
            logger.warning("Query expansion failed; skipping expansion", exc_info=True)
            return []

        text = (response.text or "").strip()
        if not text or text.startswith(_SIMULATED_PREFIX):
            return []

        variants: list[str] = []
        seen = set()

        # Normalize original for dedup.
        original = query.strip()
        seen.add(_normalize(original))

        for line in text.splitlines():
            if _EMPTY_LINE_RE.match(line):
                continue
            variant = _BULLET_RE.sub("", line).strip()
            # Remove wrapping quotes
            variant = variant.strip('"').strip("'").strip()
            if not variant:
                continue
            key = _normalize(variant)
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)
            if len(variants) >= max_variants:
                break

        if variants:
            logger.debug("Expanded query to %s variants", variants)
        return variants


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip()
