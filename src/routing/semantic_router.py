"""Deterministic, explainable query-intent router.

Only the query text is inspected, locally. No document content and no network call
is used to decide which provider to prefer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import settings
from src.routing.quota_tracker import QuotaTracker

# Ordered by priority when scores tie: analysis > summary > lookup > general.
_INTENT_PATTERNS: dict[str, list[str]] = {
    "analysis": [
        r"\bcompar\w*", r"\bdiscrepanc\w*", r"\binconsisten\w*", r"\bmismatch\w*",
        r"\brisk\w*", r"\bcomplian\w*", r"\bverify\b", r"\bvalidate\b", r"\bwhy\b",
        r"\bcalculat\w*", r"\bdoes .* (match|meet|satisfy)\b", r"\beligib\w*",
        r"\bdefault\w*", r"\bflag\w*", r"\bassess\w*", r"\bimpact\b",
    ],
    "summary": [
        r"\bsummar\w*", r"\boverview\b", r"\bexplain\b", r"\bdescribe\b",
        r"\bin brief\b", r"\bkey (points|terms)\b", r"\bwhat is this (document|file)\b",
        r"\btl;?dr\b", r"\bhighlights?\b",
    ],
    "lookup": [
        r"\bwhat is the\b", r"\bhow much\b", r"\bwhen\b", r"\bwho\b", r"\bwhich\b",
        r"\bamount\b", r"\bdate\b", r"\brate\b", r"\bname\b", r"\bnumber\b",
        r"\bterm\b", r"\btenure\b", r"\bemi\b", r"\bpage\b", r"\bwhere\b",
    ],
}
_PRIORITY = ["analysis", "summary", "lookup"]


@dataclass
class RouteDecision:
    intent: str
    provider: str                     # first choice
    provider_order: list[str]         # full fallback chain
    reason: str
    matched: list[str] = field(default_factory=list)
    skipped_cooling: list[str] = field(default_factory=list)


class SemanticRouter:
    def __init__(
        self,
        quota_tracker: QuotaTracker | None = None,
        preferred_by_intent: dict[str, str] | None = None,
        fallback_chain: list[str] | None = None,
    ) -> None:
        self.quota = quota_tracker
        self.preferred_by_intent = preferred_by_intent or settings.INTENT_PREFERRED_PROVIDER
        self.fallback_chain = list(fallback_chain or settings.FALLBACK_CHAIN)
        self._compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in pats]
            for intent, pats in _INTENT_PATTERNS.items()
        }

    def classify(self, query: str) -> tuple[str, list[str]]:
        scores: dict[str, int] = {}
        matched: dict[str, list[str]] = {}
        for intent, patterns in self._compiled.items():
            hits = [m.group(0) for p in patterns if (m := p.search(query))]
            if hits:
                scores[intent] = len(hits)
                matched[intent] = hits
        if not scores:
            return "general", []
        best = max(scores.values())
        for intent in _PRIORITY:  # deterministic tie-break
            if scores.get(intent) == best:
                return intent, matched[intent]
        return "general", []

    def route(self, query: str, available_providers: list[str]) -> RouteDecision:
        """`available_providers` = providers that have credentials/adapters configured."""
        if not available_providers:
            raise ValueError("No LLM providers are configured.")

        intent, matched = self.classify(query)
        # The intent chooses only the FIRST provider; the remainder always follows
        # the configured FALLBACK_CHAIN, then any other configured provider.
        first = self.preferred_by_intent.get(intent)
        order = ([first] if first else []) + self.fallback_chain + list(available_providers)
        preferred: list[str] = []
        for p in order:
            if p in available_providers and p not in preferred:
                preferred.append(p)

        skipped: list[str] = []
        if self.quota:
            healthy = [p for p in preferred if self.quota.is_available(p)]
            skipped = [p for p in preferred if p not in healthy]
            # Cooling providers go to the back for visibility in the decision;
            # LLMHandler skips them until their cooldown ends.
            preferred = healthy + skipped

        reason = f"intent='{intent}'" + (f" (matched: {', '.join(matched)})" if matched else " (no keywords matched)")
        reason += f" -> preferred order {preferred}"
        if skipped:
            reason += f"; cooling down: {skipped}"

        return RouteDecision(
            intent=intent,
            provider=preferred[0],
            provider_order=preferred,
            reason=reason,
            matched=matched,
            skipped_cooling=skipped,
        )