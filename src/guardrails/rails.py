"""Safety gate for the agent and the RAG pipeline.

Two directions, both backed by Llama Guard (``guard_model``):

  * ``guard()``        — input. Blocks jailbreaks and off-policy prompts before
                         any retrieval or generation happens.
  * ``guard_output()`` — output. Blocks a generated answer that contains a
                         hazardous category before it is returned or persisted.

Both return ``(blocked, message)`` so existing callers keep working. The previous
implementation judged intent by substring-matching the chat model's reply against
five hardcoded sentences and therefore never fired in practice.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.config import settings
from src.guardrails.guard_model import (
    HAZARDS,
    REFUSAL,
    Verdict,
    classify,
    is_available,
)

logger = logging.getLogger(__name__)

OUTPUT_REFUSAL = (
    "That answer was withheld by the safety filter. "
    "Ask me something else about the documents in your file."
)


def initialize_rails() -> None:
    """Warm the guard client. Kept for startup call sites that pre-build rails."""
    if not settings.GUARD_ENABLED:
        logger.info("Safety guard disabled (ANEXUS_GUARD_ENABLED=false).")
        return
    if not is_available():
        logger.warning(
            "Safety guard has no provider: set OPENROUTER_API_KEY to enable "
            "Llama Guard. Running without a gate."
        )
        return
    logger.info("Safety guard ready (%s).", settings.GUARD_MODEL)


def _describe(verdict: Verdict) -> str:
    if verdict.categories:
        return f"{verdict.reason} [{verdict.labels}]"
    return verdict.reason


def guard(message: str) -> tuple[bool, Optional[str]]:
    """Input gate. Returns (fired, response); response is None when clean."""
    if not settings.GUARD_ENABLED:
        return False, None

    verdict = classify(
        message, purpose="input", ignore=settings.GUARD_IGNORE_CATEGORIES
    )
    if verdict.safe:
        return False, None

    logger.info(
        "Input blocked by guard | categories=%s", verdict.categories or "unspecified"
    )
    return True, REFUSAL


def guard_output(text: str) -> tuple[bool, Optional[str]]:
    """Output gate. Returns (fired, replacement_text)."""
    if not settings.GUARD_ENABLED:
        return False, None

    verdict = classify(
        text, purpose="output", ignore=settings.GUARD_OUTPUT_IGNORE_CATEGORIES
    )
    if verdict.safe:
        return False, None

    logger.info(
        "Output blocked by guard | categories=%s", verdict.categories or "unspecified"
    )
    return True, OUTPUT_REFUSAL


def guard_verdict(text: str, *, purpose: str = "input") -> Verdict:
    """Expose the structured verdict for callers that want the categories."""
    return classify(text, purpose=purpose)


def format_hazards(codes: tuple[str, ...]) -> str:
    """Human-readable hazard list, e.g. 'S9 indiscriminate weapons'."""
    return ", ".join(f"{code} {HAZARDS.get(code, 'unknown')}" for code in codes)


__all__ = [
    "guard",
    "guard_output",
    "guard_verdict",
    "format_hazards",
    "initialize_rails",
]