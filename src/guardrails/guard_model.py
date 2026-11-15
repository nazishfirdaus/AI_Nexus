"""Llama Guard safety classification for prompts and generated answers.

Replaces the previous prompt-judged Colang gate, which only fired when the chat
model happened to repeat one of five hardcoded sentences. Llama Guard returns a
structured verdict ("safe" or "unsafe" plus hazard codes), so the decision does
not depend on the chat model's phrasing.

Availability: Llama Guard 4 is served by OpenRouter. Groq no longer hosts any
Llama Guard model, so a Groq-only deployment runs without the gate and says so.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

HAZARDS: dict[str, str] = {
    "S1": "violent crimes",
    "S2": "non-violent crimes",
    "S3": "sex-related crimes",
    "S4": "child sexual exploitation",
    "S5": "defamation",
    "S6": "specialized advice",
    "S7": "privacy",
    "S8": "intellectual property",
    "S9": "indiscriminate weapons",
    "S10": "hate",
    "S11": "self-harm",
    "S12": "sexual content",
    "S13": "elections",
}

REFUSAL = (
    "I can't help with that request. I'm a loan-document assistant — "
    "ask me about the documents in your file."
)


@dataclass(frozen=True)
class Verdict:
    """Outcome of one classification call."""

    safe: bool
    categories: tuple[str, ...] = ()
    reason: str = ""
    unavailable: bool = False

    @property
    def labels(self) -> str:
        return ", ".join(
            f"{code} {HAZARDS.get(code, 'unknown')}" for code in self.categories
        )


_client = None
_client_lock = threading.Lock()
_warned_unavailable = False


def is_available() -> bool:
    """True when a Llama Guard model can be called."""
    return bool(settings.OPENROUTER_API_KEY)


def _client_or_none():
    global _client
    with _client_lock:
        if _client is None:
            if not settings.OPENROUTER_API_KEY:
                return None
            from openai import OpenAI

            _client = OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
            )
        return _client


def _parse(raw: str) -> Verdict:
    """Llama Guard replies 'safe' or 'unsafe' followed by hazard codes."""
    text = (raw or "").strip()
    lowered = text.lower()
    if "unsafe" not in lowered:
        return Verdict(safe=True, reason=text[:200])

    codes = tuple(
        dict.fromkeys(
            code
            for token in lowered.replace(",", " ").split()
            for code in [token.strip(".;:")]
            if code in HAZARDS
        )
    )
    if not codes:
        codes = tuple(
            code
            for code in HAZARDS
            if code.lower() in lowered
        )
    return Verdict(safe=False, categories=codes, reason=text[:200])


@lru_cache(maxsize=512)
def _classify_cached(text: str, purpose: str) -> Verdict:
    client = _client_or_none()
    if client is None:
        return Verdict(safe=True, unavailable=True, reason="no Llama Guard provider")

    try:
        response = client.chat.completions.create(
            model=settings.GUARD_MODEL,
            messages=[{"role": "user", "content": text}],
            temperature=0,
            max_tokens=64,
        )
        return _parse(response.choices[0].message.content or "")
    except Exception as exc:  # noqa: BLE001 - policy decided by the caller
        logger.warning("Llama Guard call failed (%s): %s", purpose, exc)
        if settings.GUARD_FAIL_OPEN:
            return Verdict(safe=True, reason=f"guard unavailable: {type(exc).__name__}")
        return Verdict(
            safe=False,
            reason=f"guard unavailable (fail-closed): {type(exc).__name__}",
            unavailable=True,
        )


def classify(
    text: str,
    *,
    purpose: str = "input",
    ignore: tuple[str, ...] = (),
) -> Verdict:
    """Classify ``text``; never raises.

    ``ignore`` drops hazard codes before the verdict is decided, so a reply that
    only trips an ignored category stays allowed.
    """
    global _warned_unavailable

    if not settings.GUARD_ENABLED:
        return Verdict(safe=True, reason="guard disabled")

    body = (text or "").strip()
    if not body:
        return Verdict(safe=True)

    verdict = _classify_cached(body[: settings.GUARD_MAX_CHARS], purpose)
    if verdict.unavailable:
        if not _warned_unavailable:
            _warned_unavailable = True
            logger.warning(
                "Llama Guard unavailable (%s); running without the safety gate. "
                "Set OPENROUTER_API_KEY or ANEXUS_GUARD_FAIL_OPEN=false to change this.",
                verdict.reason,
            )
        return verdict

    if verdict.safe or not ignore:
        return verdict

    kept = tuple(code for code in verdict.categories if code not in ignore)
    if len(kept) == len(verdict.categories):
        return verdict
    if not kept:
        return Verdict(
            safe=True,
            reason=f"only ignored categories: {', '.join(verdict.categories)}",
        )
    return Verdict(safe=False, categories=kept, reason=verdict.reason)