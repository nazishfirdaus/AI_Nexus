"""Pure history-window helpers: trimming, truncation, follow-up detection.

No I/O and no LLM calls here, so everything is trivially unit-testable.
`build_history` decides what the model actually sees for a turn; the pipeline
scrubs whatever it returns before anything reaches a provider.
"""
from __future__ import annotations

from typing import Optional, Sequence

from config import settings

# Prefixes that mark a question as referring back to the conversation.
_FOLLOWUP_MARKERS = (
    "what about",
    "how about",
    "and the",
    "and its",
    "and their",
    "is it",
    "is that",
    "is there",
    "are they",
    "are there",
    "are these",
    "are those",
    "does it",
    "do they",
    "did it",
    "can it",
    "can they",
    "tell me more",
    "what else",
    "why is",
    "why does",
    "why did",
    "same ",
    "again",
    "the second",
    "the first",
    "the third",
    "that one",
    "this one",
    "those",
    "them",
)


def looks_like_followup(query: str) -> bool:
    """Heuristic gate for the rewrite step.

    True for short questions (<= REWRITE_MAX_WORDS words) and for questions
    starting with a follow-up marker, both of which usually depend on prior
    turns to be understood. Long, self-contained questions skip the extra
    LLM call entirely.
    """
    cleaned = (query or "").strip().lower()
    if not cleaned:
        return False
    if len(cleaned.split()) <= settings.REWRITE_MAX_WORDS:
        return True
    return cleaned.startswith(_FOLLOWUP_MARKERS)


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def build_history(
    chat_history: Optional[Sequence[dict]],
    summary: Optional[str] = None,
    max_turns: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> tuple[Optional[str], list[dict]]:
    """Window the conversation for one generation call.

    Returns (summary, turns): the summary to inject into the system prompt and
    the most recent `max_turns` messages as [{"role", "content"}] pairs, each
    truncated to `max_chars`. Citations and other keys are dropped; empty
    turns and non user/assistant roles are ignored.
    """
    max_turns = settings.HISTORY_MAX_TURNS if max_turns is None else max_turns
    max_chars = settings.HISTORY_MAX_MESSAGE_CHARS if max_chars is None else max_chars

    turns: list[dict] = []
    for msg in chat_history or []:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        turns.append({"role": role, "content": _truncate(content, max_chars)})

    if max_turns is not None and max_turns >= 0:
        turns = turns[len(turns) - max_turns:] if max_turns else []
    clean_summary = (summary or "").strip() or None
    return clean_summary, turns


def history_transcript(turns: Sequence[dict]) -> str:
    """Render turns as plain text for the rewrite / summarizer prompts."""
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels.get(t.get('role'), t.get('role'))}: {t.get('content', '')}"
        for t in turns
    )
