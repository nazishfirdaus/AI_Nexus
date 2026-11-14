"""Serializers for stored chat rows.

Messages are persisted with the text that was displayed at the time. Answers
written before placeholder neutralisation existed can still contain
``<PERSON>``-style tokens, so every read path cleans them here rather than
requiring a database migration.
"""
from __future__ import annotations

from src.api.schemas import MessageOut
from src.redaction.placeholders import display_text, repair_mojibake


def message_out(row: dict) -> MessageOut:
    """Build a ``MessageOut`` with placeholders neutralised in text/snippets."""
    data = dict(row)
    content = str(data.get("content") or "")
    # Never rewrite a user's own words; only repair old encoding damage. The
    # assistant answer is where placeholders, IDs and addresses must be hidden.
    if data.get("role") == "user":
        data["content"] = repair_mojibake(content)
    else:
        data["content"] = display_text(content)

    citations = data.get("citations")
    if citations:
        cleaned = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            item = dict(citation)
            item["snippet"] = display_text(str(item.get("snippet") or ""))
            cleaned.append(item)
        data["citations"] = cleaned

    return MessageOut(**data)
