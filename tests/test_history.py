"""Tests for the pure history-window helpers (no I/O, no LLM)."""
from config import settings
from src.memory.history import (
    build_history,
    history_transcript,
    looks_like_followup,
)


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


# ------------------------------------------------------------------ build_history
def test_empty_history():
    summary, turns = build_history(None, None)
    assert summary is None and turns == []
    summary, turns = build_history([], "old summary")
    assert summary == "old summary" and turns == []


def test_keeps_role_and_content_only():
    history = [
        {"role": "user", "content": "q1", "citations": None, "id": 7},
        {"role": "assistant", "content": "a1", "citations": [{"page_number": 2}]},
    ]
    _, turns = build_history(history, None)
    assert turns == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_trims_to_most_recent_turns():
    history = _msgs(*[(("user" if i % 2 == 0 else "assistant"), f"m{i}") for i in range(10)])
    _, turns = build_history(history, None, max_turns=4)
    assert [t["content"] for t in turns] == ["m6", "m7", "m8", "m9"]


def test_max_turns_zero_keeps_nothing():
    _, turns = build_history(_msgs(("user", "q"), ("assistant", "a")), None, max_turns=0)
    assert turns == []


def test_negative_max_turns_keeps_everything():
    history = _msgs(*[(("user"), f"m{i}") for i in range(8)])
    _, turns = build_history(history, None, max_turns=-1)
    assert len(turns) == 8


def test_truncates_long_messages():
    history = [{"role": "user", "content": "x" * 5000}]
    _, turns = build_history(history, None, max_chars=100)
    assert len(turns[0]["content"]) <= 100
    assert turns[0]["content"].endswith("…")


def test_skips_empty_and_unknown_roles():
    history = [
        {"role": "user", "content": ""},
        {"role": "user", "content": "   "},
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "ok"},
    ]
    _, turns = build_history(history, None)
    assert turns == [{"role": "assistant", "content": "ok"}]


def test_blank_summary_becomes_none():
    summary, _ = build_history([], "   ")
    assert summary is None


def test_defaults_come_from_settings():
    history = _msgs(*[(("user"), f"m{i}") for i in range(20)])
    _, turns = build_history(history, None)
    assert len(turns) == settings.HISTORY_MAX_TURNS
    _, turns = build_history([{"role": "user", "content": "y" * 100000}], None)
    assert len(turns[0]["content"]) <= settings.HISTORY_MAX_MESSAGE_CHARS


# ---------------------------------------------------------------- looks_like_followup
def test_short_queries_are_followups():
    assert looks_like_followup("What about the second one?")
    assert looks_like_followup("And the rate?")
    assert looks_like_followup("Is it fixed?")


def test_marker_prefixes_are_followups():
    assert looks_like_followup("Tell me more about the prepayment charges and penalties")
    assert looks_like_followup("Does it change after the fixed period ends for this loan?")


def test_long_selfcontained_queries_are_not_followups():
    assert not looks_like_followup("What is the interest rate charged on this loan agreement?")
    assert not looks_like_followup("Show me the repayment schedule for the entire tenure.")
    # longer than the word threshold and not a marker
    assert not looks_like_followup("When was this mortgage loan agreement executed and signed by the borrower?")


def test_empty_query_is_not_a_followup():
    assert not looks_like_followup("")
    assert not looks_like_followup("   ")


# ------------------------------------------------------------------ history_transcript
def test_transcript_labels_roles():
    turns = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert history_transcript(turns) == "User: q\nAssistant: a"
