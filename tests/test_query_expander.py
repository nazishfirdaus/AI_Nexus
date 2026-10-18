import pytest

from config import settings
from src.generation.llm import ScrubbedText
from src.retrieval.query_expander import QueryExpander, should_expand
from tests.test_pipeline import (  # reuse fakes
    FakeLLMHandler,
    FakeScrubber,
)


def test_should_expand_respects_words_and_enabled(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 7)
    assert not should_expand("rate?")
    assert should_expand("What is the interest rate on the loan agreement?")


def test_should_expand_disabled(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", False)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 1)
    assert not should_expand("What is the interest rate on the loan agreement?")


def test_expand_returns_paraphrases(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 7)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_VARIANTS", 2)
    monkeypatch.setattr(settings, "LLM_TEMPERATURE_EXPANSION", 0.4)

    handler = FakeLLMHandler()
    handler.message_reply = "How much interest does this loan charge?\nWhat rate applies to this loan?"
    expander = QueryExpander(handler, scrubber=FakeScrubber())
    out = expander.expand("What is the interest rate on the loan agreement?", ["fake"])
    assert out == ["How much interest does this loan charge?", "What rate applies to this loan?"]
    assert handler.message_temperatures[0] == 0.4
    # sent scrubbed messages
    assert isinstance(handler.message_calls[0][0]["content"], ScrubbedText)
    assert isinstance(handler.message_calls[0][1]["content"], ScrubbedText)


def test_expand_dedupes_and_caps(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 7)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_VARIANTS", 2)

    handler = FakeLLMHandler()
    handler.message_reply = (
        "- How much interest is charged?\n"
        "How much interest is charged?\n"
        "What is the rate?\n"
        "Fourth one"
    )
    expander = QueryExpander(handler)
    out = expander.expand("What is the interest rate on the loan agreement?")
    assert len(out) == 2
    # first two distinct beyond original are returned; original deduped
    assert out[0] in ("How much interest is charged?", "What is the rate?")
    # normalised dedup works
    assert len(set(s.lower() for s in out)) == len(out)


def test_expand_simulated_declines(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 7)
    handler = FakeLLMHandler()
    handler.message_reply = "[SIMULATED] no expansion"
    expander = QueryExpander(handler)
    assert expander.expand("What is the interest rate on the loan agreement?") == []


def test_expand_failure_declines(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 7)
    handler = FakeLLMHandler()
    handler.message_fail = True
    expander = QueryExpander(handler)
    assert expander.expand("What is the interest rate on the loan agreement?") == []


def test_expand_scrubs_user_message(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(settings, "QUERY_EXPANSION_MIN_WORDS", 4)
    handler = FakeLLMHandler()
    handler.message_reply = "What is the aadhaar mentioned in the record?"
    expander = QueryExpander(handler, scrubber=FakeScrubber())
    expander.expand("Tell me about the Aadhaar number 2345 6789 0124 in detail", ["fake"])
    assert isinstance(handler.message_calls[0][1]["content"], ScrubbedText)
    assert "<AADHAAR_NUMBER>" in str(handler.message_calls[0][1]["content"])
