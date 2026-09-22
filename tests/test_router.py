"""Tests for the rule-based semantic router."""
import pytest

from src.routing.quota_tracker import QuotaTracker
from src.routing.semantic_router import SemanticRouter

ALL = ["nvidia", "gemini_flash", "groq"]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize("query,intent", [
    ("What is the loan amount?", "lookup"),
    ("How much is the EMI?", "lookup"),
    ("Summarize this document", "summary"),
    ("Give me an overview of the key terms", "summary"),
    ("Compare the income to the stated loan and flag any discrepancy", "analysis"),
    ("Why was the risk rated high?", "analysis"),
    ("hello there", "general"),
])
def test_classify(query, intent):
    assert SemanticRouter().classify(query)[0] == intent


def test_tie_prefers_analysis_over_lookup():
    # one analysis hit ("risk") and one lookup hit ("date")
    assert SemanticRouter().classify("risk date")[0] == "analysis"


def test_route_uses_intent_preference():
    d = SemanticRouter().route("Summarize this document", ALL)
    assert d.intent == "summary"
    assert d.provider == "gemini_flash"
    # intent picks the first provider; the rest follows FALLBACK_CHAIN
    assert d.provider_order == ["gemini_flash", "nvidia", "groq"]


def test_lookup_prefers_groq_then_chain():
    d = SemanticRouter().route("What is the loan amount?", ALL)
    assert d.provider_order == ["groq", "nvidia", "gemini_flash"]


def test_analysis_prefers_nvidia_then_chain():
    d = SemanticRouter().route("Compare income and flag any discrepancy", ALL)
    assert d.provider_order == ["nvidia", "gemini_flash", "groq"]


def test_route_is_explainable():
    d = SemanticRouter().route("What is the loan amount?", ALL)
    assert "lookup" in d.reason and d.matched


def test_general_follows_fallback_chain():
    d = SemanticRouter().route("hello there", ALL)
    assert d.provider_order == ["nvidia", "gemini_flash", "groq"]


def test_only_configured_providers_used():
    d = SemanticRouter().route("Summarize this document", ["groq"])
    assert d.provider_order == ["groq"]


def test_unknown_provider_appended_at_end():
    d = SemanticRouter().route("hello there", ["simulated"])
    assert d.provider_order == ["simulated"]


def test_no_providers_raises():
    with pytest.raises(ValueError):
        SemanticRouter().route("anything", [])


def test_cooling_provider_moved_to_back():
    clock = Clock()
    q = QuotaTracker(state_path=None, clock=clock)
    q.record_failure("gemini_flash", RuntimeError("HTTP 429 rate limit"))
    d = SemanticRouter(q).route("Summarize this document", ALL)
    assert d.provider != "gemini_flash"
    assert d.provider_order[-1] == "gemini_flash"
    assert d.skipped_cooling == ["gemini_flash"]
    assert "cooling" in d.reason


def test_provider_returns_after_cooldown():
    clock = Clock()
    q = QuotaTracker(state_path=None, clock=clock)
    q.record_failure("gemini_flash", RuntimeError("429"))
    clock.t += 3600
    d = SemanticRouter(q).route("Summarize this document", ALL)
    assert d.provider == "gemini_flash"


def test_router_is_deterministic():
    r = SemanticRouter()
    a = r.route("Compare the risk", ALL)
    b = r.route("Compare the risk", ALL)
    assert a.provider_order == b.provider_order