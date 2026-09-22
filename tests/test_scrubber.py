"""Security tests for the Presidio scrubber. These use the real Presidio engine and the
spaCy model (en_core_web_sm). They must FAIL, not skip, if the dependencies are missing."""
import logging

import pytest

from src.generation.llm import AllProvidersFailed, LLMHandler, ScrubbedText, SimulatedAdapter
from src.redaction import presidio_scrubber as ps
from src.redaction.presidio_scrubber import (
    PresidioScrubber, ScrubError, build_llm_inputs, get_scrubber, verhoeff_valid,
)
from src.routing.quota_tracker import QuotaTracker

VALID_AADHAAR = "2345 6789 0124"      # passes Verhoeff
INVALID_AADHAAR = "2345 6789 0125"    # fails Verhoeff


@pytest.fixture(scope="module")
def scrubber():
    return PresidioScrubber()


class Chunk:
    def __init__(self, content, page):
        self.content, self.metadata = content, {"page_number": page}


# ---------------- redaction ----------------
def test_redacts_common_pii(scrubber):
    text = ("Borrower Rahul Sharma, phone +91 98765 43210, email rahul.sharma@example.com, "
            "card 4111 1111 1111 1111, IP 192.168.1.10.")
    out, report = scrubber.scrub_with_report(text)
    for secret in ["Rahul", "98765", "rahul.sharma@example.com", "4111", "192.168.1.10"]:
        assert secret not in out
    assert {"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IP_ADDRESS"} <= set(report)


def test_redacts_valid_aadhaar_anywhere(scrubber):
    out = scrubber.scrub(f"Reference {VALID_AADHAAR} on file")
    assert "6789" not in out and "<AADHAAR_NUMBER>" in out


def test_redacts_valid_aadhaar_without_spaces_or_with_hyphens(scrubber):
    assert "<AADHAAR_NUMBER>" in scrubber.scrub("id 234567890124 here")
    assert "<AADHAAR_NUMBER>" in scrubber.scrub("id 2345-6789-0124 here")


def test_invalid_aadhaar_redacted_only_with_context(scrubber):
    assert "<AADHAAR_NUMBER>" in scrubber.scrub(f"Aadhaar no: {INVALID_AADHAAR}")
    assert INVALID_AADHAAR in scrubber.scrub(f"Random reference {INVALID_AADHAAR} on the schedule")


def test_redacts_pan(scrubber):
    out = scrubber.scrub("PAN ABCDE1234F submitted")
    assert "ABCDE1234F" not in out and "<PAN_NUMBER>" in out


def test_loan_facts_are_not_redacted(scrubber):
    text = "Loan amount is 4,50,000 at 7.25 percent, first EMI due on 5 January 2027 in Lucknow."
    assert scrubber.scrub(text) == text


def test_verhoeff():
    assert verhoeff_valid("234567890124")
    assert not verhoeff_valid("234567890125")


# ---------------- output contract ----------------
def test_returns_scrubbed_text_type(scrubber):
    assert isinstance(scrubber.scrub("hello"), ScrubbedText)
    assert isinstance(scrubber.scrub(""), ScrubbedText)


def test_idempotent(scrubber):
    once = scrubber.scrub("Rahul Sharma called")
    assert scrubber.scrub(once) == once


def test_rejects_non_string(scrubber):
    with pytest.raises(ScrubError):
        scrubber.scrub(None)


# ---------------- fail closed ----------------
def test_engine_failure_raises_never_returns_raw(scrubber, monkeypatch):
    scrubber._engines()
    monkeypatch.setattr(scrubber._analyzer, "analyze", lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ScrubError):
        scrubber.scrub("Rahul Sharma 2345 6789 0124")


def test_missing_spacy_model_raises():
    with pytest.raises(ScrubError):
        PresidioScrubber(spacy_model="no_such_model").scrub("Rahul Sharma")


# ---------------- logging must not leak PII ----------------
def test_logs_contain_counts_not_text(scrubber, caplog):
    with caplog.at_level(logging.DEBUG):
        scrubber.scrub("Rahul Sharma, email rahul.sharma@example.com")
    assert "Rahul" not in caplog.text and "example.com" not in caplog.text
    assert "EMAIL_ADDRESS" in caplog.text


# ---------------- pipeline helper ----------------
def test_build_llm_inputs_scrubs_query_and_chunks_keeps_pages(scrubber):
    chunks = [Chunk("Borrower Rahul Sharma, Aadhaar 2345 6789 0124.", 2),
              Chunk("Loan amount is 4,50,000.", 5)]
    q, ctx = build_llm_inputs("What is Rahul Sharma's loan amount?", chunks, scrubber)
    assert isinstance(q, ScrubbedText) and isinstance(ctx, ScrubbedText)
    assert "Rahul" not in q and "Rahul" not in ctx and "6789" not in ctx
    assert "[Page 2]" in ctx and "[Page 5]" in ctx and "4,50,000" in ctx


def test_scrubbed_inputs_are_accepted_by_llm_handler(scrubber):
    q, ctx = build_llm_inputs("What is the loan amount?", [Chunk("Rahul owes 4,50,000", 1)], scrubber)
    h = LLMHandler({"simulated": SimulatedAdapter()}, QuotaTracker(state_path=None))
    assert h.generate(q, ctx, ["simulated"]).provider == "simulated"


def test_raw_text_cannot_reach_llm_even_in_simulated_mode():
    h = LLMHandler({"simulated": SimulatedAdapter()}, QuotaTracker(state_path=None))
    with pytest.raises(TypeError):
        h.generate("raw query", "raw context with Rahul Sharma", ["simulated"])


def test_get_scrubber_is_singleton():
    assert get_scrubber() is get_scrubber()