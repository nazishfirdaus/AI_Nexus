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


# ---------------- Indian names / addresses ----------------
def test_redacts_indian_name_with_honorific(scrubber):
    out = scrubber.scrub("Borrower's name Mr. Rahul Sharma.")
    assert "Rahul" not in out and "Sharma" not in out


@pytest.mark.parametrize("address,secret", [
    ("Address: S/o Ramesh Gupta, H.No. 12, Gomti Nagar, Lucknow - 226010", "Ramesh"),
    ("H.No. 42, Sector 18, Gurugram", "Sector 18"),
])
def test_redacts_indian_address(scrubber, address, secret):
    out = scrubber.scrub(address)
    assert secret not in out and "<IN_ADDRESS>" in out


def test_bare_city_name_survives(scrubber):
    """LOCATION is off on purpose - a property city is a fact the user asked for."""
    text = "Property is located in Noida, Uttar Pradesh."
    assert scrubber.scrub(text) == text


def test_address_redaction_does_not_bleed_into_the_next_sentence(scrubber):
    """An over-long address pattern would swallow the PAN and Aadhaar label after it."""
    text = ("Address: S/o Ram Nath Sharma, H.No. 12, Gomti Nagar, Lucknow - 226010. "
            "PAN ABCDE1234F. Aadhaar 2345 6789 0124.")
    out = scrubber.scrub(text)
    assert "Ram Nath" not in out and "226010" not in out
    # The next sentence keeps its own recognizers rather than vanishing into the address.
    assert "<PAN_NUMBER>" in out and "<AADHAAR_NUMBER>" in out
    assert out.startswith("Address: <IN_ADDRESS>")


def test_pincode_only_redacted_next_to_an_address_word(scrubber):
    assert "<IN_PINCODE>" in scrubber.scrub("PIN code 226010, District Lucknow.")
    assert "226010" in scrubber.scrub("Score reference 226010 was verified.")


# ---------------- Indian financial identifiers ----------------
@pytest.mark.parametrize("text,placeholder,secret", [
    ("A/c No. 30123456789 is the savings account.", "<IN_ACCOUNT_NO>", "30123456789"),
    ("Account Number: 12345678901", "<IN_ACCOUNT_NO>", "12345678901"),
    ("Beneficiary A/c No. 50100234567890", "<IN_ACCOUNT_NO>", "50100234567890"),
    ("Loan A/c No. 77654321001 was disbursed.", "<IN_LOAN_NO>", "77654321001"),
    ("Loan No: LA2024000123", "<IN_LOAN_NO>", "LA2024000123"),
    ("IFSC SBIN0001234", "<IFSC_CODE>", "SBIN0001234"),
    ("MICR 110002", "<MICR_CODE>", "110002"),
    ("GSTIN 27AAPFU0939F1ZV", "<GSTIN>", "27AAPFU0939F1ZV"),
    ("UPI id rahul@okhdfcbank for collections", "<UPI_ID>", "rahul@okhdfcbank"),
    ("Pay to priya@okaxis", "<UPI_ID>", "priya@okaxis"),
    ("Cheque No. 0045121", "<CHEQUE_NO>", "0045121"),
    ("Policy No. LIC-IND-4455621", "<POLICY_NO>", "LIC-IND-4455621"),
    ("Application No. LA/2026/8891", "<APPLICATION_ID>", "LA/2026/8891"),
])
def test_redacts_indian_financial_identifiers(scrubber, text, placeholder, secret):
    out = scrubber.scrub(text)
    assert secret not in out and placeholder in out


def test_account_label_is_kept_so_the_answer_stays_readable(scrubber):
    out = scrubber.scrub("A/c No. 30123456789 is the savings account.")
    assert out.startswith("A/c No. ") and "30123456789" not in out


@pytest.mark.parametrize("text,placeholder,secret", [
    ("Voter ID ABC1234567", "<VOTER_ID>", "ABC1234567"),
    ("TAN ABCD1234E5", "<TAN_NUMBER>", "ABCD1234E5"),
    ("Driving Licence MH1220110001234", "<DL_NO>", "MH1220110001234"),
    ("Passport No. Z4578901", "<PASSPORT_NO>", "Z4578901"),
])
def test_redacts_indian_identity_documents(scrubber, text, placeholder, secret):
    out = scrubber.scrub(text)
    assert secret not in out and placeholder in out


# ---------------- false-positive guards ----------------
# Every identifier added for Indian forms also matches a shape that ordinary loan
# figures share. These must survive, or answers stop being useful.
@pytest.mark.parametrize("text", [
    "Loan amount is 4,50,000 at 7.25 percent, first EMI due on 5 January 2027 in Lucknow.",
    "The EMI is 12500 per month for 240 months.",
    "Property is located in Noida, Uttar Pradesh.",
    "The sanctioned amount was 450000 rupees.",
    "Loan tenure 240 months, margin 25%.",
    "Interest rate 7.25% for 20 years.",
    "The amount 12,00,000 was sanctioned in favour of the applicant.",
    "Property value assessed at 1.2 crore in Gurgaon.",
    "Repayment starts January 2027 and ends December 2046.",
])
def test_loan_facts_survive_the_indian_patterns(scrubber, text):
    assert scrubber.scrub(text) == text


def test_bare_digit_run_is_not_an_account_number(scrubber):
    """Account/loan numbers are label-anchored; an unlabelled run is left alone."""
    text = "Serial 12345678901234 was noted during verification."
    assert scrubber.scrub(text) == text


def test_prose_account_of_amount_is_not_an_account_number(scrubber):
    text = "The account of 450000 rupees was frozen by the lender."
    assert scrubber.scrub(text) == text


def test_loan_amount_is_not_read_as_a_loan_number(scrubber):
    text = "Loan amount 450000 at 7.25 percent."
    assert scrubber.scrub(text) == text


def test_specific_entity_wins_over_generic_person(scrubber):
    """spaCy calls "Passport No. Z4578901" a PERSON; the precise label must survive."""
    out = scrubber.scrub("Passport No. Z4578901")
    assert "<PASSPORT_NO>" in out and "Z4578901" not in out


def test_every_configured_entity_has_a_recognizer(scrubber):
    """Guards against a name in settings.PII_ENTITIES that nothing can ever detect."""
    from config import settings

    analyzer, _ = scrubber._engines()
    supported = set()
    for recognizer in analyzer.registry.recognizers:
        supported.update(recognizer.get_supported_entities())
    assert set(settings.PII_ENTITIES) <= supported, (
        set(settings.PII_ENTITIES) - supported
    )


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
    q, ctx, pages = build_llm_inputs("What is Rahul Sharma's loan amount?", chunks, scrubber)
    assert isinstance(q, ScrubbedText) and isinstance(ctx, ScrubbedText)
    assert "Rahul" not in q and "Rahul" not in ctx and "6789" not in ctx
    assert "[Page 2]" in ctx and "[Page 5]" in ctx and "4,50,000" in ctx
    # The page map carries the same scrubbed text the aggregator builds snippets
    # from, keyed by (document_id, page) so two documents never collide.
    assert set(pages) == {("", 2), ("", 5)}
    assert "Rahul" not in pages[("", 2)] and "4,50,000" in pages[("", 5)]


def test_build_llm_inputs_labels_blocks_with_the_source_document(scrubber):
    """Multi-doc context: every block shows which document it came from."""
    chunk = Chunk("Loan amount is 4,50,000.", 1)
    chunk.metadata.update({"document_id": "agreement", "filename": "agreement.pdf"})
    _, ctx, pages = build_llm_inputs("What is the loan amount?", [chunk], scrubber)
    assert "[Doc: agreement.pdf | Page 1]" in ctx
    assert set(pages) == {("agreement", 1)}
    assert "4,50,000" in pages[("agreement", 1)]


def test_account_number_is_redacted_when_chunking_splits_it_from_its_label(scrubber):
    """Regression: the chunker put "A/c No" and the digits in different chunks.

    Without the carry-over the second chunk holds a bare 11-digit run with nothing to
    anchor on, and a label-anchored pattern legitimately finds nothing to match.
    """
    chunks = [Chunk("Borrower savings A/c No", 1),
              Chunk("30123456789 held with the lender", 1),
              Chunk("Loan amount is 4,50,000.", 2)]
    _, ctx, _ = build_llm_inputs("What is the account number?", chunks, scrubber)
    assert "30123456789" not in ctx
    assert "<IN_ACCOUNT_NO>" in ctx
    assert "4,50,000" in ctx


def test_context_prefix_is_not_returned(scrubber):
    """The carried prefix must stay out of the output; only the value is redacted."""
    out = scrubber.scrub("30123456789 held with the lender", context_prefix="A/c No")
    assert out == "<IN_ACCOUNT_NO> held with the lender"


def test_context_prefix_does_not_break_ordinary_text(scrubber):
    text = "Loan amount is 4,50,000."
    assert scrubber.scrub(text, context_prefix="Application No") == text


def test_scrubbed_inputs_are_accepted_by_llm_handler(scrubber):
    q, ctx, _ = build_llm_inputs("What is the loan amount?", [Chunk("Rahul owes 4,50,000", 1)], scrubber)
    h = LLMHandler({"simulated": SimulatedAdapter()}, QuotaTracker(state_path=None))
    assert h.generate(q, ctx, ["simulated"]).provider == "simulated"


def test_raw_text_cannot_reach_llm_even_in_simulated_mode():
    h = LLMHandler({"simulated": SimulatedAdapter()}, QuotaTracker(state_path=None))
    with pytest.raises(TypeError):
        h.generate("raw query", "raw context with Rahul Sharma", ["simulated"])


def test_get_scrubber_is_singleton():
    assert get_scrubber() is get_scrubber()