"""Tests for response/citation aggregation."""
from src.generation.aggregator import (
    NO_CONTEXT_ANSWER,
    Aggregator,
    Citation,
    FinalAnswer,
)
from src.generation.llm import LLMResponse
from src.redaction.presidio_scrubber import PresidioScrubber, ScrubError
from src.retrieval.reranker import RankedChunk


class StubScrubber:
    """Stand-in that redacts one marker string, so aggregation stays fast."""

    def __init__(self, marker="Mr. Rahul Sharma", replacement="[REDACTED]"):
        self.marker, self.replacement = marker, replacement

    def scrub(self, text):
        return text.replace(self.marker, self.replacement)


def make_response(text="The rate is 7.25%."):
    return LLMResponse(
        text=text,
        provider="nvidia",
        model="m",
        latency_s=1.2,
        fallbacks_tried=["groq"],
    )


def make_chunks():
    return [
        RankedChunk("Interest rate is 7.25 percent.", {"page_number": 3, "document_id": "d"}, 0.9),
        RankedChunk("Repayment tenure is 20 years.", {"page_number": 7, "document_id": "d"}, 0.8),
        RankedChunk("Interest rate is 7.25 percent (dup).", {"page_number": 3, "document_id": "d"}, 0.5),
    ]


def test_aggregate_attaches_citations_from_retrieved_only():
    answer = Aggregator().aggregate(make_response(), make_chunks())
    assert answer.text == "The rate is 7.25%."
    assert answer.provider == "nvidia"
    assert answer.fallbacks_tried == ["groq"]
    pages = [c.page_number for c in answer.citations]
    assert pages == [3, 7]  # deduped, highest score first


def test_pages_sorted_by_score_descending():
    answer = Aggregator().aggregate(make_response(), make_chunks())
    assert answer.citations[0].score == 0.9
    assert answer.citations[1].score == 0.8


def test_citation_snippets_truncated():
    answer = Aggregator().aggregate(make_response(), make_chunks(), snippet_chars=10)
    assert all(len(c.snippet) <= 10 for c in answer.citations)


def test_chunks_without_page_are_skipped():
    chunks = [RankedChunk("no page here", {"document_id": "d"}, 0.5)]
    answer = Aggregator().aggregate(make_response(), chunks)
    assert answer.citations == []


def test_unstringable_page_is_skipped():
    chunks = [RankedChunk("x", {"page_number": "unknown"}, 0.5)]
    answer = Aggregator().aggregate(make_response(), chunks)
    assert answer.citations == []


def test_no_context_has_no_citations():
    answer = Aggregator().no_context()
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.citations == []
    assert answer.provider == ""


def test_graceful_failure_has_message_and_no_citations():
    answer = Aggregator().graceful_failure("All providers down.")
    assert "providers down" in answer.text
    assert answer.citations == []


def test_to_dict_is_json_safe():
    answer = Aggregator().aggregate(make_response(), make_chunks())
    data = answer.to_dict()
    assert data["citations"][0] == {"page_number": 3, "score": 0.9, "snippet": data["citations"][0]["snippet"]}
    import json

    json.dumps(data)  # must not raise


# ---------------- post-LLM redaction of what the user actually sees ----------------
SECRET = "Mr. Rahul Sharma"


def pii_chunks():
    return [RankedChunk(f"Borrower {SECRET}, A/c No. 30123456789.", {"page_number": 2}, 0.9)]


def test_citation_snippets_are_scrubbed():
    """Snippets are built from raw stored chunks, so they must be scrubbed on the way out."""
    scrubber = StubScrubber(marker=SECRET)
    answer = Aggregator(scrubber=scrubber).aggregate(make_response(), pii_chunks())
    assert SECRET not in answer.citations[0].snippet
    assert "[REDACTED]" in answer.citations[0].snippet


def test_answer_text_is_scrubbed():
    response = make_response(f"The borrower is {SECRET} and the rate is 7.25%.")
    answer = Aggregator(scrubber=StubScrubber(marker=SECRET)).aggregate(response, make_chunks())
    assert SECRET not in answer.text and "7.25%" in answer.text


def test_scrubber_failure_degrades_instead_of_raising():
    """Nothing leaves the machine post-LLM, so a scrub failure must not kill the answer."""

    class BrokenScrubber:
        def scrub(self, text):
            raise ScrubError("boom")

    response = make_response(f"Borrower {SECRET}.")
    answer = Aggregator(scrubber=BrokenScrubber()).aggregate(response, pii_chunks())
    assert SECRET in answer.text and SECRET in answer.citations[0].snippet


def test_aggregator_without_scrubber_is_unchanged():
    """Back-compat: no scrubber wired means snippets pass through, as before."""
    answer = Aggregator().aggregate(make_response(), pii_chunks())
    assert SECRET in answer.citations[0].snippet


def test_real_scrubber_cleans_citation_snippet_end_to_end():
    chunks = [RankedChunk(
        "Borrower Mr. Rahul Sharma, S/o Ram Gupta, H.No. 9, Sector 18, Gurugram, "
        "Loan A/c No. 77654321001, IFSC SBIN0001234.",
        {"page_number": 4}, 0.9,
    )]
    answer = Aggregator(scrubber=PresidioScrubber()).aggregate(make_response(), chunks)
    snippet = answer.citations[0].snippet
    for secret in ("Rahul Sharma", "77654321001", "SBIN0001234"):
        assert secret not in snippet


def test_snippet_uses_the_pre_scrubbed_page_text():
    """A snippet sliced out of a raw chunk can lose the label its value needs.

    When the pipeline supplies the page map, snippets come from that instead, so they
    show exactly the text the model was given.
    """
    chunks = [RankedChunk("50100234567890 is the payout account.", {"page_number": 3}, 0.9)]
    answer = Aggregator().aggregate(
        make_response(), chunks,
        scrubbed_by_page={3: "Disbursal A/c No\n<IN_LOAN_NO> is the payout account."},
    )
    snippet = answer.citations[0].snippet
    assert "50100234567890" not in snippet
    assert snippet.startswith("Disbursal A/c No")