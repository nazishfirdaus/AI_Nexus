"""Tests for response/citation aggregation."""
from src.generation.aggregator import (
    NO_CONTEXT_ANSWER,
    Aggregator,
    Citation,
    FinalAnswer,
)
from src.generation.llm import LLMResponse
from src.retrieval.reranker import RankedChunk


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