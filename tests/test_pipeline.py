"""End-to-end pipeline tests with fake embedder/reranker/LLM.

Includes the mandatory security test (§16): the mock provider must only ever receive
scrubbed (redacted) context, never raw PII.
"""
import re

import pytest
import pymupdf

from src.generation.aggregator import NO_CONTEXT_ANSWER
from src.generation.llm import (
    AllProvidersFailed,
    LLMResponse,
    ScrubbedText,
)
from src.ingestion.chunker import Chunker
from src.rag_pipeline import RAGPipeline
from src.retrieval.reranker import RankedChunk
from src.routing.semantic_router import RouteDecision

AADHAAR = "2345 6789 0124"
_AADHAAR_RE = re.compile(r"\b\d{4} \d{4} \d{4}\b")


# ------------------------------------------------------------------ fakes
class FakeVectorDB:
    def __init__(self):
        self.store: list = []

    def count(self):
        return len(self.store)

    def reset(self):
        self.store = []

    def add_chunks(self, documents, embeddings):
        self.store = list(documents)
        return len(documents)

    def search(self, query_embedding, k):
        return self.store[:k]

    def get_all_documents(self):
        return list(self.store)


class FakeRetriever:
    def __init__(self):
        self.documents: list = []
        self.rebuilt: list = []

    def reset(self):
        self.documents = []

    def rebuild_bm25(self, documents):
        self.documents = list(documents)
        self.rebuilt = list(documents)

    def hybrid_retrieve(self, query):
        return list(self.documents)


class FakeReranker:
    def __init__(self, always_return=None):
        self.always_return = always_return

    def rerank(self, query, docs, top_k=None):
        if self.always_return is not None:
            return self.always_return
        ranked = [
            RankedChunk(d.page_content, dict(d.metadata or {}), 1.0 / (i + 1))
            for i, d in enumerate(docs)
        ]
        return ranked[: top_k or 5]


class FakeScrubber:
    """Realistic stand-in: redacts 4-4-4-4 Aadhaar-like numbers."""

    @staticmethod
    def scrub(text):
        if isinstance(text, ScrubbedText):
            return text
        cleaned = _AADHAAR_RE.sub("<AADHAAR_NUMBER>", text)
        return ScrubbedText(cleaned)


class FakeRouter:
    def route(self, query, available_providers):
        return RouteDecision(
            intent="lookup",
            provider=available_providers[0],
            provider_order=list(available_providers),
            reason="fake routing",
            matched=["what is"],
        )


class FakeLLMHandler:
    def __init__(self):
        self.available_providers = ["fake"]
        self.captured = None
        self.fail = False

    def generate(self, query, context, provider_order):
        assert isinstance(query, ScrubbedText), "provider must receive ScrubbedText"
        assert isinstance(context, ScrubbedText), "provider must receive ScrubbedText"
        self.captured = (str(query), str(context))
        if self.fail:
            raise AllProvidersFailed({p: "boom" for p in provider_order})
        return LLMResponse(
            text="7.25 percent",
            provider="fake",
            model="fake-1",
            latency_s=0.1,
            fallbacks_tried=[],
        )


class FakeEmbedder:
    def embed_documents(self, documents, batch_size=8):
        return [[0.1, 0.2, 0.3, 0.4] for _ in documents]


# ------------------------------------------------------------------ helpers
def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Borrower is Rahul Sharma. Loan amount is 4,50,000. Interest rate is "
        "7.25 per cent per annum. Repayment tenure is 20 years.",
    )
    doc.save(str(path))
    doc.close()


def build(**overrides):
    defaults = {
        "vector_db": FakeVectorDB(),
        "retriever": FakeRetriever(),
        "reranker": FakeReranker(),
        "scrubber": FakeScrubber(),
        "router": FakeRouter(),
        "llm_handler": FakeLLMHandler(),
        "chunker": Chunker(chunk_size=50, chunk_overlap=10),
        "embedder": FakeEmbedder(),
    }
    defaults.update(overrides)
    return RAGPipeline(**defaults)


@pytest.fixture()
def pipeline():
    return build()


@pytest.fixture()
def pdf_path(tmp_path):
    path = tmp_path / "tiny_loan.pdf"
    make_pdf(path)
    return path


# ------------------------------------------------------------------ tests
def test_ingest_document_sets_state(pipeline, pdf_path):
    result = pipeline.ingest_document(pdf_path)
    assert result.pages >= 1
    assert result.chunks >= 1
    assert result.vectors_added == result.chunks
    assert pipeline.active_document_id == "tiny_loan"
    assert pipeline.has_document


def test_answer_returns_cited_answer(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    answer = pipeline.answer_query("What is the interest rate?")
    assert answer.text == "7.25 percent"
    assert answer.provider == "fake"
    assert answer.citations, "expected citations from retrieved pages"
    assert answer.citations[0].page_number >= 1


def test_answer_toplevel_uses_fallback_chain(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    answer = pipeline.answer_query("Set the rate?")
    assert "fake" in answer.provider


def test_no_document_returns_safe_answer(pipeline):
    answer = pipeline.answer_query("Anything")
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.citations == []


def test_empty_query_is_rejected(pipeline):
    answer = pipeline.answer_query("   ")
    assert answer.text == NO_CONTEXT_ANSWER
    assert "question" in answer.routing_reason.lower()


def test_no_reranked_context(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.reranker = FakeReranker(always_return=[])
    answer = pipeline.answer_query("What is the rate?")
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.citations == []


def test_all_providers_failing_is_graceful(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.llm_handler.fail = True
    answer = pipeline.answer_query("What is the rate?")
    assert "unavailable" in answer.text
    assert answer.citations == []


def test_provider_never_receives_raw_pii(pipeline, pdf_path):
    """Security gate (§16): mock provider must get scrubbed context only."""
    page = pymupdf.open()
    p = page.new_page()
    p.insert_text(
        (72, 72),
        f"Reference id {AADHAAR} on file for this loan. "
        "This is a confidential document prepared for the underwriting "
        "and credit review process only.",
    )
    page.save(str(pdf_path))
    page.close()

    pipeline.ingest_document(pdf_path)
    pipeline.answer_query("What reference is on file?")
    query_text, context_text = pipeline.llm_handler.captured
    assert AADHAAR not in context_text
    assert AADHAAR not in query_text
    assert "<AADHAAR_NUMBER>" in context_text