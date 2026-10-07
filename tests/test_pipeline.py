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
_ACCOUNT_RE = re.compile(r"(?:A/c|Account)\s*(?:No\.?|Number)?\s*:?\s*\d{8,20}", re.IGNORECASE)


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
        self.queries: list = []

    def reset(self):
        self.documents = []

    def rebuild_bm25(self, documents):
        self.documents = list(documents)
        self.rebuilt = list(documents)

    def hybrid_retrieve(self, query):
        self.queries.append(query)
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
    """Realistic stand-in: redacts Aadhaar-like numbers and an account number.

    Honours context_prefix so a label carried over from the previous chunk still
    counts, exactly as the real scrubber does.
    """

    @staticmethod
    def scrub(text, context_prefix=""):
        if isinstance(text, ScrubbedText):
            return text
        combined = f"{context_prefix}\n{text}"
        combined = _AADHAAR_RE.sub("<AADHAAR_NUMBER>", combined)
        combined = _ACCOUNT_RE.sub("<IN_ACCOUNT_NO>", combined)
        boundary = len(context_prefix) + 1
        # A substitution that began inside the carry-over crosses the boundary; drop
        # the carried part of it, the way the real scrubber rebases its results.
        for placeholder in re.finditer(r"<[A-Z_]+>", combined):
            if placeholder.start() < boundary < placeholder.end():
                boundary = placeholder.end()
        return ScrubbedText(combined[boundary:])


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
        self.captured_history = None
        self.captured_summary = None
        self.message_calls: list = []
        self.message_reply: str = "[SIMULATED] no rewrite"
        self.message_fail = False
        self.fail = False

    def generate(self, query, context, provider_order, history=None, summary=None):
        assert isinstance(query, ScrubbedText), "provider must receive ScrubbedText"
        assert isinstance(context, ScrubbedText), "provider must receive ScrubbedText"
        for turn in history or []:
            assert isinstance(turn.get("content"), ScrubbedText), (
                "provider must receive scrubbed history"
            )
        if summary is not None:
            assert isinstance(summary, ScrubbedText), "provider must receive scrubbed summary"
        self.captured = (str(query), str(context))
        self.captured_history = history
        self.captured_summary = summary
        if self.fail:
            raise AllProvidersFailed({p: "boom" for p in provider_order})
        return LLMResponse(
            text="7.25 percent",
            provider="fake",
            model="fake-1",
            latency_s=0.1,
            fallbacks_tried=[],
        )

    def generate_messages(self, messages, provider_order):
        for message in messages:
            assert isinstance(message.get("content"), ScrubbedText), (
                "provider must receive scrubbed rewrite/summary prompts"
            )
        self.message_calls.append(list(messages))
        if self.message_fail:
            raise AllProvidersFailed({p: "boom" for p in provider_order})
        return LLMResponse(
            text=self.message_reply,
            provider="fake",
            model="fake-1",
            latency_s=0.05,
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


def test_provider_never_receives_account_number(pipeline, pdf_path):
    """Same gate, for an Indian account number rather than an Aadhaar."""
    page = pymupdf.open()
    p = page.new_page()
    p.insert_text(
        (72, 72),
        "Borrower savings A/c No. 30123456789 held with the lender. "
        "This is a confidential document prepared for the underwriting "
        "and credit review process only.",
    )
    page.save(str(pdf_path))
    page.close()

    pipeline.ingest_document(pdf_path)
    pipeline.answer_query("What is the account number?")
    _, context_text = pipeline.llm_handler.captured
    assert "30123456789" not in context_text


# ------------------------------------------------------------------ multi-turn memory
def _history(*pairs):
    return [{"role": role, "content": content} for role, content in pairs]


class SelectiveRetriever(FakeRetriever):
    """Returns nothing for queries containing 'gibberish' (rewrite-miss tests)."""

    def hybrid_retrieve(self, query):
        self.queries.append(query)
        if "gibberish" in query:
            return []
        return list(self.documents)


def test_history_is_passed_to_the_llm(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    history = _history(("user", "What is the repayment tenure?"), ("assistant", "20 years"))
    answer = pipeline.answer_query("What is the interest rate?", chat_history=history)
    assert answer.text == "7.25 percent"
    assert [t["content"] for t in pipeline.llm_handler.captured_history] == [
        "What is the repayment tenure?",
        "20 years",
    ]


def test_history_pii_is_scrubbed_before_the_llm(pipeline, pdf_path):
    """The PII gate must cover conversation history, not just retrieval context."""
    pipeline.ingest_document(pdf_path)
    history = _history(
        ("user", f"My Aadhaar is {AADHAAR}"),
        ("assistant", "Noted."),
    )
    pipeline.answer_query("What is the rate?", chat_history=history)
    captured_history = pipeline.llm_handler.captured_history
    joined = " ".join(t["content"] for t in captured_history)
    assert AADHAAR not in joined
    assert "<AADHAAR_NUMBER>" in joined
    query_text, context_text = pipeline.llm_handler.captured
    assert AADHAAR not in query_text
    assert AADHAAR not in context_text


def test_empty_history_and_summary_reach_the_llm_as_empty(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.answer_query("What is the rate?")
    assert pipeline.llm_handler.captured_history == []
    assert pipeline.llm_handler.captured_summary is None
    assert pipeline.llm_handler.message_calls == []  # no rewrite, no summary


def test_summary_is_passed_to_the_llm(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.answer_query(
        "What is the rate?", conversation_summary="User focuses on interest rates."
    )
    assert pipeline.llm_handler.captured_summary == "User focuses on interest rates."


def test_blank_summary_stays_none(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.answer_query("What is the rate?", conversation_summary="   ")
    assert pipeline.llm_handler.captured_summary is None


def test_followup_query_is_rewritten_for_retrieval(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.llm_handler.message_reply = "What is the interest rate on the loan?"
    history = _history(("user", "Tell me about the loan"), ("assistant", "It exists."))
    answer = pipeline.answer_query("What about the rate?", chat_history=history)

    assert answer.text == "7.25 percent"
    # retrieval, routing and the prompt question all use the standalone query
    assert pipeline.retriever.queries[0] == "What is the interest rate on the loan?"
    assert pipeline.llm_handler.captured[0] == "What is the interest rate on the loan?"
    # the rewrite prompt saw the conversation it needs to resolve the reference
    rewrite_messages = pipeline.llm_handler.message_calls[0]
    assert rewrite_messages[0]["role"] == "system"
    assert "Tell me about the loan" in rewrite_messages[1]["content"]
    assert "What about the rate?" in rewrite_messages[1]["content"]


def test_failed_rewrite_falls_back_to_the_raw_query(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    history = _history(("user", "previous question"), ("assistant", "previous answer"))
    pipeline.answer_query("What about the rate?", chat_history=history)
    # default message_reply is the simulated marker -> rewriter declines
    assert pipeline.retriever.queries[0] == "What about the rate?"


def test_rewrite_errors_fall_back_to_the_raw_query(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    pipeline.llm_handler.message_fail = True
    history = _history(("user", "previous question"), ("assistant", "previous answer"))
    pipeline.answer_query("What about the rate?", chat_history=history)
    assert pipeline.retriever.queries == ["What about the rate?"]


def test_selfcontained_query_is_not_rewritten(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    history = _history(("user", "q"), ("assistant", "a"))
    pipeline.answer_query(
        "What is the interest rate charged on this loan agreement?",
        chat_history=history,
    )
    assert pipeline.llm_handler.message_calls == []
    assert pipeline.retriever.queries == [
        "What is the interest rate charged on this loan agreement?"
    ]


def test_rewrite_that_retrieves_nothing_retries_the_raw_query(pipeline, pdf_path):
    pipeline.ingest_document(pdf_path)
    selective = SelectiveRetriever()
    selective.documents = list(pipeline.retriever.documents)
    pipeline.retriever = selective
    pipeline.llm_handler.message_reply = "gibberish unrelated rewrite"

    history = _history(("user", "a"), ("assistant", "b"))
    answer = pipeline.answer_query("What about the rate?", chat_history=history)

    assert answer.text == "7.25 percent"
    assert selective.queries == ["gibberish unrelated rewrite", "What about the rate?"]
    # raw query wins the fallback, so the prompt question reverts to it too
    assert pipeline.llm_handler.captured[0] == "What about the rate?"


# ------------------------------------------------------------------ summarization
def _turns(n):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(n)
    ]


def test_summarize_below_trigger_is_a_noop(pipeline):
    from config import settings

    history = _turns(settings.HISTORY_SUMMARY_TRIGGER)
    assert pipeline.summarize_history(history, "existing summary") is None
    assert pipeline.llm_handler.message_calls == []


def test_summarize_folds_old_turns_into_a_new_summary(pipeline):
    from config import settings

    pipeline.llm_handler.message_reply = "User asked about rates; assistant cited page 3."
    history = _turns(settings.HISTORY_SUMMARY_TRIGGER + 8)
    result = pipeline.summarize_history(history, "old summary")

    assert result == "User asked about rates; assistant cited page 3."
    user_content = str(pipeline.llm_handler.message_calls[0][1]["content"])
    assert "old summary" in user_content          # previous summary is carried forward
    assert "turn 0" in user_content               # folded-in turns are in the transcript
    assert f"turn {len(history) - 1}" not in user_content  # recent turns stay verbatim


def test_summarize_llm_failure_returns_none(pipeline):
    from config import settings

    pipeline.llm_handler.message_fail = True
    history = _turns(settings.HISTORY_SUMMARY_TRIGGER + 8)
    assert pipeline.summarize_history(history, "keep me") is None


def test_summarize_simulated_reply_returns_none(pipeline):
    from config import settings

    history = _turns(settings.HISTORY_SUMMARY_TRIGGER + 8)
    # default message_reply starts with [SIMULATED] -> not a usable summary
    assert pipeline.summarize_history(history, "keep me") is None


# ------------------------------------------------------------------ doc restore
def _vector_db_with(*metadatas):
    from langchain_core.documents import Document

    db = FakeVectorDB()
    db.store = [
        Document(page_content=f"chunk {i}", metadata=dict(meta))
        for i, meta in enumerate(metadatas)
    ]
    return db


def test_restore_active_document_from_chunk_metadata():
    db = _vector_db_with(
        {"document_id": "loan_doc", "filename": "loan.pdf"},
        {"document_id": "loan_doc", "filename": "loan.pdf"},
    )
    pipe = build(vector_db=db)
    assert pipe.active_document_id is None
    assert pipe.restore_active_document() is True
    assert pipe.active_document_id == "loan_doc"
    assert pipe.active_filename == "loan.pdf"
    assert pipe.has_document
    # already active: no-op
    assert pipe.restore_active_document() is False


def test_restore_refuses_to_guess_with_mixed_documents():
    db = _vector_db_with(
        {"document_id": "a", "filename": "a.pdf"},
        {"document_id": "b", "filename": "b.pdf"},
    )
    pipe = build(vector_db=db)
    assert pipe.restore_active_document() is False
    assert pipe.active_document_id is None


def test_restore_refuses_incomplete_metadata():
    db = _vector_db_with({"page_number": 1}, {"page_number": 2})
    pipe = build(vector_db=db)
    assert pipe.restore_active_document() is False
    assert pipe.active_document_id is None


def test_restore_with_empty_vectordb_is_false():
    pipe = build()
    assert pipe.restore_active_document() is False