"""Tests for the cross-encoder reranker. The real model is replaced by a fake,
so nothing is downloaded and tests run instantly."""
import pytest
from langchain_core.documents import Document

from src.retrieval import reranker as rr
from src.retrieval.reranker import CrossEncoderReranker


class FakeModel:
    """Scores a pair by how many query words appear in the passage."""
    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        self.calls.append((len(pairs), batch_size))
        return [sum(w in p.lower() for w in q.lower().split()) for q, p in pairs]


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(rr, "_load_model", lambda name, device: model)
    return model


def docs():
    return [
        Document(page_content="signature page", metadata={"page_number": 5}),
        Document(page_content="loan amount and interest rate", metadata={"page_number": 2}),
        Document(page_content="loan appraisal", metadata={"page_number": 7}),
    ]


def test_sorted_by_score_descending(fake):
    out = CrossEncoderReranker().rerank("loan amount rate", docs())
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0].metadata["page_number"] == 2


def test_top_k_applied(fake):
    assert len(CrossEncoderReranker(top_k=2).rerank("loan", docs())) == 2


def test_top_k_override(fake):
    assert len(CrossEncoderReranker(top_k=1).rerank("loan", docs(), top_k=3)) == 3


def test_returns_content_and_metadata(fake):
    out = CrossEncoderReranker().rerank("loan amount", docs())
    assert out[0].content == "loan amount and interest rate"
    assert out[0].metadata == {"page_number": 2}
    d = out[0].to_dict()
    assert set(d) == {"content", "metadata", "score"}


def test_empty_inputs(fake):
    r = CrossEncoderReranker()
    assert r.rerank("loan", []) == []
    assert r.rerank("   ", docs()) == []


def test_blank_chunks_skipped(fake):
    d = docs() + [Document(page_content="   ", metadata={})]
    out = CrossEncoderReranker(top_k=10).rerank("loan", d)
    assert len(out) == 3


def test_batch_size_passed_to_model(fake):
    CrossEncoderReranker(batch_size=4).rerank("loan", docs())
    assert fake.calls[-1] == (3, 4)


def test_model_loaded_lazily(monkeypatch):
    called = []
    monkeypatch.setattr(rr, "_load_model", lambda n, d: called.append(1) or FakeModel())
    r = CrossEncoderReranker()
    assert called == []          # constructing must not load the model
    r.rerank("loan", docs())
    assert called == [1]