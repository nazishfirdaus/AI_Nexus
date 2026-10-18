from langchain_core.documents import Document

from config import settings
from tests.test_retriever import FakeDB, make


def _doc(text, document_id="d1", chunk_id=1):
    return Document(
        page_content=text,
        metadata={"document_id": document_id, "chunk_id": chunk_id, "page_number": 1},
    )


def test_multi_query_retrieve_fuses_variants(monkeypatch):
    monkeypatch.setattr(settings, "DENSE_WEIGHT", 1.0)
    monkeypatch.setattr(settings, "BM25_WEIGHT", 1.0)
    monkeypatch.setattr(settings, "RRF_K", 60)

    docs = [
        _doc("rate interest loan", document_id="d1", chunk_id="c1"),
        _doc("tenure years", document_id="d1", chunk_id="c2"),
    ]
    retr = make(db=FakeDB([]), docs=docs)
    retr.dense_retrieve = lambda q, document_id=None: [docs[0]] if "rate" in q else []
    retr.bm25_retrieve = lambda q, document_id=None: [docs[1]] if "tenure" in q else []
    fused = retr.multi_query_retrieve(["rate question", "tenure question"])
    assert {f.metadata["chunk_id"] for f in fused} == {"c1", "c2"}


def test_multi_query_retrieve_dedupes_variants(monkeypatch):
    monkeypatch.setattr(settings, "RRF_K", 60)
    docs = [_doc("rate", chunk_id="c1")]
    retr = make(db=FakeDB([]), docs=docs)
    calls = []

    def dense(q, document_id=None):
        calls.append(q)
        return docs

    retr.dense_retrieve = dense
    retr.bm25_retrieve = lambda q, document_id=None: []
    retr.multi_query_retrieve(["rate", " rate ", "RATE"])
    assert calls == ["rate"]  # blank/duplicate variants collapsed


def test_multi_query_retrieve_scope(monkeypatch):
    monkeypatch.setattr(settings, "RRF_K", 60)
    docs = [
        _doc("rate", document_id="d1", chunk_id="c1"),
        _doc("rate alt", document_id="d1", chunk_id="c2"),
        _doc("rate", document_id="d2", chunk_id="c3"),
    ]
    retr = make(db=FakeDB([]), docs=docs)
    retr.dense_retrieve = lambda q, document_id=None: [
        d for d in docs if document_id is None or d.metadata.get("document_id") == document_id
    ]
    retr.bm25_retrieve = lambda q, document_id=None: []
    fused = retr.multi_query_retrieve(["rate", "interest"], document_id="d1")
    assert {f.metadata["chunk_id"] for f in fused} == {"c1", "c2"}


def test_multi_query_retrieve_caps(monkeypatch):
    monkeypatch.setattr(settings, "HYBRID_CANDIDATES", 2)
    monkeypatch.setattr(settings, "RRF_K", 60)
    docs = [_doc(f"rate {i}", chunk_id=f"c{i}") for i in range(3)]
    retr = make(db=FakeDB([]), docs=docs)
    retr.dense_retrieve = lambda q, document_id=None: list(docs)
    retr.bm25_retrieve = lambda q, document_id=None: list(reversed(docs))
    fused = retr.multi_query_retrieve(["rate", "interest"])
    assert len(fused) == 2


def test_multi_query_retrieve_empty(monkeypatch):
    monkeypatch.setattr(settings, "RRF_K", 60)
    retr = make(db=FakeDB([]), docs=[_doc("rate")])
    retr.hybrid_retrieve = lambda q, document_id=None: (_ for _ in ()).throw(
        AssertionError("must not retrieve for empty input")
    )
    assert retr.multi_query_retrieve([]) == []
    assert retr.multi_query_retrieve(["", "  "]) == []
