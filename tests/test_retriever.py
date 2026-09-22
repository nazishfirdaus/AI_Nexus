"""Tests for the user's HybridRetriever API (dense_retrieve / bm25_retrieve / hybrid_retrieve).
No BGE model or Chroma needed: both are replaced by fakes."""
import pytest
from langchain_core.documents import Document

from src.retrieval.retriever import HybridRetriever


def chunks():
    return [
        Document(page_content="The loan amount is 4,50,000 at an interest rate of 7.25 percent.",
                 metadata={"document_id": "d1", "page_number": 2, "chunk_id": 1}),
        Document(page_content="Borrower signature page and date of signing.",
                 metadata={"document_id": "d1", "page_number": 5, "chunk_id": 2}),
        Document(page_content="Property address and appraisal value details.",
                 metadata={"document_id": "d1", "page_number": 7, "chunk_id": 3}),
    ]


class Vec(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, fail=False):
        self.seen, self.fail = [], fail

    def encode(self, text, normalize_embeddings=True, convert_to_numpy=True):
        if self.fail:
            raise RuntimeError("model down")
        self.seen.append(text)
        return Vec([0.1, 0.2])


class FakeDB:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query_embedding, k):
        self.calls.append((query_embedding, k))
        return self.results[:k]


def make(db=None, docs="default", model=None, k=10):
    docs = chunks() if docs == "default" else docs
    return HybridRetriever(db or FakeDB(), docs, k=k, embedding_model=model or FakeModel())


def test_construction_does_not_load_real_embedder():
    make()          # would raise ImportError/slow download if it tried to load BGE


def test_bm25_finds_exact_number():
    out = make().bm25_retrieve("interest rate 7.25")
    assert out and out[0].metadata["chunk_id"] == 1


def test_bm25_ignores_trailing_punctuation():
    assert make().bm25_retrieve("What is the loan amount?")[0].metadata["chunk_id"] == 1


def test_bm25_no_overlap_returns_nothing():
    assert make().bm25_retrieve("zzzz qqqq") == []


def test_bm25_respects_k():
    assert len(make(k=1).bm25_retrieve("loan borrower property")) == 1


def test_empty_query_and_empty_corpus():
    r = make()
    assert r.dense_retrieve("  ") == [] and r.bm25_retrieve("") == [] and r.hybrid_retrieve("") == []
    assert make(docs=[]).bm25_retrieve("loan") == []


def test_dense_query_gets_bge_prefix_and_k():
    model, db = FakeModel(), FakeDB()
    make(db=db, model=model, k=4).dense_retrieve("loan amount")
    assert model.seen[0].endswith("loan amount") and model.seen[0].startswith("Represent this sentence")
    assert db.calls[0][1] == 4


def test_dense_failure_falls_back_to_bm25():
    r = make(model=FakeModel(fail=True))
    out = r.hybrid_retrieve("appraisal value")
    assert [d.metadata["chunk_id"] for d in out] == [3]


def test_chunk_found_by_both_ranks_first():
    c = chunks()
    r = make(db=FakeDB([c[1], c[0]]))            # dense: chunk 2 then 1
    out = r.hybrid_retrieve("loan amount interest rate")   # bm25 favours chunk 1
    assert out[0].metadata["chunk_id"] == 1


def test_no_duplicates_in_hybrid():
    c = chunks()
    out = make(db=FakeDB(c)).hybrid_retrieve("loan borrower property")
    ids = [d.metadata["chunk_id"] for d in out]
    assert len(ids) == len(set(ids))


def test_same_chunk_id_on_different_pages_kept():
    docs = [Document(page_content="loan amount one", metadata={"document_id": "d", "page_number": 1, "chunk_id": 0}),
            Document(page_content="loan amount two", metadata={"document_id": "d", "page_number": 2, "chunk_id": 0})]
    assert len(make(docs=docs).hybrid_retrieve("loan amount")) == 2


def test_chunks_without_metadata_are_not_collapsed():
    docs = [Document(page_content="loan amount one"), Document(page_content="loan amount two")]
    assert len(make(docs=docs).hybrid_retrieve("loan amount")) == 2


def test_bm25_rebuilt_from_vector_db_after_restart():
    class DB(FakeDB):
        n = 0

        def get_all_documents(self):
            DB.n += 1
            return chunks()

    r = HybridRetriever(DB(), None, embedding_model=FakeModel())   # fresh process, no documents
    assert [d.metadata["chunk_id"] for d in r.bm25_retrieve("borrower signature")] == [2]
    r.bm25_retrieve("loan")
    assert DB.n == 1


def test_reset_clears_bm25():
    r = make()
    r.reset()
    assert r.bm25_retrieve("loan amount") == []