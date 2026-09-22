"""Tests for the ChromaDB vector store (uses a real temp PersistentClient)."""
import pytest
from langchain_core.documents import Document

from src.ingestion.vectordb import VectorDB


def make_docs(n=3):
    return [
        Document(
            page_content=f"Chunk {i} — loan amount for page {i}.",
            metadata={
                "chunk_id": f"d1_p{i}_c1",
                "document_id": "d1",
                "page_number": i,
                "source": "loan.pdf",
                "ocr_used": i % 2 == 0,
            },
        )
        for i in range(1, n + 1)
    ]


def make_embeddings(n=3, dim=4):
    return [[float(i + 1), 0.0, 0.0, float(i)] for i in range(n)]


@pytest.fixture()
def db(tmp_path):
    store = VectorDB(persist_directory=str(tmp_path), collection_name="test_col")
    yield store
    store.close()


def test_initial_state_empty(db):
    assert db.count() == 0


def test_add_and_count(db):
    assert db.add_chunks(make_docs(), make_embeddings()) == 3
    assert db.count() == 3


def test_len_mismatch_raises(db):
    with pytest.raises(ValueError):
        db.add_chunks(make_docs(3), make_embeddings(2))


def test_add_empty_is_noop(db):
    assert db.add_chunks([], []) == 0


def test_search_returns_top_k(db):
    db.add_chunks(make_docs(), make_embeddings())
    out = db.search([1.0, 0.0, 0.0, 1.0], k=2)
    assert len(out) == 2
    assert out[0].page_content.startswith("Chunk 1")


def test_search_preserves_metadata(db):
    db.add_chunks(make_docs(), make_embeddings())
    out = db.search([1.0, 0.0, 0.0, 1.0], k=3)
    by_page = {d.metadata["page_number"]: d for d in out}
    assert set(by_page) == {1, 2, 3}
    assert by_page[2].metadata["ocr_used"] is True
    assert by_page[3].metadata["ocr_used"] is False
    assert "distance" in by_page[1].metadata


def test_search_zero_k_returns_empty(db):
    db.add_chunks(make_docs(), make_embeddings())
    assert db.search([1.0, 0.0, 0.0, 1.0], k=0) == []


def test_get_all_documents_round_trips(db):
    db.add_chunks(make_docs(), make_embeddings())
    all_docs = db.get_all_documents()
    assert len(all_docs) == 3
    ids = {d.metadata["chunk_id"] for d in all_docs}
    assert ids == {"d1_p1_c1", "d1_p2_c1", "d1_p3_c1"}


def test_reset_clears_collection(db):
    db.add_chunks(make_docs(), make_embeddings())
    db.reset()
    assert db.count() == 0
    # Store is usable after reset.
    db.add_chunks(make_docs(1), make_embeddings(1))
    assert db.count() == 1