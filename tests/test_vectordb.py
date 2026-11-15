"""Tests for the Qdrant vector store (real in-memory Qdrant client)."""
import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from config import settings
from src.ingestion.qdrant_vectordb import (
    EmbeddingDimensionMismatch,
    QdrantVectorDB,
    point_id_for,
)


def make_docs(n=3, document_id="d1", prefix="d1"):
    return [
        Document(
            page_content=f"Chunk {i} — loan amount for page {i}.",
            metadata={
                "chunk_id": f"{prefix}_p{i}_c1",
                "document_id": document_id,
                "page_number": i,
                "filename": f"{document_id}.pdf",
                "source": f"{document_id}.pdf",
                "ocr_used": i % 2 == 0,
            },
        )
        for i in range(1, n + 1)
    ]


def make_embeddings(n=3, dim=4):
    vectors = []
    for i in range(n):
        vector = [0.0] * dim
        vector[i % dim] = 1.0 + 0.1 * i
        vectors.append(vector)
    return vectors


@pytest.fixture()
def db():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorDB(collection_name="test_col", client=client)
    yield store
    client.close()


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


def test_dimension_mismatch_on_add_raises_and_keeps_data(db):
    db.add_chunks(make_docs(2), make_embeddings(2))
    with pytest.raises(EmbeddingDimensionMismatch) as exc:
        db.add_chunks(
            make_docs(1, document_id="d2", prefix="d2"),
            [[0.1, 0.2, 0.3]],  # 3-dim vs the stored 4-dim
        )
    assert "re-ingest" in str(exc.value)
    assert db.count() == 2


def test_mixed_dimensions_in_one_batch_raise(db):
    with pytest.raises(EmbeddingDimensionMismatch):
        db.add_chunks(make_docs(2), [[0.1, 0.2], [0.1, 0.2, 0.3]])


def test_search_dimension_mismatch_raises(db):
    db.add_chunks(make_docs(), make_embeddings())
    with pytest.raises(EmbeddingDimensionMismatch):
        db.search([1.0, 0.0], k=1)


def test_search_returns_top_k(db):
    db.add_chunks(make_docs(), make_embeddings())
    out = db.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(out) == 2
    assert out[0].page_content.startswith("Chunk 1")


def test_search_preserves_metadata(db):
    db.add_chunks(make_docs(), make_embeddings())
    out = db.search([1.0, 0.0, 0.0, 0.0], k=3)
    by_page = {d.metadata["page_number"]: d for d in out}
    assert set(by_page) == {1, 2, 3}
    assert by_page[2].metadata["ocr_used"] is True
    assert by_page[3].metadata["ocr_used"] is False
    assert "similarity" in by_page[1].metadata


def test_search_zero_k_returns_empty(db):
    db.add_chunks(make_docs(), make_embeddings())
    assert db.search([1.0, 0.0, 0.0, 0.0], k=0) == []


def test_get_all_documents_round_trips(db):
    db.add_chunks(make_docs(), make_embeddings())
    all_docs = db.get_all_documents()
    assert len(all_docs) == 3
    ids = {d.metadata["chunk_id"] for d in all_docs}
    assert ids == {"d1_p1_c1", "d1_p2_c1", "d1_p3_c1"}


def test_reingest_same_chunks_does_not_duplicate(db):
    docs, vectors = make_docs(), make_embeddings()
    assert db.add_chunks(docs, vectors) == 3
    assert db.add_chunks(docs, vectors) == 3
    assert db.count() == 3
    assert db.document_ids() == ["d1"]
    assert {d.metadata["chunk_id"] for d in db.get_all_documents()} == {
        "d1_p1_c1",
        "d1_p2_c1",
        "d1_p3_c1",
    }


def test_point_id_is_deterministic_uuid(db):
    first = point_id_for("d1_p1_c1")
    assert first == point_id_for("d1_p1_c1")
    assert first != point_id_for("d1_p1_c2")


def test_large_corpus_batches_add_and_paginates_get(db, monkeypatch):
    """More than one ADD_BATCH_SIZE write and more than one GET_BATCH_SIZE read."""
    monkeypatch.setattr(settings, "QDRANT_GET_BATCH_SIZE", 100)
    n = int(settings.QDRANT_ADD_BATCH_SIZE) + 100
    docs = [
        Document(
            page_content=f"Chunk {i}.",
            metadata={
                "chunk_id": f"bulk_p{i}_c1",
                "document_id": "bulk",
                "page_number": i,
            },
        )
        for i in range(1, n + 1)
    ]
    embeddings = [[float(i % 3) + 1.0, 0.0, 0.0, float(i % 7)] for i in range(n)]
    assert db.add_chunks(docs, embeddings) == n
    assert db.count() == n

    all_docs = db.get_all_documents()
    assert len(all_docs) == n
    assert {d.metadata["chunk_id"] for d in all_docs} == {
        d.metadata["chunk_id"] for d in docs
    }


def test_reset_clears_collection(db):
    db.add_chunks(make_docs(), make_embeddings())
    db.reset()
    assert db.count() == 0
    # Store is usable after reset.
    db.add_chunks(make_docs(1), make_embeddings(1))
    assert db.count() == 1


# ---------------------------------------------------------- multi-document
def test_two_documents_coexist(db):
    db.add_chunks(make_docs(3), make_embeddings(3))
    db.add_chunks(
        make_docs(2, document_id="d2", prefix="d2"),
        make_embeddings(2),
    )
    assert db.count() == 5
    assert db.document_ids() == ["d1", "d2"]


def test_delete_document_removes_only_its_chunks(db):
    db.add_chunks(make_docs(3), make_embeddings(3))
    db.add_chunks(
        make_docs(2, document_id="d2", prefix="d2"),
        make_embeddings(2),
    )
    db.delete_document("d2")
    assert db.count() == 3
    assert db.document_ids() == ["d1"]
    assert {d.metadata["document_id"] for d in db.get_all_documents()} == {"d1"}


def test_delete_unknown_document_is_harmless(db):
    db.add_chunks(make_docs(1), make_embeddings(1))
    db.delete_document("never_uploaded")
    assert db.count() == 1


def test_search_with_metadata_filter_restricts_corpus(db):
    db.add_chunks(make_docs(3), make_embeddings(3))
    db.add_chunks(
        make_docs(2, document_id="d2", prefix="d2"),
        make_embeddings(2),
    )
    out = db.search([1.0, 0.0, 0.0, 0.0], k=10, where={"document_id": "d2"})
    assert out
    assert {d.metadata["document_id"] for d in out} == {"d2"}
