"""Tests for the SQLite document registry."""
import pytest

from src.ingestion.document_store import DocumentStore


@pytest.fixture()
def store():
    s = DocumentStore(db_path=":memory:")
    yield s
    s.close()


def test_add_and_list(store):
    store.add("loan", "loan.pdf", pages=3, chunks=9, vectors=9, duration_s=1.5)
    store.add("agreement", "agreement.pdf", pages=2, chunks=5, vectors=5)

    documents = store.list()
    assert [d["id"] for d in documents] == ["loan", "agreement"]
    assert documents[0]["filename"] == "loan.pdf"
    assert documents[0]["pages"] == 3
    assert documents[0]["chunks"] == 9
    assert documents[0]["duration_s"] == 1.5


def test_get(store):
    store.add("loan", "loan.pdf", pages=1, chunks=2, vectors=2)
    assert store.get("loan")["filename"] == "loan.pdf"
    assert store.get("missing") is None


def test_add_same_id_replaces_the_record(store):
    store.add("loan", "loan.pdf", pages=1, chunks=2, vectors=2)
    store.add("loan", "loan_v2.pdf", pages=4, chunks=8, vectors=8)

    documents = store.list()
    assert len(documents) == 1
    assert documents[0]["filename"] == "loan_v2.pdf"
    assert documents[0]["pages"] == 4


def test_remove(store):
    store.add("a", "a.pdf")
    store.add("b", "b.pdf")
    store.remove("a")
    assert [d["id"] for d in store.list()] == ["b"]
    store.remove("never_added")  # no-op, must not raise


def test_clear(store):
    store.add("a", "a.pdf")
    store.add("b", "b.pdf")
    store.clear()
    assert store.list() == []


def test_isolated_in_memory_stores_do_not_share_state():
    first = DocumentStore(db_path=":memory:")
    second = DocumentStore(db_path=":memory:")
    try:
        first.add("a", "a.pdf")
        assert second.list() == []
        assert first.list() != []
    finally:
        first.close()
        second.close()


def test_file_backed_store_persists(tmp_path):
    path = str(tmp_path / "docs.sqlite3")
    writer = DocumentStore(db_path=path)
    writer.add("loan", "loan.pdf", pages=2, chunks=4, vectors=4)

    reader = DocumentStore(db_path=path)
    documents = reader.list()
    assert len(documents) == 1
    assert documents[0]["id"] == "loan"
    assert documents[0]["chunks"] == 4
