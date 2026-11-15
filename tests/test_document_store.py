"""Tests for the SQLite document registry."""
import sqlite3

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


def test_storage_path_roundtrip(store):
    store.add("loan", "loan.pdf")
    store.set_storage_path("loan", "C:/data/loan.pdf", managed=True)
    row = store.get("loan")
    assert row["storage_path"] == "C:/data/loan.pdf"
    assert row["storage_managed"] == 1


def test_reingest_without_storage_path_keeps_the_existing_file(store):
    store.add("loan", "loan.pdf", storage_path="C:/data/loan.pdf", storage_managed=True)
    store.add("loan", "loan.pdf", pages=2)
    row = store.get("loan")
    assert row["storage_path"] == "C:/data/loan.pdf"
    assert row["storage_managed"] == 1


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


def test_existing_database_is_migrated_for_source_storage(tmp_path):
    path = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE documents (
            id           TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            pages        INTEGER NOT NULL DEFAULT 0,
            chunks       INTEGER NOT NULL DEFAULT 0,
            vectors      INTEGER NOT NULL DEFAULT 0,
            duration_s   REAL NOT NULL DEFAULT 0.0,
            created_at   TEXT NOT NULL
        );
        INSERT INTO documents (id, filename, created_at)
        VALUES ('old', 'old.pdf', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    store = DocumentStore(db_path=path)
    try:
        row = store.get("old")
        assert row["storage_path"] is None
        assert row["storage_managed"] == 0

        store.set_storage_path("old", "C:/data/old.pdf", managed=True)
        assert store.get("old")["storage_path"] == "C:/data/old.pdf"
    finally:
        store.close()


def test_file_backed_store_persists(tmp_path):
    path = str(tmp_path / "docs.sqlite3")
    writer = DocumentStore(db_path=path)
    writer.add("loan", "loan.pdf", pages=2, chunks=4, vectors=4)

    reader = DocumentStore(db_path=path)
    documents = reader.list()
    assert len(documents) == 1
    assert documents[0]["id"] == "loan"
    assert documents[0]["chunks"] == 4
