"""Hash-verified discovery of legacy source files."""
import hashlib

from src.ingestion.document_store import DocumentStore
from src.ingestion.source_files import backfill_source_paths, find_source_file


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_find_source_file_requires_a_hash_match(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "legacy-loan-test.pdf"
    source.write_bytes(b"%PDF-1.4 legacy")

    assert find_source_file(
        source.name, _sha256(source), extra_roots=[corpus]
    ) == source.resolve()
    assert find_source_file(source.name, "0" * 64, extra_roots=[corpus]) is None


def test_find_source_file_accepts_a_unique_match_without_a_hash(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "legacy-loan-no-hash.pdf"
    source.write_bytes(b"legacy")

    assert find_source_file(
        source.name, None, extra_roots=[corpus]
    ) == source.resolve()


def test_backfill_records_an_unmanaged_path(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "legacy-loan-backfill.pdf"
    source.write_bytes(b"%PDF-1.4 backfill")

    store = DocumentStore(tmp_path / "app.sqlite3")
    store.add(
        "doc-legacy",
        source.name,
        content_hash=_sha256(source),
    )

    attached = backfill_source_paths(
        "tenant-test", store, extra_roots=[corpus]
    )

    assert attached == 1
    row = store.get("doc-legacy")
    assert row is not None
    assert row["storage_path"] == str(source.resolve())
    assert row["storage_managed"] == 0


def test_backfill_leaves_an_existing_path_alone(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "legacy-loan-existing.pdf"
    source.write_bytes(b"existing")

    store = DocumentStore(tmp_path / "app.sqlite3")
    store.add(
        "doc-existing",
        source.name,
        content_hash=_sha256(source),
        storage_path=str(source),
        storage_managed=True,
    )

    assert backfill_source_paths(
        "tenant-test", store, extra_roots=[corpus]
    ) == 0
    row = store.get("doc-existing")
    assert row is not None
    assert row["storage_managed"] == 1
