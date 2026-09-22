"""Tests for the self-contained recursive chunker."""
import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import Chunker, chunk_id_for

PAGE_TEXT = (
    "The first paragraph explains the purpose of the loan.\n\n"
    "The second paragraph contains the interest rate of 7.25 percent per annum "
    "and the loan amount. Another long sentence about the repayment tenure and EMI "
    "details that pushes the text well past the configured chunk size for testing."
)


def make_chunker(chunk_size=50, chunk_overlap=10, **kw):
    return Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kw)


# ----------------------------------------------------------------- split_text
def test_short_text_is_single_chunk():
    assert make_chunker().split_text("Hello loan") == ["Hello loan"]


def test_empty_and_blank_text():
    assert make_chunker().split_text("") == []
    assert make_chunker().split_text("   \n  ") == []


def _shared_overlap_len(a: str, b: str) -> int:
    """Length of the shared window between the tail of a and the head of b."""
    limit = min(len(a), len(b))
    for i in range(limit, 0, -1):
        if a[-i:] == b[:i]:
            return i
    return 0


def test_no_chunk_longer_than_chunk_size():
    parts = make_chunker().split_text(PAGE_TEXT)
    assert parts
    assert all(len(p) <= 50 for p in parts)


def test_consecutive_chunks_share_overlap():
    chunker = make_chunker(chunk_size=50, chunk_overlap=10)
    parts = chunker.split_text(PAGE_TEXT)
    overlapping = sum(1 for a, b in zip(parts, parts[1:]) if _shared_overlap_len(a, b) >= 3)
    assert overlapping >= max(1, (len(parts) - 1) // 2)


def test_overlap_carries_whole_words():
    chunker = Chunker(chunk_size=30, chunk_overlap=10)
    parts = chunker.split_text("one two three four five six seven eight nine ten")
    for a, b in zip(parts, parts[1:]):
        assert _shared_overlap_len(a, b) > 0
    # At least one boundary must carry a real word (not a mid-word fragment).
    assert any(_shared_overlap_len(a, b) >= 3 for a, b in zip(parts, parts[1:]))


def test_all_source_text_preserved_approximately():
    chunker = make_chunker(chunk_size=50, chunk_overlap=10)
    parts = chunker.split_text(PAGE_TEXT)
    # Word-aligned overlap means key phrases stay contiguous as tokens.
    joined = " ".join(p.strip() for p in parts)
    assert "interest rate" in joined
    assert "7.25 percent per annum" in joined
    assert "repayment tenure" in joined
    assert "configured chunk size" in joined


def test_no_separator_hard_split():
    chunker = make_chunker(chunk_size=20, chunk_overlap=5)
    parts = chunker.split_text("a" * 100)
    assert all(len(p) <= 20 for p in parts)
    assert len(parts) >= 5


# ------------------------------------------------------------------- split()
def test_split_preserves_metadata_and_adds_chunk_id():
    docs = [
        Document(
            page_content="Title line.\n\n" + PAGE_TEXT,
            metadata={"document_id": "LOAN1", "page_number": 3, "filename": "loan.pdf", "source": "loan.pdf"},
        )
    ]
    chunks = make_chunker().split(docs)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "LOAN1"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["filename"] == "loan.pdf"
        assert chunk.metadata["chunk_id"].startswith("LOAN1_p3_c")


def test_chunk_ids_are_unique():
    docs = [Document(page_content="word " * 30, metadata={"document_id": "D", "page_number": 1})]
    ids = [c.metadata["chunk_id"] for c in make_chunker().split(docs)]
    assert len(ids) == len(set(ids))


def test_blank_pages_are_skipped():
    docs = [
        Document(page_content="Some content that is long enough to chunk. " * 4, metadata={"document_id": "A", "page_number": 1}),
        Document(page_content="   ", metadata={"document_id": "A", "page_number": 2}),
    ]
    chunks = make_chunker().split(docs)
    assert all(c.metadata["page_number"] == 1 for c in chunks)


def test_invalid_config():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, chunk_overlap=10)


def test_chunk_id_for():
    assert chunk_id_for("D", 3, 1) == "D_p3_c1"