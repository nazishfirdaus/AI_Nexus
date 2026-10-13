"""Tests for the multi-track hardening work.

Track A (scale/batching)   — Chroma batch sizes, paginated get_all_documents
Track B (encoding/language)— unicode BM25 tokenizer (kept in retriever tests)
Track C (tables)           — render_table_lines hardening, merged xlsx cells,
                             pdfplumber table gating helpers
Track D (dedup/quality)    — content_hash registry + lookup, dedup skip/replace,
                             batch duplicate-chunk suppression, quality stats
"""
import importlib
import types

import pytest
from langchain_core.documents import Document

from src.ingestion.document_store import DocumentStore
from src.ingestion.parsers import pdf as pdf_mod
from src.ingestion.parsers.common import render_table_lines
from src.ingestion.parsers.tables import parse_xlsx
from src.rag_pipeline import DuplicateDocumentError, RAGPipeline


# ------------------------------------------------------------------ settings
def test_settings_multitrack_defaults():
    from config import settings as s

    assert s.CHROMA_ADD_BATCH_SIZE == 512
    assert s.CHROMA_GET_BATCH_SIZE == 1000
    assert s.EMBEDDING_MODEL_NAME == "BAAI/bge-large-en"
    assert s.PDF_TABLE_EXTRACTION == "auto"
    assert s.DEDUP_ACTION == "skip"


def test_settings_multitrack_env_overrides(monkeypatch):
    import config.settings as s

    monkeypatch.setenv("ANEXUS_EMBEDDING_MODEL_NAME", "jinaai/jina-embeddings-v3")
    monkeypatch.setenv("ANEXUS_DEDUP_ACTION", "replace")
    monkeypatch.setenv("ANEXUS_PDF_TABLE_EXTRACTION", "off")
    importlib.reload(s)
    try:
        assert s.EMBEDDING_MODEL_NAME == "jinaai/jina-embeddings-v3"
        assert s.DEDUP_ACTION == "replace"
        assert s.PDF_TABLE_EXTRACTION == "off"
    finally:
        for name in (
            "ANEXUS_EMBEDDING_MODEL_NAME",
            "ANEXUS_DEDUP_ACTION",
            "ANEXUS_PDF_TABLE_EXTRACTION",
        ):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(s)
        assert s.EMBEDDING_MODEL_NAME == "BAAI/bge-large-en"
        assert s.DEDUP_ACTION == "skip"
        assert s.PDF_TABLE_EXTRACTION == "auto"


# ------------------------------------------------------------------ track C
def test_render_table_lines_pads_ragged_rows_to_the_widest_row():
    rows = [
        ["Name", "Rate", "State"],
        ["Loan A", "7.25"],            # short row: extra header columns dropped
        ["Loan B", "8.10", "KL", "z"],  # wide row: extra value with no header dropped
    ]
    lines = render_table_lines(rows)
    assert lines == [
        "Name: Loan A | Rate: 7.25",
        "Name: Loan B | Rate: 8.10 | State: KL",
    ]


def test_render_table_lines_keeps_columns_by_position():
    rows = [
        ["Name", "Rate"],
        ["", "7.25"],   # value sits under Rate -> only that pair survives
        ["Loan A", ""],  # name has no value -> its pair is dropped
    ]
    assert render_table_lines(rows) == ["Rate: 7.25", "Name: Loan A"]


def test_render_table_lines_skips_separator_rows():
    rows = [
        ["Name", "Rate"],
        ["---", "---"],
        ["Loan A", "7.25"],
        ["|---|---|", "|---|---|"],
        ["Loan B", "8.10"],
    ]
    assert render_table_lines(rows) == [
        "Name: Loan A | Rate: 7.25",
        "Name: Loan B | Rate: 8.10",
    ]


def test_render_table_lines_all_separators_returns_empty():
    assert render_table_lines([["---"], ["|---|---|"]]) == []


def test_render_table_lines_header_only_fallback_preserved():
    assert render_table_lines([["Name", "State", ""]]) == ["Name | State"]


def test_parse_xlsx_merged_cells_fill_down(tmp_path):
    from openpyxl import Workbook

    workbook = Workbook()
    ws = workbook.active
    ws.title = "Merged"
    ws.append(["Category", "Loan", "Rate"])
    ws.append(["Housing", "Loan A", "7.25"])
    ws.append([None, "Loan B", "8.10"])
    ws.merge_cells("A2:A3")

    path = tmp_path / "merged.xlsx"
    workbook.save(str(path))

    documents = parse_xlsx(path)
    content = "\n".join(d.page_content for d in documents)
    # the merged category label repeats for both records instead of being blank
    assert "Category: Housing | Loan: Loan A | Rate: 7.25" in content
    assert "Category: Housing | Loan: Loan B | Rate: 8.10" in content


def test_parse_xlsx_horizontal_merges_repeat_in_the_header(tmp_path):
    from openpyxl import Workbook

    workbook = Workbook()
    ws = workbook.active
    ws.append(["Rate Structure", None])
    ws.append(["Rate", "Amount"])
    ws.append(["7.25", "450000"])
    ws.merge_cells("A1:B1")

    path = tmp_path / "spanning.xlsx"
    workbook.save(str(path))

    content = "\n".join(d.page_content for d in parse_xlsx(path))
    assert "Rate Structure" in content


def test_looks_tabular_classifies_prose_as_false():
    assert not pdf_mod._looks_tabular(
        "Borrower Rahul Sharma obtained a home loan from the lender."
    )


def test_looks_tabular_classifies_columns_as_true():
    text = "Borrower        Amount        Rate\nRahul           450000         7.25\n"
    assert pdf_mod._looks_tabular(text)


def test_looks_tabular_requires_multiple_lines():
    assert not pdf_mod._looks_tabular("Borrower   Amount   Rate")


def test_tables_engine_returns_empty_without_pdfplumber(tmp_path):
    fake_pdf = types.SimpleNamespace(name=str(tmp_path / "x.pdf"))
    tables_cache = {"loaded": False, "pdf": None}
    text = pdf_mod._engine_text(
        "tables",
        page=None,
        page_number=1,
        plain_text="short",
        cache={"loaded": True, "pages": None},
        tables_cache=tables_cache,
        pdf=fake_pdf,
    )
    assert text == ""
    assert tables_cache["loaded"] is True
    assert tables_cache["pdf"] is None


def test_tables_engine_is_registered():
    assert "tables" in pdf_mod._PDF_ENGINES


# ------------------------------------------------------------------ track D
def test_find_by_content_hash(tmp_path):
    path = tmp_path / "docs.sqlite3"
    store = DocumentStore(db_path=str(path))
    store.add("loan", "loan.pdf", pages=1, chunks=2, vectors=2, content_hash="abc123")
    assert store.find_by_content_hash("abc123")["id"] == "loan"
    assert store.find_by_content_hash("missing") is None
    assert store.find_by_content_hash("") is None
    store.close()

    # Reopening (the migration path re-runs) still resolves the hash.
    reopened = DocumentStore(db_path=str(path))
    assert reopened.find_by_content_hash("abc123")["id"] == "loan"
    reopened.close()


def test_add_content_hash_updates_replaced_record():
    from src.ingestion.document_store import DocumentStore

    store = DocumentStore(":memory:")
    store.add("loan", "loan.pdf", pages=1, chunks=2, vectors=2, content_hash="aaa")
    store.add("loan", "loan_v2.pdf", pages=4, chunks=8, vectors=8, content_hash="bbb")
    assert store.find_by_content_hash("aaa") is None
    assert store.find_by_content_hash("bbb")["id"] == "loan"
    store.close()


# ------------------------------------------------------------------ dedup end-to-end
def make_pdf(path):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Borrower is Rahul Sharma. Loan amount is 4,50,000. Interest rate is "
        "7.25 per cent per annum. Repayment tenure is 20 years.",
    )
    doc.save(str(path))
    doc.close()


class FakeVectorDB:
    def __init__(self):
        self.store = []

    def count(self):
        return len(self.store)

    def reset(self):
        self.store = []

    def add_chunks(self, documents, embeddings):
        self.store.extend(documents)
        return len(documents)

    def delete_document(self, document_id):
        before = len(self.store)
        self.store = [
            d for d in self.store
            if (d.metadata or {}).get("document_id") != document_id
        ]
        return before - len(self.store)

    def get_all_documents(self):
        return list(self.store)


class FakeRetriever:
    def __init__(self):
        self.documents = []
        self.rebuilt = []

    def reset(self):
        self.rebuilt = []

    def rebuild_bm25(self, documents):
        self.documents = list(documents)
        self.rebuilt = list(documents)

    def add_documents(self, chunks):
        self.documents.extend(chunks)


class FakeEmbedder:
    def embed_documents(self, documents, batch_size=8):
        return [[0.1, 0.2, 0.3, 0.4] for _ in documents]


class _Noop:
    def __getattr__(self, item):
        return lambda *a, **k: None


def build_pipeline(**overrides):
    from src.ingestion.chunker import Chunker

    defaults = {
        "vector_db": FakeVectorDB(),
        "retriever": FakeRetriever(),
        "reranker": _Noop(),
        "scrubber": _Noop(),
        "router": _Noop(),
        "llm_handler": _Noop(),
        "chunker": Chunker(chunk_size=200, chunk_overlap=20),
        "embedder": FakeEmbedder(),
    }
    defaults.update(overrides)
    return RAGPipeline(**defaults)


@pytest.fixture()
def pdf_path(tmp_path):
    path = tmp_path / "tiny_loan.pdf"
    make_pdf(path)
    return path


def test_dedup_skip_rejects_same_content_under_new_name(pipeline_like, pdf_path, tmp_path):
    pipeline_like.ingest_document(pdf_path)
    renamed = tmp_path / "renamed_copy.pdf"
    renamed.write_bytes(pdf_path.read_bytes())

    with pytest.raises(DuplicateDocumentError):
        pipeline_like.ingest_document(renamed, source_name="renamed_copy.pdf")

    assert len(pipeline_like.list_documents()) == 1  # original untouched


@pytest.fixture()
def pipeline_like():
    return build_pipeline()


def test_dedup_replace_brings_in_the_new_name(pipeline_like, pdf_path, tmp_path):
    from config import settings

    pipeline_like.ingest_document(pdf_path)
    renamed = tmp_path / "renamed_copy.pdf"
    renamed.write_bytes(pdf_path.read_bytes())

    previous = settings.DEDUP_ACTION
    settings.DEDUP_ACTION = "replace"
    try:
        result = pipeline_like.ingest_document(renamed, source_name="renamed_copy.pdf")
    finally:
        settings.DEDUP_ACTION = previous

    documents = pipeline_like.list_documents()
    assert [d["id"] for d in documents] == ["renamed_copy.pdf"]
    assert result.content_hash
    assert result.pages_empty_dropped == 0
    assert result.pages_ocr == 0


def test_dedup_reports_quality_stats(pipeline_like):
    pages = [
        Document(page_content="", metadata={"page_number": 1}),
        Document(page_content="  ", metadata={"page_number": 2}),
        Document(
            page_content="The rate is 7.25 percent per annum.",
            metadata={"page_number": 3, "ocr_used": True},
        ),
    ]
    result = pipeline_like.ingest_parsed_documents(
        pages, filename="quality.pdf", document_id="quality.pdf"
    )
    assert result.pages_empty_dropped == 2
    assert result.pages_ocr == 1
    assert result.content_hash
    assert result.pages_overcleaned == 0
    assert result.chunks_dropped_short == 0
    assert result.boilerplate_lines_removed == 0


def test_duplicate_chunks_within_batch_are_dropped(pipeline_like):
    identical = "Mortgage loan boilerplate repeated verbatim on every page. "
    pages = [
        Document(page_content=identical, metadata={"page_number": 1}),
        Document(page_content=identical, metadata={"page_number": 2}),
    ]
    result = pipeline_like.ingest_parsed_documents(
        pages, filename="boiler.pdf", document_id="boiler.pdf"
    )
    assert result.chunks == 1
    assert result.vectors_added == result.chunks
    assert pipeline_like.vector_db.count() == 1


def test_hash_driven_dedup_repairs_an_empty_page_pair(pipeline_like):
    # A doc whose only text differs in whitespace/casing must still collapse.
    pages = [
        Document(page_content="Loan amount is 4,50,000.", metadata={"page_number": 1}),
        Document(
            page_content="   loan   AMOUNT is 4,50,000.  ", metadata={"page_number": 2}
        ),
    ]
    result = pipeline_like.ingest_parsed_documents(
        pages, filename="casey.pdf", document_id="casey.pdf"
    )
    assert result.chunks == 1