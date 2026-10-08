"""Multi-format parser tests.

Fixtures are generated with the same libraries the parsers use, so no binary
sample files are committed. The image test needs Tesseract and skips when it
is unavailable.
"""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import Chunker
from src.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_document
from src.ingestion.parsers.common import (
    document_id_for,
    group_sections,
    render_table_lines,
)

METADATA_KEYS = {"document_id", "filename", "page_number", "source", "ocr_used"}


def _assert_standard_schema(documents):
    assert documents, "parser returned no documents"
    for document in documents:
        assert isinstance(document, Document)
        assert METADATA_KEYS <= set(document.metadata)
        assert document.metadata["filename"]
        assert document.page_content.strip() == document.page_content
        assert "\r\n" not in document.page_content
    pages = [d.metadata["page_number"] for d in documents]
    assert pages == list(range(1, len(documents) + 1))


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------- txt/md
@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_parse_text_sections(tmp_path, suffix):
    path = tmp_path / f"notes{suffix}"
    path.write_text(
        "Interest rate is 7.25 percent per annum.\n\n"
        "Repayment tenure is 20 years.\n\n"
        "Prepayment penalty is 2 percent.",
        encoding="utf-8",
    )

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert documents[0].metadata["document_id"] == f"notes{suffix}"
    assert "Interest rate is 7.25" in documents[0].page_content


def test_parse_text_packs_sections_into_pages(tmp_path):
    sections = [f"Section {i} " + ("word " * 60) for i in range(20)]
    path = tmp_path / "long.txt"
    path.write_text("\n\n".join(sections), encoding="utf-8")

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert len(documents) > 1  # long files split into multiple pseudo-pages


def test_parse_text_empty_file_returns_no_documents(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  \n", encoding="utf-8")
    assert parse_document(path) == []


def test_parse_text_decodes_non_utf8(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes("Borrower: Jos\u00e9 Garc\u00eda".encode("latin-1"))

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert "José García" in documents[0].page_content


# ---------------------------------------------------------------------- docx
def test_parse_docx_paragraphs_headings_and_tables(tmp_path):
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_heading("Loan Terms", level=1)
    doc.add_paragraph("Interest rate is 7.25 percent per annum.")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Borrower"
    table.rows[0].cells[1].text = "Amount"
    table.rows[1].cells[0].text = "Rahul Sharma"
    table.rows[1].cells[1].text = "450000"
    path = tmp_path / "agreement.docx"
    doc.save(str(path))

    documents = parse_document(path)

    _assert_standard_schema(documents)
    content = "\n".join(d.page_content for d in documents)
    assert "Loan Terms" in content
    assert "Interest rate is 7.25" in content
    assert "Borrower: Rahul Sharma | Amount: 450000" in content


def test_parse_docx_heading_starts_new_page(tmp_path):
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_heading("Section One", level=1)
    doc.add_paragraph("First section body.")
    doc.add_heading("Section Two", level=1)
    doc.add_paragraph("Second section body.")
    path = tmp_path / "headings.docx"
    doc.save(str(path))

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert len(documents) >= 2
    assert "Section One" in documents[0].page_content
    assert "Section Two" in documents[-1].page_content


def test_parse_docx_empty_returns_no_documents(tmp_path):
    from docx import Document as DocxDocument

    path = tmp_path / "empty.docx"
    DocxDocument().save(str(path))
    assert parse_document(path) == []


# ---------------------------------------------------------------------- csv
def test_parse_csv_rows_rendered_as_header_value(tmp_path):
    path = tmp_path / "rates.csv"
    path.write_text("City,Rate\nMumbai,7.25\nDelhi,7.40\n", encoding="utf-8")

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert documents[0].page_content == (
        "City: Mumbai | Rate: 7.25\nCity: Delhi | Rate: 7.40"
    )
    assert "sheet" not in documents[0].metadata


def test_parse_csv_groups_rows_into_pages(tmp_path):
    rows = ["Borrower,EMI"] + [f"Borrower {i},{3000 + i}" for i in range(130)]
    path = tmp_path / "big.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert len(documents) == 3  # 129 data rows / 50 rows per page


# --------------------------------------------------------------------- xlsx
def test_parse_xlsx_one_page_set_per_sheet(tmp_path):
    from openpyxl import Workbook

    workbook = Workbook()
    emi = workbook.active
    emi.title = "EMI"
    emi.append(["Borrower", "EMI"])
    emi.append(["Rahul", "3500"])
    fees = workbook.create_sheet("Fees")
    fees.append(["Type", "Amount"])
    fees.append(["Processing", "5000"])
    path = tmp_path / "schedule.xlsx"
    workbook.save(str(path))

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert [d.metadata["sheet"] for d in documents] == ["EMI", "Fees"]
    # page numbers are document-global so chunk ids / citation keys never collide
    assert [d.metadata["page_number"] for d in documents] == [1, 2]
    assert "[Sheet: EMI]" in documents[0].page_content
    assert "Borrower: Rahul | EMI: 3500" in documents[0].page_content


# --------------------------------------------------------------------- pptx
def test_parse_pptx_one_page_per_slide(tmp_path):
    from pptx import Presentation

    presentation = Presentation()
    for title, body in [
        ("Overview", "Fixed rate 7.25 percent"),
        ("Fees", "Processing fee 5000 rupees"),
    ]:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    path = tmp_path / "deck.pptx"
    presentation.save(str(path))

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert len(documents) == 2
    assert "Overview" in documents[0].page_content
    assert "Fees" in documents[1].page_content
    assert documents[1].metadata["page_number"] == 2


def test_parse_pptx_no_text_raises(tmp_path):
    from pptx import Presentation

    presentation = Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[6])  # blank
    path = tmp_path / "blank.pptx"
    presentation.save(str(path))

    with pytest.raises(ValueError, match="No text could be extracted"):
        parse_document(path)


# -------------------------------------------------------------------- images
@pytest.mark.skipif(not _tesseract_available(), reason="tesseract not installed")
def test_parse_image_ocr(tmp_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "Loan account number 1234567890 interest rate 7.25", fill="black")
    path = tmp_path / "scan.png"
    image.save(str(path))

    documents = parse_document(path)

    _assert_standard_schema(documents)
    assert len(documents) == 1
    assert documents[0].metadata["ocr_used"] is True
    assert documents[0].metadata["page_number"] == 1
    assert "1234567890" in documents[0].page_content


# ----------------------------------------------------------------- dispatcher
def test_dispatcher_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "malware.exe"
    path.write_bytes(b"MZ")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(path)


def test_dispatcher_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "nope.txt")


def test_supported_extensions_cover_every_parser():
    assert set(SUPPORTED_EXTENSIONS) == {
        ".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".pptx",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    }


def test_parse_pdf_through_dispatcher_still_works():
    documents = parse_document(Path("Synthetic_Mortgage_Loan_File_TEST.pdf"))
    _assert_standard_schema(documents)
    assert len(documents) == 12


# -------------------------------------------------------------------- helpers
def test_document_id_for_keeps_extension():
    assert document_id_for("notes.txt") == "notes.txt"
    assert document_id_for("notes.pdf") == "notes.pdf"
    assert document_id_for("C:/tmp/report.docx") == "report.docx"


def test_group_sections_respects_target():
    sections = ["aaa", "bbb", "ccc"]
    assert group_sections(sections, target_chars=1000) == ["aaa\n\nbbb\n\nccc"]
    pages = group_sections(["x" * 600, "y" * 600], target_chars=700)
    assert pages == ["x" * 600, "y" * 600]


def test_render_table_lines_skips_empty_rows_and_cells():
    rows = [
        ["Name", "Rate", "Notes"],
        ["", "", ""],
        ["Loan A", "7.25", ""],
    ]
    assert render_table_lines(rows) == ["Name: Loan A | Rate: 7.25"]


def test_chunker_accepts_non_pdf_documents(tmp_path):
    path = tmp_path / "chunk_me.txt"
    path.write_text("word " * 300, encoding="utf-8")

    documents = parse_document(path)
    chunks = Chunker(chunk_size=50, chunk_overlap=10).split(documents)

    assert chunks
    assert all(c.metadata["chunk_id"].startswith("chunk_me.txt_p1_c") for c in chunks)
