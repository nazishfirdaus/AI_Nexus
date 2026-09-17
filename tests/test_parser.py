from pathlib import Path

from src.ingestion.parser import parse_pdf


PDF_PATH = Path(
    "Synthetic_Mortgage_Loan_File_TEST.pdf"
)


def test_parse_pdf():

    documents = parse_pdf(PDF_PATH)

    # The test document contains 12 pages.
    assert len(documents) == 12

    # Check first page
    first_page = documents[0]

    assert first_page.metadata["page_number"] == 1

    assert (
        first_page.metadata["filename"]
        == "Synthetic_Mortgage_Loan_File_TEST.pdf"
    )

    assert (
        first_page.metadata["document_id"]
        == "Synthetic_Mortgage_Loan_File_TEST"
    )

    assert first_page.page_content != ""


def test_page_numbers_are_preserved():

    documents = parse_pdf(PDF_PATH)

    page_numbers = [
        document.metadata["page_number"]
        for document in documents
    ]

    assert page_numbers == list(range(1, 13))


def test_every_page_has_metadata():

    documents = parse_pdf(PDF_PATH)

    for document in documents:

        assert "document_id" in document.metadata

        assert "filename" in document.metadata

        assert "page_number" in document.metadata

        assert "source" in document.metadata

        assert "ocr_used" in document.metadata


def test_text_is_normalized():

    documents = parse_pdf(PDF_PATH)

    for document in documents:

        text = document.page_content

        assert text == text.strip()

        assert "\r\n" not in text