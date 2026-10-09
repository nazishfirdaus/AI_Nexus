"""PDF parsing: PyMuPDF text extraction with a Tesseract OCR fallback."""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from config.settings import MAX_PDF_PAGES, MIN_TEXT_LENGTH_FOR_OCR, OCR_DPI

from .common import (
    has_sufficient_text,
    make_document,
    normalize_text,
    validate_file,
)

ALLOWED_EXTENSIONS = (".pdf",)


def render_page_to_image(
    page: pymupdf.Page,
    dpi: int = OCR_DPI,
):

    """
    Render a PDF page as a PIL image.

    The image is used as input to Tesseract OCR.
    """

    zoom = dpi / 72

    matrix = pymupdf.Matrix(
        zoom,
        zoom,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

    from PIL import Image

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    return image


def extract_text_with_ocr(
    page: pymupdf.Page,
    dpi: int = OCR_DPI,
) -> str:
    """
    Extract text from a PDF page using Tesseract OCR.
    """

    image = render_page_to_image(
        page,
        dpi=dpi,
    )

    import pytesseract

    text = pytesseract.image_to_string(
        image
    )

    return text


def parse_pdf(
    pdf_path: str | Path,
    min_text_length: int = MIN_TEXT_LENGTH_FOR_OCR,
    ocr_dpi: int = OCR_DPI,
) -> list[Document]:
    """
    Parse a PDF page-by-page.

    Processing flow:

        PDF
          ↓
        PyMuPDF
          ↓
        Text extraction
          ↓
        Sufficient text?
          ↓
        OCR fallback if necessary
          ↓
        Text normalization
          ↓
        Metadata
          ↓
        LangChain Documents

    Returns:
        One LangChain Document for each PDF page.
    """

    pdf_path = validate_file(pdf_path, ALLOWED_EXTENSIONS)

    # ---------------------------------------------
    # Open PDF
    # ---------------------------------------------

    pdf = pymupdf.open(pdf_path)

    documents = []

    try:

        if len(pdf) == 0:
            raise ValueError(
                "The PDF contains no pages."
            )

        if len(pdf) > MAX_PDF_PAGES:
            raise ValueError(
                f"The PDF has {len(pdf)} pages, which exceeds the "
                f"{MAX_PDF_PAGES} page limit."
            )

        filename = pdf_path.name

        # -----------------------------------------
        # Process every page
        # -----------------------------------------

        for page_index, page in enumerate(pdf):

            page_number = page_index + 1

            # -------------------------------------
            # Normal PyMuPDF extraction
            # -------------------------------------

            extracted_text = page.get_text(
                "text"
            )

            ocr_used = False

            # -------------------------------------
            # OCR fallback
            # -------------------------------------

            if not has_sufficient_text(
                extracted_text,
                min_text_length=min_text_length,
            ):

                extracted_text = extract_text_with_ocr(
                    page,
                    dpi=ocr_dpi,
                )

                ocr_used = True

            # -------------------------------------
            # Normalize + metadata + Document
            # -------------------------------------

            documents.append(
                make_document(
                    normalize_text(extracted_text),
                    filename=filename,
                    page_number=page_number,
                    ocr_used=ocr_used,
                )
            )

    finally:
        pdf.close()

    return documents
