from pathlib import Path
import io
import re

import pymupdf
import pytesseract
from PIL import Image
from langchain_core.documents import Document

from config.settings import (
    MIN_TEXT_LENGTH_FOR_OCR,
    OCR_DPI,
)


def normalize_text(text: str) -> str:
    """
    Normalize text extracted from a PDF or OCR.

    The goal is to remove unnecessary whitespace while
    preserving meaningful line structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove spaces and tabs at the beginning/end of lines
    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if line:
            lines.append(line)

    # Rebuild text
    text = "\n".join(lines)

    # Normalize repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Avoid excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def has_sufficient_text(
    text: str,
    min_text_length: int = MIN_TEXT_LENGTH_FOR_OCR,
) -> bool:
    """
    Check whether normal PDF extraction produced
    enough text.

    If not, the page will be sent to OCR.
    """

    normalized_text = normalize_text(text)

    return len(normalized_text) >= min_text_length


def render_page_to_image(
    page: pymupdf.Page,
    dpi: int = OCR_DPI,
) -> Image.Image:
    """
    Render a PDF page as a PIL image.

    The image is used as input to Tesseract OCR.
    """

    zoom = dpi / 72

    matrix = fitz.Matrix(
        zoom,
        zoom,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

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

    pdf_path = Path(pdf_path)

    # ---------------------------------------------
    # Validate file
    # ---------------------------------------------

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"Path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path.suffix}"
        )

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

        # Use filename stem as internal document ID.
        document_id = pdf_path.stem
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
            # Normalize
            # -------------------------------------

            normalized_text = normalize_text(
                extracted_text
            )

            # -------------------------------------
            # Metadata
            # -------------------------------------

            metadata = {
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "source": filename,
                "ocr_used": ocr_used,
            }

            # -------------------------------------
            # Page-level Document
            # -------------------------------------

            document = Document(
                page_content=normalized_text,
                metadata=metadata,
            )

            documents.append(document)

    finally:
        pdf.close()

    return documents