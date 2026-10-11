"""PDF parsing: PyMuPDF text extraction, with optional higher-accuracy fallbacks.

Extraction order per page:

    PyMuPDF plain text
      -> pymupdf4llm layout-aware Markdown (only if the plain text is too short
         AND the optional package is installed)
      -> OCR (Tesseract, or RapidOCR as fallback)

The baseline (PyMuPDF + Tesseract) is unchanged and is what runs by default;
the extra engines are additive fallbacks.
"""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from config import settings
from config.settings import MAX_PDF_PAGES, MIN_TEXT_LENGTH_FOR_OCR, OCR_DPI

from . import ocr
from .common import (
    has_sufficient_text,
    looks_like_gibberish,
    make_document,
    normalize_text,
    strip_markdown_artifacts,
    validate_file,
)

ALLOWED_EXTENSIONS = (".pdf",)

_PDF_ENGINES = ("pymupdf", "pymupdf4llm")


def render_page_to_image(
    page: pymupdf.Page,
    dpi: int = OCR_DPI,
):
    """Render a PDF page as a PIL image (input to OCR)."""

    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image_bytes = pixmap.tobytes("png")

    from PIL import Image

    return Image.open(io.BytesIO(image_bytes))


def extract_text_with_ocr(
    page: pymupdf.Page,
    dpi: int = OCR_DPI,
) -> str:
    """Extract text from a PDF page using OCR (primary engine + fallback)."""

    return ocr.ocr_image(render_page_to_image(page, dpi=dpi))


def _normalize_engine(engine: str | None) -> str:
    name = (engine or settings.PDF_TEXT_ENGINE or "pymupdf").strip().lower()
    return name if name in _PDF_ENGINES else "pymupdf"


def _should_strip_markdown(extractor: str) -> bool:
    """Whether to strip Markdown relics for text produced by ``extractor``.

    "auto" (default) strips only the layout-aware pymupdf4llm output; uploaded
    .md files and PyMuPDF plain text are left as-is.
    """

    mode = settings.STRIP_MARKDOWN_ARTIFACTS
    if mode == "true":
        return True
    if mode == "false":
        return False
    return extractor == "pymupdf4llm"


def _load_pymupdf4llm_pages(pdf: pymupdf.Document) -> dict[int, str] | None:
    """Return ``{page_number: markdown}`` via pymupdf4llm, or None if unusable."""

    try:
        import pymupdf4llm
    except Exception:
        return None

    try:
        chunks = pymupdf4llm.to_markdown(
            pdf,
            page_chunks=True,
            use_ocr=False,
            header=False,
            footer=False,
        )
    except Exception:
        return None

    pages: dict[int, str] = {}
    for chunk in chunks:
        number = (chunk.get("metadata") or {}).get("page_number")
        if number is None:
            continue
        pages[int(number)] = chunk.get("text") or ""
    return pages


def _ensure_md_cache(cache: dict, pdf: pymupdf.Document) -> dict[int, str] | None:
    if not cache["loaded"]:
        cache["loaded"] = True
        cache["pages"] = _load_pymupdf4llm_pages(pdf)
    return cache["pages"]


def _engine_text(
    engine: str,
    *,
    page: pymupdf.Page,
    page_number: int,
    plain_text: str,
    cache: dict,
    pdf: pymupdf.Document,
) -> str:
    if engine == "pymupdf4llm":
        pages = _ensure_md_cache(cache, pdf)
        return pages.get(page_number, "") if pages else ""
    return plain_text


def parse_pdf(
    pdf_path: str | Path,
    min_text_length: int = MIN_TEXT_LENGTH_FOR_OCR,
    ocr_dpi: int = OCR_DPI,
    engine: str | None = None,
    ocr_engine: str | None = None,
) -> list[Document]:
    """Parse a PDF page-by-page.

    The primary engine is ``settings.PDF_TEXT_ENGINE`` (default ``pymupdf``);
    when a page's text is too short the other engine is tried, then OCR.
    """

    pdf_path = validate_file(pdf_path, ALLOWED_EXTENSIONS)

    primary = _normalize_engine(engine)
    fallback = "pymupdf4llm" if primary == "pymupdf" else "pymupdf"

    pdf = pymupdf.open(pdf_path)
    documents: list[Document] = []
    md_cache = {"loaded": False, "pages": None}

    try:
        if len(pdf) == 0:
            raise ValueError("The PDF contains no pages.")

        if len(pdf) > MAX_PDF_PAGES:
            raise ValueError(
                f"The PDF has {len(pdf)} pages, which exceeds the "
                f"{MAX_PDF_PAGES} page limit."
            )

        filename = pdf_path.name

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            plain_text = page.get_text("text")

            chosen_engine: str | None = None
            chosen_text = ""

            for name in (primary, fallback):
                text = _engine_text(
                    name,
                    page=page,
                    page_number=page_number,
                    plain_text=plain_text,
                    cache=md_cache,
                    pdf=pdf,
                )
                if has_sufficient_text(
                    text, min_text_length=min_text_length
                ) and not looks_like_gibberish(text):
                    chosen_engine, chosen_text = name, text
                    break

            if chosen_engine is not None:
                if _should_strip_markdown(chosen_engine):
                    chosen_text = strip_markdown_artifacts(chosen_text)
                documents.append(
                    make_document(
                        normalize_text(chosen_text),
                        filename=filename,
                        page_number=page_number,
                        ocr_used=False,
                        extractor=chosen_engine,
                    )
                )
                continue

            ocr_text, ocr_used_engine = ocr.ocr_image_with_engine(
                render_page_to_image(page, dpi=ocr_dpi),
                engine=ocr_engine,
            )
            documents.append(
                make_document(
                    normalize_text(ocr_text),
                    filename=filename,
                    page_number=page_number,
                    ocr_used=True,
                    extractor="ocr",
                    ocr_engine=ocr_used_engine,
                )
            )

    finally:
        pdf.close()

    return documents
