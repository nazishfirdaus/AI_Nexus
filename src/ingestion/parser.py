"""Back-compat shim: the implementation moved to ``src/ingestion/parsers/``.

Existing importers (tests, evaluation) keep working:

    from src.ingestion.parser import parse_pdf
"""
from __future__ import annotations

from src.ingestion.parsers import parse_document
from src.ingestion.parsers.common import has_sufficient_text, normalize_text
from src.ingestion.parsers.pdf import (
    extract_text_with_ocr,
    parse_pdf,
    render_page_to_image,
)

__all__ = [
    "extract_text_with_ocr",
    "has_sufficient_text",
    "normalize_text",
    "parse_document",
    "parse_pdf",
    "render_page_to_image",
]
