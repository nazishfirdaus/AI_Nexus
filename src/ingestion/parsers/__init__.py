"""Format dispatcher: one entry point for every supported document type.

``parse_document`` is what the pipeline calls; individual parsers import their
heavy libraries lazily so importing this package stays cheap.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config.settings import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from langchain_core.documents import Document

__all__ = ["SUPPORTED_EXTENSIONS", "parse_document"]


def parse_document(file_path: str | Path) -> "list[Document]":
    """Parse any supported file into page-level LangChain Documents.

    Raises FileNotFoundError/ValueError with a clear message for missing,
    empty or unsupported files.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        from .pdf import parse_pdf

        return parse_pdf(path)

    if extension in (".txt", ".md"):
        from .text import parse_text

        return parse_text(path)

    if extension == ".docx":
        from .docx import parse_docx

        return parse_docx(path)

    if extension == ".csv":
        from .tables import parse_csv

        return parse_csv(path)

    if extension == ".xlsx":
        from .tables import parse_xlsx

        return parse_xlsx(path)

    if extension == ".pptx":
        from .slides import parse_pptx

        return parse_pptx(path)

    if extension in IMAGE_EXTENSIONS:
        from .image import parse_image

        return parse_image(path)

    raise ValueError(
        f"Unsupported file type '{extension or '(no extension)'}'. "
        f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
