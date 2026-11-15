"""Format dispatcher: one entry point for every supported document type.

``parse_document`` is what the pipeline calls; individual parsers import their
heavy libraries lazily so importing this package stays cheap.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.observability import hash_id, info, span

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
    with span(
        "ingestion.parsers.parse_document",
        file_id=hash_id(str(path)),
        extension=path.suffix.lower(),
    ):
        extension = path.suffix.lower()

        if extension == ".pdf":
            from .pdf import parse_pdf

            documents = parse_pdf(path)

        elif extension in (".txt", ".md"):
            from .text import parse_text

            documents = parse_text(path)

        elif extension in (".html", ".htm"):
            from .html import parse_html

            documents = parse_html(path)

        elif extension == ".docx":
            from .docx import parse_docx

            documents = parse_docx(path)

        elif extension == ".csv":
            from .tables import parse_csv

            documents = parse_csv(path)

        elif extension == ".xlsx":
            from .tables import parse_xlsx

            documents = parse_xlsx(path)

        elif extension == ".pptx":
            from .slides import parse_pptx

            documents = parse_pptx(path)

        elif extension in IMAGE_EXTENSIONS:
            from .image import parse_image

            documents = parse_image(path)

        else:
            raise ValueError(
                f"Unsupported file type '{extension or '(no extension)'}'. "
                f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        info(
            "Document parsed",
            extension=extension,
            page_count=len(documents),
        )
        return documents
