"""HTML parsing: strip scripts/styles and return readable page Documents."""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.observability import error, hash_id, info, span

from .common import clean_page_text, make_document, read_text_bytes, validate_file

ALLOWED_EXTENSIONS = (".html", ".htm")


def parse_html(file_path: str | Path) -> list[Document]:
    """Parse one HTML file into a single page-level Document."""
    path = validate_file(file_path, ALLOWED_EXTENSIONS)
    with span(
        "ingestion.parsers.html.parse_html",
        file_id=hash_id(str(path)),
        file_ext=path.suffix.lower(),
    ):
        try:
            soup = BeautifulSoup(read_text_bytes(path.read_bytes()), "html.parser")
            for tag in soup(["script", "style", "meta", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            phrases = (p.strip() for line in lines for p in line.split("  "))
            raw = "\n".join(p for p in phrases if p)
            cleaned, stats = clean_page_text(raw)

            info("HTML parsed", char_count=len(cleaned), page_count=1 if cleaned else 0)
            if not cleaned:
                return []

            return [
                make_document(
                    cleaned,
                    filename=path.name,
                    page_number=1,
                    doc_type="html",
                    raw_chars=stats["raw_chars"],
                    clean_chars=stats["clean_chars"],
                )
            ]
        except Exception as exc:
            error("HTML parse failed", error_type=type(exc).__name__)
            raise