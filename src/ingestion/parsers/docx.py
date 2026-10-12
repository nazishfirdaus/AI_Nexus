"""DOCX parsing via python-docx.

Body elements are walked in document order so tables stay interleaved with the
paragraphs around them. Headings start a new pseudo-page, which keeps citation
granularity useful for long contracts.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from config import settings
from config.settings import TEXT_PAGE_TARGET_CHARS

from .common import (
    clean_page_text,
    make_document,
    render_table_lines,
    validate_file,
)

ALLOWED_EXTENSIONS = (".docx",)


def _paragraph_style_name(paragraph) -> str:
    try:
        style = paragraph.style
        return (style.name or "") if style is not None else ""
    except (KeyError, ValueError):
        return ""


def _is_heading(style_name: str) -> bool:
    lowered = style_name.lower()
    return lowered.startswith("heading") or lowered.startswith("title")


def _table_rows(table) -> list[list[str]]:
    rows = []
    for row in table.rows:
        rows.append([cell.text for cell in row.cells])
    return rows


def _embedded_metadata(document) -> dict:
    """Core-property metadata (title/author/created/keywords) if present."""

    if not settings.EXTRACT_EMBEDDED_METADATA:
        return {}
    try:
        props = document.core_properties
    except Exception:
        return {}
    meta: dict = {}
    if props.title:
        meta["doc_title"] = str(props.title).strip()
    if props.author:
        meta["doc_author"] = str(props.author).strip()
    if props.created:
        meta["doc_created"] = props.created.isoformat()
    if props.modified:
        meta["doc_modified"] = props.modified.isoformat()
    if props.keywords:
        meta["doc_keywords"] = str(props.keywords).strip()
    return meta


def _group_blocks(
    blocks: list[tuple[bool, str, int, int]],
    target_chars: int,
) -> list[tuple[str, int, int]]:
    """Pack ``(is_heading, text, raw_chars, clean_chars)`` blocks into pages.

    A heading always opens a new page; otherwise pages fill to about
    ``target_chars``. Cleaning stats are summed per page.
    """

    pages: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    size = raw = clean = 0

    def flush() -> None:
        nonlocal buffer, size, raw, clean
        if buffer:
            pages.append(("\n\n".join(buffer), raw, clean))
        buffer, size, raw, clean = [], 0, 0, 0

    for is_heading, text, raw_chars, clean_chars in blocks:
        if is_heading and buffer:
            flush()
        if buffer and size + len(text) > target_chars:
            flush()
        buffer.append(text)
        size += len(text) + 2
        raw += raw_chars
        clean += clean_chars
        if size >= target_chars:
            flush()

    flush()
    return pages


def parse_docx(
    file_path: str | Path,
    target_chars: int = TEXT_PAGE_TARGET_CHARS,
) -> list[Document]:
    """Parse a .docx file into heading/size-delimited pseudo-pages."""

    path = validate_file(file_path, ALLOWED_EXTENSIONS)

    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = DocxDocument(str(path))
    doc_meta = _embedded_metadata(document)

    blocks: list[tuple[bool, str, int, int]] = []

    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text, stats = clean_page_text(paragraph.text)
            if not text:
                continue
            blocks.append(
                (
                    _is_heading(_paragraph_style_name(paragraph)),
                    text,
                    stats["raw_chars"],
                    stats["clean_chars"],
                )
            )

        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            for line in render_table_lines(_table_rows(table)):
                length = len(line)
                blocks.append((False, line, length, length))

    pages = _group_blocks(blocks, target_chars=target_chars)

    return [
        make_document(
            page_text,
            filename=path.name,
            page_number=page_number,
            raw_chars=raw_chars,
            clean_chars=clean_chars,
            **doc_meta,
        )
        for page_number, (page_text, raw_chars, clean_chars) in enumerate(
            pages, start=1
        )
    ]
