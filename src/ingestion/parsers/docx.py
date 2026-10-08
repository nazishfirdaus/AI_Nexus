"""DOCX parsing via python-docx.

Body elements are walked in document order so tables stay interleaved with the
paragraphs around them. Headings start a new pseudo-page, which keeps citation
granularity useful for long contracts.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from config.settings import TEXT_PAGE_TARGET_CHARS

from .common import make_document, normalize_text, render_table_lines, validate_file

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


def _group_blocks(
    blocks: list[tuple[bool, str]],
    target_chars: int,
) -> list[str]:
    """Pack (is_heading, text) blocks into pseudo-pages.

    A heading always opens a new page; otherwise pages fill to about
    ``target_chars``.
    """

    pages: list[str] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            pages.append("\n\n".join(buffer))
        buffer, size = [], 0

    for is_heading, text in blocks:
        if is_heading and buffer:
            flush()
        if buffer and size + len(text) > target_chars:
            flush()
        buffer.append(text)
        size += len(text) + 2
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

    blocks: list[tuple[bool, str]] = []

    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = normalize_text(paragraph.text)
            if not text:
                continue
            blocks.append((_is_heading(_paragraph_style_name(paragraph)), text))

        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            for line in render_table_lines(_table_rows(table)):
                blocks.append((False, line))

    pages = _group_blocks(blocks, target_chars=target_chars)

    return [
        make_document(
            page_text,
            filename=path.name,
            page_number=page_number,
        )
        for page_number, page_text in enumerate(pages, start=1)
    ]
