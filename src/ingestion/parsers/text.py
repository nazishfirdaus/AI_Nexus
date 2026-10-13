"""Plain-text parsing: .txt and .md files split on blank lines into sections."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from config.settings import TEXT_PAGE_TARGET_CHARS

from .common import (
    clean_page_text,
    make_document,
    read_text_bytes,
    validate_file,
)

ALLOWED_EXTENSIONS = (".txt", ".md")


def _group_with_stats(
    sections: list[tuple[str, int, int]],
    target_chars: int,
) -> list[tuple[str, int, int]]:
    """Greedily pack ``(text, raw_chars, clean_chars)`` sections into pages.

    Mirrors ``common.group_sections`` but carries the cleaning stats through so
    each pseudo-page reports how much of its source survived cleaning.
    """

    pages: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    size = raw = clean = 0

    def flush() -> None:
        nonlocal buffer, size, raw, clean
        if buffer:
            pages.append(("\n\n".join(buffer), raw, clean))
        buffer, size, raw, clean = [], 0, 0, 0

    for text, raw_chars, clean_chars in sections:
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


def parse_text(
    file_path: str | Path,
    target_chars: int = TEXT_PAGE_TARGET_CHARS,
) -> list[Document]:
    """
    Parse a text/Markdown file into pseudo-pages.

    Blank lines delimit sections; sections are packed greedily into pages of
    about ``target_chars`` so citations stay meaningful on long files.

    Sections are split *before* normalization because ``normalize_text`` (by
    design) drops empty lines.
    """

    path = validate_file(file_path, ALLOWED_EXTENSIONS)

    raw = read_text_bytes(path.read_bytes())
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    sections: list[tuple[str, int, int]] = []
    for block in re.split(r"\n{2,}", raw):
        cleaned, stats = clean_page_text(block)
        if cleaned:
            sections.append(
                (cleaned, stats["raw_chars"], stats["clean_chars"])
            )

    if not sections:
        return []

    pages = _group_with_stats(sections, target_chars=target_chars)

    return [
        make_document(
            page_text,
            filename=path.name,
            page_number=page_number,
            raw_chars=raw_chars,
            clean_chars=clean_chars,
        )
        for page_number, (page_text, raw_chars, clean_chars) in enumerate(
            pages, start=1
        )
    ]
