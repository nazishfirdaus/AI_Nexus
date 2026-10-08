"""Plain-text parsing: .txt and .md files split on blank lines into sections."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from config.settings import TEXT_PAGE_TARGET_CHARS

from .common import (
    group_sections,
    make_document,
    normalize_text,
    read_text_bytes,
    validate_file,
)

ALLOWED_EXTENSIONS = (".txt", ".md")


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

    sections = [
        normalized
        for block in re.split(r"\n{2,}", raw)
        if (normalized := normalize_text(block))
    ]

    if not sections:
        return []

    pages = group_sections(sections, target_chars=target_chars)

    return [
        make_document(
            page_text,
            filename=path.name,
            page_number=page_number,
        )
        for page_number, page_text in enumerate(pages, start=1)
    ]
