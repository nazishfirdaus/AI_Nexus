"""Shared helpers every format parser builds on.

Every parser produces LangChain Documents with the same metadata schema as the
PDF parser (document_id, filename, page_number, source, ocr_used) so the
chunker, aggregator, scrubber and UI stay format-agnostic.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from config.settings import (
    MIN_TEXT_LENGTH_FOR_OCR,
    TEXT_PAGE_TARGET_CHARS,
)


def normalize_text(text: str) -> str:
    """
    Normalize extracted text.

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
    Check whether extraction produced enough text.

    If not, a PDF page will be sent to OCR.
    """

    normalized_text = normalize_text(text)

    return len(normalized_text) >= min_text_length


def validate_file(file_path: str | Path, allowed: tuple[str, ...]) -> Path:
    """Common existence/type validation for every parser."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in allowed:
        raise ValueError(
            f"Expected one of {', '.join(allowed)}, got: {path.suffix or '(no extension)'}"
        )

    return path


def document_id_for(filename: str) -> str:
    """Stable document identity: the full filename (stem + extension).

    The extension keeps ``notes.txt`` and ``notes.pdf`` from aliasing each
    other into one corpus entry. The pipeline re-binds this to the original
    upload name, so temp paths never leak into identities.
    """

    return Path(filename).name


def make_document(
    text: str,
    *,
    filename: str,
    page_number: int,
    ocr_used: bool = False,
    **extra,
) -> Document:
    """Build one page-level Document with the standard metadata schema."""

    metadata = {
        "document_id": document_id_for(filename),
        "filename": filename,
        "page_number": page_number,
        "source": filename,
        "ocr_used": ocr_used,
    }
    metadata.update(extra)

    return Document(page_content=text, metadata=metadata)


def read_text_bytes(data: bytes) -> str:
    """Decode file bytes, trying sensible encodings in order."""

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def group_sections(
    sections: list[str],
    target_chars: int = TEXT_PAGE_TARGET_CHARS,
) -> list[str]:
    """Greedily pack text sections into pseudo-pages of about ``target_chars``.

    A section that already reaches the target becomes its own page; the chunker
    handles any oversized page downstream.
    """

    pages: list[str] = []
    buffer: list[str] = []
    size = 0

    for section in sections:
        if buffer and size + len(section) > target_chars:
            pages.append("\n\n".join(buffer))
            buffer, size = [], 0
        buffer.append(section)
        size += len(section) + 2
        if size >= target_chars:
            pages.append("\n\n".join(buffer))
            buffer, size = [], 0

    if buffer:
        pages.append("\n\n".join(buffer))

    return pages


def render_table_lines(rows: list[list[str]]) -> list[str]:
    """Render a table (first row = header) as ``Header: value | ...`` lines.

    Row-per-record rendering keeps column context attached to each value, which
    is what dense + BM25 retrieval matches on.
    """

    cleaned = [
        [str(cell).strip() for cell in row]
        for row in rows
        if any(str(cell).strip() for cell in row)
    ]
    if not cleaned:
        return []

    header = cleaned[0]
    lines: list[str] = []

    for row in cleaned[1:]:
        pairs = [
            f"{name}: {value}"
            for name, value in zip(header, row)
            if name and value
        ]
        if pairs:
            lines.append(" | ".join(pairs))

    if not lines:
        # Header-only table: keep whatever content exists instead of dropping it.
        lines = [" | ".join(cell for cell in row if cell) for row in cleaned]

    return lines
