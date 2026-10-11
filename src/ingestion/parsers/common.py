"""Shared helpers every format parser builds on.

Every parser produces LangChain Documents with the same metadata schema as the
PDF parser (document_id, filename, page_number, source, ocr_used) so the
chunker, aggregator, scrubber and UI stay format-agnostic.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

from config.settings import (
    MIN_TEXT_LENGTH_FOR_OCR,
    NORMALIZE_DEHYPHENATE,
    OCR_GIBBERISH_MIN_ALNUM_RATIO,
    TEXT_PAGE_TARGET_CHARS,
)

# Zero-width / invisible characters that carry no meaning and only corrupt tokens.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
# Control characters that should never survive extraction (keep \n and \t).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Word split across a line break by a hyphen ("loan ac-\ncount" -> "loan account").
_DEHYPHEN_RE = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")


def normalize_text(text: str) -> str:
    """
    Normalize extracted text.

    Unicode is NFKC-normalized (ligatures, full-width forms and NBSP become their
    plain ASCII equivalents), then soft hyphens, zero-width and control characters
    are removed, so downstream tokenizers see clean word forms. Whitespace is
    tidied while meaningful line structure is preserved.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Canonical Unicode form: "ﬁnance" -> "finance", full-width digits -> ASCII,
    # NBSP -> space. Runs before the whitespace pass so NBSP is collapsed too.
    text = unicodedata.normalize("NFKC", text)

    # Strip invisible characters that would otherwise split or corrupt tokens.
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.translate(_ZERO_WIDTH)  # zero-width space/joiner, BOM
    text = _CONTROL_RE.sub("", text)

    # Rejoin words hyphenated across a line break (optional).
    if NORMALIZE_DEHYPHENATE:
        text = _DEHYPHEN_RE.sub("", text)

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


# Markdown relics emitted by layout-aware extractors (pymupdf4llm). Stripped only
# where requested, so uploaded .md files keep their authoring syntax.
_MD_FENCE_RE = re.compile(r"^[ \t]*```.*$", re.MULTILINE)
_MD_SEPARATOR_ROW_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$", re.MULTILINE
)


def strip_markdown_artifacts(text: str) -> str:
    """Remove Markdown syntax so only the readable content remains.

    Deliberately conservative: headers and list markers are only stripped at line
    start, emphasis is limited to ``*``/``**`` (never ``_``, which is common in
    identifiers such as ``IN_LOAN_NO``), and a leading ``-`` before a digit is
    preserved so negative amounts survive. Line structure is left for
    ``normalize_text`` to tidy.
    """

    if not text:
        return ""

    text = _MD_FENCE_RE.sub("", text)                          # ``` fence lines
    text = re.sub(r"`([^`\n]+)`", r"\1", text)                 # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)           # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)       # links -> label
    # Line-relative whitespace is [ \t] (never \s) so a match cannot swallow the
    # newline that separates it from the next line.
    text = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+", "", text, flags=re.MULTILINE)   # headers
    text = re.sub(
        r"^[ \t]{0,3}[-*+][ \t]+(?=\D)", "", text, flags=re.MULTILINE
    )                                                          # bullets
    text = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", text, flags=re.MULTILINE)   # rules
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)           # bold
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)  # italic
    text = _MD_SEPARATOR_ROW_RE.sub("", text)                  # |---|---| rows
    text = re.sub(
        r"^[ \t]*\|(.*)\|[ \t]*$", lambda m: m.group(1).strip(), text, flags=re.MULTILINE
    )                                                          # table pipes

    return text


def looks_like_gibberish(
    text: str,
    min_alnum_ratio: float = OCR_GIBBERISH_MIN_ALNUM_RATIO,
) -> bool:
    """Heuristic: does ``text`` look like OCR/extraction noise rather than prose?

    Uses the alphanumeric ratio over non-space characters. Real text (including
    number-heavy financial tables) sits well above 0.5; symbol soup from a bad
    scan falls below it. Strings shorter than ``MIN_TEXT_LENGTH_FOR_OCR`` return
    False -- ``has_sufficient_text`` already rejects those, and short snippets are
    too small to judge.
    """

    normalized = normalize_text(text)
    if len(normalized) < MIN_TEXT_LENGTH_FOR_OCR:
        return False

    compact = "".join(ch for ch in normalized if not ch.isspace())
    if not compact:
        return True

    alnum = sum(ch.isalnum() for ch in compact)
    return (alnum / len(compact)) < min_alnum_ratio


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


_SEPARATOR_CELL_RE = re.compile(r"^[\-\|:=_.·~]+$")


def _padded(cells: list[str], width: int) -> list[str]:
    if len(cells) < width:
        return cells + [""] * (width - len(cells))
    return cells


def render_table_lines(rows: list[list[str]]) -> list[str]:
    """Render a table (first row = header) as ``Header: value | ...`` lines.

    Row-per-record rendering keeps column context attached to each value, which
    is what dense + BM25 retrieval matches on.

    Hardening for real-world sheets: ragged rows are padded to the widest row so
    values stay aligned with their headers (pdfplumber tables and merged xlsx
    ranges routinely produce uneven widths), empty rows are skipped, and pure
    separator rows (``---`` / ``|---|---|``) are dropped instead of polluting
    the corpus.
    """

    cleaned: list[list[str]] = []
    for row in rows:
        cells = [str(cell).strip() for cell in row]
        if not any(cells):
            continue
        separators = [c for c in cells if c]
        if all(_SEPARATOR_CELL_RE.match(c) for c in separators):
            continue
        cleaned.append(cells)

    if not cleaned:
        return []

    width = max(len(row) for row in cleaned)
    header = _padded(cleaned[0], width)
    lines: list[str] = []

    for row in cleaned[1:]:
        data = _padded(row, width)
        pairs = [
            f"{name}: {value}"
            for name, value in zip(header, data)
            if name and value
        ]
        if pairs:
            lines.append(" | ".join(pairs))

    if not lines:
        # Header-only table: keep whatever content exists instead of dropping it.
        lines = [" | ".join(cell for cell in row if cell) for row in cleaned]

    return lines
