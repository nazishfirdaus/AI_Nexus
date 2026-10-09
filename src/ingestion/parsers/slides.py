"""PPTX parsing via python-pptx: one pseudo-page per slide."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from .common import make_document, normalize_text, render_table_lines, validate_file

ALLOWED_EXTENSIONS = (".pptx",)


def _slide_text(slide) -> str:
    parts: list[str] = []

    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            for paragraph in shape.text_frame.paragraphs:
                text = normalize_text(paragraph.text)
                if text:
                    parts.append(text)
        if getattr(shape, "has_table", False):
            rows = [
                [cell.text for cell in row.cells]
                for row in shape.table.rows
            ]
            parts.extend(render_table_lines(rows))

    if getattr(slide, "has_notes_slide", False):
        notes = normalize_text(slide.notes_slide.notes_text_frame.text)
        if notes:
            parts.append(f"Notes: {notes}")

    return normalize_text("\n".join(parts))


def parse_pptx(file_path: str | Path) -> list[Document]:
    """Parse a .pptx deck; each non-empty slide is one page (slide number)."""

    path = validate_file(file_path, ALLOWED_EXTENSIONS)

    from pptx import Presentation

    presentation = Presentation(str(path))

    documents: list[Document] = []
    empty = 0

    for slide_number, slide in enumerate(presentation.slides, start=1):
        text = _slide_text(slide)
        if not text:
            empty += 1
            continue
        documents.append(
            make_document(
                text,
                filename=path.name,
                page_number=slide_number,
            )
        )

    if not documents:
        raise ValueError(
            f"No text could be extracted from {path.name} "
            f"({empty} slide(s) contained no text)."
        )

    return documents
