"""Tabular parsing: .csv (stdlib) and .xlsx (openpyxl).

Each data row is rendered as ``Header: value | ...`` and rows are packed into
pseudo-pages of ``TABLE_ROWS_PER_PAGE`` so every citation points at a small,
readable slice of the sheet.
"""
from __future__ import annotations

import csv
from pathlib import Path

from langchain_core.documents import Document

from config.settings import MAX_TABLE_ROWS, TABLE_ROWS_PER_PAGE

from .common import make_document, normalize_text, render_table_lines, validate_file

CSV_EXTENSIONS = (".csv",)
XLSX_EXTENSIONS = (".xlsx",)


def _rows_to_documents(
    rows: list[list[str]],
    *,
    filename: str,
    rows_per_page: int = TABLE_ROWS_PER_PAGE,
    sheet: str | None = None,
    page_start: int = 1,
) -> list[Document]:
    lines = render_table_lines(rows)
    if not lines:
        return []

    if len(lines) > MAX_TABLE_ROWS:
        lines = lines[:MAX_TABLE_ROWS]

    prefix = f"[Sheet: {sheet}]\n" if sheet else ""
    pages = [
        prefix + "\n".join(lines[i : i + rows_per_page])
        for i in range(0, len(lines), rows_per_page)
    ]

    extra = {"sheet": sheet} if sheet else {}
    return [
        make_document(
            page_text,
            filename=filename,
            page_number=page_number,
            **extra,
        )
        for page_number, page_text in enumerate(pages, start=page_start)
    ]


def parse_csv(
    file_path: str | Path,
    rows_per_page: int = TABLE_ROWS_PER_PAGE,
) -> list[Document]:
    """Parse a CSV file; the first row is treated as the header."""

    path = validate_file(file_path, CSV_EXTENSIONS)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    return _rows_to_documents(rows, filename=path.name, rows_per_page=rows_per_page)


def parse_xlsx(
    file_path: str | Path,
    rows_per_page: int = TABLE_ROWS_PER_PAGE,
) -> list[Document]:
    """Parse an .xlsx workbook; every sheet becomes its own set of pages."""

    path = validate_file(file_path, XLSX_EXTENSIONS)

    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)

    try:
        documents: list[Document] = []
        # Page numbers run across the whole workbook: chunk ids and citation
        # keys are (document_id, page_number), so per-sheet restarts would
        # collide.
        page_start = 1
        for sheet in workbook.worksheets:
            rows = [
                ["" if cell is None else str(cell) for cell in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            sheet_documents = _rows_to_documents(
                rows,
                filename=path.name,
                rows_per_page=rows_per_page,
                sheet=sheet.title,
                page_start=page_start,
            )
            documents.extend(sheet_documents)
            page_start += len(sheet_documents)
        return documents
    finally:
        workbook.close()
