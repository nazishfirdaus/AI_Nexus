"""Text cleaning / preprocessing tests.

Covers the Unicode + control-character cleanup, optional de-hyphenation,
Markdown-artifact stripping and the OCR gibberish guard added in
``src/ingestion/parsers/common.py`` and ``pdf.py``.
"""
from pathlib import Path

from src.ingestion.parsers import common
from src.ingestion.parsers import pdf as pdf_mod
from src.ingestion.parsers.common import (
    CleaningStats,
    clean_page_text,
    looks_like_gibberish,
    normalize_text,
    normalize_text_with_stats,
    strip_markdown_artifacts,
)

PDF_PATH = Path("Synthetic_Mortgage_Loan_File_TEST.pdf")


# --------------------------------------------------------------- normalize_text
def test_nfkc_folds_ligature():
    assert normalize_text("of\ufb01ce") == "office"


def test_nfkc_folds_fullwidth_digits():
    assert normalize_text("Total \uff11\uff12\uff13") == "Total 123"


def test_nbsp_becomes_space():
    assert normalize_text("Amount\u00a0due") == "Amount due"


def test_soft_hyphen_removed():
    assert normalize_text("loan\u00adaccount") == "loanaccount"


def test_zero_width_and_bom_removed():
    assert normalize_text("a\u200bb\u200dc\ufeffd") == "abcd"


def test_control_characters_removed_but_newlines_kept():
    assert normalize_text("a\x07b\x1fc\nnext") == "abc\nnext"


def test_dehyphenation_joins_split_word():
    assert normalize_text("loan ac-\ncount details") == "loan account details"


def test_dehyphenation_off_keeps_hyphen(monkeypatch):
    monkeypatch.setattr(common, "NORMALIZE_DEHYPHENATE", False)
    result = normalize_text("loan ac-\ncount details")
    assert "ac-" in result
    assert "count details" in result


def test_normalize_is_idempotent():
    messy = "  \ufb01nance\u00a0\u200b  ac-\ncount \r\n\r\n\r\n end  "
    once = normalize_text(messy)
    assert normalize_text(once) == once


# ------------------------------------------------------- normalize_text_with_stats
def test_stats_report_removed_characters():
    cleaned, stats = normalize_text_with_stats("loan\u00adaccount")
    assert cleaned == "loanaccount"
    assert isinstance(stats, CleaningStats)
    assert stats.input_chars == len("loan\u00adaccount")
    assert stats.output_chars == len("loanaccount")
    assert stats.removed == 1


def test_stats_empty_input():
    cleaned, stats = normalize_text_with_stats("")
    assert cleaned == ""
    assert stats.removed == 0


def test_clean_page_text_returns_metadata():
    cleaned, meta = clean_page_text("  Loan   amount\u00a0due  ")
    assert cleaned == "Loan amount due"
    assert meta["raw_chars"] == len("  Loan   amount\u00a0due  ")
    assert meta["clean_chars"] == len(cleaned)
    # metadata is ready to splat into make_document
    assert set(meta) == {"raw_chars", "clean_chars"}


# ------------------------------------------------------- strip_markdown_artifacts
def test_strip_headers():
    assert strip_markdown_artifacts("## Loan Schedule") == "Loan Schedule"


def test_strip_emphasis():
    assert strip_markdown_artifacts("**bold** and *italic*") == "bold and italic"


def test_strip_links_and_images():
    assert strip_markdown_artifacts("see [docs](http://x) now") == "see docs now"
    assert strip_markdown_artifacts("![alt](http://x)text") == "text"


def test_strip_inline_code():
    assert strip_markdown_artifacts("run `pip install` now") == "run pip install now"


def test_bullet_text_marker_removed():
    assert strip_markdown_artifacts("- first item") == "first item"


def test_bullet_before_digit_is_preserved():
    # A leading "-" before a number is a negative value, not a Markdown bullet.
    assert strip_markdown_artifacts("- 45000") == "- 45000"


def test_strip_table_pipes_and_separator_row():
    table = "| Name | Amount |\n|------|--------|\n| Ravi | 45000 |"
    stripped = strip_markdown_artifacts(table)
    assert "|------" not in stripped
    lines = [line for line in stripped.splitlines() if line]
    assert lines == ["Name | Amount", "Ravi | 45000"]


def test_strip_horizontal_rule():
    assert strip_markdown_artifacts("above\n---\nbelow") == "above\n\nbelow"


# --------------------------------------------------------- looks_like_gibberish
def test_gibberish_detected():
    assert looks_like_gibberish("@@@ ### !!! ... " * 20) is True


def test_normal_text_is_not_gibberish():
    prose = "The borrower shall repay the loan in equal monthly instalments. " * 5
    assert looks_like_gibberish(prose) is False


def test_number_heavy_table_is_not_gibberish():
    rows = "Sanction Amount: 4,50,000 | ROI: 7.25% | Tenure: 240 months\n" * 5
    assert looks_like_gibberish(rows) is False


def test_short_string_never_flagged():
    assert looks_like_gibberish("@@@ ###") is False


# ------------------------------------------------------------- pdf integration
def test_garbled_page_falls_through_to_pymupdf4llm(monkeypatch):
    """A low-alnum page is rejected by the gibberish guard and retried."""

    def fake_engine_text(engine, *, page, page_number, plain_text, cache, tables_cache, pdf):
        if engine == "pymupdf":
            return "@@@ ### !!! ... " * 20
        return (
            "## Loan Schedule\n| Field | Value |\n|-------|-------|\n| ROI | 7.25 |\n"
        ) * 3

    monkeypatch.setattr(pdf_mod, "_engine_text", fake_engine_text)

    documents = pdf_mod.parse_pdf(PDF_PATH)

    assert documents
    assert {d.metadata["extractor"] for d in documents} == {"pymupdf4llm"}
    assert all(d.metadata["ocr_used"] is False for d in documents)
    # Markdown relics are stripped for the pymupdf4llm extractor.
    for document in documents:
        assert "#" not in document.page_content
        assert "|-------|" not in document.page_content
