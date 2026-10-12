"""Repeated-line / boilerplate removal tests (``drop_repeated_boilerplate``).

Covers the conservative contract: only multi-page documents, only lines that
repeat on most pages, and never money amounts or labelled fields.
"""
from langchain_core.documents import Document

from src.ingestion.parsers.common import drop_repeated_boilerplate


def _pages(texts):
    return [
        Document(page_content=text, metadata={"page_number": i})
        for i, text in enumerate(texts, start=1)
    ]


def test_drops_running_header_across_pages():
    pages = _pages(
        [
            "ACME Mortgage Ltd - Confidential\nInterest rate is 7.25 percent.\nPage 1",
            "ACME Mortgage Ltd - Confidential\nLoan tenure is 20 years.\nPage 2",
            "ACME Mortgage Ltd - Confidential\nPenalty is 2 percent.\nPage 3",
        ]
    )
    kept, removed = drop_repeated_boilerplate(pages)
    assert removed == 3  # the running header on each page
    assert kept[0].page_content == "Interest rate is 7.25 percent.\nPage 1"
    for page in kept:
        assert "ACME Mortgage Ltd - Confidential" not in page.page_content


def test_single_page_document_is_untouched():
    pages = _pages(["ACME Mortgage Ltd - Confidential\nRate is 7.25."])
    kept, removed = drop_repeated_boilerplate(pages)
    assert removed == 0
    assert kept[0].page_content.startswith("ACME Mortgage")


def test_two_page_document_below_min_pages_is_untouched():
    pages = _pages(["BOILER HEADER\nA", "BOILER HEADER\nB"])
    kept, removed = drop_repeated_boilerplate(pages)
    assert removed == 0


def test_amounts_are_protected_even_when_repeated():
    # A repeated total row must survive: it is a value, not boilerplate.
    pages = _pages(
        [
            "Statement\nTotal: Rs. 4,50,000\nLine A",
            "Statement\nTotal: Rs. 4,50,000\nLine B",
            "Statement\nTotal: Rs. 4,50,000\nLine C",
        ]
    )
    kept, removed = drop_repeated_boilerplate(pages)
    assert "Total: Rs. 4,50,000" in "\n".join(p.page_content for p in kept)


def test_labelled_fields_are_protected():
    pages = _pages(
        [
            "Header\nA/c No: 12345678\nA",
            "Header\nA/c No: 12345678\nB",
            "Header\nA/c No: 12345678\nC",
        ]
    )
    kept, removed = drop_repeated_boilerplate(pages)
    assert "A/c No: 12345678" in "\n".join(p.page_content for p in kept)


def test_tiny_repeated_lines_are_not_dropped():
    # Short tokens (e.g. section labels) below the char threshold are kept.
    pages = _pages(["Hdr\nX\nA", "Hdr\nX\nB", "Hdr\nX\nC"])
    kept, removed = drop_repeated_boilerplate(pages)
    assert "Hdr" in "\n".join(p.page_content for p in kept)


def test_page_is_never_cleaned_down_to_nothing():
    # Every line repeats, but removing them would leave the page empty: revert.
    pages = _pages(
        [
            "SAME LINE EVERYWHERE",
            "SAME LINE EVERYWHERE",
            "SAME LINE EVERYWHERE",
        ]
    )
    kept, removed = drop_repeated_boilerplate(pages)
    assert removed == 0
    assert all(p.page_content == "SAME LINE EVERYWHERE" for p in kept)


def test_reports_removed_count_in_metadata():
    pages = _pages(
        [
            "Running Footer Text Here\nReal content number one.",
            "Running Footer Text Here\nReal content number two.",
            "Running Footer Text Here\nReal content number three.",
        ]
    )
    kept, removed = drop_repeated_boilerplate(pages)
    assert removed == 3
    assert all(p.metadata["boilerplate_lines_removed"] == 1 for p in kept)
