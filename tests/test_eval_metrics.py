"""Tests for the eval harness metric helpers (no pipeline run)."""
from evaluation.run_eval import _norm_doc, _parse_temperatures, parse_citations, score_answer


def test_parse_inline_and_bibliography_citations():
    text = (
        "The rate is 7.25% (agreement.pdf, p.2). Tenure is 20 years "
        "(agreement.pdf, page 4).\nSources: (agreement.pdf, p.2); (policy.pdf, p.4)"
    )
    assert parse_citations(text) == [
        ("agreement.pdf", 2),
        ("agreement.pdf", 4),
        ("agreement.pdf", 2),
        ("policy.pdf", 4),
    ]


def test_parse_citations_none_when_text_is_empty():
    assert parse_citations("") == []
    assert parse_citations("No sources were used.") == []


def test_norm_doc_reduces_to_basename_lowercased():
    assert _norm_doc("C:/Users/me/Agreement.PDF") == "agreement.pdf"
    assert _norm_doc("mydir\\Policy.pdf") == "policy.pdf"


# ---------------- answer scoring ----------------
_ITEM = {
    "category": "direct_fact_lookup",
    "expected_answer": "$320,000",
    "acceptable_answer_points": ["$320,000", "320000"],
}


def test_score_answer_matches_curated_point():
    correct, matched = score_answer("The requested loan amount is $320,000.", _ITEM)
    assert correct and matched in _ITEM["acceptable_answer_points"]


def test_score_answer_matches_numeric_variant():
    assert score_answer("It is 320000 dollars.", _ITEM)[0] is True


def test_score_answer_is_case_and_whitespace_insensitive():
    item = {"category": "definition_meaning", "acceptable_answer_points": ["Debt-to-income ratio"]}
    assert score_answer("This is a  DEBT-TO-INCOME   RATIO.", item)[0] is True


def test_score_answer_wrong_and_empty_are_incorrect():
    assert score_answer("The loan amount is $250,000.", _ITEM) == (False, None)
    assert score_answer("", _ITEM) == (False, None)


def test_score_answer_accepts_refusal_for_negative_unknown():
    item = {"category": "negative_unknown",
            "acceptable_answer_points": ["Not provided", "Not stated"]}
    correct, matched = score_answer(
        "The document does not provide the spouse's income.", item
    )
    assert correct and matched.startswith("refusal:")


def test_score_answer_does_not_grant_refusal_to_positive_items():
    assert score_answer("The document does not provide the amount.", _ITEM)[0] is False


# ---------------- temperature parsing ----------------
def test_parse_temperatures_list_and_single():
    assert _parse_temperatures("0.0,0.1,0.3") == [0.0, 0.1, 0.3]
    assert _parse_temperatures("0.2") == [0.2]
    assert _parse_temperatures(" 0.1 , 0.5 ") == [0.1, 0.5]

