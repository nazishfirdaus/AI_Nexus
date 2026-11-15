"""Display-side placeholder neutralisation."""
from src.api.serializers import message_out
from src.redaction.placeholders import (
    display_text,
    neutralize_placeholders,
    repair_mojibake,
    strip_internal_identifiers,
)


def test_known_placeholders_become_natural_nouns():
    text = (
        "Borrower <PERSON> holds account <IN_ACCOUNT_NO>, loan <IN_LOAN_NO>, "
        "PAN <PAN_NUMBER> and lives at <IN_ADDRESS>."
    )
    result = neutralize_placeholders(text)
    assert "<" not in result and ">" not in result
    assert "the borrower" in result.lower()
    assert "the account number" in result
    assert "the loan number" in result
    assert "the PAN" in result
    assert "the property address" in result


def test_unknown_uppercase_tokens_are_neutralised():
    assert neutralize_placeholders("Value <WIDGET_ID> here.") == (
        "Value the redacted value here."
    )


def test_lowercase_markup_is_left_alone():
    text = "Use <context> and <br> literally."
    assert neutralize_placeholders(text) == text


def test_duplicate_label_and_honorific_are_cleaned():
    assert neutralize_placeholders("Borrower <PERSON> applied.") == (
        "The borrower applied."
    )
    assert neutralize_placeholders("Signed by Mr. <PERSON>.") == (
        "Signed by the borrower."
    )


def test_sentence_start_is_capitalised():
    assert neutralize_placeholders("<PERSON> signed the form.") == (
        "The borrower signed the form."
    )


def test_spacing_and_punctuation_are_cleaned():
    result = neutralize_placeholders("Name: <PERSON> , account <IN_ACCOUNT_NO> .")
    assert "  " not in result
    assert " ," not in result and " ." not in result
    assert result == "Name: the borrower, the account number."


def test_redundant_field_label_is_collapsed():
    assert neutralize_placeholders("Aadhaar <AADHAAR_NUMBER> was verified.") == (
        "The Aadhaar number was verified."
    )
    assert neutralize_placeholders("Loan A/c No: <IN_LOAN_NO>") == "The loan number"


def test_text_without_placeholders_is_unchanged():
    text = "The rate is 7.25%."
    assert neutralize_placeholders(text) is text


def test_message_serializer_cleans_content_and_citation_snippets():
    out = message_out(
        {
            "id": 1,
            "role": "assistant",
            "content": "Borrower <PERSON> has loan <IN_LOAN_NO>.",
            "citations": [
                {
                    "page_number": 2,
                    "score": 0.9,
                    "snippet": "A/c No <IN_ACCOUNT_NO>.",
                    "document_id": "loan.pdf",
                    "filename": "loan.pdf",
                }
            ],
            "feedback": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    assert "<PERSON>" not in out.content
    assert out.citations is not None
    assert "<IN_ACCOUNT_NO>" not in out.citations[0].snippet


def test_repairs_utf8_decoded_as_latin1_and_normalises_typography():
    text = "Document\u00e2\u0080\u00afID\u00e2\u0080\u00afANX\u00e2\u0080\u0091MTG\u00e2\u0080\u00910001"
    repaired = repair_mojibake(text)
    assert "â" not in repaired
    assert "\u202f" not in repaired
    assert repaired == "Document ID ANX-MTG-0001"


def test_strip_internal_identifiers_removes_parenthetical_and_line_forms():
    text = (
        "The file (Document ID: ANX-MTG-0001) covers the loan.\n"
        "Document ID: ANX-MTG-0001\n"
        "Document ID is a field, not a value."
    )
    cleaned = strip_internal_identifiers(text)
    assert "ANX-MTG-0001" not in cleaned
    assert "Document ID is a field" in cleaned
    assert "The file covers the loan." in cleaned


def test_display_text_hides_ids_addresses_and_placeholders_together():
    text = (
        "The file (Document ID: ANX-MTG-0001) is at 1847 Willow <PERSON>, "
        "<PERSON>, TX 78745."
    )
    cleaned = display_text(text)
    assert "ANX-MTG-0001" not in cleaned
    assert "1847" not in cleaned and "78745" not in cleaned
    assert "<PERSON>" not in cleaned
    assert "the property address" in cleaned


def test_display_text_preserves_markdown():
    assert display_text("**Loan terms**\nThe rate is 7.25%.") == (
        "**Loan terms**\nThe rate is 7.25%."
    )
