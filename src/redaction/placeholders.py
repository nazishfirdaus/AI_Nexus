"""Turn PII placeholders into fluent generic wording for user-visible text.

The scrubber replaces sensitive values with ``<ENTITY_TYPE>`` tokens before any
text reaches a model. Those tokens are useful *inside* the pipeline (the model
must know a value was removed), but they must never be shown to the user: an
answer that reads "the account number is <IN_ACCOUNT_NO>" is both ugly and
confusing.

This module is the display-side counterpart to the scrubber. It leaves the
stored/embedded text untouched and only rewrites text on its way to a human
(answer text, citation snippets, conversation history read back from SQLite).
"""
from __future__ import annotations

import re

# Entity type -> natural phrase. The actual value is never reconstructed; these
# are generic nouns, not guesses. Domain wording ("the borrower") is preferred
# over "the person" because every document in this app is a mortgage document.
_REPLACEMENTS: dict[str, str] = {
    "PERSON": "the borrower",
    "IN_PERSON_NAME": "the borrower",
    "IN_ADDRESS": "the property address",
    "US_ADDRESS": "the property address",
    "IN_PINCODE": "the PIN code",
    "PHONE_NUMBER": "the phone number",
    "EMAIL_ADDRESS": "the email address",
    "PAN_NUMBER": "the PAN",
    "AADHAAR_NUMBER": "the Aadhaar number",
    "US_SSN": "the Social Security number",
    "VOTER_ID": "the voter ID",
    "DL_NO": "the driving licence number",
    "PASSPORT_NO": "the passport number",
    "TAN_NUMBER": "the TAN",
    "IN_ACCOUNT_NO": "the account number",
    "US_BANK_NUMBER": "the account number",
    "IBAN_CODE": "the account number",
    "IN_LOAN_NO": "the loan number",
    "CHEQUE_NO": "the cheque number",
    "POLICY_NO": "the policy number",
    "APPLICATION_ID": "the application number",
    "IFSC_CODE": "the IFSC code",
    "MICR_CODE": "the MICR code",
    "GSTIN": "the GSTIN",
    "UPI_ID": "the UPI ID",
    "CREDIT_CARD": "the card number",
    "IP_ADDRESS": "the IP address",
}

# Presidio's default anonymizer emits <UPPER_SNAKE_CASE>. The lower-case tags a
# model might write (<br>, <context>) deliberately do not match.
_PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]{1,64})>")

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?%])")
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
_DOUBLE_ARTICLE_RE = re.compile(r"\bthe\s+the\b", re.IGNORECASE)
# "Borrower the borrower applied" / "The applicant the individual ..." -> one noun.
_LABEL_DUPLICATE_RE = re.compile(
    r"\b(Borrower|Applicant|Guarantor|Co-?borrower|Customer|Person|Individual)"
    r"\s+the\s+(?:borrower|individual|person)\b",
    re.IGNORECASE,
)
# "Mr. the borrower" -> "the borrower"; the honorific was part of the name span.
_HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt|Sri)\.?\s+(the\s+(?:borrower|individual|person))\b",
    re.IGNORECASE,
)
# "Aadhaar the Aadhaar number" / "Loan A/c No the loan number" -> one phrase.
_LABEL_VALUE_RE = re.compile(
    r"\b(?:Loan\s+A/c|A/c|Aadhaar|Aadhar|PAN|TAN|GSTIN|UPI|IFSC|MICR|Account|"
    r"Loan|Card|Cheque|Policy|Application|Phone|Email|Address|PIN)\b"
    r"(?:\s*(?:No\.?|Number|ID|Code|#))?"
    r"\s*:?\s*"
    r"(the\s+(?:Aadhaar number|PAN|TAN|GSTIN|UPI ID|IFSC code|MICR code|"
    r"account number|loan number|card number|cheque number|policy number|"
    r"application number|phone number|email address|property address|PIN code))\b",
    re.IGNORECASE,
)
# Capitalise a replacement that landed at the start of a sentence/line.
_SENTENCE_START_RE = re.compile(r"(^|[.!?]\s+|\n\s*)(the\s+)")

# Streaming responses are decoded as UTF-8 now, but rows written before that fix
# (and cached answers) can still contain UTF-8 bytes that were decoded as
# Latin-1/CP1252. Repair only runs that form a valid multi-byte sequence, so
# ordinary accented text is left alone.
_MOJIBAKE_RE = re.compile(
    r"[\u00c2-\u00f4]"
    r"[\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6"
    r"\u02dc\u2013\u2014\u2018-\u201e\u2020-\u2022\u2026\u2030\u2039\u203a"
    r"\u20ac\u2122]+"
)
# Invisible typography that caused the "â" reports: narrow/no-break spaces and
# non-breaking hyphens are normalised so copied answers are searchable and
# stable across fonts.
_DISPLAY_TRANSLATION = str.maketrans({
    "\u00a0": " ",
    "\u2007": " ",
    "\u2009": " ",
    "\u202f": " ",
    "\u2010": "-",
    "\u2011": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
})

# Internal document IDs are not useful to the reader and must not be echoed
# from the source into an answer or citation snippet. The patterns deliberately
# only consume an uppercase/alphanumeric value, so a sentence such as
# "Document ID is a field" is not damaged.
_DOC_ID_PAREN_RE = re.compile(
    r"[ \t]*\([ \t]*(?i:document|doc)[ \t]*(?i:id)[ \t]*:?[ \t]*"
    r"[A-Z0-9][A-Z0-9 \t\-/.]*\)"
)
_DOC_ID_LINE_RE = re.compile(
    r"^[ \t]*(?i:document|doc)[ \t]*(?i:id)[ \t]*:?[ \t]*"
    r"[A-Z0-9][A-Z0-9 \t\-/.]*[ \t]*$",
    re.MULTILINE,
)
_DOC_ID_INLINE_RE = re.compile(
    r"\b(?i:document|doc)[ \t]*(?i:id)[ \t]*(?i:is|:)?[ \t]*"
    r"[A-Z0-9][A-Z0-9\-/]*(?:[ \t]+[A-Z0-9][A-Z0-9\-/]*)*"
)

# Safety net for old stored answers whose US address was only partially
# redacted (for example "1847 Willow <PERSON>, <PERSON>, TX 78745"). New text
# is caught by USAddressRecognizer before it reaches the model; this pattern
# cleans up history that predates it.
_RESIDUAL_ADDRESS_RE = re.compile(
    r"\b\d{1,6}[ \t]+[A-Za-z0-9 .'-]+[ \t]*,[ \t]*[A-Z]{2}[ \t]+\d{5}(?:-\d{4})?\b"
)


def _repair_mojibake_match(match: re.Match[str]) -> str:
    raw = match.group(0)
    for encoding in ("latin-1", "cp1252"):
        try:
            return raw.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return raw


def repair_mojibake(text: str) -> str:
    """Undo UTF-8-decoded-as-Latin-1/CP1252 runs and normalise typography."""
    if not text:
        return text
    repaired = _MOJIBAKE_RE.sub(_repair_mojibake_match, text)
    return repaired.translate(_DISPLAY_TRANSLATION)


def strip_internal_identifiers(text: str) -> str:
    """Remove internal document-ID labels without touching ordinary prose."""
    if not text:
        return text
    cleaned = _DOC_ID_PAREN_RE.sub("", text)
    cleaned = _DOC_ID_LINE_RE.sub("", cleaned)
    cleaned = _DOC_ID_INLINE_RE.sub("the document reference", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return _clean(cleaned).strip()


def redact_residual_addresses(text: str) -> str:
    """Replace address-shaped runs that survived an older/partial redaction."""
    if not text:
        return text
    return _RESIDUAL_ADDRESS_RE.sub("the property address", text)


def display_text(
    text: str,
    *,
    clean: bool = True,
    strip_ids: bool = True,
    redact_addresses: bool = True,
) -> str:
    """Prepare persisted/model text for a human reader.

    This is the single display boundary: repair old encoding damage, remove
    internal identifiers, neutralise PII placeholders, and hide residual
    address fragments. The stored/embedded text is never changed.
    """
    if not text:
        return text
    cleaned = repair_mojibake(text)
    if strip_ids:
        cleaned = strip_internal_identifiers(cleaned)
    cleaned = neutralize_placeholders(cleaned, clean=clean)
    if redact_addresses:
        cleaned = redact_residual_addresses(cleaned)
    if clean:
        cleaned = _clean(cleaned)
    return cleaned


def _dedupe_label(match: re.Match[str]) -> str:
    """Turn "Borrower the borrower" into "the borrower", not a bare label."""
    return f"the {match.group(1).lower()}"


def _clean(text: str) -> str:
    """Remove the punctuation/spacing scars a token replacement can leave."""
    text = _MULTISPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _EMPTY_PARENS_RE.sub("", text)
    text = _LABEL_DUPLICATE_RE.sub(_dedupe_label, text)
    text = _HONORIFIC_RE.sub(r"\1", text)
    text = _LABEL_VALUE_RE.sub(r"\1", text)
    # Run after label de-duplication, which can produce "The the borrower".
    text = _DOUBLE_ARTICLE_RE.sub("the", text)
    text = _SENTENCE_START_RE.sub(lambda match: match.group(1) + "The ", text)
    return text


def neutralize_placeholders(text: str, *, clean: bool = True) -> str:
    """Replace every ``<ENTITY_TYPE>`` token with a fluent generic phrase.

    Unknown uppercase tokens become "the redacted value" rather than leaking a
    machine artefact. Text without placeholders is returned unchanged, so this
    is cheap enough to run at every display boundary.

    ``clean=False`` skips the context-aware tidy-up (duplicate labels, sentence
    capitalisation). Streaming uses it because those rewrites can change text
    the client has already received; the final answer is cleaned normally.
    """
    if not text or "<" not in text:
        return text

    replaced = _PLACEHOLDER_RE.sub(
        lambda match: _REPLACEMENTS.get(match.group(1), "the redacted value"),
        text,
    )
    if replaced == text:
        return text
    return _clean(replaced) if clean else replaced
