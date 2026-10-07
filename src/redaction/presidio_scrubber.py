"""PII scrubbing with Microsoft Presidio. Runs locally; nothing leaves the machine.

Position in the architecture: the security boundary sits between reranking and the
cloud LLM. Ingestion, embeddings and ChromaDB are all local, so stored chunks keep
their original text; ONLY text that is about to be sent to a provider is scrubbed.
Aggregator._scrub applies the same scrubber to the answer text and the page
citation snippets, which are built from raw chunks and would otherwise be the one
place unredacted PII reaches the screen.

What is protected: names (Presidio PERSON plus honorific-matched Indian names),
addresses, phone numbers, email, PAN, Aadhaar (Verhoeff-checked), voter ID,
driving licence, passport, TAN, and Indian bank and loan account numbers, IFSC,
MICR, GSTIN, UPI/VPA, cheque, policy and application numbers. Dates and bare place
names are deliberately preserved - see config/settings.py for the full list and the
reasoning behind each exclusion.

Guarantees:
  * Fail closed: if scrubbing errors, ScrubError is raised. Raw text is never returned.
  * Output type is ScrubbedText, the only type LLMHandler accepts.
  * Never logs document text, only entity-type counts.
  * Same scrubber (and one spaCy model in RAM) is used regardless of LLM provider.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence

# Defer heavy Presidio/transformers/torch imports to avoid 30s startup
import sys as _sys
import types as _types

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presidio_analyzer import (
        AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

# Eager base classes needed at module scope (all the custom recognizers below)
try:
    from presidio_analyzer import (  # type: ignore
        EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult,
    )
except Exception:
    EntityRecognizer = object  # type: ignore
    Pattern = object  # type: ignore
    PatternRecognizer = object  # type: ignore
    RecognizerResult = object  # type: ignore

# Stub torch/transformers before full Presidio import to avoid pulling heavy deps
_stubs: list[str] = []
for _m in ("torch", "transformers"):
    if _m not in _sys.modules:
        stub = _types.ModuleType(_m)
        stub.__spec__ = None
        stub.__getattr__ = lambda name, _mname=_m: type(name, (), {})  # type: ignore[misc]
        _sys.modules[_m] = stub
        _stubs.append(_m)

from config import settings
from src.generation.llm import ScrubbedText

logger = logging.getLogger(__name__)

# Presidio's own loggers write fragments of the analysed text at DEBUG level
# (e.g. "Context list is: rahul.sharma@example.com ..."). Never let that reach any log sink.
for _name in ("presidio-analyzer", "presidio-anonymizer"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class ScrubError(RuntimeError):
    """Raised when text could not be scrubbed. Callers must NOT send the raw text."""


# ---------------------------------------------------------------------------
# Verhoeff checksum (used by Aadhaar)
# ---------------------------------------------------------------------------
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


# ---------------------------------------------------------------------------
# Custom recognizers
# ---------------------------------------------------------------------------
class AadhaarRecognizer(PatternRecognizer):
    """12-digit Aadhaar (first digit 2-9), optionally grouped 4-4-4 with space/hyphen.

    * Verhoeff-valid   -> score 1.0 (always redacted)
    * Not valid        -> stays at 0.3, below the threshold, unless an Aadhaar-related
                          word is nearby (context boost) - so typos next to the word
                          "Aadhaar" are still redacted, but random 12-digit numbers are not.
    """

    PATTERNS = [Pattern("aadhaar_12_digit", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b", 0.3)]
    CONTEXT = ["aadhaar", "aadhar", "uidai", "uid", "unique identification"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="AADHAAR_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str):
        digits = re.sub(r"\D", "", pattern_text)
        return True if len(digits) == 12 and verhoeff_valid(digits) else None


class PanRecognizer(PatternRecognizer):
    """Indian PAN: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F)."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="PAN_NUMBER",
            patterns=[Pattern("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.6)],
            context=["pan", "permanent account number", "income tax"],
            supported_language="en",
        )


# ---------------------------------------------------------------------------
# Indian financial + identity identifiers
# ---------------------------------------------------------------------------
# How the number-like identifiers are handled, and why.
#
# An Indian bank or loan account number is 8-20 bare digits with no checksum, so a
# bare "length heuristic" cannot tell it apart from a loan amount, a date or a phone
# number - and a loan QA system needs those to stay readable. Every one of them is
# therefore LABEL-ANCHORED: the regex requires the label ("A/c no.", "Loan A/c",
# "Cheque no.") to sit immediately next to the value, and only the value itself is
# redacted. A digit run with no label next to it is left alone.
#
# The digit patterns use `\d(?:[ \-]?\d){n}` rather than `[\d -]{n}` so a run can
# never span a line break, and every pattern is a linear alternation: no nested
# quantifiers, so nothing can backtrack catastrophically over document text.
_LABEL_NO = r"(?:no\.?|number|num\.?|#)"
# 8-20 digits, each optionally preceded by a single space or hyphen (bank grouping).
_NUM = r"(\d(?:[ \-]?\d){7,19})"
# 6-20 alphanumeric identifier characters, slashes and hyphens allowed.
_ALNUM_ID = r"([A-Z0-9][A-Z0-9\-/]{5,19})"
_ACCOUNT_WORD = r"(?:acc(?:ount)?|acct)"
_AC_ABBREV = r"a\s*/?\s*c"
# handle@bank, where both halves must end on an alphanumeric. Without the trailing
# character class the "." that ends a sentence is swallowed into the handle.
_UPI_HANDLE = r"[A-Za-z0-9._\-]{2,63}[A-Za-z0-9]@[A-Za-z0-9._\-]{2,63}[A-Za-z0-9]"


class _GroupRecognizer(PatternRecognizer):
    """Redacts only the regex's first capture group, so the label survives.

    Presidio's stock PatternRecognizer redacts the entire match, so a label-anchored
    pattern such as "A/c No. 30123456789" would also swallow "A/c No.". These
    recognizers emit one result per match spanning the captured value only, which
    keeps the label readable in answers and page citations.
    """

    def analyze(self, text, entities, nlp_artifacts=None, regex_flags=None):
        flags = regex_flags if regex_flags else self.global_regex_flags
        wanted = set(entities or ())
        results: list = []
        for pattern in self.patterns:
            if wanted and self.supported_entities[0] not in wanted:
                continue
            for match in re.finditer(pattern.regex, text, flags=flags):
                start, end = match.span(1)
                if start < 0 or end <= start:
                    continue
                results.append(RecognizerResult(
                    entity_type=self.supported_entities[0],
                    start=start,
                    end=end,
                    score=pattern.score,
                    analysis_explanation=self.build_regex_explanation(
                        self.name, pattern.name, pattern.regex, pattern.score, None, flags,
                    ),
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                ))
        return EntityRecognizer.remove_duplicates(results)


class IndianPersonRecognizer(PatternRecognizer):
    """Names carrying an explicit honorific: "Mr. Rahul Sharma", "Smt. Lakshmi Rao".

    spaCy en_core_web_sm has weak PERSON recall on Indian names, so this catches the
    cases with a strong honorific marker. Names without one still rely on Presidio's
    own PERSON recognizer wherever it happens to fire.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_PERSON_NAME",
            patterns=[Pattern(
                "in_honorific_name",
                r"\b(?:Mrs|Mr|Ms|Dr|Shri|Smt|Prof)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\b",
                0.5,
            )],
            context=["name", "borrower", "applicant", "holder", "son", "daughter", "wife"],
            supported_language="en",
        )


class IndianAddressRecognizer(PatternRecognizer):
    """Pinpoint postal addresses, deliberately NOT bare place names.

    Presidio's LOCATION entity stays disabled (see config/settings.py) because
    loan and property answers depend on the city. These patterns instead target the
    precise parts of an address - the care-of line, the house/flat block, the street
    or colony phrase and the "City - 226010" tail - while a bare city name survives.

    Each pattern is deliberately short. An earlier version let the care-of line run
    60 characters past the name, and the house-number line three comma-separated
    segments long, either of which quietly swallowed the PAN and part of the Aadhaar
    label that followed. Every "run to the end of the thing" clause is therefore
    bounded, stops at a comma, and stops at a full stop, so redaction cannot bleed
    into the next sentence.
    """

    _ROADS = (
        r"(?:streets?|roads?|lanes?|avenues?|colony|nagar|sector|villas?|towers?|"
        r"apartments?|locality|estate|bhawan|society|layout|chawl|marg)"
    )

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_ADDRESS",
            patterns=[
                # "S/o Ram Nath Sharma" - the care-of line is a name, so redact just that.
                Pattern("in_care_of_line",
                        r"\b(?:s/o|d/o|w/o|c/o)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\b",
                        0.5),
                # "H.No. 12, Gomti Nagar, Lucknow" - house number plus at most three
                # short segments, none of which may contain a sentence boundary.
                Pattern("in_house_number",
                        r"\b(?:H|F|SF|FF)\.?\s*No\.?\s*[-\s]?\s*\d+[A-Za-z]?"
                        r"(?:\s*,\s*[^\n,.;]{2,30}){0,3}", 0.5),
                Pattern("in_street_phrase", rf"\b{self._ROADS}\b[^\n,.]{{0,30}}", 0.5),
                # ", Lucknow - 226010" - anchored on a separator or line start so that
                # "Instalment 12345678 - 450000" is not read as a city and a PIN code.
                Pattern("in_city_pincode_tail",
                        r"(?:^|[:,]|\n)\s*[A-Za-z][A-Za-z.]{1,20}"
                        r"(?:\s+[A-Za-z][A-Za-z.]{1,20}){0,2}\s*[-\u2013]\s*[1-9]\d{5}\b", 0.5),
            ],
            context=["address", "residing", "resides", "permanent", "present",
                     "belongs", "property", "premises"],
            supported_language="en",
        )


class IndianPincodeRecognizer(PatternRecognizer):
    """Six-digit Indian PIN code.

    Base score sits below PII_SCORE_THRESHOLD, so a bare six-digit run is never
    redacted on its own - it needs a neighbouring address word ("PIN code",
    "district", "address") to cross the threshold. That keeps a comma-less amount
    such as "450000" intact.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_PINCODE",
            patterns=[Pattern("in_pincode", r"\b[1-9]\d{5}\b", 0.35)],
            context=["pincode", "pin code", "pin", "postal", "zip", "district",
                     "address", "village", "taluka", "post office"],
            supported_language="en",
        )


class IndianAccountNumberRecognizer(_GroupRecognizer):
    """Indian bank account numbers (8-20 digits), only ever redacted next to a label.

    "A/c" is an unambiguous abbreviation so the label may be omitted; the spelled-out
    "account"/"acct" form requires an explicit "no"/"number" so that prose such as
    "account of 4500000 rupees" is not mistaken for an account number.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_ACCOUNT_NO",
            patterns=[
                Pattern("account_abbrev_digits",
                        rf"\b{_AC_ABBREV}\.?\s*(?:no\.?|number|num\.?)?\s*(?:is|of)?\s*[:\-]?\s*{_NUM}",
                        0.75),
                Pattern("account_word_label_digits",
                        rf"\b{_AC_ABBREV}\s+{_ACCOUNT_WORD}\.?\s*{_LABEL_NO}\s*(?:is|of)?\s*"
                        rf"[:\-]?\s*{_NUM}", 0.75),
                Pattern("account_word_label_digits",
                        rf"\b{_ACCOUNT_WORD}\.?\s*{_LABEL_NO}\s*(?:is|of)?\s*[:\-]?\s*{_NUM}",
                        0.75),
                Pattern("account_kind_digits",
                        rf"\b(?:savings?|current|beneficiary|primary|salary|joint)\s+"
                        rf"{_ACCOUNT_WORD}\.?\s*{_LABEL_NO}\s*(?:is|of)?\s*[:\-]?\s*{_NUM}",
                        0.75),
            ],
            context=["account", "a/c", "bank", "savings", "current", "beneficiary",
                     "salary", "neft", "rtgs", "imps"],
            supported_language="en",
        )


class IndianLoanNumberRecognizer(_GroupRecognizer):
    """Loan / loan-account numbers, disbursal and EMI account numbers.

    Scores above IndianAccountNumberRecognizer on purpose: "Loan Account No. X" is
    matched by both, and Presidio keeps the higher-scoring result on an identical
    span, so the placeholder reliably says <IN_LOAN_NO>.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_LOAN_NO",
            patterns=[
                Pattern("loan_account_digits",
                        rf"\bloan\s*(?:{_AC_ABBREV}|{_ACCOUNT_WORD})\.?\s*(?:no\.?|number|num\.?|#)?"
                        rf"\s*(?:is|of)?\s*[:\-]?\s*{_NUM}", 0.8),
                Pattern("loan_label_id",
                        rf"\bloan\s*{_LABEL_NO}\s*(?:is|of)?\s*[:\-]?\s*{_ALNUM_ID}", 0.8),
                Pattern("loan_disbursal_account",
                        rf"\b(?:disbursal|sanction|emi)\s*(?:{_AC_ABBREV}|{_ACCOUNT_WORD})"
                        rf"\.?\s*(?:no\.?|number|num\.?|#)?\s*(?:is|of)?\s*[:\-]?\s*{_NUM}", 0.8),
            ],
            context=["loan", "loan account", "disbursal", "emi", "sanction", "instalment"],
            supported_language="en",
        )


class IndianIfscRecognizer(PatternRecognizer):
    """IFSC: 4 letters, a literal 0, then 6 alphanumerics (e.g. SBIN0001234)."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IFSC_CODE",
            patterns=[Pattern("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.6)],
            context=["ifsc", "bank", "branch", "neft", "rtgs", "imps", "upi", "micr"],
            supported_language="en",
        )


class IndianMicrRecognizer(PatternRecognizer):
    """MICR routing number: 6 digits.

    Same shape as a PIN code, so it stays below the threshold and needs a bank or
    branch word nearby - exactly like IndianPincodeRecognizer.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="MICR_CODE",
            patterns=[Pattern("micr", r"\b[0-9]{6}\b", 0.35)],
            context=["micr", "branch code", "ifsc", "cheque clearing", "routing"],
            supported_language="en",
        )


class IndianGstinRecognizer(PatternRecognizer):
    """GSTIN: 15 characters - 2 state digits, a 10-char PAN, entity code, Z, checksum."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="GSTIN",
            patterns=[Pattern("gstin", r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z\d]\b", 0.6)],
            context=["gstin", "gst", "goods and services", "tax"],
            supported_language="en",
        )


class UpiIdRecognizer(_GroupRecognizer):
    """UPI / VPA handles such as rahul@okhdfcbank.

    Anchored to a "UPI id" / "VPA" / "Pay to" label on purpose. A bare handle@bank
    string is indistinguishable from an email address, and matching it here would make
    UPI and EMAIL_ADDRESS overlap on every address in the document.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="UPI_ID",
            patterns=[
                Pattern("upi_label_handle",
                        rf"\bupi\b\.?\s*(?:id|handle)?\.?\s*[:\-]?\s*({_UPI_HANDLE})", 0.75),
                Pattern("vpa_label_handle",
                        rf"\bvpa\b\.?\s*(?:id|address)?\.?\s*[:\-]?\s*({_UPI_HANDLE})", 0.75),
                Pattern("pay_to_handle",
                        rf"\bpay\s*to\b\s*[:\-]?\s*({_UPI_HANDLE})", 0.75),
            ],
            context=["upi", "vpa", "pay", "transfer", "handle", "qr", "imps"],
            supported_language="en",
        )


class ChequeNumberRecognizer(_GroupRecognizer):
    """Cheque / demand-draft numbers: 6-8 digits next to a cheque label."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CHEQUE_NO",
            patterns=[Pattern(
                "cheque_no",
                r"\b(?:cheque|cheq|chq|demand\s+draft|draft)\.?\s*(?:no\.?|number|#)?"
                r"\s*(?:is|dated)?\s*[:\-]?\s*(\d{6,8})\b", 0.7,
            )],
            context=["cheque", "chq", "draft", "payee", "clearing"],
            supported_language="en",
        )


class PolicyNumberRecognizer(_GroupRecognizer):
    """Insurance / LIC / policy numbers."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="POLICY_NO",
            patterns=[Pattern(
                "policy_no",
                r"\b(?:policy|lic|licence|license|endowment|motor|insurance)\.?\s*"
                rf"{_LABEL_NO}\s*(?:is|of)?\s*[:\-]?\s*{_ALNUM_ID}", 0.7,
            )],
            context=["policy", "lic", "licence", "license", "premium", "insured"],
            supported_language="en",
        )


class VoterIdRecognizer(PatternRecognizer):
    """EPIC / Voter ID: 3 letters + 7 digits, or the older 1 letter + 7 digits + 1 letter."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="VOTER_ID",
            patterns=[Pattern(
                "epic",
                r"\b(?:[A-Z]{3}\d{7}|[A-Z]\d{7}[A-Z])\b", 0.5,
            )],
            context=["voter", "voter's", "epic", "electoral", "electroll", "booth"],
            supported_language="en",
        )


class TanRecognizer(PatternRecognizer):
    """TAN: 4 letters, 4 digits, 1 letter, 1 digit. Distinguished from PAN by the tail digit."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="TAN_NUMBER",
            patterns=[Pattern("tan", r"\b[A-Z]{4}\d{4}[A-Z]\d\b", 0.5)],
            context=["tan", "tax account"],
            supported_language="en",
        )


class DrivingLicenceRecognizer(PatternRecognizer):
    """Indian driving licence: 2 state letters + 2 RTO digits + 11 digits."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="DL_NO",
            patterns=[Pattern("dl_no", r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", 0.5)],
            context=["driving", "licence", "license", "dl", "permit", "rto"],
            supported_language="en",
        )


class PassportNumberRecognizer(PatternRecognizer):
    """Passport number: 1-2 letters + 7 digits.

    Below the threshold on its own - this shape is far too generic to trust without a
    nearby "passport" word.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="PASSPORT_NO",
            patterns=[Pattern("passport_no", r"\b[A-Z]{1,2}[-\s]?\d{7}\b", 0.35)],
            context=["passport", "passport no"],
            supported_language="en",
        )


class ReferenceIdRecognizer(_GroupRecognizer):
    """Application / customer / proposal / reference identifiers used on loan forms."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="APPLICATION_ID",
            patterns=[Pattern(
                "reference_id",
                rf"\b(?:application|applicant|customer|proposal|reference|ref|enquiry|"
                rf"lead|file)\.?\s*{_LABEL_NO}\s*(?:is|of)?\s*[:\-]?\s*{_ALNUM_ID}", 0.7,
            )],
            context=["application", "applicant", "customer", "proposal", "reference", "id"],
            supported_language="en",
        )


# Registration order matters only in that a higher score wins on an identical span.
CUSTOM_RECOGNIZERS: tuple = (
    AadhaarRecognizer,
    PanRecognizer,
    IndianPersonRecognizer,
    IndianAddressRecognizer,
    IndianPincodeRecognizer,
    IndianAccountNumberRecognizer,
    IndianLoanNumberRecognizer,
    IndianIfscRecognizer,
    IndianMicrRecognizer,
    IndianGstinRecognizer,
    UpiIdRecognizer,
    ChequeNumberRecognizer,
    PolicyNumberRecognizer,
    VoterIdRecognizer,
    TanRecognizer,
    DrivingLicenceRecognizer,
    PassportNumberRecognizer,
    ReferenceIdRecognizer,
)


# ---------------------------------------------------------------------------
# Scrubber
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _specific_entities() -> frozenset:
    """Entity types produced by our own recognizers rather than by generic NER.

    Instantiating a PatternRecognizer is cheap - it loads no model - so this stays
    a lazy one-off rather than threading the set through the analyzer.
    """
    return frozenset(r.supported_entities[0] for r in (cls() for cls in CUSTOM_RECOGNIZERS))


def _drop_generic_spans_around_specific(results: list, specific: frozenset) -> list:
    """Let one of our recognizers win over a generic one covering the same text.

    spaCy happily reports "Passport No. Z4578901" or "Mr. Rahul Sharma" as a single
    PERSON blob scoring ~0.85. Presidio's own conflict resolution keeps the higher
    score, so the sensitive value is still redacted - but the placeholder then reads
    <PERSON>, which tells the model and the user nothing and invites the model to
    treat a passport number as a person's name.

    So: a GENERIC result (anything outside CUSTOM_RECOGNIZERS) is dropped whenever one
    of our own recognizers covers a span inside it, equal spans included. Results from
    our own recognizers are never dropped this way, so IN_ADDRESS still survives the
    IN_PINCODE spaCy finds inside it. Nothing leaks either way - the surviving
    specific span still covers the value.
    """
    spans = [(r.start, r.end, r.entity_type) for r in results]
    keep = []
    for i, result in enumerate(results):
        start, end, entity_type = spans[i]
        if entity_type in specific:
            keep.append(result)
            continue
        shadowed = any(
            other_entity in specific
            and other_entity != entity_type
            and other_start >= start
            and other_end <= end
            for other_start, other_end, other_entity in spans
        )
        if not shadowed:
            keep.append(result)
    return keep


def _shift_into(results: list, offset: int) -> list:
    """Rebase results from a prefixed analysis string back onto the text itself.

    Results that sit entirely inside the carried-over prefix are dropped - they
    belong to text this call is not returning. Results straddling the boundary are
    clamped so the in-text part is still redacted.
    """
    if not offset:
        return results
    rebased = []
    for r in results:
        if r.end <= offset:
            continue
        rebased.append(RecognizerResult(
            entity_type=r.entity_type,
            start=max(r.start - offset, 0),
            end=r.end - offset,
            score=r.score,
        ))
    return rebased


class PresidioScrubber:
    def __init__(
        self,
        entities: Sequence[str] | None = None,
        score_threshold: float | None = None,
        spacy_model: str | None = None,
    ) -> None:
        self.entities = list(entities or settings.PII_ENTITIES)
        self.score_threshold = (
            settings.PII_SCORE_THRESHOLD if score_threshold is None else score_threshold
        )
        self.spacy_model = spacy_model or settings.SPACY_MODEL
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None
        self._lock = threading.Lock()

    # -- lazy engines (loading spaCy is slow and memory-heavy: do it once) ----
    def _engines(self) -> tuple[AnalyzerEngine, AnonymizerEngine]:
        with self._lock:
            if self._analyzer is None:
                # Presidio would otherwise try to download a missing model and then
                # call sys.exit(), which would kill the whole Streamlit process.
                import spacy.util

                if not spacy.util.is_package(self.spacy_model):
                    raise ScrubError(
                        f"spaCy model '{self.spacy_model}' is not installed. "
                        f"Run: python -m spacy download {self.spacy_model}"
                    )
                try:
                    # Import heavy deps inside lock, after stubbing
                    from presidio_analyzer import AnalyzerEngine
                    from presidio_analyzer.nlp_engine import NlpEngineProvider
                    from presidio_anonymizer import AnonymizerEngine

                    nlp = NlpEngineProvider(nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": self.spacy_model}],
                    }).create_engine()
                    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
                    for recognizer_cls in CUSTOM_RECOGNIZERS:
                        analyzer.registry.add_recognizer(recognizer_cls())
                    self._anonymizer = AnonymizerEngine()
                    self._analyzer = analyzer
                except (Exception, SystemExit) as e:
                    raise ScrubError(f"Could not initialise PII scrubber: {e}") from e
                finally:
                    # Remove stubs so sentence_transformers/torch load real modules later
                    for _m in _stubs:
                        _sys.modules.pop(_m, None)
                    _stubs.clear()
            return self._analyzer, self._anonymizer

    # -- public API ------------------------------------------------------------
    def scrub(self, text: str, context_prefix: str = "") -> ScrubbedText:
        return self.scrub_with_report(text, context_prefix)[0]

    def scrub_with_report(
        self, text: str, context_prefix: str = ""
    ) -> tuple[ScrubbedText, dict[str, int]]:
        """Return (scrubbed text, {entity_type: count}). Never returns raw text on failure.

        `context_prefix` is text that precedes `text` in the document but is not part
        of the return value - typically the tail of the previous chunk. It is prepended
        for analysis only, so that a label in one chunk can still be recognised as
        belonging to a value that the chunker put in the next one.
        """
        if isinstance(text, ScrubbedText):
            return text, {}
        if not isinstance(text, str):
            raise ScrubError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            return ScrubbedText(text), {}

        analyzer, anonymizer = self._engines()
        offset = len(context_prefix)
        try:
            results = analyzer.analyze(
                text=context_prefix + text if offset else text,
                language="en",
                entities=self.entities,
                score_threshold=self.score_threshold,
            )
            results = _drop_generic_spans_around_specific(results, _specific_entities())
            results = _shift_into(results, offset)
            cleaned = anonymizer.anonymize(text=text, analyzer_results=results).text
        except Exception as e:  # fail closed
            raise ScrubError(f"PII scrubbing failed: {type(e).__name__}") from e

        report = dict(Counter(r.entity_type for r in results))
        if report:
            logger.info("PII redacted: %s", report)  # counts only, never text
        return ScrubbedText(cleaned), report


@lru_cache(maxsize=1)
def get_scrubber() -> PresidioScrubber:
    """One shared scrubber, so the spaCy model is loaded once per process."""
    return PresidioScrubber()


# ---------------------------------------------------------------------------
# Helper for the pipeline: everything that goes to a cloud LLM passes through here
# ---------------------------------------------------------------------------
def scrub_chunks(
    chunks: Sequence,
    scrubber: PresidioScrubber | None = None,
    initial_context: str = "",
) -> list[tuple]:
    """Scrub chunks in order, carrying the tail of the previous one as context.

    Returns [(chunk, ScrubbedText), ...] aligned with `chunks`. Chunk boundaries
    routinely fall between a label and the value it introduces - e.g. "A/c No" ending
    one chunk and "30123456789 ..." starting the next - where a label-anchored
    pattern would otherwise have nothing to match. Carrying the tail forward makes
    the label visible to the next chunk's patterns.
    """
    scrubber = scrubber or get_scrubber()
    carry = initial_context[-settings.PII_CONTEXT_CARRY_CHARS:] if initial_context else ""
    out = []
    for c in chunks:
        out.append((c, scrubber.scrub(c.content, context_prefix=carry)))
        carry = c.content[-settings.PII_CONTEXT_CARRY_CHARS:]
    return out


def build_llm_inputs(
    query: str,
    chunks: Iterable,
    scrubber: PresidioScrubber | None = None,
) -> tuple[ScrubbedText, ScrubbedText, dict]:
    """Scrub the query and every chunk (RankedChunk-like: .content, .metadata).

    Page labels come from metadata (not scrubbed text), so citations stay intact.
    With several documents in play each block is prefixed with the source
    filename, letting the model attribute facts to the right document.

    Returns (scrubbed_query, scrubbed_context, scrubbed_by_page) ready for
    LLMHandler.generate(). The third element maps (document_id, page number) to
    that page's already scrubbed text: the aggregator builds its citation
    snippets from it, so a snippet shows exactly what the model was given and
    can never re-expose raw PII.
    """
    scrubber = scrubber or get_scrubber()
    scrubbed_query = scrubber.scrub(query)

    parts: list[str] = []
    by_page: dict = {}
    for c, scrubbed in scrub_chunks(list(chunks), scrubber, initial_context=query):
        meta = getattr(c, "metadata", None) or {}
        page = meta.get("page_number", "?")
        filename = meta.get("filename") or meta.get("document_id")
        label = f"[Doc: {filename} | Page {page}]" if filename else f"[Page {page}]"
        try:  # match the aggregator's (document_id, int(page)) citation key
            page_key: object = int(page)
        except (TypeError, ValueError):
            page_key = page
        key = (str(meta.get("document_id") or ""), page_key)
        parts.append(f"{label}\n{scrubbed}")
        by_page.setdefault(key, []).append(str(scrubbed))
    return (
        scrubbed_query,
        ScrubbedText("\n\n".join(parts)),
        {key: "\n".join(texts) for key, texts in by_page.items()},
    )