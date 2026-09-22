"""PII scrubbing with Microsoft Presidio. Runs locally; nothing leaves the machine.

Position in the architecture: the security boundary sits between reranking and the
cloud LLM. Ingestion, embeddings and ChromaDB are all local, so stored chunks keep
their original text; ONLY text that is about to be sent to a provider is scrubbed.

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

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

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
# Scrubber
# ---------------------------------------------------------------------------
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
                    nlp = NlpEngineProvider(nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": self.spacy_model}],
                    }).create_engine()
                    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
                    analyzer.registry.add_recognizer(AadhaarRecognizer())
                    analyzer.registry.add_recognizer(PanRecognizer())
                    self._anonymizer = AnonymizerEngine()
                    self._analyzer = analyzer
                except (Exception, SystemExit) as e:
                    raise ScrubError(f"Could not initialise PII scrubber: {e}") from e
            return self._analyzer, self._anonymizer

    # -- public API ------------------------------------------------------------
    def scrub(self, text: str) -> ScrubbedText:
        return self.scrub_with_report(text)[0]

    def scrub_with_report(self, text: str) -> tuple[ScrubbedText, dict[str, int]]:
        """Return (scrubbed text, {entity_type: count}). Never returns raw text on failure."""
        if isinstance(text, ScrubbedText):
            return text, {}
        if not isinstance(text, str):
            raise ScrubError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            return ScrubbedText(text), {}

        analyzer, anonymizer = self._engines()
        try:
            results = analyzer.analyze(
                text=text,
                language="en",
                entities=self.entities,
                score_threshold=self.score_threshold,
            )
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
def build_llm_inputs(
    query: str,
    chunks: Iterable,
    scrubber: PresidioScrubber | None = None,
) -> tuple[ScrubbedText, ScrubbedText]:
    """Scrub the query and every chunk (RankedChunk-like: .content, .metadata).

    Page labels come from metadata (not scrubbed text), so citations stay intact.
    Returns (scrubbed_query, scrubbed_context) ready for LLMHandler.generate().
    """
    scrubber = scrubber or get_scrubber()
    scrubbed_query = scrubber.scrub(query)

    parts: list[str] = []
    for c in chunks:
        page = (getattr(c, "metadata", None) or {}).get("page_number", "?")
        parts.append(f"[Page {page}]\n{scrubber.scrub(c.content)}")
    return scrubbed_query, ScrubbedText("\n\n".join(parts))