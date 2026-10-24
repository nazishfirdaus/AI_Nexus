"""Tests for local NLI claim verification (Phase D).

Mostly model-free: ``NliScorer.verify`` is exercised through a stub, so the
suite never loads the ~700 MB cross-encoder. The last test is the exception -
it runs the real model when the weights are cached and skips otherwise.
"""
import json
from types import SimpleNamespace

import pytest

from evaluation import nli


def test_split_claims_splits_on_sentence_boundaries():
    text = "The rate is 6.5%. Payment is due March 1. See page 3."
    assert nli.split_claims(text) == [
        "The rate is 6.5%.",
        "Payment is due March 1.",
        "See page 3.",
    ]


def test_split_claims_drops_degenerate_fragments():
    text = "The rate is 6.5%.\nN/A\n- \nUnknown."
    claims = nli.split_claims(text)
    assert "The rate is 6.5%." in claims
    assert all(len(c.split()) >= nli.MIN_CLAIM_WORDS for c in claims)


def test_split_claims_on_empty_and_whitespace():
    assert nli.split_claims("") == []
    assert nli.split_claims(None) == []


def test_softmax_is_row_stochastic():
    rows = nli._softmax([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    for row in rows:
        assert abs(sum(row) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in row)
    # largest logit wins within its row
    assert rows[0][2] > rows[0][1] > rows[0][0]
    assert abs(rows[1][0] - 1 / 3) < 1e-9


def _stub_scorer(entail_by_claim):
    """A scorer whose verify() returns canned verdicts (no model needed)."""

    class _Stub:
        def verify(self, answer_text, contexts):
            return [
                nli.ClaimVerdict(
                    claim=claim,
                    entailed=entailed,
                    contradiction=False,
                    entailment_prob=0.99 if entailed else 0.01,
                    contradiction_prob=0.01,
                    best_context_index=0,
                )
                for claim, entailed in entail_by_claim.items()
            ]

    return _Stub()


def test_nli_metrics_averages_verdicts():
    scorer = _stub_scorer({"A claim.": True, "Another claim.": False})
    out = nli.nli_metrics("A claim. Another claim.", ["ctx"], scorer)
    assert out == {
        "nli_claim_count": 2,
        "nli_entailed_fraction": 0.5,
        "nli_contradiction_fraction": 0.0,
    }


def test_nli_metrics_returns_empty_when_there_is_nothing_to_verify():
    scorer = _stub_scorer({})
    assert nli.nli_metrics("", ["ctx"], scorer) == {}
    assert nli.nli_metrics("A claim.", [], scorer) == {}


def _stub_nli_model(label2id):
    """Subclass whose ``model`` attribute is a stub (the real one is a property)."""

    class _StubScorer(nli.NliScorer):
        model = SimpleNamespace(
            config=SimpleNamespace(id2label=label2id),
        )

    return _StubScorer(model_name="stub")


def test_scorer_maps_labels_from_model_config():
    scorer = _stub_nli_model({0: "contradiction", 1: "entailment", 2: "neutral"})
    assert scorer._label_indices() == (1, 0)


def test_scorer_rejects_a_non_nli_checkpoint():
    scorer = _stub_nli_model({0: "LABEL_0", 1: "LABEL_1"})
    try:
        scorer._label_indices()
    except RuntimeError as exc:
        assert "not an NLI checkpoint" in str(exc)
    else:  # pragma: no cover - the raise is the point
        raise AssertionError("expected RuntimeError")


def test_verify_marks_entailed_claim_and_leaves_unverifiable_one_neutral():
    """End-to-end through the real model if cached, else through the stub.

    Kept here (rather than the model-free tests above) because it documents
    the scoring contract the eval relies on: a claim present in context comes
    out entailed with high probability, an absent claim stays neutral.
    """
    pytest.importorskip("sentence_transformers")
    try:
        scorer = nli.NliScorer()
        scorer.model  # probe the load; skip if weights are not cached
    except Exception as exc:  # noqa: BLE001 - any load failure means "skip"
        pytest.skip(f"NLI model unavailable: {exc}")

    context = ["The interest rate for this loan is 6.5% per year."]
    verdicts = scorer.verify("The interest rate is 6.5%.", context)
    assert len(verdicts) == 1
    assert verdicts[0].entailed is True
    assert verdicts[0].entailment_prob > 0.9

def test_score_rows_updates_only_complete_rows():
    rows = [
        {"id": "a", "answer_text": "The rate is 6.5%.", "source_snippets": ["ctx"]},
        {"id": "b", "answer_text": "", "source_snippets": ["ctx"]},
        {"id": "c", "answer_text": "The rate is 6.5%.", "source_snippets": []},
    ]
    scorer = _stub_scorer({"The rate is 6.5%.": True})
    assert nli.score_rows(rows, scorer) == 1
    assert rows[0]["nli_claim_count"] == 1
    assert rows[0]["nli_entailed_fraction"] == 1.0
    assert "nli_claim_count" not in rows[1]
    assert "nli_claim_count" not in rows[2]


def test_score_rows_honours_limit_and_is_idempotent():
    rows = [
        {"id": "a", "answer_text": "One.", "source_snippets": ["ctx"]},
        {"id": "b", "answer_text": "Two.", "source_snippets": ["ctx"]},
    ]
    scorer = _stub_scorer({"Only claim.": True})
    assert nli.score_rows(rows, scorer, limit=1) == 1
    assert "nli_claim_count" not in rows[1]
    # a full re-run overwrites rather than duplicates or zeroes existing fields
    assert nli.score_rows(rows, scorer) == 2
    assert nli.score_rows(rows, scorer) == 2
    assert rows[0]["nli_claim_count"] == 1
    assert rows[0]["nli_entailed_fraction"] == 1.0


def test_score_rows_survives_a_scorer_failure():
    class _Boom:
        def verify(self, answer_text, contexts):
            raise RuntimeError("weights missing")

    rows = [{"id": "a", "answer_text": "One.", "source_snippets": ["ctx"]}]
    assert nli.score_rows(rows, _Boom()) == 0
    assert "nli_claim_count" not in rows[0]


def test_summarize_nli_ignores_unscored_rows():
    rows = [
        {
            "id": "a", "nli_claim_count": 2,
            "nli_entailed_fraction": 1.0, "nli_contradiction_fraction": 0.0,
        },
        {
            "id": "b", "nli_claim_count": 1,
            "nli_entailed_fraction": 0.0, "nli_contradiction_fraction": 1.0,
        },
        {"id": "c"},  # never scored (no contexts) - must not dilute the means
    ]
    assert nli.summarize_nli(rows) == {
        "nli_entailed_fraction": 0.5,
        "nli_contradiction_fraction": 0.5,
        "nli_rows_scored": 2,
    }


def test_summarize_nli_on_empty_input():
    assert nli.summarize_nli([]) == {
        "nli_entailed_fraction": None,
        "nli_contradiction_fraction": None,
        "nli_rows_scored": 0,
    }


class _FakeScorer:
    """Stands in for NliScorer in the CLI so tests never load the model."""

    def __init__(self, model_name="fake", threshold=0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.model = object()  # the fail-fast probe in main() sees a "loaded" model
        self._inner = _stub_scorer({"The rate is 6.5%.": True, "Payment is late.": False})

    def verify(self, answer_text, contexts):
        return self._inner.verify(answer_text, contexts)


def _write_results(tmp_path, rows, summary=None):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"summary": summary or {"answer_accuracy": 0.5}, "rows": rows}),
        encoding="utf-8",
    )
    return path


def test_cli_adds_nli_fields_and_summary(tmp_path, monkeypatch):
    path = _write_results(tmp_path, [
        {"id": "a", "answer_text": "The rate is 6.5%. Payment is late.",
         "source_snippets": ["The rate is 6.5%."]},
        {"id": "b", "answer_text": "No contexts here."},
    ])
    monkeypatch.setattr(nli, "NliScorer", _FakeScorer)
    nli.main(["--results", str(path)])

    data = json.loads(path.read_text(encoding="utf-8"))
    scored = data["rows"][0]
    assert scored["nli_claim_count"] == 2
    assert scored["nli_entailed_fraction"] == 0.5
    assert scored["nli_contradiction_fraction"] == 0.0
    assert "nli_claim_count" not in data["rows"][1]  # left untouched, not zeroed
    assert data["summary"]["answer_accuracy"] == 0.5  # pre-existing keys kept
    assert data["summary"]["nli_rows_scored"] == 1
    assert data["summary"]["nli_entailed_fraction"] == 0.5


def test_cli_limit_scores_only_the_first_rows(tmp_path, monkeypatch):
    rows = [
        {"id": "a", "answer_text": "The rate is 6.5%.", "source_snippets": ["ctx"]},
        {"id": "b", "answer_text": "The rate is 6.5%.", "source_snippets": ["ctx"]},
    ]
    path = _write_results(tmp_path, rows)
    monkeypatch.setattr(nli, "NliScorer", _FakeScorer)
    nli.main(["--results", str(path), "--limit", "1"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["nli_rows_scored"] == 1


def test_cli_requires_existing_results(tmp_path):
    with pytest.raises(SystemExit, match="results not found"):
        nli.main(["--results", str(tmp_path / "nope.json")])


def test_cli_rejects_a_results_file_without_the_pass(tmp_path):
    path = _write_results(tmp_path, [])
    path.write_text(json.dumps({"summary": {}, "oracle_rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no rows in"):
        nli.main(["--results", str(path), "--pass", "rows"])
