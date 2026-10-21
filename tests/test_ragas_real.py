"""Tests for the RAGAS LLM-judge runner (Phase C) and its import shim.

These are offline: nothing here calls a judge or loads an embedding model.
The two tests that import ``ragas`` go through the same compat hook the runner
uses and only *construct* metric objects.
"""
import importlib.machinery
import json
import sys
import types

import pytest

from evaluation import ragas_compat
from evaluation import ragas_real


def test_dataset_specs_default_matches_run_eval():
    # the runner keeps its own copy to avoid importing the heavyweight
    # pipeline package; fail loudly if the two ever drift apart
    from evaluation.run_eval import DEFAULT_DATASETS

    assert ragas_real._DEFAULT_SPECS == DEFAULT_DATASETS


def test_dataset_paths_default_to_both_golden_datasets():
    from evaluation.run_eval import DEFAULT_DATASETS

    assert ragas_real._dataset_paths(None) == list(DEFAULT_DATASETS)


def test_dataset_paths_strip_an_optional_document_source():
    assert ragas_real._dataset_paths(["some/other.json=Doc.pdf"]) == ["some/other.json"]
    assert ragas_real._dataset_paths(["some/other.json"]) == ["some/other.json"]


def test_load_references_prefixes_ids_and_skips_unexpected(tmp_path):
    path = tmp_path / "golden_dataset_01.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "q001", "expected_answer": "42"},
                    {"id": "q002"},  # no expected_answer -> not a reference
                ]
            }
        ),
        encoding="utf-8",
    )
    assert ragas_real._load_references([str(path)]) == {"d01_q001": "42"}


def test_load_references_errors_on_missing_dataset(tmp_path):
    with pytest.raises(SystemExit, match="dataset not found"):
        ragas_real._load_references([str(tmp_path / "golden_dataset_01.json")])


def test_build_rows_renames_to_ragas_single_turn_schema():
    rows = [
        {
            "id": "d01_q001",
            "question": "What is the payment due date?",
            "answer_text": "2024-03-01.",
            "source_snippets": ["page 3 text", "page 4 text"],
            "category": "direct_fact_lookup",
            "provider": "groq",
        }
    ]
    out = ragas_real._build_rows(rows, {"d01_q001": "2024-03-01"})
    assert out == [
        {
            "user_input": "What is the payment due date?",
            "response": "2024-03-01.",
            "retrieved_contexts": ["page 3 text", "page 4 text"],
            "reference": "2024-03-01",
            "_id": "d01_q001",
            "category": "direct_fact_lookup",
            "provider": "groq",
        }
    ]


def test_build_rows_skips_rows_without_reference_or_answer():
    base = {
        "id": "d01_q001",
        "question": "Q?",
        "answer_text": "A.",
        "source_snippets": ["ctx"],
    }
    no_reference = dict(base)
    no_answer = dict(base, id="d01_q002", answer_text="")
    no_contexts = dict(base, id="d01_q003", source_snippets=[])
    assert ragas_real._build_rows(
        [no_reference, no_answer, no_contexts], {"d01_q001": "expected"}
    ) == [
        {
            "user_input": "Q?",
            "response": "A.",
            "retrieved_contexts": ["ctx"],
            "reference": "expected",
            "_id": "d01_q001",
            "category": None,
            "provider": None,
        }
    ]


def test_build_rows_tolerates_a_single_string_context():
    rows = [
        {
            "id": "d01_q001",
            "question": "Q?",
            "answer_text": "A.",
            "source_snippets": "one big chunk",
        }
    ]
    out = ragas_real._build_rows(rows, {"d01_q001": "expected"})
    assert out[0]["retrieved_contexts"] == ["one big chunk"]


def test_finite_scrubs_failed_and_non_numeric_scores():
    assert ragas_real._finite(0.5) == 0.5
    assert ragas_real._finite(1) == 1.0
    assert ragas_real._finite(float("nan")) is None
    assert ragas_real._finite(float("inf")) is None
    assert ragas_real._finite(True) is None
    assert ragas_real._finite("0.9") is None
    assert ragas_real._finite(None) is None


def test_summarize_skips_failed_jobs_and_maps_result_columns():
    scored = [
        {
            "category": "direct",
            "provider": "groq",
            "faithfulness": 0.5,
            "nv_response_groundedness": float("nan"),
        },
        {
            "category": "direct",
            "provider": "nvidia",
            "faithfulness": float("nan"),
            "nv_response_groundedness": 1.0,
        },
    ]
    pairs = [
        ("faithfulness", "faithfulness"),
        ("response_groundedness", "nv_response_groundedness"),
    ]
    summary = ragas_real._summarize(scored, pairs)
    assert summary["items"] == 2
    assert summary["columns"] == {
        "faithfulness": "faithfulness",
        "response_groundedness": "nv_response_groundedness",
    }
    # NaN must not poison the mean: only the surviving row is averaged
    assert summary["faithfulness"] == 0.5
    assert summary["response_groundedness"] == 1.0
    groq = summary["by_provider"]["groq"]
    assert groq["count"] == 1
    assert groq["faithfulness"] == 0.5
    # groq's groundedness job failed, so that column is absent, not zero
    assert "response_groundedness" not in groq


# --- import shim -----------------------------------------------------------------


def test_finder_stubs_only_the_genuinely_missing_submodule():
    finder = ragas_compat._MissingLangchainCommunityFinder()
    spec = finder.find_spec("langchain_community.chat_models.vertexai")
    assert spec is not None
    assert isinstance(spec.loader, ragas_compat._StubModuleLoader)
    assert spec.submodule_search_locations is not None  # stubbed as a package


def test_finder_defers_to_the_real_finder_and_ignores_other_packages():
    finder = ragas_compat._MissingLangchainCommunityFinder()
    # top-level package (prefix includes the trailing dot) is never ours
    assert finder.find_spec("langchain_community") is None
    # anything outside the package is never ours
    assert finder.find_spec("json.decoder") is None
    # a submodule that really ships gets delegated to the path finder
    fullname = "langchain_community.tools"
    real_exists = importlib.machinery.PathFinder.find_spec(fullname, None) is not None
    if real_exists:
        assert finder.find_spec(fullname) is None


def test_stub_loader_synthesises_classes_but_rejects_dunders():
    module = types.ModuleType("stubbed.thing")
    module.__getattr__ = ragas_compat._synthesize
    cls = module.ChatVertexAI
    assert isinstance(cls, type)
    assert cls.__name__ == "ChatVertexAI"
    with pytest.raises(AttributeError):
        module.__wrapped__


def test_install_hook_is_idempotent():
    def installed():
        return [
            f
            for f in sys.meta_path
            if isinstance(f, ragas_compat._MissingLangchainCommunityFinder)
        ]

    before = len(installed())
    ragas_compat.install_hook()  # returns False when already present
    ragas_compat.install_hook()
    assert len(installed()) == max(before, 1)


def test_resolve_metrics_reports_result_column_names():
    ragas_compat.ensure_ragas_importable()
    metrics = ragas_real._resolve_metrics(["response_groundedness"], judge=None)
    # the metric names itself "nv_response_groundedness"; the report keeps the
    # friendly name and records the mapping in summary["columns"]
    assert [m.name for m in metrics] == ["nv_response_groundedness"]


def test_resolve_metrics_demands_embeddings_when_a_metric_needs_them():
    ragas_compat.ensure_ragas_importable()
    with pytest.raises(SystemExit, match="needs an embedding model"):
        ragas_real._resolve_metrics(["answer_relevancy"], judge=None, embeddings=None)


# --- resume / multi-day merge ----------------------------------------------------


def _scored(row_id, **cols):
    return {"_id": row_id, "category": "direct", "provider": "groq", **cols}


def test_load_scored_returns_empty_for_missing_or_corrupt_files(tmp_path):
    assert ragas_real._load_scored(tmp_path / "nope.json") == ([], {})
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert ragas_real._load_scored(bad) == ([], {})
    notdict = tmp_path / "list.json"
    notdict.write_text("[1, 2, 3]", encoding="utf-8")
    assert ragas_real._load_scored(notdict) == ([], {})


def test_load_scored_reads_a_prior_report(tmp_path):
    path = tmp_path / "ragas_real.json"
    path.write_text(
        json.dumps({
            "summary": {"items": 1, "columns": {"faithfulness": "faithfulness"},
                        "metrics": ["faithfulness"]},
            "rows": [_scored("d01_q001", faithfulness=0.75)],
        }),
        encoding="utf-8",
    )
    rows, summary = ragas_real._load_scored(path)
    assert rows[0]["_id"] == "d01_q001"
    assert summary["metrics"] == ["faithfulness"]


def test_unscored_rows_skips_rows_that_already_have_every_column():
    pairs = [("faithfulness", "faithfulness")]
    prior = [_scored("a", faithfulness=0.5), _scored("b", faithfulness=None)]
    prepared = [{"_id": "a"}, {"_id": "b"}, {"_id": "c"}]
    pending = ragas_real._unscored_rows(prepared, prior, pairs)
    # a is done; b failed (None) and c is new, so both are re-sent
    assert [r["_id"] for r in pending] == ["b", "c"]


def test_unscored_rows_treats_a_new_metric_as_not_done():
    # day 1 held only faithfulness; asking for answer_relevancy must re-score
    prior = [_scored("a", faithfulness=0.5)]
    pairs = [("answer_relevancy", "answer_relevancy")]
    assert [r["_id"] for r in ragas_real._unscored_rows([{"_id": "a"}], prior, pairs)] == ["a"]


def test_unscored_rows_returns_everything_without_prior_data():
    prepared = [{"_id": "a"}, {"_id": "b"}]
    assert ragas_real._unscored_rows(prepared, [], [("x", "x")]) == prepared


def test_merge_scored_keeps_disjoint_columns_and_prior_order():
    prior = [_scored("a", faithfulness=0.5), _scored("b", faithfulness=1.0)]
    fresh = [_scored("b", answer_relevancy=0.25), _scored("c", answer_relevancy=0.75)]
    merged = ragas_real._merge_scored(prior, fresh)
    assert [r["_id"] for r in merged] == ["a", "b", "c"]  # prior order first
    assert merged[0] == prior[0]                           # untouched row survives
    assert merged[1]["faithfulness"] == 1.0                # kept
    assert merged[1]["answer_relevancy"] == 0.25           # added
    assert merged[2]["answer_relevancy"] == 0.75


def test_merge_scored_lets_fresh_values_win_per_column():
    prior = [_scored("a", faithfulness=0.1, answer_relevancy=0.2)]
    fresh = [_scored("a", faithfulness=0.9)]
    merged = ragas_real._merge_scored(prior, fresh)
    assert merged[0]["faithfulness"] == 0.9
    assert merged[0]["answer_relevancy"] == 0.2


def test_union_pairs_preserves_a_prior_days_columns():
    current = [("answer_relevancy", "answer_relevancy")]
    prior_summary = {"columns": {"faithfulness": "faithfulness",
                                 "answer_relevancy": "answer_relevancy"}}
    union = ragas_real._union_pairs(current, prior_summary)
    names = {name for name, _ in union}
    assert names == {"faithfulness", "answer_relevancy"}
    # the freshly requested name must not be shadowed by the prior mapping
    assert ("answer_relevancy", "answer_relevancy") in union


def test_union_pairs_without_prior_summary_is_identity():
    current = [("faithfulness", "faithfulness")]
    assert ragas_real._union_pairs(current, {}) == current
