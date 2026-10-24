"""Tests for the eval harness metric helpers (no pipeline run)."""
import json

from evaluation import metrics
from evaluation.run_eval import (
    _context_recall_for_item,
    _dataset_specs,
    _norm_doc,
    _parse_temperatures,
    _posthoc_nli,
    load_datasets,
    parse_citations,
    score_answer,
)


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


# ------------- conjunctive scoring (required_fact_groups) -------------
_GROUPS_ITEM = {
    "category": "numerical_financial",
    "required_fact_groups": [
        ["$8,666.67", "8666.67"],
        ["$8,000", "8000"],
        ["$666.67", "666.67"],
    ],
}


def test_required_fact_groups_all_must_match():
    answer = "PI is $8,666.67: $8,000 principal plus $666.67 interest."
    correct, matched = score_answer(answer, _GROUPS_ITEM)
    assert correct
    assert matched == "$8,666.67; $8,000; $666.67"


def test_required_fact_groups_partial_match_fails():
    assert score_answer("The payment is $8,666.67.", _GROUPS_ITEM)[0] is False


def test_required_fact_groups_or_variants_within_group():
    answer = "PI is 8666.67, split into 8000 and 666.67."
    assert score_answer(answer, _GROUPS_ITEM)[0] is True


# ---------------- temperature parsing ----------------
def test_parse_temperatures_list_and_single():
    assert _parse_temperatures("0.0,0.1,0.3") == [0.0, 0.1, 0.3]
    assert _parse_temperatures("0.2") == [0.2]
    assert _parse_temperatures(" 0.1 , 0.5 ") == [0.1, 0.5]


# ---------------- retrieval ranking metrics ----------------
def test_hit_and_precision_at_k():
    assert metrics.hit_at_k([1, 4], 3) == 1.0
    assert metrics.hit_at_k([5], 3) == 0.0
    assert metrics.precision_at_k([2, 4], 4) == 0.5
    assert metrics.precision_at_k([2, 4], 3) == 1 / 3


def test_true_recall_at_k_counts_gold_pages():
    # 4 gold pages, 2 recovered in top-5
    assert metrics.recall_at_k([1, 3], 4, 5) == 0.5
    assert metrics.recall_at_k([1, 3], 4, 1) == 0.25
    assert metrics.recall_at_k([], 0, 5) == 0.0


def test_mrr_at_k():
    assert metrics.mrr_at_k([3], 5) == 1 / 3
    assert metrics.mrr_at_k([1], 5) == 1.0
    assert metrics.mrr_at_k([9], 5) == 0.0


def test_ndcg_at_k_perfect_and_partial():
    assert metrics.ndcg_at_k([True, True], 2) == 1.0
    # relevant at rank 2: DCG = 1/log2(3) ~= 0.6309, IDCG = 1.0
    assert round(metrics.ndcg_at_k([False, True], 2), 4) == 0.6309
    assert metrics.ndcg_at_k([False, False], 2) == 0.0


def test_context_precision_ranks_earlier_hits_higher():
    # AP: hit@1 -> 1/1, hit@4 -> 2/4 => (1 + 0.5) / 3 gold = 0.5
    assert metrics.context_precision([1, 4], 3, 5) == round((1 + 0.5) / 3, 4)
    assert metrics.context_precision([], 3, 5) == 0.0


# ---------------- context quality ----------------
def test_points_in_context_and_context_recall():
    context = "The APR is 37.05%. Principal is $350,000."
    assert metrics.points_in_context(["37.05", "$350,000"], context) == 2
    assert metrics.context_recall(["37.05", "$350,000", "15 years"], context) == 2 / 3
    assert metrics.context_recall([], context) == 0.0


def test_context_recall_for_item_supports_groups():
    item = {"required_fact_groups": [["TILA", "Truth in Lending"], ["RESPA"]]}
    assert _context_recall_for_item(item, "Under TILA and RESPA rules...") == 1.0
    assert _context_recall_for_item(item, "Only TILA applies here.") == 0.5
    assert _context_recall_for_item(item, "") == 0.0


def test_context_recall_for_item_flat_points():
    item = {"acceptable_answer_points": ["$320,000"], "expected_answer": "$320,000"}
    assert _context_recall_for_item(item, "the amount is $320,000.") == 1.0


# ---------------- answer grounding ----------------
def test_split_claims_keeps_decimals():
    claims = metrics.split_claims("The APR is 37.05%. Tenure is 20 years.")
    assert claims == ["The APR is 37.05%.", "Tenure is 20 years."]


def test_groundedness_scores_supported_claims_high():
    answer = "The APR is 37.05%. Tenure is 20 years."
    context = "The APR is 37.05% per the disclosure. Tenure is 20 years for the plan."
    result = metrics.groundedness(answer, context)
    # Sentence-level F1 penalizes the extra words in the context sentences
    # ("per the disclosure", "for the plan"), so support lands ~0.7, not 1.0.
    assert result["score"] > 0.6
    assert result["ungrounded_fraction"] == 0.0
    assert result["claims"] == 2


def test_groundedness_flags_fabricated_claims():
    answer = "The APR is 37.05%. The borrower lives in Springfield."
    context = "The APR is 37.05% per the disclosure table."
    result = metrics.groundedness(answer, context)
    assert result["ungrounded_fraction"] == 0.5
    assert result["score"] < 0.9


def test_groundedness_empty_answer():
    assert metrics.groundedness("", "context")["claims"] == 0


def test_token_f1_and_cosine():
    assert metrics.token_f1("the apr is 37.05", "the APR is 37.05 percent") > 0.8
    assert metrics.token_f1("hello", "goodbye") == 0.0
    assert metrics.cosine([1, 0], [1, 0]) == 1.0
    assert metrics.cosine([1, 0], [0, 1]) == 0.0


def test_answer_relevance_with_fake_embedder():
    def fake_embed(texts):
        # deterministic: same string -> same vector
        return [[float(len(t)), 1.0] for t in texts]

    assert metrics.answer_relevance("short", "short", fake_embed) == 1.0
    assert metrics.answer_relevance("x" * 50, "short", fake_embed) < 1.0
    assert metrics.answer_relevance("", "short", fake_embed) == 0.0


# ---------------- dataset loading ----------------
def test_load_datasets_dual_format_prefix_and_normalize(tmp_path):
    list_path = tmp_path / "golden_dataset_01.json"
    list_path.write_text(json.dumps([
        {"id": "q001", "category": "direct", "question": "Q1?", "source_pages": [2]},
        {"id": "q002", "category": "multi_hop", "question": "Q2?", "source_pages": [3]},
    ]), encoding="utf-8")
    dict_path = tmp_path / "golden_dataset_02.json"
    dict_path.write_text(json.dumps({
        "dataset_name": "ds02",
        "questions": [
            {"id": "q001", "category": "numerical", "question": "Q1 again?"},
        ],
    }), encoding="utf-8")

    merged = load_datasets({
        str(list_path): "docA.pdf",
        str(dict_path): "docB.pdf",
    })

    assert [q["id"] for q in merged] == [
        "d01_q001", "d01_q002", "d02_q001",
    ]
    assert [q["category"] for q in merged] == [
        "direct_fact_lookup", "multi_hop", "numerical_financial",
    ]
    assert [q["source_file"] for q in merged] == ["docA.pdf", "docA.pdf", "docB.pdf"]
    assert merged[0]["source_pages"] == [2]
    assert "source_pages" not in merged[2]


def test_load_datasets_missing_file_exits(tmp_path):
    try:
        load_datasets({str(tmp_path / "nope.json"): "doc.pdf"})
    except SystemExit as exc:
        assert "Dataset not found" in str(exc)
    else:
        raise AssertionError("expected SystemExit for a missing dataset file")


def test_load_datasets_rejects_wrong_shape(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_questions": True}), encoding="utf-8")
    try:
        load_datasets({str(bad): "doc.pdf"})
    except SystemExit as exc:
        assert "expected a list" in str(exc)
    else:
        raise AssertionError("expected SystemExit for a wrong-shaped dataset")


def test_dataset_specs_defaults_and_explicit(monkeypatch, tmp_path):
    fake = tmp_path / "golden_dataset_01.json"
    fake.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "evaluation.run_eval.DEFAULT_DATASETS",
        {str(fake): "Synthetic_Mortgage_Loan_File_TEST_01.pdf"},
    )
    specs = _dataset_specs(None)
    assert specs == {str(fake): "Synthetic_Mortgage_Loan_File_TEST_01.pdf"}

    specs = _dataset_specs(["my.json=that.pdf"])
    assert specs == {"my.json": "that.pdf"}


# ---------------- cost meter (evaluation/cost.py) ----------------
from types import SimpleNamespace  # noqa: E402

from evaluation.cost import CostMeter  # noqa: E402
from src.generation.llm import LLMResponse, TokenUsage  # noqa: E402


def test_cost_meter_accumulates_tokens_and_priced_cost(monkeypatch):
    # USD per 1M tokens: prompt 1.00, completion 2.00
    monkeypatch.setenv("ANEXUS_PRICE_GROQ", "1.0,2.0")
    meter = CostMeter()
    meter.record(SimpleNamespace(
        provider="groq", model="m",
        usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=100_000,
                         total_tokens=1_100_000),
    ))
    assert meter.calls == 1
    assert meter.prompt_tokens == 1_000_000
    assert meter.completion_tokens == 100_000
    # (1M * 1.0 + 0.1M * 2.0) / 1M = $1.20
    assert round(meter.cost_usd, 4) == 1.2
    assert meter.summary()["by_provider"]["groq"]["calls"] == 1


def test_cost_meter_counts_calls_without_usage_and_falls_back_to_total():
    meter = CostMeter()
    meter.record(SimpleNamespace(provider="simulated", model="m", usage=None))
    # usage with total_tokens omitted -> total falls back to prompt+completion
    meter.record(SimpleNamespace(
        provider="groq", model="m",
        usage=TokenUsage(prompt_tokens=4, completion_tokens=6, total_tokens=0),
    ))
    assert meter.calls == 2
    assert meter.prompt_tokens == 4
    assert meter.completion_tokens == 6
    assert meter.total_tokens == 10  # 4 + 6


def test_cost_meter_snapshot_delta_and_budget(monkeypatch):
    monkeypatch.setenv("ANEXUS_PRICE_GROQ", "1.0,2.0")
    meter = CostMeter()
    _one_call = lambda: meter.record(SimpleNamespace(  # noqa: E731
        provider="groq", model="m",
        usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000),
    ))
    _one_call()
    before = meter.snapshot()
    _one_call()

    delta = meter.delta(before)
    assert delta["calls"] == 1
    assert delta["cost_usd"] == 1.0
    assert delta["by_provider"]["groq"]["calls"] == 1

    assert meter.exceeds(None) is False
    assert meter.exceeds(0.0) is False
    assert meter.exceeds(1.5) is True
    assert meter.exceeds(0.5) is True


def test_cost_meter_unpriced_providers_cost_zero():
    meter = CostMeter()  # no ANEXUS_PRICE_* set -> free
    meter.record(SimpleNamespace(
        provider="nvidia", model="m",
        usage=TokenUsage(prompt_tokens=100_000, completion_tokens=50_000,
                         total_tokens=150_000),
    ))
    assert meter.total_tokens == 150_000
    assert meter.cost_usd == 0.0


# ---------------- usage capture in adapters / handler ----------------
def test_adapter_usage_parsers():
    from src.generation.llm import BaseAdapter

    assert BaseAdapter._openai_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    ) == TokenUsage(10, 5, 15)
    assert BaseAdapter._openai_usage({}) is None
    assert BaseAdapter._openai_usage({"usage": {}}) is None
    assert BaseAdapter._gemini_usage(
        {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3,
                           "totalTokenCount": 10}}
    ) == TokenUsage(7, 3, 10)
    assert BaseAdapter._gemini_usage({}) is None


def test_openai_compatible_chat_captures_last_usage(monkeypatch):
    import src.generation.llm as llm_mod

    payload = {
        "choices": [{"message": {"content": " ok "}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return payload

    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: _FakeResponse())

    adapter = llm_mod.GroqAdapter("model-x", "key")
    assert adapter.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert adapter.last_usage == TokenUsage(12, 3, 15)


def test_handler_reports_usage_and_notifies_observer():
    from src.generation.llm import LLMHandler, ScrubbedText
    from src.routing.quota_tracker import QuotaTracker
    import src.generation.llm as llm_mod

    class _Fake(llm_mod.BaseAdapter):
        name = "fake"

        def __init__(self):
            super().__init__("fake-1", "key")

        def chat(self, messages, temperature=None):
            self.last_usage = TokenUsage(1, 2, 3)
            return "ok"

    seen: list = []
    handler = LLMHandler(
        {"fake": _Fake()}, QuotaTracker(state_path=None), on_response=seen.append
    )
    response = handler.generate(
        ScrubbedText("q"), ScrubbedText("c"), provider_order=["fake"]
    )
    assert response.text == "ok"
    assert response.usage == TokenUsage(1, 2, 3)
    assert seen == [response]


def test_llm_response_usage_defaults_to_none():
    response = LLMResponse(
        text="t", provider="p", model="m", latency_s=0.1, fallbacks_tried=[]
    )
    assert response.usage is None


# ---------------- provider pinning / checkpointing ----------------
def test_pinned_router_forces_single_provider():
    from evaluation.run_eval import _PinnedRouter

    class _Inner:
        def route(self, query, available_providers=None):
            return SimpleNamespace(
                provider="nvidia", provider_order=["nvidia", "groq"], reason="intent"
            )

    decision = _PinnedRouter(_Inner(), "groq").route("q", ["nvidia", "groq"])
    assert decision.provider == "groq"
    assert decision.provider_order == ["groq"]


def test_run_checkpoint_round_trip_and_clear(tmp_path):
    from evaluation.run_eval import _RunCheckpoint

    path = tmp_path / "run.partial.json"
    ckpt = _RunCheckpoint(path)
    ckpt.add({"id": "q1"})
    ckpt.add({"id": "q2"})

    reloaded = _RunCheckpoint(path).load()
    assert [r["id"] for r in reloaded] == ["q1", "q2"]

    _RunCheckpoint(path).clear()
    assert not path.exists()
    assert _RunCheckpoint(path).load() == []


def test_run_checkpoint_prior_rows_are_not_aliased(tmp_path):
    """The resume pattern in main() must not double-count newly run rows.

    main() does ``rows = prior + rows_new`` where ``rows_new`` was produced by a
    pass that reports each row through ``on_row=checkpoint.add``. If ``load()``
    handed back the checkpoint's own list, ``prior`` would grow by every row of
    that pass and each new row would be counted twice.
    """
    from evaluation.run_eval import _RunCheckpoint

    path = tmp_path / "run.partial.json"
    seed = _RunCheckpoint(path)
    seed.add({"id": "q1"})
    seed.add({"id": "q2"})

    ckpt = _RunCheckpoint(path)
    prior = ckpt.load()
    assert [r["id"] for r in prior] == ["q1", "q2"]

    # the pass reports its new rows through the same checkpoint
    new_rows = []
    for i in (3, 4, 5):
        row = {"id": f"q{i}"}
        new_rows.append(row)
        ckpt.add(row)

    assert [r["id"] for r in prior] == ["q1", "q2"], "prior grew during the pass"
    merged = prior + new_rows
    assert [r["id"] for r in merged] == ["q1", "q2", "q3", "q4", "q5"]
    assert len({r["id"] for r in merged}) == len(merged)
    assert [r["id"] for r in ckpt.load()] == ["q1", "q2", "q3", "q4", "q5"]


# ---------------- summary aggregation ----------------
def _row(**overrides):
    row = {
        "id": "q1", "category": "direct_fact_lookup",
        "hit_at_k": 1.0, "recall_at_k": 1.0, "precision_at_k": 0.2,
        "ndcg_at_k": 1.0, "mrr_at_k": 1.0, "context_precision": 1.0,
        "context_recall": 1.0, "answer_correct": True,
        "answer_text": "The rate is 7.25%.", "groundedness": 0.9,
        "ungrounded_fraction": 0.0, "answer_claims": 1, "answer_relevance": 0.8,
        "latency_s": 1.0, "provider": "groq", "fallbacks_tried": [],
        "cited_count": 1, "cited_ok": 1, "pii_leaked": False,
    }
    row.update(overrides)
    return row


def test_summarize_aggregates_rows_and_cost():
    from evaluation.run_eval import _summarize

    rows = [
        _row(),
        _row(
            id="q2", answer_correct=False,
            answer_text="The answer is not available in the provided documents.",
            groundedness=0.0, answer_claims=2, answer_relevance=None,
            latency_s=3.0, provider="nvidia", fallbacks_tried=["groq"],
            cited_count=0, cited_ok=0, pii_leaked=True,
        ),
    ]
    summary = _summarize(rows, 0.1, "end_to_end", {"cost_usd": 0.5, "calls": 2})

    assert summary["items"] == 2
    assert summary["answer_accuracy"] == 0.5
    assert summary["answered_rate"] == 0.5
    assert summary["no_context_rate"] == 0.5
    assert summary["pii_leak_rate"] == 0.5
    assert summary["citation_support_rate"] == 1.0  # 1 of 1 supported
    assert summary["citation_coverage"] == 1.0       # 1 citing row / 1 answered row
    assert summary["fallback_rate"] == 0.5
    assert summary["groundedness"] == 0.45           # mean over both rows with claims
    assert summary["answer_relevance"] == 0.8        # None rows excluded
    assert summary["median_e2e_s"] == 2.0
    assert summary["cost"]["cost_usd"] == 0.5
    assert summary["by_provider"]["groq"]["correct"] == 1


def test_posthoc_nli_invokes_the_standalone_cli(tmp_path, monkeypatch):
    import subprocess as _sp

    seen = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="[nli] scored 1 rows\n", stderr="")

    monkeypatch.setattr(_sp, "run", _fake_run)
    path = tmp_path / "results.json"
    assert _posthoc_nli(path) is True
    assert seen["cmd"][-4:] == ["--results", str(path)] or str(path) in seen["cmd"]
    assert "-m" in seen["cmd"] and "evaluation.nli" in seen["cmd"]
    assert seen["kwargs"].get("timeout", 0) > 0


def test_posthoc_nli_failure_keeps_the_run_valid(tmp_path, monkeypatch):
    import subprocess as _sp

    monkeypatch.setattr(
        _sp, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="no weights"),
    )
    assert _posthoc_nli(tmp_path / "results.json") is False


def test_posthoc_nli_survives_a_subprocess_that_never_starts(tmp_path, monkeypatch):
    import subprocess as _sp

    def _boom(*a, **k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(_sp, "run", _boom)
    assert _posthoc_nli(tmp_path / "results.json") is False
