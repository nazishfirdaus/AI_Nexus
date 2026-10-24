"""Tests for the TruthfulQA closed-book benchmark (Phase E, model-free)."""
import json
from types import SimpleNamespace

import pytest

from evaluation.benchmarks import truthfulqa as tqa


def _item(question="What is 2+2?", incorrect=("3", "5"),
          category="arithmetic", type_="numeric"):
    return {
        "question": question,
        "category": category,
        "type": type_,
        "mc1_targets": {"choices": ["4"] + list(incorrect), "label": 0},
    }


def test_build_question_keeps_the_label_consistent_with_the_shuffle():
    choices, label = tqa.build_question(_item(), seed=7)
    assert choices[label] == "4"
    assert sorted(choices) == ["3", "4", "5"]


def test_build_question_is_deterministic_per_seed():
    item = _item(incorrect=("a", "b", "c", "d", "e"))
    assert tqa.build_question(item, seed=3) == tqa.build_question(item, seed=3)
    assert tqa.build_question(item, seed=3) != tqa.build_question(item, seed=4)


def test_build_question_caps_incorrect_choices():
    item = _item(incorrect=("a", "b", "c", "d", "e"))
    choices, label = tqa.build_question(item, n_incorrect=2, seed=0)
    assert len(choices) == 3
    assert choices[label] == "4"


def test_parse_choice_reads_a_letter():
    assert tqa.parse_choice("B", ["x", "y", "z"]) == 1
    assert tqa.parse_choice("The answer is C.", ["x", "y", "z"]) == 2


def test_parse_choice_rejects_letters_outside_the_options():
    assert tqa.parse_choice("E", ["x", "y", "z"]) is None


def test_parse_choice_falls_back_to_the_full_answer_text():
    assert tqa.parse_choice("  the second one  ", ["first", "the second one"]) == 1


def test_parse_choice_returns_none_for_garbage():
    assert tqa.parse_choice("", ["a"]) is None
    assert tqa.parse_choice(None, ["a"]) is None
    simulated = "[SIMULATED] Answer to 'Question: what' based on 10 chars of context."
    assert tqa.parse_choice(simulated, ["a", "b"]) is None


def test_load_items_reads_a_local_cache_and_drops_broken_records(tmp_path):
    path = tmp_path / "tqa.json"
    path.write_text(
        json.dumps([_item("q1"), _item("q2"), {"question": "no targets"}]),
        encoding="utf-8",
    )
    items = tqa.load_items(path=path)
    assert [i["question"] for i in items] == ["q1", "q2"]


def test_load_items_respects_limit_and_seed(tmp_path):
    path = tmp_path / "tqa.json"
    path.write_text(
        json.dumps([_item(f"q{i}") for i in range(10)]), encoding="utf-8"
    )
    first = [i["question"] for i in tqa.load_items(path=path, limit=3, seed=1)]
    again = [i["question"] for i in tqa.load_items(path=path, limit=3, seed=1)]
    other = [i["question"] for i in tqa.load_items(path=path, limit=3, seed=2)]
    assert first == again
    assert first != other


def test_load_items_errors_on_a_missing_path(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        tqa.load_items(path=tmp_path / "nope.json")


def test_summarize_accuracy_and_breakdowns():
    rows = [
        {"id": "a", "category": "cat1", "type": "true",
         "parsed_index": 0, "label": 0, "correct": True,
         "provider": "groq", "latency_s": 1.0},
        {"id": "b", "category": "cat1", "type": "false",
         "parsed_index": 0, "label": 1, "correct": False,
         "provider": "nvidia", "latency_s": 3.0},
        {"id": "c", "category": "cat2", "type": "true",
         "parsed_index": None, "label": 0, "correct": None,
         "provider": None, "latency_s": 2.0},
    ]
    out = tqa.summarize(rows, cost={"calls": 2}, pinned="groq")
    assert out["items"] == 3
    assert out["answered_rate"] == round(2 / 3, 4)
    assert out["accuracy"] == 0.5
    assert out["by_category"] == {"cat1": 0.5, "cat2": None}
    assert out["by_type"] == {"false": 0.0, "true": 1.0}
    assert out["median_latency_s"] == 2.0
    assert out["fallback_rate"] == round(1 / 3, 4)  # nvidia is the only fallback


def test_summarize_empty_input():
    out = tqa.summarize([])
    assert out["items"] == 0
    assert out["accuracy"] is None


class _StubHandler:
    """Answers each call with a canned reply (an Exception to raise)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_messages(self, messages, provider_order, temperature=None):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            text=reply, provider=provider_order[0], model="stub",
            fallbacks_tried=[], usage=None,
        )


def test_run_closed_book_scores_letters_and_survives_errors():
    items = [_item("q1"), _item("q2"), _item("q3")]
    letters = [tqa.LETTERS[tqa.build_question(item, seed=0)[1]] for item in items]
    handler = _StubHandler([letters[0], RuntimeError("boom"), "not a letter"])
    rows = tqa.run_closed_book(items, handler, ["groq"])
    assert handler.calls == 3
    assert rows[0]["correct"] is True
    assert rows[1]["error"].startswith("RuntimeError")
    assert rows[2]["parsed_index"] is None
    assert rows[2]["correct"] is None


def test_run_closed_book_stops_on_budget():
    items = [_item(f"q{i}") for i in range(3)]
    handler = _StubHandler(["A", "A", "A"])

    class _OverBudget:
        def exceeds(self, limit):
            return True

    rows = tqa.run_closed_book(
        items, handler, ["groq"], meter=_OverBudget(), max_cost_usd=0.01
    )
    assert len(rows) == 1

def test_normalize_accepts_the_hf_raw_shape():
    raw = {
        "question": "q", "category": "c", "type": "t",
        "best_answer": "yes", "correct_answers": ["yes", "probably"],
        "incorrect_answers": ["no", "maybe", "yes"],
    }
    item = tqa._normalize(raw)
    assert item["mc1_targets"]["choices"][0] == "yes"  # correct first, before shuffle
    assert item["mc1_targets"]["label"] == 0
    assert item["mc1_targets"]["choices"][1:] == ["no", "maybe"]


def test_normalize_prefers_mc1_targets_and_drops_unusable_rows():
    with_targets = {
        "question": "q",
        "mc1_targets": {"choices": ["b", "a"], "label": 1},
    }
    assert tqa._normalize(with_targets)["mc1_targets"]["label"] == 1
    assert tqa._normalize({"question": "q"}) is None
    assert tqa._normalize({"best_answer": "only", "incorrect_answers": ["only"]}) is None

def test_summarize_counts_zero_latency_rows_in_the_median():
    # a 0.0 latency is real data: skipped it, [0.0, 0.0, 2.0] would collapse
    # to [2.0] and the median would wrongly read 2.0
    rows = [
        {"id": "a", "category": "c", "type": "t", "parsed_index": 0, "label": 0,
         "correct": True, "provider": "simulated", "latency_s": 0.0},
        {"id": "b", "category": "c", "type": "t", "parsed_index": 1, "label": 1,
         "correct": True, "provider": "simulated", "latency_s": 0.0},
        {"id": "c", "category": "c", "type": "t", "parsed_index": 0, "label": 0,
         "correct": True, "provider": "simulated", "latency_s": 2.0},
    ]
    out = tqa.summarize(rows, pinned="simulated")
    assert out["median_latency_s"] == 0.0

def test_cooldown_wait_parses_both_message_shapes():
    handler_error = ("AllProvidersFailed: All LLM providers failed: "
                     "nvidia: cooling down (20s left)")
    tracker_log = "Provider nvidia cooling down for 7s (HTTP 404: ...)"
    assert tqa._cooldown_wait(handler_error) == 20.5
    assert tqa._cooldown_wait(tracker_log) == 7.5
    assert tqa._cooldown_wait(None) is None
    assert tqa._cooldown_wait("RuntimeError: boom") is None
    # too long to park a single row - fail honestly instead of freezing
    assert tqa._cooldown_wait("nvidia: cooling down (120s left)") is None


def test_run_closed_book_retries_once_after_a_cooldown():
    items = [_item("q1")]
    letter = tqa.LETTERS[tqa.build_question(items[0], seed=0)[1]]
    handler = _StubHandler([
        RuntimeError("AllProvidersFailed: nvidia: cooling down (1s left)"),
        letter,
    ])
    rows = tqa.run_closed_book(items, handler, ["nvidia"])
    assert handler.calls == 2
    assert rows[0]["error"] is None
    assert rows[0]["correct"] is True


def test_run_closed_book_does_not_retry_ordinary_errors():
    items = [_item("q1")]
    handler = _StubHandler([RuntimeError("HTTP 500"), "A"])
    rows = tqa.run_closed_book(items, handler, ["nvidia"])
    assert handler.calls == 1
    assert rows[0]["error"].startswith("RuntimeError")
