"""Tests for the local quota / cooldown tracker."""
import json

import pytest

from config import settings
from src.generation.llm import ProviderError
from src.routing.quota_tracker import QuotaTracker, is_quota_error


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def tracker(clock):
    return QuotaTracker(state_path=None, clock=clock)


def test_new_provider_is_available(tracker):
    assert tracker.is_available("nvidia")
    assert tracker.seconds_remaining("nvidia") == 0


def test_rate_limit_uses_longer_cooldown(tracker):
    assert tracker.record_failure("nvidia", RuntimeError("HTTP 429")) == settings.RATE_LIMIT_COOLDOWN_S
    assert tracker.record_failure("groq", RuntimeError("boom")) == settings.ERROR_COOLDOWN_S


def test_cooldown_blocks_then_expires(tracker, clock):
    tracker.record_failure("nvidia", RuntimeError("429"))
    assert not tracker.is_available("nvidia")
    clock.t += settings.RATE_LIMIT_COOLDOWN_S + 1
    assert tracker.is_available("nvidia")


def test_repeated_failures_back_off_and_cap(tracker):
    first = tracker.record_failure("nvidia", RuntimeError("429"))
    second = tracker.record_failure("nvidia", RuntimeError("429"))
    assert second == 2 * first
    for _ in range(20):
        last = tracker.record_failure("nvidia", RuntimeError("429"))
    assert last == settings.MAX_COOLDOWN_S


def test_success_resets_failures(tracker, clock):
    tracker.record_failure("nvidia", RuntimeError("429"))
    tracker.record_success("nvidia")
    assert tracker.is_available("nvidia")
    assert tracker.record_failure("nvidia", RuntimeError("429")) == settings.RATE_LIMIT_COOLDOWN_S


def test_providers_are_independent(tracker):
    tracker.record_failure("nvidia", RuntimeError("429"))
    assert tracker.is_available("groq")


def test_status_reports_remaining(tracker):
    tracker.record_failure("nvidia", RuntimeError("429"))
    assert tracker.status()["nvidia"] > 0


def test_state_persists_across_instances(tmp_path, clock):
    path = str(tmp_path / "q.json")
    QuotaTracker(state_path=path, clock=clock).record_failure("nvidia", RuntimeError("429"))
    assert not QuotaTracker(state_path=path, clock=clock).is_available("nvidia")


# ---- robustness: a foreign / corrupt state file must never crash a query ----
@pytest.mark.parametrize("content", [
    {"nvidia": {"daily_limit": 100}},          # different schema (no 'until'/'failures')
    {"nvidia": "oops"},                        # wrong value type
    {"nvidia": {"until": "abc", "failures": "x"}},
    ["not", "a", "dict"],
    "just a string",
])
def test_foreign_state_file_does_not_crash(tmp_path, clock, content):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(content))
    t = QuotaTracker(state_path=str(path), clock=clock)
    assert t.is_available("nvidia")
    t.record_failure("nvidia", RuntimeError("429"))      # previously: KeyError
    assert not t.is_available("nvidia")


def test_corrupt_json_file_ignored(tmp_path, clock):
    path = tmp_path / "q.json"
    path.write_text("{not json")
    assert QuotaTracker(state_path=str(path), clock=clock).is_available("nvidia")


def test_unwritable_path_does_not_crash(clock):
    t = QuotaTracker(state_path="/nonexistent_dir/q.json", clock=clock)
    t.record_failure("nvidia", RuntimeError("429"))      # save fails silently
    assert not t.is_available("nvidia")


@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("HTTP 429: too many requests"), True),
    (RuntimeError("Rate limit reached"), True),
    (RuntimeError("quota exceeded"), True),
    (ProviderError("x", "p", 429), True),
    (ProviderError("bad request", "p", 400), False),
    (RuntimeError("connection reset"), False),
])
def test_is_quota_error(exc, expected):
    assert is_quota_error(exc) is expected