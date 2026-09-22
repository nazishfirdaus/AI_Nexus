"""Tests for provider adapters and the fallback handler. No network access:
HTTP calls are replaced by a fake requests.post."""
import pytest
import requests

from src.generation import llm
from src.generation.llm import (
    AllProvidersFailed, BaseAdapter, GeminiAdapter, GroqAdapter, LLMHandler,
    NvidiaAdapter, ProviderError, ScrubbedText, SimulatedAdapter,
)
from src.routing.quota_tracker import QuotaTracker, is_quota_error

Q, C = ScrubbedText("what is the loan amount?"), ScrubbedText("Loan amount is <AMOUNT>.")


class FakeAdapter(BaseAdapter):
    def __init__(self, name, result="ok", error=None, configured=True):
        super().__init__(model=f"{name}-model", api_key="k" if configured else None)
        self.name = name
        self.result, self.error, self.calls = result, error, 0

    def generate(self, query, context):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def handler(*adapters):
    return LLMHandler({a.name: a for a in adapters}, QuotaTracker(state_path=None))


# ---------------- handler ----------------
def test_rejects_unscrubbed_text():
    h = handler(FakeAdapter("groq"))
    with pytest.raises(TypeError):
        h.generate("raw query", C, ["groq"])
    with pytest.raises(TypeError):
        h.generate(Q, "raw context with PII", ["groq"])


def test_first_provider_success():
    a, b = FakeAdapter("nvidia", "A"), FakeAdapter("groq", "B")
    r = handler(a, b).generate(Q, C, ["nvidia", "groq"])
    assert (r.text, r.provider, r.fallbacks_tried) == ("A", "nvidia", [])
    assert b.calls == 0


def test_falls_back_on_provider_error():
    a = FakeAdapter("nvidia", error=ProviderError("boom", "nvidia", 500))
    b = FakeAdapter("groq", "B")
    r = handler(a, b).generate(Q, C, ["nvidia", "groq"])
    assert r.provider == "groq" and r.fallbacks_tried == ["nvidia"]


def test_falls_back_on_unexpected_exception():
    a = FakeAdapter("nvidia", error=ValueError("adapter bug"))
    b = FakeAdapter("groq", "B")
    assert handler(a, b).generate(Q, C, ["nvidia", "groq"]).provider == "groq"


def test_empty_response_treated_as_failure():
    a, b = FakeAdapter("nvidia", result=""), FakeAdapter("groq", "B")
    assert handler(a, b).generate(Q, C, ["nvidia", "groq"]).provider == "groq"


def test_unconfigured_provider_skipped():
    a, b = FakeAdapter("nvidia", configured=False), FakeAdapter("groq", "B")
    r = handler(a, b).generate(Q, C, ["nvidia", "groq"])
    assert r.provider == "groq" and a.calls == 0


def test_all_fail_raises_with_reasons():
    a = FakeAdapter("nvidia", error=ProviderError("down", "nvidia"))
    b = FakeAdapter("groq", error=ProviderError("429", "groq", 429))
    with pytest.raises(AllProvidersFailed) as e:
        handler(a, b).generate(Q, C, ["nvidia", "groq"])
    assert set(e.value.errors) == {"nvidia", "groq"}


def test_failed_provider_enters_cooldown_and_is_not_retried():
    a = FakeAdapter("nvidia", error=ProviderError("HTTP 429", "nvidia", 429))
    b = FakeAdapter("groq", "B")
    h = handler(a, b)
    h.generate(Q, C, ["nvidia", "groq"])
    assert not h.quota.is_available("nvidia")
    h.generate(Q, C, ["nvidia", "groq"])
    assert a.calls == 1          # second query skipped the dead provider


def test_cooling_provider_never_called_even_if_only_one_left():
    a = FakeAdapter("nvidia", "A")
    h = handler(a)
    h.quota.record_failure("nvidia", RuntimeError("429"))
    with pytest.raises(AllProvidersFailed) as e:
        h.generate(Q, C, ["nvidia"])
    assert a.calls == 0
    assert "cooling down" in e.value.errors["nvidia"]


def test_success_clears_failure_state():
    a = FakeAdapter("nvidia", "A")
    h = handler(a)
    h.quota.record_failure("nvidia", RuntimeError("boom"))
    h.quota._state["nvidia"]["until"] = 0
    h.generate(Q, C, ["nvidia"])
    assert h.quota._state["nvidia"]["failures"] == 0


def test_available_providers_lists_only_configured():
    h = handler(FakeAdapter("nvidia", configured=False), FakeAdapter("groq"))
    assert h.available_providers == ["groq"]


def test_simulated_adapter():
    h = LLMHandler({"simulated": SimulatedAdapter()}, QuotaTracker(state_path=None))
    r = h.generate(Q, C, ["simulated"])
    assert r.text.startswith("[SIMULATED]")


def test_build_default_handler_simulated(monkeypatch):
    monkeypatch.setattr(llm.settings, "SIMULATE_LLM", True)
    h = llm.build_default_handler(QuotaTracker(state_path=None))
    assert h.available_providers == ["simulated"]


def test_build_default_handler_reads_env(monkeypatch):
    monkeypatch.setattr(llm.settings, "SIMULATE_LLM", False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    h = llm.build_default_handler(QuotaTracker(state_path=None))
    assert h.available_providers == ["groq"]


# ---------------- quota error detection ----------------
@pytest.mark.parametrize("msg", ["HTTP 429: too many", "Rate limit reached", "quota exceeded", "RESOURCE_EXHAUSTED"])
def test_is_quota_error_true(msg):
    assert is_quota_error(RuntimeError(msg))


def test_is_quota_error_by_status_code():
    assert is_quota_error(ProviderError("x", "p", 429))
    assert not is_quota_error(ProviderError("bad request", "p", 400))


# ---------------- HTTP adapters ----------------
class FakeResp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def patch_post(monkeypatch, resp=None, exc=None):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json, timeout=timeout)
        if exc:
            raise exc
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    return seen


@pytest.mark.parametrize("cls,url_part", [(GroqAdapter, "groq.com"), (NvidiaAdapter, "nvidia.com")])
def test_openai_compatible_adapters_parse_and_send(monkeypatch, cls, url_part):
    seen = patch_post(monkeypatch, FakeResp(body={"choices": [{"message": {"content": " Hi "}}]}))
    out = cls("m", "KEY").generate("q", "ctx")
    assert out == "Hi"
    assert url_part in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer KEY"
    assert seen["json"]["model"] == "m"
    assert "ctx" in seen["json"]["messages"][1]["content"]


def test_gemini_adapter_parses(monkeypatch):
    body = {"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}
    seen = patch_post(monkeypatch, FakeResp(body=body))
    assert GeminiAdapter("gemini-x", "KEY").generate("q", "ctx") == "AB"
    assert "gemini-x:generateContent" in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "KEY"


def test_gemini_blocked_response_raises(monkeypatch):
    patch_post(monkeypatch, FakeResp(body={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(ProviderError):
        GeminiAdapter("m", "K").generate("q", "c")


def test_http_429_becomes_provider_error_with_status(monkeypatch):
    patch_post(monkeypatch, FakeResp(status=429, text="slow down"))
    with pytest.raises(ProviderError) as e:
        GroqAdapter("m", "K").generate("q", "c")
    assert e.value.status_code == 429 and is_quota_error(e.value)


def test_timeout_becomes_provider_error(monkeypatch):
    patch_post(monkeypatch, exc=requests.Timeout())
    with pytest.raises(ProviderError, match="timeout"):
        NvidiaAdapter("m", "K").generate("q", "c")


def test_network_error_becomes_provider_error(monkeypatch):
    patch_post(monkeypatch, exc=requests.ConnectionError("dns"))
    with pytest.raises(ProviderError, match="network"):
        GroqAdapter("m", "K").generate("q", "c")


def test_bad_json_and_bad_shape(monkeypatch):
    patch_post(monkeypatch, FakeResp(body=None))
    with pytest.raises(ProviderError):
        GroqAdapter("m", "K").generate("q", "c")
    patch_post(monkeypatch, FakeResp(body={"unexpected": 1}))
    with pytest.raises(ProviderError):
        GroqAdapter("m", "K").generate("q", "c")


def test_timeout_setting_is_passed(monkeypatch):
    seen = patch_post(monkeypatch, FakeResp(body={"choices": [{"message": {"content": "x"}}]}))
    GroqAdapter("m", "K", timeout=7).generate("q", "c")
    assert seen["timeout"] == 7


# ---------------- SIMULATE_LLM parsing (CI sets SIMULATE_LLM=true) ----------------
@pytest.fixture
def reload_settings(monkeypatch):
    import importlib
    from config import settings
    yield lambda: importlib.reload(settings)
    monkeypatch.undo()
    importlib.reload(settings)


@pytest.mark.parametrize("val,expected", [
    ("true", True), ("True", True), ("TRUE", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("", False), ("no", False),
])
def test_simulate_llm_env_parsing(monkeypatch, reload_settings, val, expected):
    monkeypatch.setenv("SIMULATE_LLM", val)
    assert reload_settings().SIMULATE_LLM is expected