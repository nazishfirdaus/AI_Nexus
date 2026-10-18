"""Streaming tests: SSE adapters, fallback-before-first-token, and the
PII-preserving streaming scrubber.

Everything runs offline: HTTP is replaced by a fake requests.post, and the
pipeline uses the same fakes as tests/test_pipeline.py (imported, not copied).
"""
import json

import pytest
import requests

from src.generation.llm import (
    AllProvidersFailed,
    BaseAdapter,
    GeminiAdapter,
    GroqAdapter,
    LLMHandler,
    LLMStream,
    NvidiaAdapter,
    ProviderError,
    ScrubbedText,
)
from src.rag_pipeline import _StreamingScrubber
from src.routing.quota_tracker import QuotaTracker
from test_pipeline import (
    FakeEmbedder,
    FakeRetriever,
    FakeReranker,
    FakeRouter,
    FakeScrubber,
    FakeVectorDB,
    build,
    make_pdf,
)

Q, C = ScrubbedText("what is the loan amount?"), ScrubbedText("Loan amount is <AMOUNT>.")


# ------------------------------------------------------------------ fakes
class FakeStreamAdapter(BaseAdapter):
    """Streams a canned list of chunks; can be told to fail at a given point."""

    supports_streaming = True

    def __init__(self, name="fake", chunks=None, fail_after=None, error=None):
        super().__init__(model=f"{name}-model", api_key="k")
        self.name = name
        self.chunks = list(chunks if chunks is not None else ["ok"])
        self.fail_after = fail_after  # raise after this many chunks (None = never)
        self.error = error or ProviderError("boom", name)
        self.calls = 0

    def chat_stream(self, messages, temperature=None):
        self.calls += 1
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise self.error
            yield chunk


class NonStreamingAdapter(BaseAdapter):
    name = "nostream"

    def chat(self, messages, temperature=None):
        return "ok"


def stream_handler(*adapters):
    return LLMHandler({a.name: a for a in adapters}, QuotaTracker(state_path=None))


# ------------------------------------------------------------------ SSE adapters
class FakeStreamResp:
    """Minimal stand-in for a streaming requests.Response."""

    def __init__(self, status=200, lines=None, text=""):
        self.status_code = status
        self._lines = lines or []
        self.text = text

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


def patch_stream_post(monkeypatch, resp=None, exc=None):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        seen.update(url=url, headers=headers, json=json, timeout=timeout, stream=stream)
        if exc:
            raise exc
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    return seen


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def test_openai_compatible_stream_parses_delta_chunks(monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {"content": "He"}}]}),
        _sse({"choices": [{"delta": {"content": "llo"}}]}),
        "data: [DONE]",
    ]
    seen = patch_stream_post(monkeypatch, FakeStreamResp(lines=lines))
    out = list(GroqAdapter("m", "KEY").chat_stream([{"role": "user", "content": "hi"}]))
    assert out == ["He", "llo"]
    assert seen["stream"] is True
    assert seen["json"]["stream"] is True
    assert seen["json"]["model"] == "m"


def test_nvidia_stream_parses_delta_chunks(monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {"content": "A"}}]}),
        _sse({"choices": [{"delta": {}}]}),  # empty delta is skipped
        _sse({"choices": [{"delta": {"content": "B"}}]}),
        "data: [DONE]",
    ]
    patch_stream_post(monkeypatch, FakeStreamResp(lines=lines))
    out = list(NvidiaAdapter("m", "KEY").chat_stream([{"role": "user", "content": "x"}]))
    assert out == ["A", "B"]


def test_gemini_stream_parses_sse_parts(monkeypatch):
    lines = [
        _sse({"candidates": [{"content": {"parts": [{"text": "A"}, {"text": "B"}]}}]}),
        _sse({"candidates": [{"content": {"parts": [{"text": "C"}]}}]}),
        "data: [DONE]",
    ]
    seen = patch_stream_post(monkeypatch, FakeStreamResp(lines=lines))
    out = list(GeminiAdapter("gemini-x", "KEY").chat_stream([{"role": "user", "content": "hi"}]))
    assert out == ["A", "B", "C"]
    assert "streamGenerateContent?alt=sse" in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "KEY"
    assert seen["stream"] is True


def test_stream_http_error_raises_provider_error(monkeypatch):
    patch_stream_post(monkeypatch, FakeStreamResp(status=429, text="slow down"))
    with pytest.raises(ProviderError) as e:
        list(GroqAdapter("m", "K").chat_stream([{"role": "user", "content": "x"}]))
    assert e.value.status_code == 429


def test_stream_network_error_raises_provider_error(monkeypatch):
    patch_stream_post(monkeypatch, exc=requests.ConnectionError("dns"))
    with pytest.raises(ProviderError, match="network"):
        list(GroqAdapter("m", "K").chat_stream([{"role": "user", "content": "x"}]))


def test_streaming_unsupported_adapter_is_flagged():
    assert GroqAdapter("m", "k").supports_streaming is True
    assert NonStreamingAdapter("nostream", "k").supports_streaming is False


# ------------------------------------------------------------------ handler stream
def test_stream_rejects_unscrubbed_text():
    h = stream_handler(FakeStreamAdapter())
    with pytest.raises(TypeError):
        h.generate_stream("raw query", C, ["fake"])
    with pytest.raises(TypeError):
        h.generate_stream(Q, "raw context", ["fake"])


def test_stream_first_provider_success():
    a = FakeStreamAdapter("nvidia", ["A1", "A2"])
    b = FakeStreamAdapter("groq", ["B"])
    s = stream_handler(a, b).generate_stream(Q, C, ["nvidia", "groq"])
    out = list(s)
    assert out == ["A1", "A2"]
    assert (s.provider, s.model, s.fallbacks_tried) == ("nvidia", "nvidia-model", [])
    assert b.calls == 0


def test_stream_falls_back_before_first_token():
    a = FakeStreamAdapter("nvidia", ["A"], fail_after=0)  # fails before any token
    b = FakeStreamAdapter("groq", ["B1", "B2"])
    s = stream_handler(a, b).generate_stream(Q, C, ["nvidia", "groq"])
    assert list(s) == ["B1", "B2"]
    assert s.provider == "groq" and s.fallbacks_tried == ["nvidia"]


def test_stream_midstream_failure_stops_and_keeps_partial():
    a = FakeStreamAdapter("nvidia", ["A1", "A2", "A3"], fail_after=2)
    b = FakeStreamAdapter("groq", ["B"])
    h = stream_handler(a, b)
    s = h.generate_stream(Q, C, ["nvidia", "groq"])
    assert list(s) == ["A1", "A2"]  # partial, then stops
    assert s.provider == "nvidia"
    assert b.calls == 0  # never splices a second provider mid-answer


def test_stream_skips_unconfigured_and_non_streaming():
    nostream = NonStreamingAdapter("nostream", "k")
    good = FakeStreamAdapter("groq", ["G"])
    h = LLMHandler({"nostream": nostream, "groq": good}, QuotaTracker(state_path=None))
    s = h.generate_stream(Q, C, ["nostream", "groq"])
    assert list(s) == ["G"]
    assert s.provider == "groq"


def test_stream_all_fail_raises():
    a = FakeStreamAdapter("nvidia", ["x"], fail_after=0)
    b = FakeStreamAdapter("groq", ["x"], fail_after=0)
    h = stream_handler(a, b)
    with pytest.raises(AllProvidersFailed):
        list(h.generate_stream(Q, C, ["nvidia", "groq"]))


def test_stream_quotas_are_recorded():
    a = FakeStreamAdapter("nvidia", ["x"], fail_after=0)
    b = FakeStreamAdapter("groq", ["G"])
    h = stream_handler(a, b)
    list(h.generate_stream(Q, C, ["nvidia", "groq"]))
    assert not h.quota.is_available("nvidia")  # failure recorded
    assert h.quota.is_available("groq")


# ------------------------------------------------------------------ streaming scrubber
class CountingScrubber:
    """Scrubs nothing but counts calls, so boundary behaviour is observable."""

    def __init__(self):
        self.calls = []

    def scrub(self, text, context_prefix=""):
        self.calls.append(text)
        return ScrubbedText(text)


def test_scrubber_holds_back_tail_until_flush():
    scrubber = CountingScrubber()
    s = _StreamingScrubber(scrubber, tail=10)
    out = s.feed("x" * 30)
    # 30 raw chars, tail 10 -> 20 released, 10 held back
    assert len(out) == 20
    assert s.flush() == "x" * 10


def test_scrubber_never_emits_more_than_it_has():
    scrubber = CountingScrubber()
    s = _StreamingScrubber(scrubber, tail=5)
    assert s.feed("ab") == ""        # 2 < 5, everything stays in the tail
    assert s.feed("cd") == ""        # 4 < 5, still held
    assert s.feed("efghij") == "abcde"  # 10 - 5 = 5 stable
    assert s.flush() == "fghij"


def test_scrubber_splits_pii_across_chunk_boundary():
    """A PII value split across two chunks must never be emitted unscrubbed."""

    class AadhaarScrubber:
        def scrub(self, text, context_prefix=""):
            import re

            return ScrubbedText(re.sub(r"\b\d{4} \d{4} \d{4}\b", "<AADHAAR_NUMBER>", text))

    aadhaar = "2345 6789 0124"
    raw = f"The reference is {aadhaar} on file."
    # feed one character at a time: the value is guaranteed to straddle chunks
    s = _StreamingScrubber(AadhaarScrubber(), tail=20)
    emitted = []
    for ch in raw:
        delta = s.feed(ch)
        if delta:
            emitted.append(delta)
    emitted.append(s.flush())
    shown = "".join(emitted)
    assert aadhaar not in shown
    assert "<AADHAAR_NUMBER>" in shown


def test_scrubber_passes_through_when_scrub_fails():
    class Boom:
        def scrub(self, text, context_prefix=""):
            raise RuntimeError("scrubber down")

    s = _StreamingScrubber(Boom(), tail=2)
    assert s.feed("hello") == "hel"  # 5 - 2 tail = 3 released, pass-through
    assert s.flush() == "lo"


# ------------------------------------------------------------------ pipeline stream
class StreamingFakeLLMHandler:
    """Fake handler that streams the canned answer text in fixed chunks."""

    def __init__(self):
        self.available_providers = ["fake"]
        self.chunks = ["7.2", "5 percent"]
        self.fail = False
        self.stream_calls = 0
        self.message_calls: list = []

    def generate(self, query, context, provider_order, history=None, summary=None,
                 temperature=None):
        from src.generation.llm import LLMResponse

        return LLMResponse(
            text="7.25 percent", provider="fake", model="fake-1",
            latency_s=0.1, fallbacks_tried=[],
        )

    def generate_messages(self, messages, provider_order, temperature=None):
        """Rewrite/summary calls decline, matching the module's other fakes."""
        from src.generation.llm import LLMResponse

        self.message_calls.append(list(messages))
        return LLMResponse(
            text="[SIMULATED] no rewrite", provider="fake", model="fake-1",
            latency_s=0.05, fallbacks_tried=[],
        )

    def generate_stream(self, query, context, provider_order, history=None, summary=None,
                        temperature=None):
        assert isinstance(query, ScrubbedText), "provider must receive ScrubbedText"
        assert isinstance(context, ScrubbedText), "provider must receive ScrubbedText"
        self.stream_calls += 1
        if self.fail:
            raise AllProvidersFailed({p: "boom" for p in provider_order})
        handler = stream_handler(FakeStreamAdapter("fake", self.chunks))
        return handler.generate_stream(query, context, provider_order, history, summary,
                                       temperature)


@pytest.fixture(autouse=True)
def _disable_query_expansion(monkeypatch):
    """Match tests/test_pipeline.py: expansion is opt-in per test."""
    from config import settings

    monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", False)


def _stream_pipeline(handler=None):
    return build(
        vector_db=FakeVectorDB(),
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        scrubber=FakeScrubber(),
        router=FakeRouter(),
        llm_handler=handler or StreamingFakeLLMHandler(),
        embedder=FakeEmbedder(),
    )


def test_pipeline_stream_matches_nonstreaming(tmp_path):
    pdf = tmp_path / "loan.pdf"
    make_pdf(pdf)
    pipe = _stream_pipeline()
    pipe.ingest_document(pdf)

    nonstream = pipe.answer_query("What is the interest rate?")
    stream_handle = pipe.answer_query_stream("What is the interest rate?")
    shown = "".join(stream_handle)
    final = stream_handle.final

    assert shown == "7.25 percent"
    assert final.text == nonstream.text == "7.25 percent"
    assert [c.page_number for c in final.citations] == [
        c.page_number for c in nonstream.citations
    ]


def test_pipeline_stream_no_document(tmp_path):
    pipe = _stream_pipeline()
    handle = pipe.answer_query_stream("Anything")
    assert "".join(handle)  # yields the safe no-context message
    assert handle.final.citations == []


def test_pipeline_stream_provider_never_receives_raw_pii(tmp_path):
    pdf = tmp_path / "pii.pdf"
    doc = __import__("pymupdf").open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Reference id 2345 6789 0124 on file for this loan. "
        "This is a confidential document prepared for the underwriting "
        "and credit review process only.",
    )
    doc.save(str(pdf))
    doc.close()

    pipe = _stream_pipeline()
    pipe.ingest_document(pdf)
    handle = pipe.answer_query_stream("What reference is on file?")
    shown = "".join(handle)
    # The gate: the model only ever saw scrubbed context, so nothing raw leaks.
    assert "2345 6789 0124" not in shown
    assert handle.final is not None


def test_pipeline_stream_failure_is_graceful(tmp_path):
    pdf = tmp_path / "loan.pdf"
    make_pdf(pdf)
    handler = StreamingFakeLLMHandler()
    pipe = _stream_pipeline(handler)
    pipe.ingest_document(pdf)
    handler.fail = True

    handle = pipe.answer_query_stream("What is the rate?")
    assert "".join(handle)  # the stream still yields something safe
    assert "unavailable" in handle.final.text
    assert handle.final.citations == []
