"""LLM provider adapters + sequential fallback handler.

Design rules:
  * One adapter per provider, all returning the same LLMResponse.
  * Only ScrubbedText is accepted for query/context/history/summary/messages,
    so unredacted content cannot reach a cloud provider by accident.
    presidio_scrubber must return it.
  * Adapters implement chat(messages); BaseAdapter.generate assembles the
    system + history + question message list for QA calls.
  * SIMULATE_LLM=1 swaps in an offline adapter for tests / development.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC
from dataclasses import dataclass

import requests

from config import settings
from config.prompts import USER_PROMPT, build_system_prompt
from src.routing.quota_tracker import QuotaTracker

logger = logging.getLogger(__name__)


class ScrubbedText(str):
    """Marker type: text that has already passed through the Presidio scrubber."""


class ProviderError(Exception):
    def __init__(self, message: str, provider: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class AllProvidersFailed(Exception):
    def __init__(self, errors: dict[str, str]):
        super().__init__("All LLM providers failed: " + "; ".join(f"{p}: {e}" for p, e in errors.items()))
        self.errors = errors


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    fallbacks_tried: list[str]


def _build_user_prompt(query: str, context: str) -> str:
    return USER_PROMPT.format(question=query, context=context)


def _build_chat_messages(
    query: str,
    context: str,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> list[dict]:
    """Full message list for one generation: system (+summary), prior turns, question."""
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(summary)}]
    for turn in history or []:
        content = turn.get("content")
        if content:
            messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": _build_user_prompt(query, context)})
    return messages


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class BaseAdapter(ABC):
    name: str
    supports_streaming: bool = False

    def __init__(self, model: str, api_key: str | None, timeout: float = settings.LLM_TIMEOUT_S):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        query: str,
        context: str,
        history: list[dict] | None = None,
        summary: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Build the full message list and send it via chat()."""
        return self.chat(
            _build_chat_messages(query, context, history, summary),
            temperature=temperature,
        )

    def chat_stream(self, messages: list[dict], temperature: float | None = None):
        raise NotImplementedError(f"{type(self).__name__} does not implement chat_stream()")

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.Timeout as e:
            raise ProviderError(f"timeout after {self.timeout}s", self.name) from e
        except requests.RequestException as e:
            raise ProviderError(f"network error: {e}", self.name) from e
        if r.status_code != 200:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:300]}", self.name, r.status_code)
        try:
            return r.json()
        except ValueError as e:
            raise ProviderError("invalid JSON in response", self.name) from e

    def _post_stream(self, url: str, headers: dict, payload: dict):
        """POST with stream=True; yield each SSE `data:` payload as a string.

        Stops at the `[DONE]` sentinel. Raises the same ProviderError family as
        `_post` so the streaming fallback chain can reuse the existing handling.
        """
        try:
            r = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout, stream=True
            )
        except requests.Timeout as e:
            raise ProviderError(f"timeout after {self.timeout}s", self.name) from e
        except requests.RequestException as e:
            raise ProviderError(f"network error: {e}", self.name) from e
        if r.status_code != 200:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:300]}", self.name, r.status_code)
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == "[DONE]":
                break
            if chunk:
                yield chunk


class OpenAICompatibleAdapter(BaseAdapter):
    """Works for any OpenAI-style chat completions API (OpenAI, Groq, ...)."""

    base_url: str
    supports_streaming = True

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        data = self._post(
            self.base_url,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "messages": messages,
                "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
                "max_tokens": settings.LLM_MAX_TOKENS,
            },
        )
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}", self.name) from e

    def chat_stream(self, messages: list[dict], temperature: float | None = None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": True,
        }
        for raw in self._post_stream(
            self.base_url,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload,
        ):
            try:
                data = json.loads(raw)
            except Exception:
                continue
            try:
                content = data["choices"][0]["delta"]["content"]
                if content:
                    yield content
            except (KeyError, IndexError, TypeError):
                continue


class GroqAdapter(OpenAICompatibleAdapter):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1/chat/completions"


class NvidiaAdapter(OpenAICompatibleAdapter):
    """NVIDIA NIM hosted endpoint (OpenAI-compatible)."""

    name = "nvidia"
    base_url = "https://integrate.api.nvidia.com/v1/chat/completions"


class GeminiAdapter(BaseAdapter):
    name = "gemini_flash"
    supports_streaming = True

    def _build_payload(self, messages: list[dict], temperature: float | None) -> dict:
        system_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
        contents = [
            # Gemini alternates "user" / "model" roles.
            {"role": "model" if m.get("role") == "assistant" else "user",
             "parts": [{"text": str(m.get("content", ""))}]}
            for m in messages
            if m.get("role") != "system"
        ]
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
                "maxOutputTokens": settings.LLM_MAX_TOKENS,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return payload

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = self._post(
            url,
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            self._build_payload(messages, temperature),
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as e:
            # Safety blocks / empty candidates land here.
            raise ProviderError(f"no usable candidate: {str(data)[:200]}", self.name) from e

    def chat_stream(self, messages: list[dict], temperature: float | None = None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        for raw in self._post_stream(
            url,
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            self._build_payload(messages, temperature),
        ):
            try:
                data = json.loads(raw)
            except Exception:
                continue
            try:
                parts = data["candidates"][0]["content"]["parts"]
                for p in parts:
                    t = p.get("text")
                    if t:
                        yield t
            except (KeyError, IndexError, TypeError):
                continue


class SimulatedAdapter(BaseAdapter):
    """Offline, deterministic adapter for tests (SIMULATE_LLM=1)."""

    name = "simulated"
    supports_streaming = True

    def __init__(self) -> None:
        super().__init__(model="simulated-1", api_key="simulated")

    def _answer_text(self, messages: list[dict]) -> str:
        user = ""
        for m in messages:
            if m.get("role") == "user":
                user = str(m.get("content", ""))
        # Split the standard USER_PROMPT so the output matches the old
        # generate(query, context) form; anything else is used as-is.
        question, context = user, user
        if user.startswith("Question: "):
            rest = user[len("Question: "):]
            if "\n\nRetrieved context:\n" in rest:
                question, context = rest.split("\n\nRetrieved context:\n", 1)
        # Drop the <context> fence (and the newlines around it) before measuring,
        # so the reported size reflects the actual context, not the delimiters.
        context = context.strip()
        if context.startswith("<context>"):
            context = context[len("<context>"):].strip()
        if context.endswith("</context>"):
            context = context[: -len("</context>")].strip()
        return f"[SIMULATED] Answer to '{question[:80]}' based on {len(context)} chars of context."

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        return self._answer_text(messages)

    def chat_stream(self, messages: list[dict], temperature: float | None = None):
        text = self._answer_text(messages)
        n = 32
        for i in range(0, len(text), n):
            yield text[i:i + n]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
class LLMHandler:
    def __init__(
        self,
        adapters: dict[str, BaseAdapter],
        quota_tracker: QuotaTracker | None = None,
    ) -> None:
        self.adapters = adapters
        self.quota = quota_tracker or QuotaTracker()

    @property
    def available_providers(self) -> list[str]:
        return [n for n, a in self.adapters.items() if a.is_configured()]

    def generate(
        self,
        query: ScrubbedText,
        context: ScrubbedText,
        provider_order: list[str],
        history: list[dict] | None = None,
        summary: ScrubbedText | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not isinstance(query, ScrubbedText) or not isinstance(context, ScrubbedText):
            raise TypeError("LLMHandler only accepts ScrubbedText for query and context.")
        if history:
            for turn in history:
                if not isinstance(turn.get("content"), ScrubbedText):
                    raise TypeError("LLMHandler only accepts ScrubbedText history content.")
        if summary is not None and not isinstance(summary, ScrubbedText):
            raise TypeError("LLMHandler only accepts ScrubbedText for summary.")

        if history or summary:
            return self._run(
                provider_order,
                lambda adapter: adapter.generate(
                    str(query), str(context), history=history, summary=summary,
                    temperature=temperature,
                ),
            )
        return self._run(
            provider_order,
            lambda adapter: adapter.generate(
                str(query), str(context), temperature=temperature
            ),
        )

    def generate_messages(
        self,
        messages: list[dict],
        provider_order: list[str],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send a pre-built message list (rewrite / summarization calls).

        Every content must already be ScrubbedText - this path exists for the
        non-QA prompts and inherits the same PII guarantee.
        """
        if not messages or not isinstance(messages, list):
            raise TypeError("generate_messages requires a non-empty list of messages.")
        for message in messages:
            if message.get("role") not in ("system", "user", "assistant"):
                raise TypeError(f"invalid message role: {message.get('role')!r}")
            if not isinstance(message.get("content"), ScrubbedText):
                raise TypeError("LLMHandler only accepts ScrubbedText message content.")
        return self._run(
            provider_order,
            lambda adapter: adapter.chat(messages, temperature=temperature),
        )

    def generate_stream(
        self,
        query: ScrubbedText,
        context: ScrubbedText,
        provider_order: list[str],
        history: list[dict] | None = None,
        summary: ScrubbedText | None = None,
        temperature: float | None = None,
    ) -> "LLMStream":
        """Streaming twin of generate(): same ScrubbedText gate, streamed chunks.

        Returns an LLMStream (iterable of text chunks) that exposes
        provider / model / latency_s / fallbacks_tried once iteration completes.
        """
        if not isinstance(query, ScrubbedText) or not isinstance(context, ScrubbedText):
            raise TypeError("LLMHandler only accepts ScrubbedText for query and context.")
        if history:
            for turn in history:
                if not isinstance(turn.get("content"), ScrubbedText):
                    raise TypeError("LLMHandler only accepts ScrubbedText history content.")
        if summary is not None and not isinstance(summary, ScrubbedText):
            raise TypeError("LLMHandler only accepts ScrubbedText for summary.")

        messages = _build_chat_messages(str(query), str(context), history, summary)
        return LLMStream(self, messages, provider_order, temperature)

    def _run(self, provider_order: list[str], invoke) -> LLMResponse:
        """Sequential fallback over `provider_order`; `invoke(adapter)` -> text."""
        errors: dict[str, str] = {}
        tried: list[str] = []

        for name in provider_order:
            adapter = self.adapters.get(name)
            if adapter is None or not adapter.is_configured():
                errors[name] = "not configured"
                continue
            # Never retry a known-dead provider during its cooldown.
            if not self.quota.is_available(name):
                errors[name] = f"cooling down ({self.quota.seconds_remaining(name):.0f}s left)"
                continue

            tried.append(name)
            start = time.perf_counter()
            try:
                text = invoke(adapter)
                if not text:
                    raise ProviderError("empty response", name)
            except ProviderError as e:
                self.quota.record_failure(name, e)
                errors[name] = str(e)
                logger.warning("Provider %s failed, trying next: %s", name, e)
                continue
            except Exception as e:  # never let an adapter bug crash the chain
                self.quota.record_failure(name, e)
                errors[name] = f"unexpected: {e}"
                continue

            self.quota.record_success(name)
            return LLMResponse(
                text=text,
                provider=name,
                model=adapter.model,
                latency_s=time.perf_counter() - start,
                fallbacks_tried=tried[:-1],
            )

        raise AllProvidersFailed(errors)


class LLMStream:
    """Iterator of streamed answer chunks with fallback-after-first-token.

    Fallback only happens before the first token: a provider that fails before
    emitting anything falls through to the next one. Once a token has been
    yielded the stream is committed to that provider - a mid-stream error stops
    the stream (logged + quota-failed) rather than splicing two providers'
    text together. After exhaustion, ``provider`` / ``model`` / ``latency_s`` /
    ``fallbacks_tried`` describe whichever provider actually answered.
    """

    def __init__(self, handler: "LLMHandler", messages: list[dict],
                 provider_order: list[str], temperature: float | None) -> None:
        self._handler = handler
        self._messages = messages
        self._provider_order = provider_order
        self._temperature = temperature
        self.provider = ""
        self.model = ""
        self.latency_s = 0.0
        self.fallbacks_tried: list[str] = []
        self.text = ""

    def __iter__(self):
        errors: dict[str, str] = {}
        tried: list[str] = []

        for name in self._provider_order:
            adapter = self._handler.adapters.get(name)
            if adapter is None or not adapter.is_configured():
                errors[name] = "not configured"
                continue
            if not self._handler.quota.is_available(name):
                errors[name] = (
                    f"cooling down ({self._handler.quota.seconds_remaining(name):.0f}s left)"
                )
                continue
            if not getattr(adapter, "supports_streaming", False):
                errors[name] = "streaming unsupported"
                continue

            tried.append(name)
            start = time.perf_counter()
            emitted = False
            try:
                for chunk in adapter.chat_stream(
                    self._messages, temperature=self._temperature
                ):
                    if not chunk:
                        continue
                    emitted = True
                    self.provider = name
                    self.model = adapter.model
                    self.text += chunk
                    yield chunk
            except ProviderError as e:
                self._handler.quota.record_failure(name, e)
                errors[name] = str(e)
                if emitted:
                    logger.warning(
                        "Stream from %s failed after the first token; stopping. %s", name, e
                    )
                    self.latency_s = time.perf_counter() - start
                    self.fallbacks_tried = tried[:-1]
                    return
                logger.warning("Provider %s failed, trying next: %s", name, e)
                continue
            except Exception as e:  # never let an adapter bug crash the chain
                self._handler.quota.record_failure(name, e)
                errors[name] = f"unexpected: {e}"
                if emitted:
                    logger.warning(
                        "Stream from %s failed after the first token; stopping. %s", name, e
                    )
                    self.latency_s = time.perf_counter() - start
                    self.fallbacks_tried = tried[:-1]
                    return
                continue

            self._handler.quota.record_success(name)
            self.latency_s = time.perf_counter() - start
            self.fallbacks_tried = tried[:-1]
            return

        raise AllProvidersFailed(errors)


# Provider -> (credential dict field, environment variable).
_PROVIDER_KEY_FIELDS = {
    "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY"),
    "gemini_flash": ("gemini_api_key", "GEMINI_API_KEY"),
    "groq": ("groq_api_key", "GROQ_API_KEY"),
}


def build_handler(
    api_keys: dict[str, str | None],
    quota_tracker: QuotaTracker | None = None,
) -> LLMHandler:
    """Construct the handler from explicit per-provider keys (provider -> key)."""
    tracker = quota_tracker or QuotaTracker()
    if settings.SIMULATE_LLM:
        return LLMHandler({"simulated": SimulatedAdapter()}, tracker)

    models = settings.PROVIDER_MODELS
    return LLMHandler(
        {
            "nvidia": NvidiaAdapter(models["nvidia"], api_keys.get("nvidia")),
            "gemini_flash": GeminiAdapter(models["gemini_flash"], api_keys.get("gemini_flash")),
            "groq": GroqAdapter(models["groq"], api_keys.get("groq")),
        },
        tracker,
    )


def build_default_handler(quota_tracker: QuotaTracker | None = None) -> LLMHandler:
    """Construct the handler from environment variables (see .env.example)."""
    keys = {provider: os.getenv(env) for provider, (_, env) in _PROVIDER_KEY_FIELDS.items()}
    return build_handler(keys, quota_tracker)


def build_handler_from_credentials(
    credentials: dict | None,
    quota_tracker: QuotaTracker | None = None,
) -> LLMHandler:
    """Build the handler from a user's stored credentials.

    When the user has supplied at least one provider key, ONLY those keys are
    used - the deployment's environment keys are never spent on a user's behalf.
    With no user keys, the handler falls back to the environment defaults so a
    freshly-onboarded account still works.
    """
    creds = credentials or {}
    user_keys = {
        provider: creds.get(field)
        for provider, (field, _env) in _PROVIDER_KEY_FIELDS.items()
    }
    if any(user_keys.values()):
        return build_handler(user_keys, quota_tracker)
    return build_default_handler(quota_tracker)