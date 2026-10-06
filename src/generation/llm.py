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
    ) -> str:
        """Build the full message list and send it via chat()."""
        return self.chat(_build_chat_messages(query, context, history, summary))

    def chat(self, messages: list[dict]) -> str:
        """Send a ready-made [{role, content}, ...] list; return answer text.

        Subclasses implement this (or override generate directly). Raising
        ProviderError marks the attempt as a failure for the fallback chain.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement chat()")

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


class OpenAICompatibleAdapter(BaseAdapter):
    """Works for any OpenAI-style chat completions API (OpenAI, Groq, ...)."""

    base_url: str

    def chat(self, messages: list[dict]) -> str:
        data = self._post(
            self.base_url,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "messages": messages,
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS,
            },
        )
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ProviderError(f"unexpected response shape: {str(data)[:200]}", self.name) from e


class GroqAdapter(OpenAICompatibleAdapter):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1/chat/completions"


class NvidiaAdapter(OpenAICompatibleAdapter):
    """NVIDIA NIM hosted endpoint (OpenAI-compatible)."""

    name = "nvidia"
    base_url = "https://integrate.api.nvidia.com/v1/chat/completions"


class GeminiAdapter(BaseAdapter):
    name = "gemini_flash"

    def chat(self, messages: list[dict]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
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
                "temperature": settings.LLM_TEMPERATURE,
                "maxOutputTokens": settings.LLM_MAX_TOKENS,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        data = self._post(
            url,
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            payload,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as e:
            # Safety blocks / empty candidates land here.
            raise ProviderError(f"no usable candidate: {str(data)[:200]}", self.name) from e


class SimulatedAdapter(BaseAdapter):
    """Offline, deterministic adapter for tests (SIMULATE_LLM=1)."""

    name = "simulated"

    def __init__(self) -> None:
        super().__init__(model="simulated-1", api_key="simulated")

    def chat(self, messages: list[dict]) -> str:
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
        return f"[SIMULATED] Answer to '{question[:80]}' based on {len(context)} chars of context."


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
                    str(query), str(context), history=history, summary=summary
                ),
            )
        return self._run(provider_order, lambda adapter: adapter.generate(str(query), str(context)))

    def generate_messages(
        self, messages: list[dict], provider_order: list[str]
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
        return self._run(provider_order, lambda adapter: adapter.chat(messages))

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


def build_default_handler(quota_tracker: QuotaTracker | None = None) -> LLMHandler:
    """Construct the handler from environment variables (see .env.example)."""
    tracker = quota_tracker or QuotaTracker()
    if settings.SIMULATE_LLM:
        return LLMHandler({"simulated": SimulatedAdapter()}, tracker)

    models = settings.PROVIDER_MODELS
    return LLMHandler(
        {
            "nvidia": NvidiaAdapter(models["nvidia"], os.getenv("NVIDIA_API_KEY")),
            "gemini_flash": GeminiAdapter(models["gemini_flash"], os.getenv("GEMINI_API_KEY")),
            "groq": GroqAdapter(models["groq"], os.getenv("GROQ_API_KEY")),
        },
        tracker,
    )