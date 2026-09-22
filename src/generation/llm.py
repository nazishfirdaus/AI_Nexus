"""LLM provider adapters + sequential fallback handler.

Design rules:
  * One adapter per provider, all returning the same LLMResponse.
  * Only ScrubbedText is accepted for query/context, so unredacted content
    cannot reach a cloud provider by accident. presidio_scrubber must return it.
  * SIMULATE_LLM=1 swaps in an offline adapter for tests / development.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from config import settings
from src.routing.quota_tracker import QuotaTracker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer ONLY from the provided "
    "context. If the answer is not in the context, say you cannot find it. Be concise. "
    "Some values are redacted placeholders like <PERSON>; never guess them."
)


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
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


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

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Return plain answer text or raise ProviderError."""

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

    def generate(self, query: str, context: str) -> str:
        data = self._post(
            self.base_url,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(query, context)},
                ],
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

    def generate(self, query: str, context: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = self._post(
            url,
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": _build_user_prompt(query, context)}]}],
                "generationConfig": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "maxOutputTokens": settings.LLM_MAX_TOKENS,
                },
            },
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

    def generate(self, query: str, context: str) -> str:
        return f"[SIMULATED] Answer to '{query[:80]}' based on {len(context)} chars of context."


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
        self, query: ScrubbedText, context: ScrubbedText, provider_order: list[str]
    ) -> LLMResponse:
        if not isinstance(query, ScrubbedText) or not isinstance(context, ScrubbedText):
            raise TypeError("LLMHandler only accepts ScrubbedText for query and context.")

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
                text = adapter.generate(str(query), str(context))
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