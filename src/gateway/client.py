"""LLM backend for the LangGraph agent.

When ``OPENROUTER_API_KEY`` is set, all chat traffic goes through OpenRouter
(OpenAI-compatible), which performs ordered model fallback via the ``models``
request field and exact-match response caching via the ``X-OpenRouter-Cache``
headers. Otherwise the agent calls Groq directly with ``GROQ_API_KEY`` and uses
a secondary Groq key/model when configured.

``llm_client`` keeps the ``client.chat.completions.create(...)`` surface used by
the responder; ``get_langchain_llm`` / ``get_guard_llm`` provide LangChain
Runnables for the planner and NeMo guardrails.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from groq import Groq
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


def _openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
    if settings.OPENROUTER_APP_NAME:
        headers["X-Title"] = settings.OPENROUTER_APP_NAME
    if settings.OPENROUTER_CACHE_ENABLED:
        # Exact-match response cache at the OpenRouter layer. Hits are free.
        headers["X-OpenRouter-Cache"] = "true"
        headers["X-OpenRouter-Cache-TTL"] = str(settings.OPENROUTER_CACHE_TTL)
    return headers


def _openrouter_models() -> list[str]:
    models = [settings.OPENROUTER_MODEL]
    fallback = settings.OPENROUTER_FALLBACK_MODEL
    if fallback and fallback != settings.OPENROUTER_MODEL:
        models.append(fallback)
    return models


def _openrouter_model(model: str, *, fallback: bool = False) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": settings.OPENROUTER_API_KEY,
        "base_url": settings.OPENROUTER_BASE_URL,
        "temperature": 0,
        "max_retries": 2,
        "default_headers": _openrouter_headers(),
    }
    if fallback:
        kwargs["extra_body"] = {
            "models": _openrouter_models(),
            "route": "fallback",
        }
    return ChatOpenAI(**kwargs)


def _groq_model(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0,
        max_retries=2,
    )


def get_langchain_llm(feature: str = "rag") -> Runnable:
    """Planner/responder model with gateway- or key-level fallback.

    ``feature`` is kept for call-site compatibility; LangSmith traces already
    carry the node name.
    """
    del feature
    if settings.OPENROUTER_API_KEY:
        return _openrouter_model(settings.OPENROUTER_MODEL, fallback=True)

    primary = _groq_model(settings.GROQ_API_KEY, settings.GROQ_MODEL)
    if settings.GROQ_FALLBACK_API_KEY:
        fallback = _groq_model(
            settings.GROQ_FALLBACK_API_KEY, settings.GROQ_FALLBACK_MODEL
        )
        return primary.with_fallbacks([fallback])
    return primary


def get_guard_llm() -> Runnable:
    """Small model for NeMo guardrail intent classification.

    NeMo's async adapter hangs with the OpenRouter ``ChatOpenAI`` path in
    practice, so prefer a direct Groq model for the gate when that key exists.
    """
    if settings.GROQ_API_KEY:
        return _groq_model(settings.GROQ_API_KEY, settings.GROQ_GUARD_MODEL)
    if settings.OPENROUTER_API_KEY:
        return _openrouter_model(settings.OPENROUTER_GUARD_MODEL)
    raise RuntimeError(
        "Set GROQ_API_KEY or OPENROUTER_API_KEY to run the guardrails."
    )


class _Completions:
    def __init__(self, owner: "OpenRouterLLMClient") -> None:
        self._owner = owner

    def create(
        self,
        *,
        messages: list[dict],
        temperature: float = 0.1,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        return self._owner._create(messages, temperature, model, **kwargs)


class _Chat:
    def __init__(self, owner: "OpenRouterLLMClient") -> None:
        self.completions = _Completions(owner)


class OpenRouterLLMClient:
    """OpenAI-compatible client with OpenRouter-side model fallback."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            max_retries=2,
            default_headers=_openrouter_headers(),
        )
        self.chat = _Chat(self)

    def _create(
        self,
        messages: list[dict],
        temperature: float,
        model: Optional[str],
        **kwargs: Any,
    ):
        extra_body = {
            "models": _openrouter_models(),
            "route": "fallback",
        }
        caller_extra = kwargs.pop("extra_body", None)
        if caller_extra:
            extra_body.update(caller_extra)
        raw_response = self._client.chat.completions.with_raw_response.create(
            model=model or settings.OPENROUTER_MODEL,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
            **kwargs,
        )
        cache_status = raw_response.headers.get("X-OpenRouter-Cache-Status")
        if cache_status:
            logger.info("OpenRouter response cache: %s", cache_status)
        return raw_response.parse()


class _GroqCompletions:
    def __init__(self, owner: "GroqLLMClient") -> None:
        self._owner = owner

    def create(
        self,
        *,
        messages: list[dict],
        temperature: float = 0.1,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        return self._owner._create(messages, temperature, model, **kwargs)


class _GroqChat:
    def __init__(self, owner: "GroqLLMClient") -> None:
        self.completions = _GroqCompletions(owner)


class GroqLLMClient:
    """``client.chat.completions.create`` with a default model and fallback."""

    def __init__(self) -> None:
        self._primary = Groq(api_key=settings.GROQ_API_KEY, max_retries=2)
        self._fallback = (
            Groq(api_key=settings.GROQ_FALLBACK_API_KEY, max_retries=2)
            if settings.GROQ_FALLBACK_API_KEY
            else None
        )
        self.chat = _GroqChat(self)

    def _create(
        self,
        messages: list[dict],
        temperature: float,
        model: Optional[str],
        **kwargs: Any,
    ):
        target = model or settings.GROQ_MODEL
        try:
            return self._primary.chat.completions.create(
                model=target,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        except Exception:  # noqa: BLE001 - fall back to the secondary key/model
            if self._fallback is None:
                raise
            logger.warning(
                "Primary Groq model %s failed; trying %s",
                target,
                settings.GROQ_FALLBACK_MODEL,
                exc_info=True,
            )
            return self._fallback.chat.completions.create(
                model=settings.GROQ_FALLBACK_MODEL,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )


def _build_client():
    """Pick a provider client, or explain what to configure."""
    if settings.OPENROUTER_API_KEY:
        return OpenRouterLLMClient()
    if settings.GROQ_API_KEY:
        return GroqLLMClient()
    raise RuntimeError(
        "Set OPENROUTER_API_KEY (gateway) or GROQ_API_KEY (direct) "
        "to run the agent backend."
    )


class _LazyLLMClient:
    """Proxy that builds the provider client on first use.

    Constructing eagerly at import time meant importing this module required a
    key, which made the whole agent path un-importable in environments that
    only need part of it (tests, the API's health route, guardrail-only paths).
    The error is still raised, but only when a completion is actually attempted.
    """

    def __init__(self) -> None:
        self._client = None

    def _resolve(self):
        if self._client is None:
            self._client = _build_client()
        return self._client

    def __getattr__(self, item: str):
        return getattr(self._resolve(), item)


llm_client = _LazyLLMClient()
