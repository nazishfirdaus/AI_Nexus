"""Token / cost accounting for real-provider evaluation runs.

Every successful ``LLMResponse`` carries optional ``TokenUsage`` reported by the
provider (see ``src/generation/llm.py``). A ``CostMeter`` is attached to the
``LLMHandler`` as an ``on_response`` observer, so it counts **all** provider
calls in a pass - answer generation *and* the rewrite / query-expansion /
summary calls - not just the final answer.

Prices are USD per **one million** tokens and default to zero for the three
free-tier providers this project uses. Override any provider (or the fallback
``default``) with an environment variable:

    ANEXUS_PRICE_GROQ="0.59,0.79"       # prompt,completion per 1M tokens
    ANEXUS_PRICE_NVIDIA="0.20,0.60"
    ANEXUS_PRICE_DEFAULT="1.00,2.00"

Providers that do not report usage contribute calls but zero tokens, so treat
the cost as a lower bound in that case.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class _Price:
    prompt_per_million: float = 0.0
    completion_per_million: float = 0.0


@dataclass
class _ProviderTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def _price_from_env(provider: str) -> _Price:
    raw = os.getenv(f"ANEXUS_PRICE_{provider.upper()}")
    if not raw:
        return _Price()
    try:
        prompt, completion = (float(x) for x in raw.split(",", 1))
        return _Price(prompt, completion)
    except (ValueError, TypeError):
        return _Price()


def _default_price() -> _Price:
    raw = os.getenv("ANEXUS_PRICE_DEFAULT")
    if not raw:
        return _Price()
    try:
        prompt, completion = (float(x) for x in raw.split(",", 1))
        return _Price(prompt, completion)
    except (ValueError, TypeError):
        return _Price()


class CostMeter:
    """Accumulates token usage and cost across an evaluation pass."""

    def __init__(self) -> None:
        self._by_provider: dict[str, _ProviderTotals] = {}

    # -- recording ---------------------------------------------------------
    def record(self, response) -> None:
        """Observe one successful ``LLMResponse`` (duck-typed)."""
        provider = str(getattr(response, "provider", "") or "unknown")
        usage = getattr(response, "usage", None)
        totals = self._by_provider.setdefault(provider, _ProviderTotals())
        totals.calls += 1
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
        price = _price_from_env(provider)
        if price.prompt_per_million == 0.0 and price.completion_per_million == 0.0:
            price = _default_price()
        totals.prompt_tokens += prompt
        totals.completion_tokens += completion
        totals.total_tokens += total
        totals.cost_usd += (
            prompt * price.prompt_per_million + completion * price.completion_per_million
        ) / 1_000_000

    # -- reporting ---------------------------------------------------------
    @property
    def calls(self) -> int:
        return sum(t.calls for t in self._by_provider.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self._by_provider.values())

    @property
    def completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self._by_provider.values())

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self._by_provider.values())

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self._by_provider.values())

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "by_provider": {
                name: {
                    "calls": t.calls,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "total_tokens": t.total_tokens,
                    "cost_usd": round(t.cost_usd, 6),
                }
                for name, t in sorted(self._by_provider.items())
            },
        }

    def snapshot(self) -> dict:
        """Cheap copy used to compute per-pass deltas."""
        return {
            name: _ProviderTotals(
                calls=t.calls,
                prompt_tokens=t.prompt_tokens,
                completion_tokens=t.completion_tokens,
                total_tokens=t.total_tokens,
                cost_usd=t.cost_usd,
            )
            for name, t in self._by_provider.items()
        }

    def delta(self, before: dict) -> dict:
        """Usage/cost accumulated since ``before`` (a ``snapshot()``)."""
        by_provider: dict[str, dict] = {}
        for name, after in self._by_provider.items():
            prior = before.get(name, _ProviderTotals())
            calls = after.calls - prior.calls
            if calls <= 0:
                continue
            by_provider[name] = {
                "calls": calls,
                "prompt_tokens": after.prompt_tokens - prior.prompt_tokens,
                "completion_tokens": after.completion_tokens - prior.completion_tokens,
                "total_tokens": after.total_tokens - prior.total_tokens,
                "cost_usd": round(after.cost_usd - prior.cost_usd, 6),
            }
        return {
            "calls": sum(p["calls"] for p in by_provider.values()),
            "prompt_tokens": sum(p["prompt_tokens"] for p in by_provider.values()),
            "completion_tokens": sum(p["completion_tokens"] for p in by_provider.values()),
            "total_tokens": sum(p["total_tokens"] for p in by_provider.values()),
            "cost_usd": round(sum(p["cost_usd"] for p in by_provider.values()), 6),
            "by_provider": by_provider,
        }

    def exceeds(self, limit_usd: float | None) -> bool:
        return limit_usd is not None and limit_usd > 0 and self.cost_usd >= limit_usd
