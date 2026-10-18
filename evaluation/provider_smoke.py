"""Preflight smoke test: one tiny real call per configured provider.

Validates API keys and model IDs *before* a long eval run. Loads ``.env``
(shell env wins) and calls each configured adapter directly, bypassing the
fallback chain, so a failure is attributed to the provider it belongs to.

    python -m evaluation.provider_smoke

Exit code is non-zero if no provider is configured or every call failed.
"""
from __future__ import annotations

try:  # load .env before settings reads the environment
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

import sys
import time

from config import settings
from src.generation.llm import build_default_handler
from src.routing.quota_tracker import QuotaTracker

_PROBE = [{"role": "user", "content": "Reply with the single word: ok"}]


def main() -> int:
    if settings.SIMULATE_LLM:
        print("[smoke] SIMULATE_LLM is on; unset it to test real providers.")
        return 1

    handler = build_default_handler(QuotaTracker(state_path=None))
    configured = [name for name, a in handler.adapters.items() if a.is_configured()]
    print(f"[smoke] configured providers: {configured or 'none'}")
    if not configured:
        print("[smoke] no API keys found in the environment (.env not loaded?)")
        return 1

    failures = 0
    for name in configured:
        adapter = handler.adapters[name]
        start = time.perf_counter()
        try:
            text = adapter.chat(_PROBE)
            elapsed = time.perf_counter() - start
            preview = (text or "").strip().replace("\n", " ")[:60]
            print(f"[smoke] {name:<13} model={adapter.model!r} OK {elapsed:.1f}s -> {preview!r}")
        except Exception as exc:  # noqa: BLE001 - surface any provider error
            elapsed = time.perf_counter() - start
            failures += 1
            print(f"[smoke] {name:<13} model={adapter.model!r} FAIL {elapsed:.1f}s: {exc}")

    ok = len(configured) - failures
    print(f"[smoke] {ok}/{len(configured)} providers responded")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
