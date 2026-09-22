"""Local, simple provider health tracking: cooldowns after failures, longer after rate limits."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable

from config import settings

logger = logging.getLogger(__name__)

_QUOTA_MARKERS = (
    "429", "rate limit", "rate_limit", "quota", "resource_exhausted",
    "too many requests", "insufficient_quota", "exceeded your current",
)


def is_quota_error(exc: BaseException) -> bool:
    """True if the failure looks like a rate-limit / quota problem."""
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return any(m in text for m in _QUOTA_MARKERS)


class QuotaTracker:
    def __init__(
        self,
        state_path: str | None = settings.QUOTA_STATE_PATH,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = state_path
        self._clock = clock
        self._lock = threading.Lock()
        # provider -> {"until": epoch, "failures": int}
        self._state: dict[str, dict] = self._load()

    # -- public API ----------------------------------------------------------
    def is_available(self, provider: str) -> bool:
        return self.seconds_remaining(provider) <= 0

    def seconds_remaining(self, provider: str) -> float:
        with self._lock:
            until = self._state.get(provider, {}).get("until", 0.0)
        return max(0.0, until - self._clock())

    def record_failure(self, provider: str, exc: BaseException) -> float:
        """Record a failure and return the cooldown (seconds) applied."""
        base = (
            settings.RATE_LIMIT_COOLDOWN_S if is_quota_error(exc) else settings.ERROR_COOLDOWN_S
        )
        with self._lock:
            entry = self._state.setdefault(provider, {"until": 0.0, "failures": 0})
            entry["failures"] += 1
            cooldown = min(base * 2 ** (entry["failures"] - 1), settings.MAX_COOLDOWN_S)
            entry["until"] = self._clock() + cooldown
            self._save()
        logger.warning("Provider %s cooling down for %.0fs (%s)", provider, cooldown, exc)
        return cooldown

    def record_success(self, provider: str) -> None:
        with self._lock:
            if provider in self._state:
                self._state[provider] = {"until": 0.0, "failures": 0}
                self._save()

    def status(self) -> dict[str, float]:
        return {p: self.seconds_remaining(p) for p in list(self._state)}

    # -- persistence (best effort, must never break a query) -----------------
    def _load(self) -> dict:
        if not self._path or not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        return self._sanitize(data)

    @staticmethod
    def _sanitize(data) -> dict:
        """Keep only well-formed entries so a corrupt/foreign file can never crash a query."""
        clean: dict[str, dict] = {}
        if not isinstance(data, dict):
            return clean
        for provider, entry in data.items():
            if not isinstance(entry, dict):
                continue
            try:
                clean[str(provider)] = {
                    "until": float(entry.get("until", 0.0)),
                    "failures": int(entry.get("failures", 0)),
                }
            except (TypeError, ValueError):
                continue
        return clean

    def _save(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._state, f)
        except Exception:
            logger.debug("Could not persist quota state", exc_info=True)