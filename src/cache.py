"""Response cache for RAG answers and LLM generations.

Two levels, both optional and safe to use in any worker model:

* **Memory** (always on when caching is enabled): a bounded TTL + LRU cache
  local to the process. No extra dependency; this is enough for a single
  Uvicorn worker and for tests.
* **Redis** (opt-in via ``ANEXUS_REDIS_URL``): a shared L2 so several workers
  or Cloud Run instances see each other's entries. ``redis`` is imported
  lazily; if the package is missing or the server is unreachable the cache
  logs once and continues with memory only.

Keys are SHA-256 digests of the supplied parts, so prompts, questions and
answers never appear in keys. The namespace (tenant id) is part of the key, so
one tenant can never read another tenant's entry. Values are JSON, which keeps
the cache inspectable and avoids pickle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "anexus:response-cache:"


class _MemoryBackend:
    """Small thread-safe TTL + LRU store holding JSON strings."""

    def __init__(
        self,
        max_entries: int,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._now = now
        self._items: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, raw = item
            if expires_at <= self._now():
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return raw

    def set(self, key: str, raw: str, ttl_s: float) -> None:
        with self._lock:
            self._items[key] = (self._now() + max(1.0, ttl_s), raw)
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _connect_redis(url: str):
    """Return a connected client, or None when redis is unavailable.

    Redis is optional: we never make it a hard startup requirement.
    """
    try:
        import redis  # type: ignore[import-not-found]
    except Exception:
        logger.info(
            "Response cache: redis package not installed; using in-process memory only"
        )
        return None

    try:
        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client.ping()
        logger.info("Response cache: Redis L2 enabled")
        return client
    except Exception as exc:  # noqa: BLE001 - cache must never break a request
        logger.warning(
            "Response cache: Redis unavailable (%s); using in-process memory only",
            type(exc).__name__,
        )
        return None


class ResponseCache:
    """Namespaced TTL cache with a memory L1 and an optional Redis L2."""

    def __init__(
        self,
        namespace: str = "default",
        *,
        ttl_s: int = 300,
        max_entries: int = 1024,
        redis_url: str = "",
        enabled: bool = True,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.namespace = str(namespace or "default")
        self.ttl_s = max(1, int(ttl_s))
        self.enabled = bool(enabled)
        self._memory = _MemoryBackend(max_entries, now=now)
        self._redis = _connect_redis(redis_url) if self.enabled and redis_url else None
        self._stats_lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "redis_hits": 0, "errors": 0}

    # ------------------------------------------------------------------ keys
    def make_key(self, *parts: Any) -> str:
        """Deterministic, tenant-scoped key from arbitrary JSON-ish parts."""
        try:
            payload = json.dumps(
                parts, ensure_ascii=False, sort_keys=True, default=str
            )
        except (TypeError, ValueError):
            payload = repr(parts)
        digest = hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()
        return f"{self.namespace}:{digest}"

    # --------------------------------------------------------------- public
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        raw = self._memory.get(key)
        if raw is None and self._redis is not None:
            raw = self._redis_get(key)
            if raw is not None:
                self._memory.set(key, raw, self.ttl_s)

        if raw is None:
            self._bump("misses")
            return None

        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            self._bump("errors")
            logger.warning("Discarding malformed cached entry in %s", self.namespace)
            return None

        self._bump("hits")
        return value

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        if not self.enabled:
            return

        ttl = max(1, int(ttl_s if ttl_s is not None else self.ttl_s))
        try:
            raw = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            self._bump("errors")
            logger.warning("Skipping non-JSON cache value in %s", self.namespace)
            return

        self._memory.set(key, raw, ttl)
        if self._redis is not None:
            self._redis_set(key, raw, ttl)
        self._bump("sets")

    def delete(self, key: str) -> None:
        self._memory.delete(key)
        if self._redis is not None:
            try:
                self._redis.delete(_REDIS_KEY_PREFIX + key)
            except Exception:  # noqa: BLE001 - cache must never break a request
                self._disable_redis()

    def clear(self) -> None:
        """Clear the local L1. Redis entries expire naturally via their TTL."""
        self._memory.clear()

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            data = dict(self._stats)
        data["entries"] = len(self._memory)
        return data

    # --------------------------------------------------------------- redis
    def _redis_get(self, key: str) -> Optional[str]:
        try:
            raw = self._redis.get(_REDIS_KEY_PREFIX + key)
            if raw is not None:
                self._bump("redis_hits")
            return raw
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Redis cache read failed (%s); disabling Redis for this process",
                type(exc).__name__,
            )
            self._disable_redis()
            return None

    def _redis_set(self, key: str, raw: str, ttl: int) -> None:
        try:
            self._redis.setex(_REDIS_KEY_PREFIX + key, ttl, raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Redis cache write failed (%s); disabling Redis for this process",
                type(exc).__name__,
            )
            self._disable_redis()

    def _disable_redis(self) -> None:
        self._redis = None

    def _bump(self, name: str) -> None:
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + 1


# ---------------------------------------------------------------------------
# Registry: one cache per tenant namespace for the life of the process.
# ---------------------------------------------------------------------------
_CACHES: dict[str, ResponseCache] = {}
_CACHES_LOCK = threading.Lock()


def get_cache(namespace: str = "default") -> ResponseCache:
    """Return the process-wide cache for ``namespace``, creating it from settings."""
    from config import settings

    with _CACHES_LOCK:
        cache = _CACHES.get(namespace)
        if cache is None:
            cache = ResponseCache(
                namespace,
                ttl_s=settings.CACHE_TTL_S,
                max_entries=settings.CACHE_MAX_ENTRIES,
                redis_url=settings.CACHE_REDIS_URL,
                enabled=settings.CACHE_ENABLED,
            )
            _CACHES[namespace] = cache
        return cache


def reset_response_caches() -> None:
    """Drop every registered cache (tests, config reloads, tenant deletion)."""
    with _CACHES_LOCK:
        caches = list(_CACHES.values())
        _CACHES.clear()
    for cache in caches:
        cache.clear()
