"""Tests for the response cache (memory L1, optional Redis L2)."""
from __future__ import annotations

from src.cache import ResponseCache, get_cache, reset_response_caches


def _clock():
    now = [0.0]
    return now, lambda: now[0]


def test_namespaced_keys_do_not_collide():
    a = ResponseCache("tenant-a", ttl_s=60, max_entries=8)
    b = ResponseCache("tenant-b", ttl_s=60, max_entries=8)
    key = a.make_key("same", "parts")
    a.set(key, {"answer": "a"})

    assert a.get(key) == {"answer": "a"}
    assert b.get(key) is None


def test_ttl_expiry():
    now, clock = _clock()
    cache = ResponseCache("ttl", ttl_s=10, max_entries=8, now=clock)
    key = cache.make_key("q")
    cache.set(key, "value")

    now[0] = 9.9
    assert cache.get(key) == "value"
    now[0] = 10.1
    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1


def test_lru_eviction_is_bounded():
    cache = ResponseCache("lru", ttl_s=60, max_entries=2)
    for name in ("a", "b", "c"):
        cache.set(cache.make_key(name), name)

    assert cache.get(cache.make_key("a")) is None  # evicted first
    assert cache.get(cache.make_key("b")) == "b"
    assert cache.get(cache.make_key("c")) == "c"
    assert cache.stats()["entries"] == 2


def test_values_round_trip_through_json():
    cache = ResponseCache("json", ttl_s=60, max_entries=8)
    key = cache.make_key("answer")
    value = {
        "text": "7.25 percent",
        "citations": [{"page_number": 1, "score": 0.9, "snippet": "rate"}],
        "pii_redactions": {"IN_ACCOUNT_NO": 1},
    }
    cache.set(key, value)
    assert cache.get(key) == value


def test_disabled_cache_is_a_noop():
    cache = ResponseCache("off", enabled=False)
    key = cache.make_key("q")
    cache.set(key, "value")
    assert cache.get(key) is None
    assert cache.stats()["sets"] == 0


def test_redis_unavailable_falls_back_to_memory():
    # Port 1 is closed; the constructor must degrade to memory, not raise.
    cache = ResponseCache(
        "redis-fallback", ttl_s=60, max_entries=8, redis_url="redis://127.0.0.1:1/0"
    )
    key = cache.make_key("q")
    cache.set(key, "value")
    assert cache.get(key) == "value"


def test_registry_is_per_namespace_and_resettable():
    reset_response_caches()
    one = get_cache("one")
    two = get_cache("two")
    assert one is not two
    assert get_cache("one") is one

    reset_response_caches()
    assert get_cache("one") is not one
