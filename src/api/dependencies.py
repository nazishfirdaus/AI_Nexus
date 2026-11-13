"""FastAPI dependencies: identity, tenant-scoped pipeline and chat store.

The pipeline and the SQLite store are cached per user for the life of the
process. Heavy models inside the pipeline (embedder, cross-encoder, spaCy) are
already process-cached by their own modules, so only tenant-bound pieces are
rebuilt on first use.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request

from src.api.config import api_settings
from src.memory.chat_store import ChatStore
from src.pipeline_factory import PipelineBundle, build_user_pipeline

logger = logging.getLogger(__name__)

# Collection suffixes and directory names are built from this id, so keep it
# path-safe and Qdrant-name-safe.
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_USER_ID = os.getenv("ANEXUS_DEV_USER_ID", "dev").strip() or "dev"


def get_user_id(
    x_user_id: Annotated[Optional[str], Header(alias="X-User-Id")] = None,
) -> str:
    """Resolve the tenant.

    In production the React client (or an auth proxy) sends the provisioned
    account id in ``X-User-Id``. Locally it defaults to ``dev``, which maps to
    the ``<collection>__dev`` namespace already used by the ingestion CLI.
    """
    user_id = (x_user_id or _DEFAULT_USER_ID).strip()
    if not _USER_ID_RE.fullmatch(user_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "X-User-Id must be 1-64 characters of letters, digits, '-' or '_'."
            ),
        )
    return user_id


_pipelines: dict[str, PipelineBundle] = {}
_pipelines_guard = threading.Lock()
_pipeline_locks: dict[str, threading.Lock] = {}


def _pipeline_lock(user_id: str) -> threading.Lock:
    with _pipelines_guard:
        return _pipeline_locks.setdefault(user_id, threading.Lock())


def get_pipeline(
    user_id: Annotated[str, Depends(get_user_id)],
) -> PipelineBundle:
    # One lock per tenant: a 70-second first build for user A must not block
    # requests for user B.
    with _pipeline_lock(user_id):
        bundle = _pipelines.get(user_id)
        if bundle is not None:
            return bundle
        try:
            bundle = build_user_pipeline(user_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as a 503 to the client
            logger.exception("Could not build the RAG pipeline for %s", user_id)
            raise HTTPException(
                status_code=503,
                detail=f"RAG pipeline unavailable: {exc}",
            ) from exc
        try:
            documents = bundle.pipeline.restore_documents()
            logger.info(
                "Pipeline for %s restored %d document(s)", user_id, len(documents)
            )
        except Exception:  # noqa: BLE001 - dense retrieval still works
            logger.exception("Could not restore documents for %s", user_id)
        _pipelines[user_id] = bundle
        logger.info("Built RAG pipeline for user %s", user_id)
        return bundle


_chat_stores: dict[str, ChatStore] = {}
_chat_stores_lock = threading.Lock()


def get_chat_store(
    user_id: Annotated[str, Depends(get_user_id)],
) -> ChatStore:
    from src.auth.tenancy import user_db_path

    with _chat_stores_lock:
        store = _chat_stores.get(user_id)
        if store is None:
            store = ChatStore(user_db_path(user_id))
            _chat_stores[user_id] = store
        return store


class _SlidingWindowLimiter:
    """Dependency-free per-key request limiter."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window_s = float(window_s)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = _SlidingWindowLimiter(api_settings.rate_limit_per_minute)


def rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "unknown"
    if not _limiter.allow(key):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry shortly.",
            headers={"Retry-After": "60"},
        )


def reset_caches() -> None:
    """Drop cached pipelines/stores (tests, credential changes, reconnects)."""
    with _pipelines_guard:
        _pipelines.clear()
        _pipeline_locks.clear()
    with _chat_stores_lock:
        _chat_stores.clear()
    _limiter.clear()
