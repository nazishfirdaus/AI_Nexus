"""Embedding service for ANexus.

Primary provider: Google Gemini embeddings. When no key is configured (or the
probe fails) the process falls back to a local sentence-transformers model
(``all-mpnet-base-v2``, 768-dim, by default).

The legacy public API is preserved so the retriever, pipeline, evaluation
harness and tests keep working unchanged:

    embed_texts, embed_documents, embed_query, get_embedding_model,
    get_embedding_dim, get_active_model_info, token_length,
    EMBEDDING_MODEL_NAME, QUERY_INSTRUCTION

Privacy: document text, query text and API keys never reach observability.
Only counts, provider/model names and vector dimensions are recorded.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from typing import Any, Optional

from config import settings
from src.observability import (
    error as obs_error,
    info as obs_info,
    span,
    warning as obs_warning,
)

logger = logging.getLogger(__name__)

# Kept for backwards compatibility with code/tests that import these names.
# EMBEDDING_MODEL_NAME remains the legacy BGE setting (used by the retrieval
# evaluation harness); the *active* local fallback is EMBEDDING_FALLBACK_MODEL.
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
QUERY_INSTRUCTION = settings.QUERY_INSTRUCTION
BATCH_SIZE = max(1, int(settings.EMBEDDING_BATCH_SIZE))

# Longest BGE window is 512 tokens; all-mpnet-base-v2 is 384. The chunker uses
# token_length() only when explicitly enabled, so this stays a soft default.
_MAX_TOKENS = 512

_LOCK = threading.RLock()
_active_model: Any = None
_model_type: Optional[str] = None  # "gemini" | "local"
_model_name: Optional[str] = None
_model_dim: Optional[int] = None

_GEMINI_DIM = int(settings.GEMINI_EMBEDDING_DIM)
_FALLBACK_DIM = int(settings.EMBEDDING_FALLBACK_DIM)
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _l2_normalize(vector: Any) -> list[float]:
    """Return a unit-length float list; safe for zero/invalid vectors."""
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return values
    return [v / norm for v in values]


def _provider_mode() -> str:
    """Resolve the configured provider mode: auto | gemini | local."""
    mode = str(getattr(settings, "EMBEDDING_PROVIDER", "auto") or "auto").strip().lower()
    if mode in ("local", "fallback", "sentence-transformers", "sentence_transformers"):
        return "local"
    if mode == "gemini":
        return "gemini"
    return "auto"


def _probe_gemini() -> tuple[Any, Optional[str], Optional[int]]:
    """Try one short Gemini embedding call.

    Returns ``(model, model_name, dimension)`` on success or all-None when the
    key is missing / the SDK is unavailable / the call fails. The exception
    message is deliberately not logged (it may echo request details).
    """
    api_key = str(getattr(settings, "GEMINI_API_KEY", "") or "").strip()
    if not api_key:
        obs_info("Gemini embeddings skipped: GEMINI_API_KEY is empty")
        return None, None, None

    model_name = str(getattr(settings, "GEMINI_EMBEDDING_MODEL", "") or "").strip()
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception as exc:  # optional dependency / import error
        obs_warning(
            "Gemini embeddings unavailable; using local fallback",
            error_type=type(exc).__name__,
        )
        return None, None, None

    try:
        model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )
        probe = model.embed_query("dimension probe")
        dim = len(probe)
        if dim <= 0:
            raise ValueError("Gemini returned an empty probe embedding")
        obs_info("Gemini embeddings ready", model=model_name, dimension=dim)
        return model, model_name, dim
    except Exception as exc:
        obs_warning(
            "Gemini embedding probe failed; using local fallback",
            model=model_name,
            error_type=type(exc).__name__,
        )
        return None, None, None


def _load_local_model() -> tuple[Any, str, int]:
    """Load the sentence-transformers fallback (CPU, 4 GB-RAM friendly)."""
    from sentence_transformers import SentenceTransformer

    name = str(getattr(settings, "EMBEDDING_FALLBACK_MODEL", "") or "").strip()
    if not name:
        name = "all-mpnet-base-v2"
    obs_info("Loading local embedding fallback", model=name)
    try:
        model = SentenceTransformer(name, device="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the local embedding model {name!r}. "
            "Install sentence-transformers or set ANEXUS_FALLBACK_EMBEDDING_MODEL."
        ) from exc

    dim = None
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        try:
            dim = int(getter())
        except Exception:
            dim = None
    if not dim or dim <= 0:
        dim = int(getattr(settings, "EMBEDDING_FALLBACK_DIM", _FALLBACK_DIM))
    obs_info("Local embedding fallback ready", model=name, dimension=dim)
    return model, name, dim


def _activate_local() -> None:
    global _active_model, _model_type, _model_name, _model_dim
    model, name, dim = _load_local_model()
    _active_model = model
    _model_type = "local"
    _model_name = name
    _model_dim = dim


def _initialize() -> None:
    """Initialise the embedding model once per process (thread-safe)."""
    global _active_model, _model_type, _model_name, _model_dim

    if _active_model is not None:
        return

    with _LOCK:
        if _active_model is not None:
            return

        mode = _provider_mode()
        if mode == "local":
            _activate_local()
            return

        model, name, dim = _probe_gemini()
        if model is not None:
            _active_model = model
            _model_type = "gemini"
            _model_name = name
            _model_dim = dim
            return

        if mode == "gemini":
            raise RuntimeError(
                "ANEXUS_EMBEDDING_PROVIDER=gemini requires a working "
                "GEMINI_API_KEY, but the embedding probe failed. Set the key, "
                "use ANEXUS_EMBEDDING_PROVIDER=auto, or choose 'local'."
            )

        # auto: degraded but usable without a key/network.
        _activate_local()


# ---------------------------------------------------------------------------
# Public introspection helpers
# ---------------------------------------------------------------------------
def get_embedding_model() -> Any:
    """Return the active underlying model (Gemini client or SentenceTransformer).

    Loads the model on first use. Kept for backwards compatibility with code
    that used to call ``model.encode(...)``; new code should use
    :func:`embed_query` / :func:`embed_texts`.
    """
    _initialize()
    return _active_model


def get_embedding_dim() -> int:
    """Return the active model's vector dimension (initialises it lazily)."""
    _initialize()
    return int(_model_dim or _FALLBACK_DIM)


def get_active_model_info() -> dict[str, Any]:
    """Describe the active embedder without exposing credentials."""
    _initialize()
    return {
        "provider": _model_type,
        "model": _model_name,
        "dimension": int(_model_dim or _FALLBACK_DIM),
    }


def token_length(text: str) -> int:
    """Count model tokens for ``text`` (0 for blank input).

    Never forces the embedding model to load: if it is already active and has
    a tokenizer, that tokenizer is used; otherwise a Unicode word/punctuation
    count is a conservative proxy for chunk sizing.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    tokenizer = getattr(_active_model, "tokenizer", None) if _active_model is not None else None
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return min(_MAX_TOKENS, len(_WORD_RE.findall(text)))


# ---------------------------------------------------------------------------
# Batch embedding with retry
# ---------------------------------------------------------------------------
def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed one batch with the active provider, returning unit-length vectors."""
    if _model_type == "gemini":
        retries = max(1, int(getattr(settings, "EMBEDDING_MAX_RETRIES", 4)))
        last_exc: Optional[BaseException] = None
        for attempt in range(retries):
            try:
                return [_l2_normalize(v) for v in _active_model.embed_documents(batch)]
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                rate_limited = any(
                    marker in message
                    for marker in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")
                )
                if not rate_limited or attempt >= retries - 1:
                    obs_error(
                        "Gemini embedding call failed",
                        attempts=attempt + 1,
                        error_type=type(exc).__name__,
                    )
                    raise
                wait = min(2 ** attempt, 30)
                obs_warning(
                    "Gemini rate limit; retrying",
                    attempt=attempt + 1,
                    retry_in_s=wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            "Gemini rate limit persisted after retries."
        ) from last_exc

    encoded = _active_model.encode(
        batch,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return [_l2_normalize(v) for v in encoded.tolist()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def embed_query(query: str) -> list[float]:
    """Embed one search query. Returns [] for blank input."""
    if not isinstance(query, str) or not query.strip():
        return []

    _initialize()
    text = query
    # BGE v1 expects the instruction on the query side only. Gemini has its own
    # retrieval task type and all-mpnet-base-v2 was not trained with it.
    if _model_type == "local" and "bge" in str(_model_name or "").lower():
        text = QUERY_INSTRUCTION + query

    with span(
        "embed.query",
        provider=_model_type,
        model=_model_name,
        dimension=_model_dim,
    ):
        if _model_type == "gemini":
            vector = _active_model.embed_query(text)
        else:
            vector = _active_model.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=False,
            )[0].tolist()
    return _l2_normalize(vector)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, dropping blank entries.

    Blank inputs are filtered *before* batching so returned vectors always
    align with the non-blank inputs (never zero-vectors for blanks).
    """
    cleaned = [
        t for t in (texts or []) if isinstance(t, str) and t.strip()
    ]
    if not cleaned:
        return []

    _initialize()
    batch_size = max(1, int(getattr(settings, "EMBEDDING_BATCH_SIZE", BATCH_SIZE)))
    embeddings: list[list[float]] = []
    with span(
        "embed.texts",
        provider=_model_type,
        model=_model_name,
        dimension=_model_dim,
        text_count=len(cleaned),
        batch_size=batch_size,
    ):
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            with span(
                "embed.batch",
                provider=_model_type,
                start=start,
                size=len(batch),
            ):
                embeddings.extend(_embed_batch(batch))
    return embeddings


def embed_documents(documents) -> list[list[float]]:
    """Embed LangChain Documents (or plain strings), dropping blank content."""
    if not documents:
        return []
    texts: list[str] = []
    for document in documents:
        content = (
            document if isinstance(document, str)
            else getattr(document, "page_content", None)
        )
        if isinstance(content, str) and content.strip():
            texts.append(content)
    return embed_texts(texts)


__all__ = [
    "BATCH_SIZE",
    "EMBEDDING_MODEL_NAME",
    "QUERY_INSTRUCTION",
    "embed_documents",
    "embed_query",
    "embed_texts",
    "get_active_model_info",
    "get_embedding_dim",
    "get_embedding_model",
    "token_length",
]
