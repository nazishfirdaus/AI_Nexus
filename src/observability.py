"""Central Logfire / OpenTelemetry setup for ANexus.

One place configures tracing, logging and (optionally) HTTP/system
instrumentation, so every module gets the same service name and the same
privacy rules. Import the helpers below instead of calling ``logfire``
directly when you need a no-op fallback or PII-safe attributes.

Privacy contract:
  * Never attach document text, queries, prompts, answers, file contents,
    credentials or raw user identifiers to a span or log event.
  * Prefer counts, durations, model/provider names, status codes, scores and
    ``hash_id(...)`` values.
  * HTTP body/header capture is explicitly disabled.

If Logfire is not installed, or ``configure_observability()`` was never
called, the helpers are safe no-ops. This keeps unit tests and library use
offline by default.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:  # keep the app importable even in minimal environments
    import logfire
except Exception:  # pragma: no cover - only when the optional dep is absent
    logfire = None  # type: ignore[assignment]

_CONFIGURED = False

# Patterns the logging bridge redacts before anything reaches Logfire.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SECRET_RE = re.compile(
    r"\b(?:sk|gsk|AIza|pylf|hf|xox[baprs])[-_A-Za-z0-9]{8,}\b",
    re.IGNORECASE,
)
_LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)

# Logfire scrubbing patterns for values that might slip into an attribute.
_SCRUB_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"\b(?:sk|gsk|AIza|pylf|hf|xox[baprs])[-_A-Za-z0-9]{8,}\b",
    r"\b\d{9,}\b",
]


def hash_id(value: Any, *, length: int = 12) -> str:
    """Stable, non-reversible identifier for logs (never the raw value)."""
    if value is None:
        return ""
    digest = hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()
    return digest[: max(4, int(length))]


def redact_text(text: str, *, max_chars: int = 500) -> str:
    """Redact common secrets/PII from free-form log text and cap its length."""
    if not text:
        return ""
    redacted = _EMAIL_RE.sub("[email]", str(text))
    redacted = _SECRET_RE.sub("[secret]", redacted)
    redacted = _BEARER_RE.sub(r"\1[secret]", redacted)
    redacted = _LONG_DIGITS_RE.sub("[number]", redacted)
    if len(redacted) > max_chars:
        redacted = redacted[: max_chars - 1].rstrip() + "…"
    return redacted


def is_configured() -> bool:
    return bool(_CONFIGURED)


def _env_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


if logfire is not None:

    class _RedactingLogfireHandler(logfire.LogfireLoggingHandler):
        """Forward stdlib records to Logfire after redacting the message.

        A copy of the record is used so other handlers (console, pytest
        caplog) still see the original message.
        """

        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            try:
                copy = logging.makeLogRecord(record.__dict__)
                copy.msg = redact_text(copy.getMessage())
                copy.args = ()
                super().emit(copy)
            except Exception:
                # A broken log sink must never break the request.
                pass

else:  # pragma: no cover - optional dependency absent
    _RedactingLogfireHandler = None  # type: ignore[assignment,misc]


def configure_observability(
    *,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
    token: Optional[str] = None,
    send_to_logfire: Optional[bool] = None,
    console: Optional[bool] = None,
) -> None:
    """Configure Logfire once per process.

    Cloud export is enabled when a token is present (or
    ``LOGFIRE_SEND_TO_LOGFIRE`` is truthy). With no token everything stays
    local: helpers become no-ops and nothing leaves the machine.
    """
    global _CONFIGURED
    if _CONFIGURED or logfire is None:
        return

    # Imported lazily so importing this module never forces settings/dotenv.
    from config import settings

    resolved_token = (token if token is not None else settings.LOGFIRE_TOKEN) or ""
    resolved_token = str(resolved_token).strip()

    if send_to_logfire is None:
        raw_send = _env_bool(settings.LOGFIRE_SEND_TO_LOGFIRE)
        if raw_send is None:
            send_to_logfire = bool(resolved_token)
        else:
            send_to_logfire = raw_send
    # A token is required for export; never send without one.
    if not resolved_token:
        send_to_logfire = False
    # Test runs must never export traces, even when a developer's .env has a
    # token. This is a hard override, not a default.
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        send_to_logfire = False

    resolved_console = (
        bool(settings.LOGFIRE_CONSOLE) if console is None else bool(console)
    )

    try:
        logfire.configure(
            token=resolved_token or None,
            send_to_logfire=bool(send_to_logfire),
            service_name=service_name or settings.LOGFIRE_SERVICE_NAME,
            environment=environment or settings.LOGFIRE_ENVIRONMENT,
            console=resolved_console,
            inspect_arguments=False,
            scrubbing=logfire.ScrubbingOptions(extra_patterns=_SCRUB_PATTERNS),
        )
    except Exception:
        # Observability must never prevent the app from starting.
        return

    # Outbound HTTP spans: URLs/status only; never headers or bodies.
    try:
        logfire.instrument_requests()
    except Exception:
        pass
    try:
        logfire.instrument_httpx(
            capture_headers=False,
            capture_request_body=False,
            capture_response_body=False,
        )
    except Exception:
        pass
    if getattr(settings, "LOGFIRE_SYSTEM_METRICS", False):
        try:
            logfire.instrument_system_metrics()
        except Exception:
            pass

    # Bridge stdlib logging. The level is intentionally conservative by
    # default (WARNING) so existing verbose INFO logs do not explode trace
    # volume or leak filenames in development.
    if _RedactingLogfireHandler is not None:
        level = getattr(logging, str(settings.LOGFIRE_LOG_LEVEL), logging.WARNING)
        root = logging.getLogger()
        if not any(isinstance(h, _RedactingLogfireHandler) for h in root.handlers):
            handler = _RedactingLogfireHandler(level=level)
            handler.setLevel(level)
            root.addHandler(handler)

    _CONFIGURED = True
    try:
        logfire.info(
            "Observability configured",
            service=service_name or settings.LOGFIRE_SERVICE_NAME,
            environment=environment or settings.LOGFIRE_ENVIRONMENT,
            cloud_export=bool(send_to_logfire),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Safe wrappers
# ---------------------------------------------------------------------------
@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a Logfire span, or a no-op context when observability is off."""
    if logfire is None or not _CONFIGURED:
        yield None
        return
    with logfire.span(name, **attributes) as current:
        yield current


@contextmanager
def bind(**attributes: Any) -> Iterator[None]:
    """Attach context (e.g. hashed user id) to spans created inside the block."""
    if logfire is None or not _CONFIGURED:
        yield
        return
    with logfire.attach_context(attributes):
        yield


def info(message: str, **attributes: Any) -> None:
    if logfire is not None and _CONFIGURED:
        logfire.info(message, **attributes)


def warning(message: str, **attributes: Any) -> None:
    if logfire is not None and _CONFIGURED:
        logfire.warning(message, **attributes)


def error(message: str, **attributes: Any) -> None:
    if logfire is not None and _CONFIGURED:
        logfire.error(message, **attributes)


def exception(message: str, **attributes: Any) -> None:
    if logfire is not None and _CONFIGURED:
        logfire.exception(message, **attributes)


def debug(message: str, **attributes: Any) -> None:
    if logfire is not None and _CONFIGURED:
        logfire.debug(message, **attributes)


__all__ = [
    "bind",
    "configure_observability",
    "debug",
    "error",
    "exception",
    "hash_id",
    "info",
    "is_configured",
    "redact_text",
    "span",
    "warning",
]
