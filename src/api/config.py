"""API-only settings.

Pipeline/vector settings stay in ``config/settings.py``; this module only holds
the knobs that belong to the HTTP layer (CORS, optional API key, upload cap).
The import of ``config.settings`` also loads ``.env`` before any getenv() below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import config.settings  # noqa: F401  (loads .env as a side effect)

_DEFAULT_ORIGINS = (
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:4173",  # Vite preview
    "http://127.0.0.1:4173",
    "http://localhost:3000",  # CRA / other dev servers
    "http://127.0.0.1:3000",
    "http://localhost:8000",  # built frontend served by the API
    "http://127.0.0.1:8000",
)


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class APISettings:
    environment: str = os.getenv("ANEXUS_ENV", "development")

    # Optional shared secret. When set, every /api route except /api/health
    # requires an ``X-API-Key`` header. Leave empty for local development.
    api_key: str = (
        os.getenv("ANEXUS_API_KEY") or os.getenv("FRONTEND_API_KEY") or ""
    ).strip()

    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("ANEXUS_CORS_ORIGINS", _DEFAULT_ORIGINS)
    )

    # Sliding-window limit per client IP, across all API routes.
    rate_limit_per_minute: int = int(os.getenv("ANEXUS_API_RATE_LIMIT", "120"))

    max_upload_mb: int = int(os.getenv("ANEXUS_MAX_UPLOAD_MB", "50"))

    @property
    def cors_allow_credentials(self) -> bool:
        # The CORS spec forbids credentials with a wildcard origin.
        return "*" not in self.cors_origins


api_settings = APISettings()
