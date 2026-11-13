"""FastAPI application for the React frontend.

Start it with::

    uvicorn src.api.main:app --reload --port 8000

or::

    python -m src.api

The app exposes the RAG pipeline (``/api/chat``), the SQLite-backed
conversation store (``/api/conversations``), message feedback, and the document
library. It does not import Streamlit or ``app.py``.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.observability import configure_observability

# Configure tracing before the pipeline modules are imported.
configure_observability(service_name="anexus-api")

from src.api.config import api_settings  # noqa: E402
from src.api.routers import chat, conversations, documents, health, messages  # noqa: E402
from src.api.dependencies import get_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Backend for the ANexus React client.

* **Chat** — ask a question, get an answer with page citations; both sides of
  the turn are stored in the tenant's SQLite database.
* **Conversations** — list, create, rename, delete; fetch a conversation with
  its full message history.
* **Documents** — list, upload/ingest, and delete documents in the tenant's
  Qdrant collection.
* **Feedback** — thumbs up/down on an assistant message.

Tenant selection is via the ``X-User-Id`` header (defaults to ``dev`` locally).
"""


def _warmup_enabled() -> bool:
    return os.getenv("ANEXUS_API_WARMUP", "0").strip().lower() in ("1", "true", "yes")


def _warmup() -> None:
    user_id = os.getenv("ANEXUS_DEV_USER_ID", "dev").strip() or "dev"
    try:
        get_pipeline(user_id)
        logger.info("Warmed the RAG pipeline for %s", user_id)
    except Exception:  # noqa: BLE001 - warmup must never crash the app
        logger.exception("Pipeline warmup failed")


def create_app(*, api_key: str | None = None) -> FastAPI:
    resolved_api_key = api_settings.api_key if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            "ANexus API starting (env=%s, cors=%s)",
            api_settings.environment,
            ",".join(api_settings.cors_origins),
        )
        if _warmup_enabled():
            threading.Thread(
                target=_warmup, name="anexus-api-warmup", daemon=True
            ).start()
        yield
        logger.info("ANexus API stopped")

    app = FastAPI(
        title="ANexus API",
        version="0.1.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=api_settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )

    if resolved_api_key:

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            path = request.url.path
            if (
                path.startswith("/api")
                and path != "/api/health"
                and request.method != "OPTIONS"
            ):
                if request.headers.get("X-API-Key") != resolved_api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing X-API-Key."},
                    )
            return await call_next(request)

    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api")
    app.include_router(messages.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "service": "ANexus API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
