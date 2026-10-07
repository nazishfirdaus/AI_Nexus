"""SQLite-backed registry of ingested documents.

Chroma stores the chunks; this table stores the *documents themselves* so the
UI can list them, show per-document stats and remove them individually. It
lives in the same database as the chat store (settings.CHAT_DB_PATH) but owns
only its own table.

`db_path` is injectable so tests never touch the real database; ":memory:" is
supported for pipelines constructed without a store (unit tests).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,
    pages      INTEGER NOT NULL DEFAULT 0,
    chunks     INTEGER NOT NULL DEFAULT 0,
    vectors    INTEGER NOT NULL DEFAULT 0,
    duration_s REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class DocumentStore:
    """Create, list and delete document records."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.CHAT_DB_PATH
        self._lock = threading.Lock()
        self._anchor: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            # A plain ":memory:" database exists only for the connection that
            # opened it, but this store opens a fresh connection per operation.
            # A named shared-memory DSN keeps one isolated in-memory database
            # alive for the lifetime of this store (the anchor connection).
            self._dsn = f"file:anexus_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._anchor = sqlite3.connect(
                self._dsn, uri=True, check_same_thread=False
            )
        else:
            self._dsn = self.db_path
            self._uri = False
            directory = os.path.dirname(self.db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        """Fresh connection per operation: commit, then close. Thread-safe."""
        conn = sqlite3.connect(
            self._dsn, uri=self._uri, timeout=30, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        """Release the anchor connection of an in-memory store."""
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    # ------------------------------------------------------------------- CRUD
    def add(
        self,
        document_id: str,
        filename: str,
        pages: int = 0,
        chunks: int = 0,
        vectors: int = 0,
        duration_s: float = 0.0,
    ) -> dict:
        """Insert (or replace) one document record."""
        record = {
            "id": document_id,
            "filename": filename,
            "pages": int(pages),
            "chunks": int(chunks),
            "vectors": int(vectors),
            "duration_s": float(duration_s),
            "created_at": _now(),
        }
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO documents "
                "(id, filename, pages, chunks, vectors, duration_s, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "filename=excluded.filename, pages=excluded.pages, "
                "chunks=excluded.chunks, vectors=excluded.vectors, "
                "duration_s=excluded.duration_s, created_at=excluded.created_at",
                (
                    record["id"],
                    record["filename"],
                    record["pages"],
                    record["chunks"],
                    record["vectors"],
                    record["duration_s"],
                    record["created_at"],
                ),
            )
        return record

    def list(self) -> list[dict]:
        """All documents, oldest first."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at, id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, document_id: str) -> Optional[dict]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def remove(self, document_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        logger.info("Removed document %s from registry", document_id)

    def clear(self) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM documents")
        logger.info("Cleared document registry")
