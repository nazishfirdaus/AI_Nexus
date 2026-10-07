"""SQLite-backed chat persistence: conversations and their messages.

The store is deliberately dependency-free (stdlib sqlite3) and opens a fresh
connection per operation so it is safe under Streamlit's threaded reruns.
`db_path` is injectable so tests never touch the real database.

Schema:

    conversations(id, title, document_id, document_name, summary,
                  created_at, updated_at)
    messages(id, conversation_id, role, content, citations, created_at)

`citations` is a JSON array (or NULL). Deleting a conversation cascades to its
messages via ON DELETE CASCADE (PRAGMA foreign_keys=ON per connection).
"""
from __future__ import annotations

import json
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
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    document_id   TEXT,
    document_name TEXT,
    summary       TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);
"""

_VALID_ROLES = ("user", "assistant")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ChatStore:
    """Create, read, update and delete conversations and their messages."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.CHAT_DB_PATH
        self._lock = threading.Lock()
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        """Fresh connection per operation: commit, then close. Thread-safe."""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------------------------------------------------- conversations
    def create_conversation(
        self,
        title: str = "",
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> str:
        conv_id = uuid.uuid4().hex
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO conversations "
                "(id, title, document_id, document_name, summary, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, '', ?, ?)",
                (conv_id, title, document_id, document_name, now, now),
            )
        return conv_id

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 100) -> list[dict]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT c.*, COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id = c.id "
                "GROUP BY c.id "
                "ORDER BY c.updated_at DESC, c.rowid DESC "
                "LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conv_id: str, title: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conv_id),
            )

    def delete_conversation(self, conv_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        logger.info("Deleted conversation %s", conv_id)

    def set_document(
        self, conv_id: str, document_id: Optional[str], document_name: Optional[str]
    ) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET document_id = ?, document_name = ?, "
                "updated_at = ? WHERE id = ?",
                (document_id, document_name, _now(), conv_id),
            )

    def set_summary(self, conv_id: str, summary: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?",
                (summary, _now(), conv_id),
            )

    def touch_conversation(self, conv_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), conv_id),
            )

    # ---------------------------------------------------------------- messages
    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[list] = None,
    ) -> int:
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got {role!r}")
        now = _now()
        citations_json = json.dumps(citations) if citations else None
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, citations, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, citations_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return int(cursor.lastrowid)

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            raw = item.pop("citations", None)
            item["citations"] = json.loads(raw) if raw else None
            item.pop("conversation_id", None)
            messages.append(item)
        return messages

    def message_count(self, conversation_id: str) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["n"])
