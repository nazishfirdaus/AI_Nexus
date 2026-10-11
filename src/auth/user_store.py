"""SQLite-backed account registry: users, roles and encrypted credentials.

Mirrors the storage conventions of `DocumentStore` / `ChatStore`: stdlib sqlite3,
a fresh connection per operation (safe under Streamlit's threaded reruns) and an
injectable `db_path` so tests never touch the real database.

Schema:

    users(id, oidc_sub, email, display_name, role, status, created_at, last_login_at)
    user_secrets(user_id, ciphertext, updated_at)

`id` is a stable internal tenant key generated when the account is provisioned.
`oidc_sub` is bound on first login; accounts are matched by sub first, then by
email. Provisioning is admin-driven: an unknown email is denied unless it matches
`BOOTSTRAP_ADMIN_EMAIL` and no admin exists yet.

Credentials live in `user_secrets` as an opaque ciphertext produced by
`SecretsVault`; this store never sees plaintext keys.
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
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    oidc_sub      TEXT,
    email         TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL DEFAULT 'user',
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS user_secrets (
    user_id    TEXT PRIMARY KEY
        REFERENCES users(id) ON DELETE CASCADE,
    ciphertext TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_sub ON users(oidc_sub);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _norm_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()


class UserStore:
    """Provision accounts, resolve logins and persist encrypted credentials."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.AUTH_DB_PATH
        self._lock = threading.Lock()
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
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

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        return dict(row)

    # ------------------------------------------------------------------ reads
    def get_by_id(self, user_id: str) -> Optional[dict]:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._to_dict(row) if row else None

    def get_by_email(self, email: str) -> Optional[dict]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (_norm_email(email),)
            ).fetchone()
        return self._to_dict(row) if row else None

    def get_by_sub(self, oidc_sub: str) -> Optional[dict]:
        if not oidc_sub:
            return None
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE oidc_sub = ?", (oidc_sub,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at, email"
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    def admin_count(self) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = ? AND status = ?",
                (settings.ROLE_ADMIN, settings.STATUS_ACTIVE),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ writes
    def provision(
        self,
        email: str,
        role: str = settings.ROLE_USER,
        display_name: str = "",
        oidc_sub: Optional[str] = None,
    ) -> dict:
        """Create an account (admin action). Email must be new."""
        email = _norm_email(email)
        if not email:
            raise ValueError("email is required")
        if role not in settings.VALID_ROLES:
            raise ValueError(f"role must be one of {settings.VALID_ROLES}, got {role!r}")
        existing = self.get_by_email(email)
        if existing:
            return existing
        user_id = uuid.uuid4().hex
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO users "
                "(id, oidc_sub, email, display_name, role, status, created_at, last_login_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (user_id, oidc_sub, email, display_name, role, settings.STATUS_ACTIVE, now),
            )
        logger.info("Provisioned %s account %s (%s)", role, email, user_id)
        return self.get_by_id(user_id)  # type: ignore[return-value]

    def resolve_login(
        self, oidc_sub: Optional[str], email: str, display_name: str = ""
    ) -> Optional[dict]:
        """Map an OIDC login to an account, or return None if not allowed.

        Matching order: `oidc_sub`, then email. A disabled account is denied. An
        unknown email is auto-provisioned as an active admin only when it matches
        `BOOTSTRAP_ADMIN_EMAIL` and no admin exists yet.
        """
        email = _norm_email(email)
        if not email:
            return None
        row = self.get_by_sub(oidc_sub) if oidc_sub else None
        if row is None:
            row = self.get_by_email(email)

        if row is not None:
            if row["status"] != settings.STATUS_ACTIVE:
                logger.warning("Denied disabled account %s", email)
                return None
            if row["oidc_sub"] and oidc_sub and row["oidc_sub"] != oidc_sub:
                logger.warning("OIDC subject mismatch for %s", email)
                return None
            updates, params = [], []
            if oidc_sub and not row["oidc_sub"]:
                updates.append("oidc_sub = ?")
                params.append(oidc_sub)
            if display_name and not row["display_name"]:
                updates.append("display_name = ?")
                params.append(display_name)
            updates.append("last_login_at = ?")
            params.append(_now())
            params.append(row["id"])
            with self._lock, self._connection() as conn:
                conn.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params
                )
            return self.get_by_id(row["id"])

        if (
            settings.BOOTSTRAP_ADMIN_EMAIL
            and email == settings.BOOTSTRAP_ADMIN_EMAIL
            and self.admin_count() == 0
        ):
            logger.info("Bootstrapping first admin from %s", email)
            return self.provision(
                email, role=settings.ROLE_ADMIN, display_name=display_name,
                oidc_sub=oidc_sub,
            )

        logger.warning("Denied unknown account %s (admin must provision it)", email)
        return None

    def set_role(self, user_id: str, role: str) -> None:
        if role not in settings.VALID_ROLES:
            raise ValueError(f"role must be one of {settings.VALID_ROLES}, got {role!r}")
        with self._lock, self._connection() as conn:
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))

    def set_status(self, user_id: str, status: str) -> None:
        if status not in settings.VALID_STATUSES:
            raise ValueError(f"status must be one of {settings.VALID_STATUSES}, got {status!r}")
        with self._lock, self._connection() as conn:
            conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))

    def remove(self, user_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        logger.info("Removed account %s", user_id)

    # --------------------------------------------------------------- secrets
    def set_secrets(self, user_id: str, ciphertext: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO user_secrets (user_id, ciphertext, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "ciphertext = excluded.ciphertext, updated_at = excluded.updated_at",
                (user_id, ciphertext, _now()),
            )

    def get_secrets(self, user_id: str) -> Optional[str]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT ciphertext FROM user_secrets WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row["ciphertext"] if row else None

    def clear_secrets(self, user_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM user_secrets WHERE user_id = ?", (user_id,))
