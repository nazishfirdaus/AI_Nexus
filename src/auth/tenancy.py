"""Per-user namespace helpers.

Every tenant gets its own Chroma collection and its own SQLite database file, so
documents and chats are isolated simply by construction - there is no shared
collection that a ``where`` filter must guard, and no shared table that a missing
``WHERE user_id`` could leak across accounts.

`user_id` is always an opaque hex id (see `UserStore.provision`), which keeps it
safe to use directly as a path component and collection-name suffix.
"""
from __future__ import annotations

import os

from config import settings


def user_collection_name(user_id: str) -> str:
    """Chroma collection name dedicated to one user."""
    return f"{settings.COLLECTION_NAME}__{user_id}"


def user_data_dir(user_id: str) -> str:
    """Directory holding one user's documents + chats database."""
    return os.path.join(settings.USER_DATA_DIR, user_id)


def user_db_path(user_id: str) -> str:
    """SQLite path shared by that user's DocumentStore and ChatStore."""
    return os.path.join(user_data_dir(user_id), "app.sqlite3")
