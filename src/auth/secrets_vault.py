"""Fernet-encrypted store for per-user API credentials.

The vault encrypts a JSON blob (provider keys, S3 credentials) with a key derived
from `ANEXUS_SECRET_KEY`. The ciphertext is persisted by `UserStore`; this module
never touches the database itself, so it is trivially testable.

Security notes:
  * Key derivation is a plain SHA-256 stretch of the passphrase - adequate for a
    demonstration app, not a substitute for a real KMS. Rotate by replacing
    `ANEXUS_SECRET_KEY` and re-saving credentials.
  * Decryption failures raise `VaultError` rather than returning garbage, so a
    wrong key can never be mistaken for valid credentials.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VaultError(Exception):
    """Raised when the vault key is missing/invalid or a token cannot be read."""


def derive_key(passphrase: str) -> bytes:
    """Stretch an arbitrary passphrase into a URL-safe base64 Fernet key."""
    if not passphrase:
        raise VaultError("ANEXUS_SECRET_KEY is not set")
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def merge_nonempty(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return `base` with the non-empty values of `updates` applied.

    Blank strings leave the existing value untouched, so the settings form can
    submit masked or omitted fields without wiping stored credentials.
    """
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, str):
            if value.strip():
                merged[key] = value.strip()
        elif value is not None:
            merged[key] = value
    return merged


class SecretsVault:
    """Encrypt/decrypt a dict of credentials with a passphrase-derived key."""

    def __init__(self, passphrase: Optional[str] = None) -> None:
        from config import settings

        secret = passphrase if passphrase is not None else settings.SECRET_KEY
        from cryptography.fernet import Fernet

        try:
            self._fernet = Fernet(derive_key(secret))
        except VaultError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise VaultError(f"Could not initialise the credential vault: {exc}") from exc

    def encrypt(self, data: dict[str, Any]) -> str:
        """Serialise `data` to JSON and return an opaque ciphertext token."""
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self._fernet.encrypt(payload).decode("ascii")

    def decrypt(self, token: Optional[str]) -> dict[str, Any]:
        """Reverse :meth:`encrypt`. Missing token -> empty dict; bad token -> error."""
        if not token:
            return {}
        from cryptography.fernet import InvalidToken

        try:
            raw = self._fernet.decrypt(token.encode("ascii"))
        except InvalidToken as exc:
            raise VaultError("Stored credentials could not be decrypted (wrong key?)") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise VaultError("Stored credentials are not valid JSON") from exc
        if not isinstance(value, dict):
            raise VaultError("Stored credentials must be a JSON object")
        return value
