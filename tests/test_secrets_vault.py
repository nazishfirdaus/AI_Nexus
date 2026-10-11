"""Tests for the Fernet credential vault."""
import pytest

from src.auth.secrets_vault import SecretsVault, VaultError, derive_key, merge_nonempty


def test_round_trip_encrypt_decrypt():
    vault = SecretsVault("unit-test-passphrase")
    data = {"groq": "gsk-123", "s3": {"access_key": "AKIA", "secret": "s3cr3t"}}

    token = vault.encrypt(data)
    assert isinstance(token, str)
    assert "gsk-123" not in token  # plaintext must not leak into the ciphertext
    assert vault.decrypt(token) == data


def test_decrypt_empty_token_returns_empty_dict():
    vault = SecretsVault("pass")
    assert vault.decrypt(None) == {}
    assert vault.decrypt("") == {}


def test_same_passphrase_is_deterministic_across_instances():
    a = SecretsVault("shared-secret")
    b = SecretsVault("shared-secret")
    assert b.decrypt(a.encrypt({"k": "v"})) == {"k": "v"}


def test_wrong_key_cannot_decrypt():
    token = SecretsVault("right-key").encrypt({"k": "v"})
    other = SecretsVault("wrong-key")
    with pytest.raises(VaultError):
        other.decrypt(token)


def test_tampered_token_is_rejected():
    vault = SecretsVault("pass")
    token = vault.encrypt({"k": "v"})
    tampered = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
    with pytest.raises(VaultError):
        vault.decrypt(tampered)


def test_missing_passphrase_is_rejected():
    with pytest.raises(VaultError):
        derive_key("")
    with pytest.raises(VaultError):
        SecretsVault("")


def test_merge_nonempty_keeps_existing_on_blank():
    base = {"groq_api_key": "old", "s3_bucket": "b"}
    merged = merge_nonempty(base, {"groq_api_key": "", "s3_bucket": "new", "nvidia_api_key": "nv"})
    assert merged == {"groq_api_key": "old", "s3_bucket": "new", "nvidia_api_key": "nv"}
    assert base == {"groq_api_key": "old", "s3_bucket": "b"}  # base untouched


def test_merge_nonempty_strips_whitespace():
    assert merge_nonempty({}, {"k": "  v  "}) == {"k": "v"}
