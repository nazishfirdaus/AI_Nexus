"""Tests for the SQLite account registry + credential storage."""
import pytest

from config import settings
from src.auth.user_store import UserStore


@pytest.fixture()
def store(tmp_path):
    return UserStore(db_path=str(tmp_path / "auth.sqlite3"))


@pytest.fixture()
def no_bootstrap(monkeypatch):
    monkeypatch.setattr(settings, "BOOTSTRAP_ADMIN_EMAIL", "")


def test_provision_and_lookup(store):
    user = store.provision("alice@example.com", role=settings.ROLE_USER, display_name="Alice")
    assert user["email"] == "alice@example.com"
    assert user["role"] == "user"
    assert user["status"] == "active"
    assert store.get_by_email("ALICE@example.com")["id"] == user["id"]


def test_provision_is_idempotent(store):
    first = store.provision("bob@example.com")
    second = store.provision("bob@example.com", role=settings.ROLE_ADMIN)
    assert second["id"] == first["id"]
    assert second["role"] == "user"  # existing record is returned unchanged


def test_provision_rejects_bad_role(store):
    with pytest.raises(ValueError):
        store.provision("x@example.com", role="root")


def test_resolve_login_binds_sub(store, no_bootstrap):
    store.provision("carol@example.com", display_name="Carol")
    user = store.resolve_login("sub-123", "carol@example.com", "Carol")
    assert user is not None
    assert user["oidc_sub"] == "sub-123"
    assert user["last_login_at"] is not None
    # second login matches by sub even if the email casing changes
    again = store.resolve_login("sub-123", "CAROL@example.com")
    assert again["id"] == user["id"]


def test_resolve_login_denies_unknown_without_bootstrap(store, no_bootstrap):
    assert store.resolve_login("sub-x", "stranger@example.com") is None


def test_resolve_login_denies_disabled(store, no_bootstrap):
    user = store.provision("denied@example.com")
    store.set_status(user["id"], settings.STATUS_DISABLED)
    assert store.resolve_login(None, "denied@example.com") is None


def test_resolve_login_rejects_sub_mismatch(store, no_bootstrap):
    store.provision("dave@example.com", oidc_sub="sub-A")
    assert store.resolve_login("sub-B", "dave@example.com") is None


def test_bootstrap_admin_created_on_first_login(store, monkeypatch):
    monkeypatch.setattr(settings, "BOOTSTRAP_ADMIN_EMAIL", "owner@example.com")
    admin = store.resolve_login("sub-owner", "owner@example.com", "Owner")
    assert admin["role"] == settings.ROLE_ADMIN
    assert admin["status"] == settings.STATUS_ACTIVE


def test_bootstrap_only_grants_one_admin(store, monkeypatch):
    monkeypatch.setattr(settings, "BOOTSTRAP_ADMIN_EMAIL", "owner@example.com")
    store.resolve_login("sub-owner", "owner@example.com")
    # a different stranger is still denied even though an admin exists
    assert store.resolve_login("sub-2", "other@example.com") is None
    assert store.admin_count() == 1


def test_role_and_status_validation(store):
    user = store.provision("erin@example.com")
    store.set_role(user["id"], settings.ROLE_ADMIN)
    assert store.get_by_id(user["id"])["role"] == "admin"
    with pytest.raises(ValueError):
        store.set_role(user["id"], "superuser")
    with pytest.raises(ValueError):
        store.set_status(user["id"], "paused")


def test_secrets_round_trip_and_clear(store):
    user = store.provision("frank@example.com")
    assert store.get_secrets(user["id"]) is None
    store.set_secrets(user["id"], "ciphertext-1")
    assert store.get_secrets(user["id"]) == "ciphertext-1"
    store.set_secrets(user["id"], "ciphertext-2")
    assert store.get_secrets(user["id"]) == "ciphertext-2"
    store.clear_secrets(user["id"])
    assert store.get_secrets(user["id"]) is None


def test_removing_user_cascades_secrets(store):
    user = store.provision("gina@example.com")
    store.set_secrets(user["id"], "ciphertext")
    store.remove(user["id"])
    assert store.get_by_id(user["id"]) is None
    assert store.get_secrets(user["id"]) is None


def test_list_users_orders_by_creation(store):
    store.provision("a@example.com")
    store.provision("b@example.com")
    emails = [u["email"] for u in store.list_users()]
    assert emails == ["a@example.com", "b@example.com"]
