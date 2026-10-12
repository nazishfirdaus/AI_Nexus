"""Tests for per-user isolation: namespaces, pipeline factory, identity, keys."""
import os
from types import SimpleNamespace

import pytest

from config import settings
from src import pipeline_factory
from src.auth import identity, tenancy
from src.auth.user_store import UserStore
from src.generation import llm


# ------------------------------------------------------------------ helpers
class FakeScrubber:
    def scrub(self, text, context_prefix=""):
        return text


class FakeHandler:
    available_providers = ["fake"]


@pytest.fixture()
def chroma_client(tmp_path, monkeypatch):
    """Redirect the shared Chroma client to an in-memory store instead of ./chroma_db."""
    import chromadb

    client = chromadb.EphemeralClient()
    monkeypatch.setattr(pipeline_factory, "get_chroma_client", lambda: client)
    monkeypatch.setattr(settings, "USER_DATA_DIR", str(tmp_path / "users"))
    return client


def _make_user_pipeline(user_id):
    return pipeline_factory.build_user_pipeline(
        user_id, llm_handler=FakeHandler(), scrubber=FakeScrubber()
    )


# ------------------------------------------------------------------ namespaces
def test_collection_and_db_path_are_user_scoped():
    assert tenancy.user_collection_name("abc") == f"{settings.COLLECTION_NAME}__abc"
    assert tenancy.user_db_path("abc") == os.path.join(
        tenancy.user_data_dir("abc"), "app.sqlite3"
    )
    assert tenancy.user_data_dir("abc").endswith(os.path.join("users", "abc"))


def test_build_user_pipeline_uses_user_namespaces(chroma_client):
    bundle = _make_user_pipeline("userA")
    assert bundle.vector_db._collection_name == f"{settings.COLLECTION_NAME}__userA"
    assert bundle.document_store.db_path == tenancy.user_db_path("userA")


def test_two_users_get_distinct_namespaces(chroma_client):
    a = _make_user_pipeline("userA")
    b = _make_user_pipeline("userB")
    assert a.vector_db._collection_name != b.vector_db._collection_name
    assert a.document_store.db_path != b.document_store.db_path


def test_documents_do_not_leak_between_users(chroma_client):
    a = _make_user_pipeline("userA")
    b = _make_user_pipeline("userB")
    a.pipeline.document_store.add("secret", "secret.pdf", pages=1, chunks=1, vectors=1)

    assert [d["id"] for d in a.pipeline.list_documents()] == ["secret"]
    assert b.pipeline.list_documents() == []


# ------------------------------------------------------------------ identity
@pytest.fixture()
def user_store(tmp_path):
    return UserStore(db_path=str(tmp_path / "auth.sqlite3"))


def test_extract_claims_from_object():
    user = SimpleNamespace(is_logged_in=True, sub="s1", email="a@b.com", name="Alice")
    assert identity.extract_claims(user) == {
        "is_logged_in": True,
        "sub": "s1",
        "email": "a@b.com",
        "name": "Alice",
    }


def test_resolve_user_none_when_logged_out(user_store):
    user = SimpleNamespace(is_logged_in=False, email="a@b.com")
    assert identity.resolve_user(user_store, user) is None


def test_resolve_user_maps_to_account(user_store):
    user_store.provision("a@b.com", display_name="Alice")
    user = SimpleNamespace(is_logged_in=True, sub="s1", email="a@b.com", name="Alice")
    resolved = identity.resolve_user(user_store, user)
    assert resolved is not None
    assert resolved["email"] == "a@b.com"


def test_resolve_user_auto_provisions_unknown(user_store):
    user = SimpleNamespace(is_logged_in=True, sub="s1", email="nobody@b.com", name="")
    resolved = identity.resolve_user(user_store, user)
    assert resolved is not None
    assert resolved["email"] == "nobody@b.com"


# ------------------------------------------------------------------ per-user keys
def test_user_keys_override_env(monkeypatch):
    monkeypatch.setattr(settings, "SIMULATE_LLM", False)
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    handler = llm.build_handler_from_credentials({"groq_api_key": "user-key"})
    assert handler.adapters["groq"].api_key == "user-key"
    # supplying any user key means the env keys are not used at all
    assert handler.adapters["nvidia"].api_key is None


def test_no_user_keys_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(settings, "SIMULATE_LLM", False)
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    handler = llm.build_handler_from_credentials({})
    assert handler.adapters["groq"].api_key == "env-key"
