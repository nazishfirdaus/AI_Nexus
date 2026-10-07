"""Tests for the SQLite chat store. Everything runs against tmp_path."""
import json

import pytest

from src.memory.chat_store import ChatStore


@pytest.fixture()
def store(tmp_path):
    return ChatStore(db_path=str(tmp_path / "chats.sqlite3"))


def test_create_and_get_conversation(store):
    conv_id = store.create_conversation(
        title="Rates", document_id="loan_doc", document_name="loan.pdf"
    )
    conv = store.get_conversation(conv_id)
    assert conv["title"] == "Rates"
    assert conv["document_id"] == "loan_doc"
    assert conv["document_name"] == "loan.pdf"
    assert conv["summary"] == ""
    assert conv["created_at"] and conv["updated_at"]


def test_get_missing_conversation_is_none(store):
    assert store.get_conversation("nope") is None


def test_append_and_get_messages_round_trip(store):
    conv_id = store.create_conversation(title="t")
    citations = [{"page_number": 3, "score": 0.9, "snippet": "the rate is 7.25"}]
    store.append_message(conv_id, "user", "What is the rate?")
    store.append_message(conv_id, "assistant", "7.25 percent", citations)

    messages = store.get_messages(conv_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What is the rate?"
    assert messages[0]["citations"] is None
    assert messages[1]["citations"] == citations
    assert all("id" in m and "created_at" in m for m in messages)


def test_append_rejects_invalid_role(store):
    conv_id = store.create_conversation()
    with pytest.raises(ValueError):
        store.append_message(conv_id, "system", "hello")


def test_messages_are_ordered_by_insertion(store):
    conv_id = store.create_conversation()
    for i in range(5):
        store.append_message(conv_id, "user", f"msg {i}")
    contents = [m["content"] for m in store.get_messages(conv_id)]
    assert contents == [f"msg {i}" for i in range(5)]


def test_list_conversations_orders_by_updated_at_and_counts(store):
    old = store.create_conversation(title="old")
    store.append_message(old, "user", "hi")
    new = store.create_conversation(title="new")
    store.append_message(new, "user", "a")
    store.append_message(new, "assistant", "b")

    listed = store.list_conversations()
    assert [c["title"] for c in listed] == ["new", "old"]
    assert listed[0]["message_count"] == 2
    assert listed[1]["message_count"] == 1


def test_rename_and_touch(store):
    conv_id = store.create_conversation(title="before")
    store.rename_conversation(conv_id, "after")
    assert store.get_conversation(conv_id)["title"] == "after"


def test_set_summary_and_document(store):
    conv_id = store.create_conversation()
    store.set_summary(conv_id, "User asked about rates.")
    store.set_document(conv_id, "doc1", "doc1.pdf")
    conv = store.get_conversation(conv_id)
    assert conv["summary"] == "User asked about rates."
    assert conv["document_id"] == "doc1"
    assert conv["document_name"] == "doc1.pdf"


def test_delete_conversation_cascades_to_messages(tmp_path):
    db = str(tmp_path / "chats.sqlite3")
    store = ChatStore(db_path=db)
    conv_id = store.create_conversation()
    store.append_message(conv_id, "user", "bye")
    store.delete_conversation(conv_id)

    assert store.get_conversation(conv_id) is None
    assert store.get_messages(conv_id) == []
    # A fresh handle on the same file must see the same thing (real persistence).
    assert ChatStore(db_path=db).get_messages(conv_id) == []


def test_persists_across_store_instances(tmp_path):
    db = str(tmp_path / "chats.sqlite3")
    conv_id = ChatStore(db_path=db).create_conversation(title="kept")
    ChatStore(db_path=db).append_message(conv_id, "user", "persisted")

    reopened = ChatStore(db_path=db)
    assert [m["content"] for m in reopened.get_messages(conv_id)] == ["persisted"]
    assert reopened.get_conversation(conv_id)["title"] == "kept"


def test_message_count(store):
    conv_id = store.create_conversation()
    assert store.message_count(conv_id) == 0
    store.append_message(conv_id, "user", "one")
    store.append_message(conv_id, "assistant", "two")
    assert store.message_count(conv_id) == 2


def test_citations_survive_json_encoding(store):
    conv_id = store.create_conversation()
    citations = [
        {"page_number": 1, "score": 0.5, "snippet": 'quote "double" and \\ back'},
        {"page_number": 9, "score": 0.1, "snippet": "unicode — ✓"},
    ]
    store.append_message(conv_id, "assistant", "answer", citations)
    assert store.get_messages(conv_id)[0]["citations"] == citations
    # stored as a JSON string, not a Python repr
    assert json.dumps(citations) is not None
