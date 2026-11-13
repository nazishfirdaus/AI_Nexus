"""HTTP API tests.

The pipeline is replaced with a small in-test double (the real pipeline is
already covered by the pipeline/retrieval suites and takes ~70 s to import).
The chat store is a real SQLite database in ``tmp_path`` — the same code path
the API uses in production.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.main import create_app
from src.generation.aggregator import Citation, FinalAnswer
from src.memory.chat_store import ChatStore


def _answer(text: str) -> FinalAnswer:
    return FinalAnswer(
        text=f"Answer to: {text}",
        provider="test",
        model="test-model",
        latency_s=0.01,
        citations=[
            Citation(
                page_number=1,
                score=0.9,
                snippet="The loan amount is 250,000.",
                document_id="loan.pdf",
                filename="loan.pdf",
            )
        ],
        intent="document",
    )


@dataclass
class _IngestResult:
    pages: int
    chunks: int
    document_id: str
    filename: str
    duration_s: float
    vectors_added: int = 0


class _StubStream:
    def __init__(self) -> None:
        self.final = None

    def __iter__(self):
        yield "Hello"
        yield " world"
        self.final = _answer("Hello world")


class _StubPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.uploaded: list[str] = []

    def answer_query(
        self,
        query,
        chat_history=None,
        conversation_summary=None,
        document_id=None,
    ):
        self.calls.append(
            {
                "query": query,
                "chat_history": list(chat_history or []),
                "conversation_summary": conversation_summary,
                "document_id": document_id,
            }
        )
        return _answer(query)

    def answer_query_stream(
        self,
        query,
        chat_history=None,
        conversation_summary=None,
        document_id=None,
    ):
        self.calls.append(
            {
                "query": query,
                "chat_history": list(chat_history or []),
                "conversation_summary": conversation_summary,
                "document_id": document_id,
            }
        )
        return _StubStream()

    def ingest_document(self, file_path, source_name=None):
        self.uploaded.append(source_name)
        return _IngestResult(
            pages=1,
            chunks=1,
            document_id=source_name,
            filename=source_name,
            duration_s=0.01,
            vectors_added=1,
        )

    def remove_document(self, document_id: str) -> bool:
        return True


class _StubDocumentStore:
    def __init__(self) -> None:
        self._rows = {
            "loan.pdf": {
                "id": "loan.pdf",
                "filename": "loan.pdf",
                "pages": 2,
                "chunks": 4,
                "vectors": 4,
                "duration_s": 0.5,
                "content_hash": "abc123",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        }

    def list(self):
        return list(self._rows.values())

    def get(self, document_id):
        return self._rows.get(document_id)


@dataclass
class _StubBundle:
    pipeline: _StubPipeline = field(default_factory=_StubPipeline)
    document_store: _StubDocumentStore = field(default_factory=_StubDocumentStore)


@pytest.fixture()
def bundle() -> _StubBundle:
    return _StubBundle()


@pytest.fixture()
def client(tmp_path, bundle):
    app = create_app(api_key="")
    store = ChatStore(tmp_path / "chats.sqlite3")
    dependencies.reset_caches()
    app.dependency_overrides[dependencies.get_pipeline] = lambda: bundle
    app.dependency_overrides[dependencies.get_chat_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    dependencies.reset_caches()


# ---------------------------------------------------------------- health/auth
def test_health_names_the_tenant_collection(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["user_id"] == "dev"
    assert body["collection"].endswith("__dev")


def test_invalid_user_header_is_rejected(client):
    response = client.get("/api/health", headers={"X-User-Id": "bad id!"})
    assert response.status_code == 400


def test_explicit_user_header_scopes_the_collection(client):
    response = client.get("/api/health", headers={"X-User-Id": "alice"})
    assert response.status_code == 200
    assert response.json()["collection"].endswith("__alice")


def test_optional_api_key_is_enforced(tmp_path):
    app = create_app(api_key="secret")
    store = ChatStore(tmp_path / "chats.sqlite3")
    app.dependency_overrides[dependencies.get_chat_store] = lambda: store
    with TestClient(app) as test_client:
        assert test_client.get("/api/conversations").status_code == 401
        allowed = test_client.get(
            "/api/conversations", headers={"X-API-Key": "secret"}
        )
        assert allowed.status_code == 200


# ---------------------------------------------------------------------- chat
def test_chat_creates_a_conversation_and_persists_both_messages(client):
    response = client.post("/api/chat", json={"message": "What is the loan amount?"})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Answer to: What is the loan amount?"
    assert body["status"] == "answered"
    assert body["citations"][0]["filename"] == "loan.pdf"
    assert body["user_message_id"] != body["assistant_message_id"]

    detail = client.get(f"/api/conversations/{body['conversation_id']}").json()
    assert detail["title"] == "What is the loan amount?"
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
    ]
    assert detail["messages"][1]["citations"][0]["page_number"] == 1


def test_follow_up_reuses_the_conversation_and_sends_history(client, bundle):
    first = client.post("/api/chat", json={"message": "What is the loan amount?"}).json()
    second = client.post(
        "/api/chat",
        json={
            "message": "And the interest rate?",
            "conversation_id": first["conversation_id"],
        },
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    history = bundle.pipeline.calls[-1]["chat_history"]
    assert [message["role"] for message in history] == ["user", "assistant"]


def test_document_scope_is_forwarded(client, bundle):
    response = client.post(
        "/api/chat",
        json={"message": "Summarise it", "document_id": "loan.pdf"},
    )
    assert response.status_code == 200
    assert bundle.pipeline.calls[-1]["document_id"] == "loan.pdf"


def test_unknown_conversation_is_404(client):
    response = client.post(
        "/api/chat",
        json={"message": "Hello", "conversation_id": "missing"},
    )
    assert response.status_code == 404


def test_empty_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_stream_emits_deltas_and_persists_the_answer(client):
    response = client.post("/api/chat/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert "event: delta" in response.text
    assert "event: done" in response.text

    done = [
        line
        for line in response.text.splitlines()
        if line.startswith("data:") and '"citations"' in line
    ][-1]
    payload = json.loads(done.removeprefix("data:").strip())
    assert payload["answer"] == "Answer to: Hello world"

    detail = client.get(f"/api/conversations/{payload['conversation_id']}").json()
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
    ]


# --------------------------------------------------------------- conversations
def test_rename_and_delete_conversation(client):
    created = client.post(
        "/api/conversations", json={"title": "Loan review"}
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    renamed = client.patch(
        f"/api/conversations/{conversation_id}",
        json={"title": "Renamed"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_feedback_round_trip(client):
    chat = client.post("/api/chat", json={"message": "Hello"}).json()
    message_id = chat["assistant_message_id"]

    rated = client.post(
        f"/api/messages/{message_id}/feedback", json={"value": "up"}
    )
    assert rated.status_code == 200
    assert rated.json()["feedback"] == "up"

    cleared = client.post(
        f"/api/messages/{message_id}/feedback", json={"value": None}
    )
    assert cleared.json()["feedback"] is None


def test_feedback_rejects_unknown_message_and_invalid_value(client):
    assert (
        client.post("/api/messages/999/feedback", json={"value": "up"}).status_code
        == 404
    )

    chat = client.post("/api/chat", json={"message": "Hello"}).json()
    invalid = client.post(
        f"/api/messages/{chat['assistant_message_id']}/feedback",
        json={"value": "sideways"},
    )
    assert invalid.status_code == 422


# ----------------------------------------------------------------- documents
def test_documents_are_listed_and_fetched(client):
    listed = client.get("/api/documents")
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()] == ["loan.pdf"]

    one = client.get("/api/documents/loan.pdf")
    assert one.status_code == 200
    assert one.json()["chunks"] == 4

    assert client.get("/api/documents/missing.pdf").status_code == 404
    assert client.delete("/api/documents/loan.pdf").status_code == 204
    assert client.delete("/api/documents/missing.pdf").status_code == 404


def test_upload_ingests_the_file(client, bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.api.routers.documents.user_data_dir", lambda user_id: str(tmp_path)
    )
    response = client.post(
        "/api/documents",
        files={"file": ("loan.txt", b"hello loan", "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "loan.txt"
    assert body["vectors_added"] == 1
    assert bundle.pipeline.uploaded == ["loan.txt"]
    assert not list((tmp_path / "uploads").iterdir())


# ---------------------------------------------------------------- rate limiting
def test_rate_limit_rejects_the_second_request(client, monkeypatch):
    """The limiter is per client host across every /api router."""
    monkeypatch.setattr(dependencies._limiter, "limit", 1)
    dependencies._limiter.clear()
    try:
        assert client.get("/api/conversations").status_code == 200
        assert client.get("/api/conversations").status_code == 429
    finally:
        dependencies._limiter.clear()


# ----------------------------------------------------------------------- agent
class _StubGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def draw_mermaid(self):
        return "graph TD;\n  planner --> retriever;"


class _StubAgent:
    """Stands in for the compiled LangGraph agent."""

    def __init__(self, state=None, *, raises=False):
        from langgraph.checkpoint.memory import MemorySaver

        self.checkpointer = MemorySaver()
        self._state = state or {}
        self._raises = raises
        self.invoked = 0

    def get_graph(self):
        return _StubGraph(
            {"planner": object(), "retriever": object(), "responder": object()},
            [
                ("__start__", "planner"),
                ("planner", "retriever"),
                ("planner", "responder"),
                ("retriever", "responder"),
                ("responder", "__end__"),
            ],
        )

    def invoke(self, state, config=None):
        self.invoked += 1
        if self._raises:
            raise RuntimeError("graph exploded")
        return self._state


def _use_agent(monkeypatch, stub):
    from src.api.routers import agent as agent_router

    monkeypatch.setattr(agent_router, "_agent", lambda: stub)
    return agent_router


def test_agent_info_reports_the_compiled_graph(client, monkeypatch):
    stub = _StubAgent()
    _use_agent(monkeypatch, stub)

    response = client.get("/api/agent")
    assert response.status_code == 200
    body = response.json()

    assert body["nodes"] == ["planner", "retriever", "responder"]
    assert body["entry_point"] == "planner"
    assert body["checkpointer"] == type(stub.checkpointer).__name__
    assert {"from_node": "planner", "to_node": "retriever"} in body["edges"]
    assert "graph TD;" in body["mermaid"]
    assert isinstance(body["guardrails_ready"], bool)


def test_agent_query_runs_the_graph_behind_passing_guardrails(client, monkeypatch):
    stub = _StubAgent(
        {
            "final_answer": "hello",
            "plan": ["Intent: Technical", "Search Term: hi"],
            "status": "Response generated.",
            "documents": ["doc1"],
        }
    )
    agent_router = _use_agent(monkeypatch, stub)
    monkeypatch.setattr(agent_router, "_ensure_rails", lambda: True)
    monkeypatch.setattr("src.guardrails.guard", lambda q: (False, None))

    response = client.post("/api/agent/query", json={"q": "hi", "thread_id": "t1"})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "hello"
    assert body["status"] == "Response generated."
    assert body["sources"] == ["doc1"]
    assert body["thought_process"] == ["Intent: Technical", "Search Term: hi"]
    assert body["guardrails"] == "passed"
    assert stub.invoked == 1


def test_agent_query_returns_blocked_response_without_running_the_graph(
    client, monkeypatch
):
    stub = _StubAgent()
    agent_router = _use_agent(monkeypatch, stub)
    monkeypatch.setattr(agent_router, "_ensure_rails", lambda: True)
    monkeypatch.setattr("src.guardrails.guard", lambda q: (True, "Blocked message"))

    body = client.post("/api/agent/query", json={"q": "bad prompt"}).json()

    assert body["status"] == "Blocked by guardrails."
    assert body["answer"] == "Blocked message"
    assert body["guardrails"] == "blocked"
    assert body["sources"] == []
    assert stub.invoked == 0


def test_agent_query_runs_when_guardrails_are_unavailable(client, monkeypatch):
    stub = _StubAgent({"final_answer": "ok", "plan": [], "status": "done", "documents": []})
    agent_router = _use_agent(monkeypatch, stub)
    monkeypatch.setattr(agent_router, "_ensure_rails", lambda: False)

    body = client.post("/api/agent/query", json={"q": "hi"}).json()

    assert body["guardrails"] == "unavailable"
    assert body["answer"] == "ok"
    assert stub.invoked == 1


def test_agent_query_reports_a_failed_run(client, monkeypatch):
    stub = _StubAgent(raises=True)
    agent_router = _use_agent(monkeypatch, stub)
    monkeypatch.setattr(agent_router, "_ensure_rails", lambda: False)

    response = client.post("/api/agent/query", json={"q": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "failed" in body["answer"].lower()


def test_agent_query_rejects_an_empty_question(client):
    assert client.post("/api/agent/query", json={"q": ""}).status_code == 422


# ------------------------------------------------------------------------ eval
@pytest.fixture()
def eval_dir(tmp_path, monkeypatch):
    """Point the eval router at a synthetic evaluation/ directory."""
    from src.api.routers import eval as eval_router

    directory = tmp_path / "evaluation"
    directory.mkdir()
    (directory / "results.json").write_text(
        json.dumps(
            {
                "summary": {"items": 2, "recall_at_k": 0.5},
                "rows": [{"id": "q1", "recall_at_k": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    (directory / "results_multi.json").write_text(
        json.dumps({"runs": [{"items": 4}, {"items": 6}]}), encoding="utf-8"
    )
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")
    monkeypatch.setattr(eval_router, "EVAL_DIR", directory)
    return directory


def test_eval_runs_lists_only_result_artifacts(client, eval_dir):
    response = client.get("/api/eval/runs")
    assert response.status_code == 200
    names = [run["name"] for run in response.json()]

    assert sorted(names) == ["results.json", "results_multi.json"]


def test_eval_run_returns_rows_and_summary(client, eval_dir):
    response = client.get("/api/eval/runs/results.json")
    assert response.status_code == 200
    body = response.json()

    assert body["items"] == 2
    assert body["summary"]["recall_at_k"] == 0.5
    assert body["rows"][0]["id"] == "q1"


def test_eval_run_normalises_a_multi_run_artifact(client, eval_dir):
    body = client.get("/api/eval/runs/results_multi.json").json()

    assert body["items"] == 6
    assert body["summary"] == {"runs": 2}
    assert len(body["rows"]) == 2


def test_eval_run_rejects_unknown_and_malformed_names(client, eval_dir):
    # The name must look like results*.json to be considered at all.
    assert client.get("/api/eval/runs/notes.txt").status_code == 400
    assert client.get("/api/eval/runs/results_missing.json").status_code == 404
