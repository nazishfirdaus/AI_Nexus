"""Agent endpoints: introspect and drive the LangGraph agent.

This is the ``src/agents`` graph (planner → retriever → responder), not the RAG
pipeline behind ``/api/chat``. It has its own collection and its own memory, so
the two paths are reported separately rather than merged.

NeMo guardrails are expensive to build, so they are created on first use behind
a lock; if they fail, the gate is reported as unavailable and the graph still
runs (a missing gate must not block the agent).
"""
from __future__ import annotations

import logging
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import rate_limit
from src.api.schemas import AgentEdge, AgentGraphOut, AgentQuery, AgentResponse
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    dependencies=[Depends(rate_limit)],
)

_rails_lock = threading.Lock()
_rails_ready = False
_rails_failed = False

# Imported lazily: building the planner/responder clients at import time would
# slow every API start, including health checks.
_graph: Any = None
_graph_lock = threading.Lock()


def _agent():
    global _graph
    with _graph_lock:
        if _graph is None:
            from src.agents.graph import rag_agent

            _graph = rag_agent
        return _graph


def _ensure_rails() -> bool:
    """Build NeMo guardrails once. Returns True when the gate is usable."""
    global _rails_ready, _rails_failed
    if _rails_ready:
        return True
    if _rails_failed:
        return False
    with _rails_lock:
        if _rails_ready:
            return True
        if _rails_failed:
            return False
        try:
            from src.guardrails import initialize_rails

            initialize_rails()
            _rails_ready = True
            return True
        except Exception:  # noqa: BLE001 - the gate is optional
            _rails_failed = True
            logger.exception("Guardrails could not be initialised")
            return False


def _nodes(graph) -> list[str]:
    return [name for name in graph.nodes if not name.startswith("__")]


def _edges(graph) -> list[AgentEdge]:
    edges: list[AgentEdge] = []
    for edge in getattr(graph, "edges", None) or []:
        start, end = edge[0], edge[1]
        if not isinstance(start, str) or not isinstance(end, str):
            continue
        if start.startswith("__") and end == "__end__":
            continue
        edges.append(AgentEdge(from_node=start, to_node=end))
    return edges


@router.get("", response_model=AgentGraphOut)
def agent_info() -> AgentGraphOut:
    graph = _agent().get_graph()
    return AgentGraphOut(
        nodes=_nodes(graph),
        edges=_edges(graph),
        mermaid=graph.draw_mermaid(),
        entry_point="planner",
        checkpointer=type(_agent().checkpointer).__name__,
        guardrails_ready=_rails_ready,
        corpus=settings.COLLECTION_NAME,
    )


@router.post("/query", response_model=AgentResponse)
async def agent_query(body: AgentQuery) -> AgentResponse:
    from src.guardrails import guard

    if _ensure_rails():
        fired, rail_response = await run_in_threadpool(guard, body.q)
        if fired:
            return AgentResponse(
                question=body.q,
                answer=rail_response or "",
                thought_process=["Intent: guardrail blocked the request", "Retrieval: skipped"],
                status="Blocked by guardrails.",
                sources=[],
                guardrails="blocked",
            )
    else:
        logger.warning("Guardrails unavailable; running the agent without the gate.")

    initial_state = {
        "messages": [{"role": "user", "content": body.q}],
        "current_query": body.q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    config = {"configurable": {"thread_id": body.thread_id}}

    try:
        final = await run_in_threadpool(_agent().invoke, initial_state, config)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.exception("Agent run failed for thread %s", body.thread_id)
        return AgentResponse(
            question=body.q,
            answer="The agent run failed. See the API logs for details.",
            thought_process=["Error during graph execution."],
            status="error",
            sources=[],
            guardrails="unavailable" if not _rails_ready else "passed",
        )

    return AgentResponse(
        question=body.q,
        answer=final.get("final_answer") or "",
        thought_process=list(final.get("plan") or []),
        status=final.get("status") or "",
        sources=[str(item) for item in (final.get("documents") or [])],
        guardrails="passed" if _rails_ready else "unavailable",
    )