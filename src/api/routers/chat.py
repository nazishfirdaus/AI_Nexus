"""Chat endpoints: ask a question and persist both sides of the turn.

``POST /api/chat`` is the normal JSON path. ``POST /api/chat/stream`` emits
server-sent events (``meta``, ``delta``, ``done``/``error``) for a typing UI.
Both store the user message before generation and the assistant message after,
so a failed generation never loses the question.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_chat_store,
    get_pipeline,
    rate_limit,
)
from src.api.schemas import ChatRequest, ChatResponse, CitationOut
from src.memory.chat_store import ChatStore
from src.pipeline_factory import PipelineBundle

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(rate_limit)],
)


def _title(text: str) -> str:
    first_line = " ".join(text.split())
    if len(first_line) > 60:
        return first_line[:60].rstrip() + "…"
    return first_line or "New conversation"


def _citations(result) -> list[CitationOut]:
    return [CitationOut(**citation.to_dict()) for citation in (result.citations or [])]


def _history(store: ChatStore, conversation_id: str) -> list[dict]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in store.get_messages(conversation_id)
        if message["role"] in ("user", "assistant")
    ]


def _document_name(bundle: PipelineBundle, document_id: Optional[str]) -> Optional[str]:
    if not document_id:
        return None
    record = bundle.document_store.get(document_id)
    return record["filename"] if record else None


def _resolve_conversation(
    store: ChatStore,
    bundle: PipelineBundle,
    conversation_id: Optional[str],
    message: str,
    document_id: Optional[str],
) -> dict:
    if conversation_id:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        if document_id and document_id != conversation.get("document_id"):
            store.set_document(
                conversation_id,
                document_id,
                _document_name(bundle, document_id),
            )
            conversation = store.get_conversation(conversation_id)
        return conversation

    new_id = store.create_conversation(
        title=_title(message),
        document_id=document_id,
        document_name=_document_name(bundle, document_id),
    )
    conversation = store.get_conversation(new_id)
    if conversation is None:  # pragma: no cover - SQLite insert just succeeded
        raise HTTPException(status_code=500, detail="Could not create conversation.")
    return conversation


def _as_response(
    conversation_id: str,
    user_message_id: int,
    assistant_message_id: int,
    result,
) -> ChatResponse:
    citations = _citations(result)
    return ChatResponse(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        answer=result.text,
        citations=citations,
        provider=getattr(result, "provider", "") or "",
        model=getattr(result, "model", "") or "",
        latency_s=float(getattr(result, "latency_s", 0.0) or 0.0),
        intent=getattr(result, "intent", "") or "",
        status="answered" if citations else "no_context",
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ChatResponse:
    conversation = _resolve_conversation(
        store, bundle, body.conversation_id, body.message, body.document_id
    )
    conversation_id = conversation["id"]
    history = _history(store, conversation_id)
    document_id = body.document_id or conversation.get("document_id")

    user_message_id = store.append_message(conversation_id, "user", body.message)

    try:
        result = bundle.pipeline.answer_query(
            body.message,
            chat_history=history,
            conversation_summary=conversation.get("summary") or None,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - client-facing 502
        logger.exception("RAG pipeline failed for conversation %s", conversation_id)
        raise HTTPException(
            status_code=502,
            detail=(
                "The RAG pipeline failed to answer. The question was saved; "
                "retry or rephrase it."
            ),
        ) from exc

    citations = _citations(result)
    assistant_message_id = store.append_message(
        conversation_id,
        "assistant",
        result.text,
        citations=[citation.model_dump() for citation in citations],
    )
    return _as_response(
        conversation_id, user_message_id, assistant_message_id, result
    )


@router.post("/stream")
def chat_stream(
    body: ChatRequest,
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> StreamingResponse:
    conversation = _resolve_conversation(
        store, bundle, body.conversation_id, body.message, body.document_id
    )
    conversation_id = conversation["id"]
    history = _history(store, conversation_id)
    document_id = body.document_id or conversation.get("document_id")
    user_message_id = store.append_message(conversation_id, "user", body.message)

    try:
        handle = bundle.pipeline.answer_query_stream(
            body.message,
            chat_history=history,
            conversation_summary=conversation.get("summary") or None,
            document_id=document_id,
        )
    except Exception as exc:  # noqa: BLE001 - preparation failed before streaming
        logger.exception("RAG stream preparation failed for %s", conversation_id)
        raise HTTPException(
            status_code=502,
            detail="The RAG pipeline could not start the answer.",
        ) from exc

    def events() -> Iterator[str]:
        yield _sse(
            "meta",
            {
                "conversation_id": conversation_id,
                "user_message_id": user_message_id,
            },
        )
        try:
            for delta in handle:
                if delta:
                    yield _sse("delta", {"text": delta})

            result = handle.final
            if result is None:
                raise RuntimeError("The stream ended without a final answer.")

            citations = _citations(result)
            assistant_message_id = store.append_message(
                conversation_id,
                "assistant",
                result.text,
                citations=[citation.model_dump() for citation in citations],
            )
            payload = _as_response(
                conversation_id, user_message_id, assistant_message_id, result
            )
            yield _sse("done", payload.model_dump())
        except Exception:  # noqa: BLE001 - the response has already started
            logger.exception("RAG stream failed for conversation %s", conversation_id)
            yield _sse(
                "error",
                {
                    "detail": (
                        "The response stream failed. The question was saved; "
                        "retry or rephrase it."
                    )
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
