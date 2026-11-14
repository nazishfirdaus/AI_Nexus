"""Conversation CRUD for the React sidebar and chat view."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from src.api.dependencies import get_chat_store, rate_limit
from src.api.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationRename,
    MessageOut,
)
from src.memory.chat_store import ChatStore

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    dependencies=[Depends(rate_limit)],
)


def _conversation(row: dict) -> ConversationOut:
    return ConversationOut(
        id=row["id"],
        title=row.get("title") or "",
        document_id=row.get("document_id"),
        document_name=row.get("document_name"),
        summary=row.get("summary") or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=int(row.get("message_count") or 0),
    )


def _require(store: ChatStore, conversation_id: str) -> dict:
    row = store.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return row


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    store: Annotated[ChatStore, Depends(get_chat_store)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ConversationOut]:
    return [_conversation(row) for row in store.list_conversations(limit=limit)]


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    body: ConversationCreate,
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ConversationOut:
    conversation_id = store.create_conversation(
        title=body.title,
        document_id=body.document_id,
        document_name=body.document_name,
    )
    return _conversation(_require(store, conversation_id))


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ConversationDetail:
    row = _require(store, conversation_id)
    messages = [MessageOut(**item) for item in store.get_messages(conversation_id)]
    return ConversationDetail(**_conversation(row).model_dump(), messages=messages)


@router.patch("/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: str,
    body: ConversationRename,
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ConversationOut:
    _require(store, conversation_id)
    store.rename_conversation(conversation_id, body.title)
    return _conversation(_require(store, conversation_id))


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> Response:
    _require(store, conversation_id)
    store.delete_conversation(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
