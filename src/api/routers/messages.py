"""Message-level actions (feedback)."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_chat_store, rate_limit
from src.api.schemas import FeedbackIn, MessageOut
from src.memory.chat_store import ChatStore

router = APIRouter(
    prefix="/messages",
    tags=["messages"],
    dependencies=[Depends(rate_limit)],
)


@router.post("/{message_id}/feedback", response_model=MessageOut)
def set_feedback(
    message_id: int,
    body: FeedbackIn,
    store: Annotated[ChatStore, Depends(get_chat_store)],
) -> MessageOut:
    if store.get_message(message_id) is None:
        raise HTTPException(status_code=404, detail="Message not found.")
    store.set_feedback(message_id, body.value)
    return MessageOut(**store.get_message(message_id))
