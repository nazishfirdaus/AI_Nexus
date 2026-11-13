"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    page_number: int = 0
    score: float = 0.0
    snippet: str = ""
    document_id: str = ""
    filename: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    user_message_id: int
    assistant_message_id: int
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_s: float = 0.0
    intent: str = ""
    status: Literal["answered", "no_context"] = "answered"


class ConversationCreate(BaseModel):
    title: str = Field(default="", max_length=200)
    document_id: Optional[str] = None
    document_name: Optional[str] = None


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationOut(BaseModel):
    id: str
    title: str = ""
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    summary: str = ""
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: Optional[list[CitationOut]] = None
    feedback: Optional[Literal["up", "down"]] = None
    created_at: str


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


class FeedbackIn(BaseModel):
    value: Optional[Literal["up", "down"]] = None


class DocumentOut(BaseModel):
    id: str
    filename: str
    pages: int = 0
    chunks: int = 0
    vectors: int = 0
    duration_s: float = 0.0
    content_hash: Optional[str] = None
    created_at: str


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    duration_s: float
    vectors_added: int = 0
    content_hash: Optional[str] = None
    pages_empty_dropped: int = 0
    pages_ocr: int = 0
    pages_overcleaned: int = 0
    chunks_dropped_short: int = 0
    boilerplate_lines_removed: int = 0
    chunks_mean_chars: float = 0.0
    chunks_median_chars: float = 0.0
    chunks_max_chars: int = 0


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "anexus-api"
    environment: str = "development"
    user_id: str
    collection: str
    qdrant_configured: bool
