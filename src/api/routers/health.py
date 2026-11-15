"""Liveness/readiness information. Does not build the RAG pipeline."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from config import settings
from src.api.config import api_settings
from src.api.dependencies import get_user_id
from src.api.schemas import HealthOut
from src.auth.tenancy import user_collection_name

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(user_id: Annotated[str, Depends(get_user_id)]) -> HealthOut:
    return HealthOut(
        service="anexus-api",
        environment=api_settings.environment,
        user_id=user_id,
        collection=user_collection_name(user_id),
        qdrant_configured=bool(settings.QDRANT_URL),
    )
