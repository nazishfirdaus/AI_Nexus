"""Document library: list, upload/ingest, and delete one tenant's documents."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from src.api.config import api_settings
from src.api.dependencies import get_pipeline, get_user_id, rate_limit
from src.api.schemas import DocumentOut, IngestResponse
from src.auth.tenancy import user_data_dir
from src.errors import DuplicateDocumentError
from src.pipeline_factory import PipelineBundle

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(rate_limit)],
)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
) -> list[DocumentOut]:
    return [DocumentOut(**row) for row in bundle.document_store.list()]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: str,
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
) -> DocumentOut:
    row = bundle.document_store.get(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentOut(**row)


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, PPTX, HTML, TXT, CSV, XLSX or image")],
    user_id: Annotated[str, Depends(get_user_id)],
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
) -> IngestResponse:
    filename = Path(file.filename or "").name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required.")

    upload_dir = Path(user_data_dir(user_id)) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid4().hex}_{filename}"
    limit = api_settings.max_upload_mb * 1024 * 1024

    size = 0
    try:
        with target.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{filename} exceeds the {api_settings.max_upload_mb} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        result = await run_in_threadpool(
            bundle.pipeline.ingest_document, target, filename
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - client-facing 500
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed for {filename}.",
        ) from exc
    finally:
        # The corpus lives in Qdrant + the registry; the temporary upload is not
        # needed after parsing and would otherwise accumulate on disk.
        target.unlink(missing_ok=True)

    return IngestResponse(**asdict(result))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
):
    if bundle.document_store.get(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    bundle.pipeline.remove_document(document_id)
    return None
