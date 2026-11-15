"""Document library: list, upload/ingest, and delete one tenant's documents."""
from __future__ import annotations

import logging
from dataclasses import asdict
from mimetypes import guess_type
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from src.api.config import api_settings
from src.api.dependencies import get_pipeline, get_user_id, rate_limit
from src.api.schemas import DocumentOut, IngestResponse
from src.auth.tenancy import user_data_dir
from src.errors import DuplicateDocumentError
from src.ingestion.source_files import find_source_file, remove_source, store_source
from src.pipeline_factory import PipelineBundle

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(rate_limit)],
)

# The frontend opens the fetched file as a blob, so the response MIME type
# decides whether uploaded markup runs as a same-origin document. These types
# are shown as plain text instead.
_UNSAFE_INLINE_SUFFIXES = {
    ".htm",
    ".html",
    ".mhtml",
    ".svg",
    ".xhtml",
    ".xml",
    ".xsl",
    ".xslt",
}


def _document_out(row: dict) -> DocumentOut:
    """Hide the on-disk path but tell the client whether it can open the file."""
    data = {key: value for key, value in row.items() if key in DocumentOut.model_fields}
    path = row.get("storage_path")
    data["file_available"] = bool(path and Path(path).is_file())
    return DocumentOut(**data)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
) -> list[DocumentOut]:
    return [_document_out(row) for row in bundle.document_store.list()]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: str,
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
) -> DocumentOut:
    row = bundle.document_store.get(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _document_out(row)


@router.get("/{document_id}/file")
def get_document_file(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    bundle: Annotated[PipelineBundle, Depends(get_pipeline)],
) -> FileResponse:
    """Serve the original file inline so citations can open the cited page.

    The frontend fetches this with the tenant/API-key headers and opens a blob
    URL (a plain ``<a href>`` cannot carry those headers). ``#page=N`` on the
    resulting URL is handled by the browser's PDF viewer.
    """
    row = bundle.document_store.get(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    raw_path = row.get("storage_path")
    file_path = Path(raw_path) if raw_path else None

    # Legacy rows predate source retention. Before giving up, look for the
    # original in the tenant directory or the checkout and verify its hash.
    if file_path is None or not file_path.is_file():
        found = find_source_file(
            row.get("filename") or document_id,
            row.get("content_hash"),
            user_id=user_id,
        )
        if found is not None:
            set_storage = getattr(bundle.document_store, "set_storage_path", None)
            if set_storage is not None:
                set_storage(document_id, str(found), managed=False)
            row = dict(row)
            row["storage_path"] = str(found)
            row["storage_managed"] = 0
            file_path = found

    if file_path is None or not file_path.is_file():
        if raw_path:
            detail = "The original file for this document is no longer available. Re-upload it."
        else:
            detail = "The original file for this document is not stored. Re-upload it to enable page links."
        raise HTTPException(status_code=404, detail=detail)

    # Files we copied ourselves must stay inside the tenant's data directory.
    if row.get("storage_managed"):
        root = Path(user_data_dir(user_id)).resolve()
        try:
            file_path.resolve().relative_to(root)
        except ValueError:
            logger.warning("Rejected stored source outside tenant directory: %s", file_path)
            raise HTTPException(status_code=404, detail="Document file not found.")

    if file_path.suffix.lower() in _UNSAFE_INLINE_SUFFIXES:
        media_type = "text/plain"
    else:
        media_type = (
            guess_type(row.get("filename") or file_path.name)[0]
            or "application/octet-stream"
        )
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=row.get("filename") or file_path.name,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


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

    stored_path: Path | None = None
    try:
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

        try:
            # Keep the original so a citation can open the real page later.
            stored_path = store_source(
                Path(user_data_dir(user_id)) / "documents",
                result.document_id,
                target,
            )
            set_storage = getattr(bundle.document_store, "set_storage_path", None)
            if set_storage is not None:
                set_storage(result.document_id, str(stored_path), managed=True)
        except Exception as exc:  # noqa: BLE001 - roll back a half-made document
            try:
                bundle.pipeline.remove_document(result.document_id)
            except Exception:  # noqa: BLE001 - best-effort rollback
                logger.exception("Could not roll back failed upload %s", filename)
            if stored_path is not None:
                remove_source(stored_path)
            logger.exception("Could not retain the source for %s", filename)
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed for {filename}.",
            ) from exc
    finally:
        # store_source moved the temp file; this is a no-op on success and
        # cleans up parse failures.
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
