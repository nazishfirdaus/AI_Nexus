"""Import documents from S3-compatible object storage.

Works with real AWS S3, MinIO, Cloudflare R2 and anything else exposing the S3
API: the per-user credentials may include a custom `endpoint_url`.

`boto3` is imported lazily and the client is injectable, so the whole module is
unit-testable against a fake client with no network and no heavy import. Only
objects whose extension the parser layer supports are considered, and each object
is size-capped before download.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from config import settings

logger = logging.getLogger(__name__)


class CloudError(Exception):
    """Raised when the storage configuration or a transfer fails."""


def build_s3_client(credentials: dict | None, session=None):
    """Create a boto3 S3 client from a user's stored credentials.

    `session` accepts an existing ``boto3.Session`` (or any object exposing a
    compatible ``.client`` method) for testing.
    """
    creds = credentials or {}
    access_key = creds.get("s3_access_key_id")
    secret_key = creds.get("s3_secret_access_key")
    if not access_key or not secret_key:
        raise CloudError("S3 access key ID and secret access key are required")

    region = creds.get("s3_region") or settings.S3_REGION or None
    endpoint = creds.get("s3_endpoint_url") or settings.S3_ENDPOINT_URL or None

    client_kwargs: dict = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if region:
        client_kwargs["region_name"] = region
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint

    if session is not None:
        return session.client("s3", **client_kwargs)

    import boto3  # lazy: heavy, and absent in some test environments

    return boto3.client("s3", **client_kwargs)


def _is_supported(key: str, extensions: Iterable[str]) -> bool:
    if not key or key.endswith("/"):  # folder placeholder
        return False
    return Path(key).suffix.lower() in tuple(extensions)


def list_objects(
    client,
    bucket: str,
    prefix: str = "",
    *,
    max_files: int = settings.CLOUD_MAX_FILES,
    max_bytes: Optional[int] = None,
    extensions: Iterable[str] = settings.SUPPORTED_EXTENSIONS,
) -> list[dict]:
    """List supported objects under `bucket/prefix` (newest-agnostic, capped).

    Returns ``[{"key", "size", "filename"}]``. Folders, unsupported extensions
    and (when given) objects larger than `max_bytes` are skipped.
    """
    if not bucket:
        raise CloudError("A bucket name is required")
    paginator = client.get_paginator("list_objects_v2")
    found: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            size = int(obj.get("Size", 0) or 0)
            if not _is_supported(key, extensions):
                continue
            if max_bytes is not None and size > max_bytes:
                logger.info("Skipping %s (%d bytes > cap %d)", key, size, max_bytes)
                continue
            found.append({"key": key, "size": size, "filename": os.path.basename(key)})
            if len(found) >= max_files:
                return found
    return found


def get_object_bytes(
    client,
    bucket: str,
    key: str,
    *,
    max_bytes: Optional[int] = None,
) -> bytes:
    """Download one object fully into memory (size-capped)."""
    kwargs: dict = {"Bucket": bucket, "Key": key}
    if max_bytes is not None:
        # Inclusive byte range: ask for at most `max_bytes`.
        kwargs["Range"] = f"bytes=0-{max_bytes - 1}"
    try:
        response = client.get_object(**kwargs)
    except CloudError:
        raise
    except Exception as exc:
        raise CloudError(f"Could not download s3://{bucket}/{key}: {exc}") from exc

    body = response["Body"]
    if hasattr(body, "read"):
        data = body.read()
    else:  # pragma: no cover - defensive for unusual clients
        data = bytes(body)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    return data


def collect_supported_objects(
    client,
    bucket: str,
    prefix: str = "",
    *,
    max_files: int = settings.CLOUD_MAX_FILES,
    max_bytes: Optional[int] = None,
) -> list[tuple[bytes, str]]:
    """Download every supported object as ``(bytes, filename)`` pairs.

    Ready to hand to the existing `ingest_documents` UI flow - the bytes path
    keeps the original filename so the parser can dispatch on the extension.
    """
    objects = list_objects(
        client, bucket, prefix, max_files=max_files, max_bytes=max_bytes
    )
    items: list[tuple[bytes, str]] = []
    for obj in objects:
        data = get_object_bytes(client, bucket, obj["key"], max_bytes=max_bytes)
        items.append((data, obj["filename"]))
    return items
