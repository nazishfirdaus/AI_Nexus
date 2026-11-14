"""Store the original uploaded file alongside one tenant's corpus.

The parsed text lives in Qdrant; the registry points at this stored copy so a
citation can open the real PDF (or DOCX/XLSX/image) at the cited page. Files are
kept inside the tenant's data directory and are removed with the document.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from pathlib import Path

from config import settings
from src.auth.tenancy import user_data_dir

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PRUNE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "data",
    "processed_data",
}
_MAX_SEARCH_FILES = 50_000

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def safe_filename(name: str) -> str:
    """Return a single path component that is safe on Windows and POSIX."""
    base = Path(str(name or "")).name
    cleaned = _INVALID_CHARS.sub("_", base).strip(" .")
    if not cleaned:
        return "document"
    if cleaned.split(".")[0].upper() in _RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def store_source(directory: str | Path, document_id: str, temp_path: str | Path) -> Path:
    """Move ``temp_path`` into ``directory`` under the document's own filename."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / safe_filename(document_id)
    if destination.exists():
        destination.unlink()
    shutil.move(str(temp_path), str(destination))
    return destination.resolve()


def remove_source(path: str | Path | None) -> None:
    """Best-effort delete of a stored source file."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove stored source file %s", path, exc_info=True)


def _sha256_file(path: Path) -> str | None:
    """Stream a file through SHA-256; return None when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _configured_roots() -> list[Path]:
    raw = getattr(settings, "SOURCE_SEARCH_ROOTS", "") or ""
    if not raw:
        return []
    raw = raw.replace(os.pathsep, ",")
    roots = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        path = Path(part).expanduser()
        roots.append(path if path.is_absolute() else _PROJECT_ROOT / path)
    return roots


def _candidate_roots(
    user_id: str | None,
    extra_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> list[Path]:
    """Directories that may contain a legacy corpus file.

    Tenant directories come first (managed uploads and the stored copy), then
    the checkout itself (which includes Dataset), then explicit/configured
    roots. Only these roots are ever searched, never an arbitrary path from the
    database.
    """
    roots: list[Path] = []
    if user_id:
        tenant = Path(user_data_dir(user_id))
        roots.extend([tenant / "documents", tenant / "uploads"])
    roots.append(_PROJECT_ROOT)
    roots.extend(Path(path) for path in (extra_roots or ()))
    roots.extend(_configured_roots())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _walk_candidates(root: Path, filename: str):
    """Yield files named ``filename`` under one root, pruning heavy folders."""
    if not root.is_dir():
        return
    direct = root / filename
    if direct.is_file():
        yield direct

    inspected = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in _PRUNE_DIRS and not name.startswith(".")
        ]
        inspected += len(filenames)
        if inspected > _MAX_SEARCH_FILES:
            logger.warning("Stopped source-file search under %s: too many files", root)
            return
        if filename in filenames:
            candidate = Path(dirpath) / filename
            if candidate.is_file():
                yield candidate


def find_source_file(
    filename: str,
    content_hash: str | None,
    *,
    user_id: str | None = None,
    extra_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path | None:
    """Locate a legacy source file and verify it against the registry hash.

    A hash match is required when the registry has one, so a same-named file
    from another corpus can never be served by mistake. Without a hash, only a
    unique basename match is accepted.
    """
    name = Path(str(filename or "")).name
    if not name:
        return None

    seen: set[Path] = set()
    unverified: list[Path] = []
    for root in _candidate_roots(user_id, extra_roots):
        for candidate in _walk_candidates(root, name):
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if content_hash:
                if _sha256_file(candidate) == content_hash:
                    return resolved
            else:
                unverified.append(resolved)

    if not content_hash and len(unverified) == 1:
        return unverified[0]
    return None


def backfill_source_paths(
    user_id: str,
    store,
    *,
    extra_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> int:
    """Attach discoverable originals to legacy registry rows.

    Rows written before source retention have ``storage_path = NULL``. A
    hash-verified match is recorded as an unmanaged reference, so deleting the
    document never deletes the user's original corpus file.
    """
    attached = 0
    try:
        rows = store.list()
    except Exception:  # noqa: BLE001 - backfill must never block startup
        logger.warning("Could not list documents for source backfill", exc_info=True)
        return 0

    for row in rows:
        existing = row.get("storage_path")
        if existing and Path(existing).is_file():
            continue
        found = find_source_file(
            row.get("filename") or row.get("id") or "",
            row.get("content_hash"),
            user_id=user_id,
            extra_roots=extra_roots,
        )
        if found is None:
            continue
        set_storage = getattr(store, "set_storage_path", None)
        if set_storage is None:
            continue
        try:
            set_storage(row["id"], str(found), managed=False)
        except Exception:  # noqa: BLE001 - best-effort per row
            logger.warning(
                "Could not backfill source path for document %s",
                row.get("id"),
                exc_info=True,
            )
            continue
        attached += 1
        logger.info("Backfilled source file for document %s", row.get("id"))

    return attached
