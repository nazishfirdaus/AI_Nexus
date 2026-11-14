"""Bulk ingestion CLI for the ANexus corpus.

Walks a directory and ingests every supported file through the same pipeline the
app uses (parse -> chunk -> embed -> Qdrant -> registry -> BM25). That means:

  * chunk ids are deterministic (``<document_id>_p<page>_c<n>``) and map to
    stable Qdrant point ids, so re-ingesting overwrites instead of duplicating;
  * the payload is ``{"text": ..., "metadata": {...}}`` with document_id,
    page_number and chunk_id, which retrieval, citations and delete expect;
  * documents land in one user's collection (``<COLLECTION_NAME>__<user_id>``)
    and that user's registry, so the React app sees them.

Usage:
    python -m src.ingestion.processor <directory> --user-id <user_id>
    python -m src.ingestion.processor <directory> --user-id <user_id> --wipe
    python -m src.ingestion.processor <directory> --user-id <user_id> --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config.settings import SUPPORTED_EXTENSIONS
from src.observability import configure_observability, hash_id, info, span
from src.pipeline_factory import build_user_pipeline
from src.rag_pipeline import DuplicateDocumentError

logger = logging.getLogger(__name__)
PROCESSED_DATA_DIR = Path("processed_data")


def _supported_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _save_processed(payload: dict, source_type: str, filename: str) -> Path:
    folder = PROCESSED_DATA_DIR / source_type
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{filename}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def run_ingestion(
    base_dir: str | Path,
    user_id: str,
    *,
    wipe: bool = False,
    source_type: str = "general",
    write_json: bool = False,
) -> tuple[int, int, int]:
    """Ingest every supported file under ``base_dir``.

    Returns ``(ok, skipped, failed)``.
    """
    root = Path(base_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    bundle = build_user_pipeline(user_id)
    collection = getattr(bundle.vector_db, "collection_name", "?")
    collection_hash = hash_id(collection)

    if wipe:
        bundle.vector_db.reset()
        bundle.document_store.clear()
        info("Wiped user corpus", collection_hash=collection_hash)

    files = _supported_files(root)
    info("Ingestion started", file_count=len(files), collection_hash=collection_hash)

    ok = skipped = failed = 0
    for path in files:
        try:
            with span(
                "processor.ingest_file",
                file_id=hash_id(str(path)),
                source_type=source_type,
            ):
                result = bundle.pipeline.ingest_document(path, source_name=path.name)
                info(
                    "File ingested",
                    pages=result.pages,
                    chunks=result.chunks,
                    vectors_added=result.vectors_added,
                    duration_s=round(result.duration_s, 3),
                )
                if write_json:
                    _save_processed(
                        {
                            "filename": path.name,
                            "document_id": result.document_id,
                            "source_type": source_type,
                            "pages": result.pages,
                            "chunks": result.chunks,
                            "vectors_added": result.vectors_added,
                        },
                        source_type,
                        path.name,
                    )
            ok += 1
        except DuplicateDocumentError:
            skipped += 1
            info("Duplicate skipped", file_id=hash_id(str(path)))
        except Exception:
            failed += 1
            logger.exception("Failed to ingest %s", hash_id(str(path)))
            info("File failed", file_id=hash_id(str(path)))

    info(
        "Ingestion finished",
        ok=ok,
        skipped=skipped,
        failed=failed,
        collection_hash=collection_hash,
    )
    return ok, skipped, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest a directory into one user's corpus."
    )
    parser.add_argument("directory", help="Folder to scan (recursively)")
    parser.add_argument(
        "--user-id",
        required=True,
        help="Tenant id that owns the collection (the same id the app uses)",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Drop this user's vectors and registry before ingesting",
    )
    parser.add_argument(
        "--source-type",
        default="general",
        help="Tag used only for the optional JSON mirror",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write processed_data/<source_type>/*.json",
    )
    args = parser.parse_args(argv)

    configure_observability(service_name="anexus-bulk-ingest")
    logging.basicConfig(level=logging.INFO)

    ok, skipped, failed = run_ingestion(
        args.directory,
        args.user_id,
        wipe=args.wipe,
        source_type=args.source_type,
        write_json=args.json,
    )
    print(f"ok={ok} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())