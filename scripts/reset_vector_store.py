"""Reset Qdrant vector collections after an embedding-model change.

Existing vectors live in the embedding model's vector space and cannot be
searched with a different model (or a different dimension). This script deletes
the collection(s) so the next ingest starts clean; it never touches the document
registry, chat history or credentials.

Usage:
    python -m scripts.reset_vector_store --yes
    python -m scripts.reset_vector_store --all-users --yes
    python -m scripts.reset_vector_store --collection my_collection --yes

Re-ingest your documents after running it.
"""
from __future__ import annotations

import argparse
import sys

from config import settings


def _targets(names: list[str], *, collection: str, all_users: bool) -> list[str]:
    if all_users:
        prefix = f"{settings.COLLECTION_NAME}__"
        return sorted(
            name
            for name in names
            if name == settings.COLLECTION_NAME or name.startswith(prefix)
        )
    return [name for name in names if name == collection]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=settings.COLLECTION_NAME,
        help="Collection to delete (default: the shared collection)",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Delete the shared collection and every per-tenant collection",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation (required for non-interactive use)",
    )
    args = parser.parse_args(argv)

    from src.pipeline_factory import get_qdrant_client

    client = get_qdrant_client()
    endpoint = settings.QDRANT_URL or "(configured endpoint)"
    names = [c.name for c in client.get_collections().collections]
    targets = _targets(names, collection=args.collection, all_users=args.all_users)

    if not targets:
        print(f"[reset] no matching collections at {endpoint}")
        return 0

    print(f"[reset] will delete {len(targets)} collection(s) from {endpoint}:")
    for name in targets:
        print(f"  - {name}")

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "[reset] refusing to delete without confirmation; pass --yes"
            )
        answer = input("Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("[reset] aborted")
            return 1

    for name in targets:
        try:
            client.delete_collection(collection_name=name)
            print(f"[reset] deleted {name}")
        except Exception as exc:  # keep going so one bad namespace cannot block
            print(f"[reset] could not delete {name}: {type(exc).__name__}")

    print("[reset] done. Re-ingest your documents to rebuild the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
