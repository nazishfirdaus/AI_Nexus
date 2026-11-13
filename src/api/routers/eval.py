"""Evaluation endpoints: expose the artifacts produced by ``evaluation/run_eval.py``.

Only files matching ``results*.json`` inside ``evaluation/`` are readable, which
keeps the route from becoming a general-purpose file server.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import rate_limit
from src.api.schemas import EvalRun, EvalRunSummary

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/eval",
    tags=["evaluation"],
    dependencies=[Depends(rate_limit)],
)

EVAL_DIR = Path(__file__).resolve().parents[3] / "evaluation"
_NAME_RE = re.compile(r"^results[A-Za-z0-9_.-]*\.json$")


def _files() -> list[Path]:
    if not EVAL_DIR.is_dir():
        return []
    return sorted(
        (path for path in EVAL_DIR.iterdir() if path.is_file() and _NAME_RE.match(path.name)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not parse %s: %s", path.name, exc)
        return {}
    return payload if isinstance(payload, dict) else {"rows": payload}


def _summarise(path: Path, payload: dict[str, Any]) -> EvalRunSummary:
    """Normalise the two artifact shapes the harness writes.

    * ``{"summary": {...}, "rows": [...]}`` — a single evaluation run
    * ``{"runs": [{...}, ...]}`` — a cross-run comparison
    """
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    runs = payload.get("runs")
    rows = payload.get("rows")

    if not isinstance(rows, list):
        if isinstance(runs, list):
            summary = {"runs": len(runs)}
            rows = runs
        else:
            rows = []

    items = summary.get("items")
    if not isinstance(items, int):
        counted = [
            int(row.get("items") or 0)
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("items"), int)
        ]
        items = max(counted) if counted else len(rows)

    return EvalRunSummary(
        name=path.name,
        modified_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        items=items,
        summary=summary,
    )


@router.get("/runs", response_model=list[EvalRunSummary])
def list_runs() -> list[EvalRunSummary]:
    return [_summarise(path, _read(path)) for path in _files()]


@router.get("/runs/{name}", response_model=EvalRun)
def get_run(name: str) -> EvalRun:
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid run name.")
    path = EVAL_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run not found.")

    payload = _read(path)
    base = _summarise(path, payload)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        runs = payload.get("runs")
        rows = runs if isinstance(runs, list) else []
    return EvalRun(
        **base.model_dump(),
        rows=[row for row in rows if isinstance(row, dict)],
    )