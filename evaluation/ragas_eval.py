"""Optional grounding report over offline answer samples.

`ragas` is a heavy, network-hungry dependency, so this module stays opt-in and
works without it: it re-scores the answers produced by ``run_eval.py`` with
``evaluation.metrics.groundedness`` - a claim-level, sentence-aware token-F1
against the retrieved snippets (the offline analogue of RAGAS faithfulness).

    python -m evaluation.ragas_eval --input evaluation/results.json

``run_eval.py`` already reports the same per-row grounding; this script is the
standalone way to recompute it from a saved ``results.json`` without re-running
the pipeline. A real RAGAS faithfulness run needs an LLM-armed evaluator and is
intentionally deferred to environments that provision keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import settings
from evaluation import metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="evaluation/results.json", help="run_eval.py output")
    parser.add_argument("--out", default="evaluation/ragas_report.json")
    args = parser.parse_args()

    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.input)
    if not results_path.exists():
        raise SystemExit(f"{args.input} not found -- run evaluation.run_eval first")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["rows"]

    per_id = []
    for row in rows:
        context = "\n".join(row.get("source_snippets", []))
        ground = metrics.groundedness(row["answer_text"], context)
        per_id.append({
            "id": row["id"],
            "groundedness": ground["score"],
            "ungrounded_fraction": ground["ungrounded_fraction"],
            "claims": ground["claims"],
            "answer_chars": len(row["answer_text"]),
            "snippet_count": len(row.get("source_snippets", [])),
        })

    report = {
        "framework": "claim-level token-F1 vs retrieved snippets "
                     "(offline; no RAGAS LLM judge enabled)",
        "config": {
            "rerank_top_k": settings.TOP_K_RERANK,
            "model": settings.EMBEDDING_MODEL_NAME,
        },
        "rows": per_id,
        "summary": {
            "mean_groundedness": round(
                sum(r["groundedness"] for r in per_id) / max(1, len(per_id)), 4
            ),
            "mean_ungrounded_fraction": round(
                sum(r["ungrounded_fraction"] for r in per_id) / max(1, len(per_id)), 4
            ),
        },
    }
    Path(args.out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()