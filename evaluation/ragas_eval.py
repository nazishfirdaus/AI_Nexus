"""Optional RAGAS evaluation over offline answer samples.

`ragas` is a heavy, network-hungry dependency, so this module stays opt-in:

    pip install ragas
    python -m evaluation.ragas_eval --input evaluation/results.json

It reads the answers produced by run_eval.py and reports context relevance
(lower bound: does the answer text overlap any fetched chunk?) plus basic
token-level hallucination indicators -- all without calling an LLM again.

A real RAGAS faithfulness run needs an LLM-armed evaluator and is intentionally
deferred to environments that provision keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import settings


def context_overlap(answer: str, snippets: list[str]) -> float:
    """Fraction of answer tokens that also appear in the retrieved snippets."""
    if not snippets or not answer:
        return 0.0
    answer_tokens = {t.lower() for t in answer.split()}
    corpus = " ".join(snippets).lower()
    if not answer_tokens:
        return 0.0
    overlap = sum(1 for token in answer_tokens if token in corpus)
    return round(overlap / len(answer_tokens), 4)


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
        snippets = row.get("source_snippets", [])
        per_id.append({
            "id": row["id"],
            "context_overlap": context_overlap(row["answer_text"], snippets),
            "answer_chars": len(row["answer_text"]),
            "snippet_count": len(snippets),
        })

    report = {
        "framework": "text-overlap heuristic (offline; no RAGAS LLM judge enabled)",
        "config": {
            "rerank_top_k": settings.TOP_K_RERANK,
            "model": settings.EMBEDDING_MODEL_NAME,
        },
        "rows": per_id,
        "summary": {
            "mean_context_overlap": round(
                sum(r["context_overlap"] for r in per_id) / max(1, len(per_id)), 4
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