"""Small multi-turn probe for follow-up rewriting and rolling summarization.

`run_eval.py` sends single-turn questions, so it never exercises the two
non-QA prompts. This probe does, against the real providers:

  * rewrite temperature  - calls ``QueryRewriter.rewrite`` on context-dependent
    follow-ups and prints the standalone query it produces;
  * summary temperature  - calls ``summarize_history`` on a transcript past the
    summary trigger and prints the (concise) summary;
  * one end-to-end follow-up through ``answer_query`` to confirm the rewritten
    query is used by retrieval and the answer is grounded.

Usage:
    python -m evaluation.turn_probe --reuse-index \\
        --rewrite-temps "0.0,0.1,0.3" --summary-temps "0.0,0.1,0.3"
"""
from __future__ import annotations

try:  # load .env before settings
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

import argparse
import json
import time
from pathlib import Path

from config import settings
from src.pipeline_factory import build_pipeline, reset_pipeline_cache
from src.routing.quota_tracker import QuotaTracker

# Follow-ups that cannot be answered without the previous turn.
REWRITE_CASES = [
    {
        "turns": [("user", "What is the borrower's DTI?"),
                  ("assistant", "The borrower's DTI is 35.65%.")],
        "followup": "Is that below the maximum?",
    },
    {
        "turns": [("user", "What is the appraised value of the property?"),
                  ("assistant", "The appraised value is $410,000.")],
        "followup": "How does that compare with the purchase price?",
    },
    {
        "turns": [("user", "What is the minimum FICO score required?"),
                  ("assistant", "The minimum FICO requirement is 700.")],
        "followup": "Does the borrower meet it?",
    },
]

_SUMMARY_TRANSCRIPT = [
    ("user", f"What does section {i} say about the disclosure?")
    if i % 2 == 0 else ("assistant", f"Section {i} states the disclosure is provided.")
    for i in range(settings.HISTORY_SUMMARY_TRIGGER + 8)
]


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()] or [0.1]


def _history(pairs) -> list[dict]:
    return [{"role": role, "content": text} for role, text in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", "--pdf", dest="files", action="append")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--out", default="evaluation/turn_probe.json")
    parser.add_argument("--rewrite-temps", default="0.0,0.1,0.3")
    parser.add_argument("--summary-temps", default="0.0,0.1,0.3")
    args = parser.parse_args()

    reset_pipeline_cache()
    bundle = build_pipeline(quota_tracker=QuotaTracker(state_path=None))
    pipeline = bundle.pipeline

    if args.reuse_index:
        restored = pipeline.restore_documents()
        print(f"[probe] reused index: {len(restored)} document(s)")
    else:
        files = [Path(f) for f in (args.files or [])]
        if not files:
            raise SystemExit("--file is required unless --reuse-index is given")
        pipeline.reset()
        for file_path in files:
            ing = pipeline.ingest_document(file_path)
            print(f"[probe] ingested {ing.filename}: {ing.chunks} chunks")

    report: dict = {"rewrite": [], "summary": [], "end_to_end": []}

    # --- rewrite temperature -------------------------------------------------
    for temp in _parse_floats(args.rewrite_temps):
        settings.LLM_TEMPERATURE_REWRITE = temp
        for case in REWRITE_CASES:
            turns = _history(case["turns"])
            start = time.perf_counter()
            try:
                out = pipeline.rewriter.rewrite(case["followup"], turns, None)
                err = None
            except Exception as exc:  # noqa: BLE001
                out, err = None, str(exc)
            elapsed = time.perf_counter() - start
            print(f"[rewrite] t={temp} {case['followup']!r} -> {out!r} ({elapsed:.1f}s)")
            report["rewrite"].append({
                "temperature": temp, "followup": case["followup"],
                "rewritten": out, "error": err, "latency_s": round(elapsed, 3),
            })

    # --- summary temperature -------------------------------------------------
    for temp in _parse_floats(args.summary_temps):
        settings.LLM_TEMPERATURE_SUMMARY = temp
        history = _history(_SUMMARY_TRANSCRIPT)
        start = time.perf_counter()
        try:
            out = pipeline.summarize_history(history, "Earlier turns covered the loan terms.")
            err = None
        except Exception as exc:  # noqa: BLE001
            out, err = None, str(exc)
        elapsed = time.perf_counter() - start
        preview = (out or "")[:100].replace("\n", " ")
        print(f"[summary] t={temp} len={len(out or '')} -> {preview!r} ({elapsed:.1f}s)")
        report["summary"].append({
            "temperature": temp, "summary": out, "chars": len(out or ""),
            "error": err, "latency_s": round(elapsed, 3),
        })

    # --- one end-to-end follow-up -------------------------------------------
    settings.LLM_TEMPERATURE_REWRITE = _parse_floats(args.rewrite_temps)[0]
    case = REWRITE_CASES[0]
    start = time.perf_counter()
    try:
        answer = pipeline.answer_query(case["followup"], chat_history=_history(case["turns"]))
        err = None
    except Exception as exc:  # noqa: BLE001
        answer, err = None, str(exc)
    elapsed = time.perf_counter() - start
    print(f"[e2e] {case['followup']!r} -> {getattr(answer, 'text', err)!r} "
          f"provider={getattr(answer, 'provider', '-')} ({elapsed:.1f}s)")
    report["end_to_end"] = [{
        "followup": case["followup"],
        "answer": getattr(answer, "text", None),
        "provider": getattr(answer, "provider", None),
        "error": err,
        "latency_s": round(elapsed, 3),
    }]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
