"""Offline evaluation harness for the ANexus RAG pipeline.

Runs the golden dataset through the *real* retrieval stack and reports:

  * retrieval quality  - Recall@K, Mean Reciprocal Rank (MRR)@K, % queries with no context
  * end-to-end latency  - median / p95 ingest-est and per-query time
  * safety              - % of final answers that leaked a raw PII token
  * provider behavior   - fallback chains used (how often the primary provider wins)

Designed to work offline (SIMULATE_LLM=1) in CI and locally; when provider keys are
present it exercises the real generation path too.

Usage:
    python -m evaluation.run_eval --file "Synthetic_Mortgage_Loan_File_TEST.pdf" \
        [--k 5] [--out results.json] [--simulate]

``--pdf`` is kept as a deprecated alias of ``--file``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

from config import settings
from src.ingestion.parsers import parse_document
from src.pipeline_factory import build_pipeline, reset_pipeline_cache
from src.redaction.presidio_scrubber import scrub_chunks
from src.retrieval.reranker import RankedChunk

# PII token patterns mirrored from the scrubber so the eval can *detect* leaks
# without depending on the scrubber's own thresholds. Covers every identifier the
# scrubber is meant to protect; a pattern that fires in an answer is a failure.
_PII_PATTERNS = [
    r"\b\d{3}[- .]?\d{2}[- .]?\d{4}\b",              # US SSN
    r"\b\d{4}[- .]?\d{4}[- .]?\d{4}[- .]?\d{4}\b",    # credit card
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # email
    r"\b(?:AADHAAR)?\d{12}\b",                        # Aadhaar
    r"\b[A-Z]{5}\d{4}[A-Z]\b",                        # PAN
    r"\b[A-Z]{4}0[A-Z0-9]{6}\b",                     # IFSC
    r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z\d]\b",         # GSTIN
    r"\b[A-Z]{3}\d{7}\b",                             # EPIC / voter id
    r"\b[A-Z]{4}\d{4}[A-Z]\d\b",                      # TAN
    r"\b(?:a\s*/?\s*c|acc(?:ount)?|acct)\.?\s*(?:no\.?|number|num\.?)?\s*[:\-]?\s*\d{8,20}\b",
    r"\bloan\s*(?:a\s*/?\s*c|acc(?:ount)?|acct)\.?\s*(?:no\.?|number|num\.?)?\s*[:\-]?\s*\d{8,20}\b",
]


def leak_score(text: str) -> float:
    """Fraction of PII patterns that appear verbatim in the answer text."""
    if not text:
        return 0.0
    hits = 0
    for pattern in _PII_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits += 1
    return hits / len(_PII_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        "--pdf",
        dest="file",
        required=True,
        help="Path to the source document to ingest (any supported format)",
    )
    parser.add_argument("--k", type=int, default=settings.TOP_K_RERANK, help="Top-k for recall/MRR")
    parser.add_argument("--out", default="evaluation/results.json", help="JSON output path")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=os.getenv("SIMULATE_LLM", "0").strip().lower() in ("1", "true", "yes"),
        help="Use the simulated LLM (no provider keys needed)",
    )
    args = parser.parse_args()

    dataset_path = Path("evaluation/golden_dataset.json")
    if not dataset_path.exists():
        raise SystemExit("evaluation/golden_dataset.json not found")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    # One fresh pipeline per run; no cross-run Chroma pollution. Ingestion is
    # append-mode now, so the corpus is cleared explicitly before the eval file.
    reset_pipeline_cache()
    bundle = build_pipeline()
    pipeline = bundle.pipeline
    pipeline.reset()

    ingest_start = time.perf_counter()
    ing = pipeline.ingest_document(file_path)
    ingest_s = time.perf_counter() - ingest_start
    print(f"[eval] ingested {ing.filename}: {ing.pages} pages, {ing.chunks} chunks in {ingest_s:.1f}s")

    pages = parse_document(file_path)

    rows = []
    retrieval_latencies, e2e_latencies = [], []
    answered, no_context, leaked = 0, 0, 0

    for item in dataset:
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
        }
        t0 = time.perf_counter()
        answer = pipeline.answer_query(item["question"])
        latency = time.perf_counter() - t0

        # --- retrieval quality (independent of the LLM answer) ---
        retrieved: list[RankedChunk] = bundle.retriever.hybrid_retrieve(item["question"])
        reranked = bundle.reranker.rerank(item["question"], retrieved, top_k=args.k)
        hit_page_set = set()
        hits = []
        expected_pages = set(item.get("source_pages", []))
        for rank, chunk in enumerate(reranked, start=1):
            page = chunk.metadata.get("page_number")
            if page is None:
                continue
            if page in hit_page_set:
                continue
            hit_page_set.add(page)
            if int(page) in expected_pages:
                hits.append(rank)
        rank_of_first_hit = min(hits) if hits else None

        row["recall_at_k"] = 1.0 if rank_of_first_hit is not None else 0.0
        row["mrr_at_k"] = (1.0 / rank_of_first_hit) if rank_of_first_hit else 0.0
        row["topk_pages"] = sorted({int(c.metadata.get("page_number")) for c in reranked if c.metadata.get("page_number") is not None})
        row["first_hit_rank"] = rank_of_first_hit
        row["expected_pages"] = sorted(expected_pages)
        # Carry context across chunks so a label split from its value is still caught.
        row["source_snippets"] = [
            str(scrubbed)
            for _, scrubbed in scrub_chunks(reranked[: args.k], bundle.scrubber)
        ]

        # --- answer behaviour ---
        row["answer_text"] = answer.text
        row["latency_s"] = round(latency, 3)
        row["provider"] = answer.provider
        row["intent"] = answer.intent
        row["citations"] = [c.page_number for c in answer.citations]

        if answer.text and not answer.text.startswith("The answer is not available"):
            answered += 1
        else:
            no_context += 1
        if leak_score(answer.text) > 0:
            leaked += 1

        retrieval_latencies.append(latency)
        e2e_latencies.append(latency)
        _mark_ok = "OK " if rank_of_first_hit is not None else "MISS"
        print(f"[eval] {item['id']} {_mark_ok} recall@k={row['recall_at_k']:.0f} mrr={row['mrr_at_k']:.2f} {latency:.1f}s")

        rows.append(row)

    n = max(1, len(rows))
    results = {
        "summary": {
            "items": len(rows),
            "recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
            "mrr_at_k": round(sum(r["mrr_at_k"] for r in rows) / n, 4),
            "topk_empty_rate": round(sum(1 for r in rows if not r["topk_pages"]) / n, 4),
            "no_context_rate": round(no_context / n, 4),
            "pii_leak_rate": round(leaked / n, 4),
            "answered_rate": round(answered / n, 4),
            "ingest_s": round(ingest_s, 2),
            "median_e2e_s": round(statistics.median(e2e_latencies), 3)
            if e2e_latencies else 0,
            "p95_e2e_s": round(
                statistics.quantiles(e2e_latencies, n=100)[94], 3
            )
            if len(e2e_latencies) >= 20 else round(max(e2e_latencies or [0]), 3),
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[summary]")
    for key, value in results["summary"].items():
        print(f"  {key}: {value}")
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()