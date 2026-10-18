"""Evaluation harness for the ANexus RAG pipeline.

Runs the golden dataset through the *real* retrieval stack and reports:

  * retrieval quality  - Recall@K, Mean Reciprocal Rank (MRR)@K, % queries with no context
  * answer quality      - accuracy vs the golden `acceptable_answer_points`, per category
                          and per provider; model-emitted citation support
  * safety              - % of final answers that leaked a raw PII token
  * provider behavior   - which provider answered, fallback chains used, latency

Works offline (``SIMULATE_LLM=1``, the default in CI) and against real providers
when keys are present. ``.env`` is loaded automatically (shell env wins).

Usage:
    python -m evaluation.run_eval --file "Synthetic_Mortgage_Loan_File_TEST.pdf" \\
        [--k 5] [--out evaluation/results.json] [--simulate]

``--file`` may be repeated to ingest several documents into one corpus. When
more than one file is supplied, a golden item's optional ``source_file`` field
scopes its expected pages to that document; hits are matched by
``(document_id, page_number)`` so equal page numbers in different documents do
not inflate recall.

Temperature sweep / index reuse (for real-provider tuning):
    python -m evaluation.run_eval --reuse-index --tag sweep \\
        --temperature "0.0,0.1,0.3"

    * ``--reuse-index`` skips reset+ingest and rebuilds BM25/registry from the
      on-disk Chroma collection (use after one normal ingest run), so a
      temperature sweep does not pay the multi-minute ingest again;
    * ``--temperature`` accepts a comma-separated list; each value is swept in a
      single process and written to its own results file;
    * ``--repeat`` repeats the whole sweep (variance for stochastic temps);
    * ``--tag`` names the output files and the combined ``*_summary.json``.

``--pdf`` is kept as a deprecated alias of ``--file``.
"""
from __future__ import annotations

# Load .env *before* importing config.settings so env overrides are visible.
try:  # python-dotenv is a declared dependency but must stay optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - env simply stays as-is
    pass

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

from config import settings
from src.pipeline_factory import build_pipeline, reset_pipeline_cache
from src.redaction.presidio_scrubber import scrub_chunks
from src.retrieval.reranker import RankedChunk
from src.routing.quota_tracker import QuotaTracker

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

# A negative/unknown item is correct when the model declines to answer. The
# golden points cover the common phrasings; these markers catch the rest.
_REFUSAL_MARKERS = (
    "not provided", "not stated", "not specified", "does not provide",
    "does not state", "does not say", "does not mention", "not mentioned",
    "not available", "not in the document", "no information",
    "cannot be determined", "cannot be inferred", "not include", "not contain",
    "does not contain", "is not available", "no bank account number",
)

_WS_RE = re.compile(r"\s+")

# Inline / bibliography citation accepted by the system prompt:
# "(agreement.pdf, p.2)" or "(policy.pdf, page 4)".
_CITATION_RE = re.compile(r"\(([^()]+?),\s*(?:p\.?|page)\s*(\d+)\)", re.IGNORECASE)


def leak_score(text: str) -> float:
    """Fraction of PII patterns that appear verbatim in the answer text."""
    if not text:
        return 0.0
    hits = 0
    for pattern in _PII_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits += 1
    return hits / len(_PII_PATTERNS)


def _norm_doc(value: str) -> str:
    """Normalize a document label for citation matching (basename, lowercase)."""
    return Path(value.replace("\\", "/")).name.strip().lower()


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Extract (document, page) pairs the model cited in its answer."""
    if not text:
        return []
    return [
        (_norm_doc(m.group(1)), int(m.group(2)))
        for m in _CITATION_RE.finditer(text)
    ]


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def score_answer(answer_text: str, item: dict) -> tuple[bool, str | None]:
    """Grade an answer against the golden item's acceptable points.

    Returns ``(correct, matched_point)``. Matching is a normalized substring
    test so formatting differences (``$320,000`` vs ``320000``) still count as
    long as one of the curated variants appears. Negative/unknown items also
    accept an explicit refusal.
    """
    text = _norm_text(answer_text)
    if not text:
        return False, None
    points = list(item.get("acceptable_answer_points") or [])
    expected = item.get("expected_answer")
    if expected:
        points.append(expected)
    for point in points:
        normalized = _norm_text(point)
        if normalized and normalized in text:
            return True, point
    if item.get("category") == "negative_unknown":
        for marker in _REFUSAL_MARKERS:
            if marker in text:
                return True, f"refusal:{marker}"
    return False, None


def _breakdown(rows: list[dict], key: str) -> dict:
    """Accuracy grouped by a row field (category / provider)."""
    agg: dict[str, dict] = {}
    for row in rows:
        bucket = agg.setdefault(str(row.get(key) or "unknown"), {"count": 0, "correct": 0})
        bucket["count"] += 1
        bucket["correct"] += 1 if row.get("answer_correct") else 0
    for bucket in agg.values():
        bucket["accuracy"] = (
            round(bucket["correct"] / bucket["count"], 4) if bucket["count"] else 0.0
        )
    return agg


def _parse_temperatures(raw: str) -> list[float]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values or [float(settings.LLM_TEMPERATURE)]


def _retrieval_prep(dataset: list[dict], bundle, top_k: int, single_document_id):
    """Compute the temperature-independent retrieval facts, once."""
    prep = []
    for item in dataset:
        retrieved: list[RankedChunk] = bundle.retriever.hybrid_retrieve(item["question"])
        reranked = bundle.reranker.rerank(item["question"], retrieved, top_k=top_k)
        expected_pages = set(item.get("source_pages", []))
        expected_doc = item.get("source_file") or single_document_id
        seen, hits = set(), []
        for rank, chunk in enumerate(reranked, start=1):
            meta = chunk.metadata or {}
            page = meta.get("page_number")
            if page is None:
                continue
            doc = str(meta.get("document_id") or meta.get("filename") or "")
            if expected_doc is not None and doc and doc != expected_doc:
                continue
            ref = (doc, int(page))
            if ref in seen:
                continue
            seen.add(ref)
            if int(page) in expected_pages:
                hits.append(rank)
        rank_of_first_hit = min(hits) if hits else None

        allowed_labels = set()
        for chunk in reranked:
            meta = chunk.metadata or {}
            page = meta.get("page_number")
            if page is None:
                continue
            for label in (meta.get("filename"), meta.get("document_id")):
                if label:
                    allowed_labels.add((_norm_doc(str(label)), int(page)))

        prep.append({
            "recall_at_k": 1.0 if rank_of_first_hit is not None else 0.0,
            "mrr_at_k": (1.0 / rank_of_first_hit) if rank_of_first_hit else 0.0,
            "topk_refs": sorted(
                f"{c.metadata.get('document_id') or c.metadata.get('filename')}#"
                f"{int(c.metadata['page_number'])}"
                for c in reranked
                if c.metadata.get("page_number") is not None
            ),
            "topk_pages": sorted(
                {int(c.metadata["page_number"]) for c in reranked
                 if c.metadata.get("page_number") is not None}
            ),
            "first_hit_rank": rank_of_first_hit,
            "expected_pages": sorted(expected_pages),
            "expected_doc": expected_doc,
            "allowed_labels": allowed_labels,
            # Carry context across chunks so a label split from its value is still caught.
            "source_snippets": [
                str(scrubbed)
                for _, scrubbed in scrub_chunks(reranked[:top_k], bundle.scrubber)
            ],
        })
    return prep


def _run_pass(dataset, prep, pipeline, temperature: float):
    settings.LLM_TEMPERATURE = temperature
    rows = []
    answered = no_context = leaked = 0
    total_cites = total_cited_ok = cited_rows = 0
    correct = 0
    fallback_rows = 0
    e2e_latencies = []

    for item, base in zip(dataset, prep):
        t0 = time.perf_counter()
        answer = pipeline.answer_query(item["question"])
        latency = time.perf_counter() - t0

        row = {"id": item["id"], "category": item["category"], "question": item["question"]}
        for key in ("recall_at_k", "mrr_at_k", "topk_refs", "topk_pages",
                    "first_hit_rank", "expected_pages", "expected_doc", "source_snippets"):
            row[key] = base[key]

        row["answer_text"] = answer.text
        row["latency_s"] = round(latency, 3)
        row["provider"] = answer.provider
        row["intent"] = answer.intent
        row["fallbacks_tried"] = list(answer.fallbacks_tried)
        row["citations"] = [c.page_number for c in answer.citations]

        is_correct, matched = score_answer(answer.text, item)
        row["answer_correct"] = is_correct
        row["matched_point"] = matched

        cited = parse_citations(answer.text)
        supported = [ref for ref in cited if ref in base["allowed_labels"]]
        row["cited_refs"] = [f"{doc}#{page}" for doc, page in cited]
        row["citation_precision"] = round(len(supported) / len(cited), 4) if cited else None
        total_cites += len(cited)
        total_cited_ok += len(supported)
        if cited:
            cited_rows += 1

        if answer.text and not answer.text.startswith("The answer is not available"):
            answered += 1
        else:
            no_context += 1
        if leak_score(answer.text) > 0:
            leaked += 1
        if is_correct:
            correct += 1
        if answer.fallbacks_tried:
            fallback_rows += 1

        e2e_latencies.append(latency)
        mark = "OK  " if is_correct else ("MISS" if base["first_hit_rank"] else "NORET")
        print(
            f"[eval] t={temperature} {item['id']} {mark} "
            f"recall={row['recall_at_k']:.0f} ans={int(is_correct)} "
            f"prov={answer.provider or '-'} {latency:.1f}s"
        )
        rows.append(row)

    n = max(1, len(rows))
    summary = {
        "temperature": temperature,
        "items": len(rows),
        "answer_accuracy": round(correct / n, 4),
        "answered_rate": round(answered / n, 4),
        "no_context_rate": round(no_context / n, 4),
        "recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
        "mrr_at_k": round(sum(r["mrr_at_k"] for r in rows) / n, 4),
        "pii_leak_rate": round(leaked / n, 4),
        "citation_support_rate": round(total_cited_ok / total_cites, 4) if total_cites else 0.0,
        "citation_coverage": round(cited_rows / max(1, answered), 4),
        "fallback_rate": round(fallback_rows / n, 4),
        "median_e2e_s": round(statistics.median(e2e_latencies), 3) if e2e_latencies else 0.0,
        "by_category": _breakdown(rows, "category"),
        "by_provider": _breakdown(rows, "provider"),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", "--pdf", dest="files", action="append",
        help="Path to a source document to ingest (any supported format). "
        "Repeat the flag to ingest multiple documents. Not needed with --reuse-index.",
    )
    parser.add_argument("--k", type=int, default=settings.TOP_K_RERANK, help="Top-k for recall/MRR")
    parser.add_argument("--out", default="evaluation/results.json", help="JSON output path")
    parser.add_argument(
        "--simulate", action="store_true",
        default=os.getenv("SIMULATE_LLM", "0").strip().lower() in ("1", "true", "yes"),
        help="Use the simulated LLM (no provider keys needed)",
    )
    parser.add_argument(
        "--temperature", default=str(settings.LLM_TEMPERATURE),
        help="QA temperature, or a comma-separated list to sweep (e.g. \"0.0,0.1,0.3\")",
    )
    parser.add_argument(
        "--reuse-index", action="store_true",
        help="Skip reset+ingest; rebuild from the on-disk Chroma collection.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the whole sweep N times")
    parser.add_argument("--tag", default="", help="Output file tag (for sweeps)")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only evaluate the first N golden items (smoke-testing; 0 = all)",
    )
    args = parser.parse_args()

    settings.SIMULATE_LLM = bool(args.simulate)
    temperatures = _parse_temperatures(args.temperature)

    dataset_path = Path("evaluation/golden_dataset.json")
    if not dataset_path.exists():
        raise SystemExit("evaluation/golden_dataset.json not found")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        dataset = dataset[: args.limit]

    reset_pipeline_cache()
    # state_path=None: never carry a cooldown across runs/processes.
    bundle = build_pipeline(quota_tracker=QuotaTracker(state_path=None))
    pipeline = bundle.pipeline

    ingested, ingest_s = [], 0.0
    if args.reuse_index:
        restored = pipeline.restore_documents()
        total_chunks = bundle.vector_db.count()
        single_document_id = restored[0]["id"] if len(restored) == 1 else None
        print(
            f"[eval] reused index: {len(restored)} document(s), {total_chunks} chunks"
        )
        documents = [
            {"document_id": r["id"], "filename": r.get("filename"), "chunks": r.get("chunks")}
            for r in restored
        ]
    else:
        files = [Path(f) for f in (args.files or [])]
        if not files:
            raise SystemExit("--file is required unless --reuse-index is given")
        for file_path in files:
            if not file_path.exists():
                raise SystemExit(f"File not found: {file_path}")
        pipeline.reset()
        for file_path in files:
            start = time.perf_counter()
            ing = pipeline.ingest_document(file_path)
            ingest_s += time.perf_counter() - start
            ingested.append(ing)
            print(
                f"[eval] ingested {ing.filename}: {ing.pages} pages, {ing.chunks} chunks "
                f"(mean {ing.chunks_mean_chars:.0f} chars, max {ing.chunks_max_chars}) "
                f"in {ing.duration_s:.1f}s"
            )
        total_chunks = sum(i.chunks for i in ingested)
        single_document_id = ingested[0].document_id if len(ingested) == 1 else None
        documents = [
            {
                "filename": i.filename, "document_id": i.document_id, "pages": i.pages,
                "chunks": i.chunks, "chunk_mean_chars": round(i.chunks_mean_chars, 1),
                "chunk_median_chars": round(i.chunks_median_chars, 1),
                "chunk_max_chars": i.chunks_max_chars, "ingest_s": round(i.duration_s, 2),
            }
            for i in ingested
        ]

    prep = _retrieval_prep(dataset, bundle, args.k, single_document_id)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    multi = len(temperatures) > 1 or args.repeat > 1 or bool(args.tag)
    tag = args.tag or "sweep"
    combined = []

    for rep in range(1, args.repeat + 1):
        for temperature in temperatures:
            rows, summary = _run_pass(dataset, prep, pipeline, temperature)
            summary["repeat"] = rep
            summary["documents"] = len(documents)
            summary["chunks_total"] = total_chunks
            summary["ingest_s"] = round(ingest_s, 2)
            combined.append(summary)

            if multi:
                safe = str(temperature).replace(".", "p")
                run_out = out.with_name(f"{out.stem}_{tag}_t{safe}_r{rep}{out.suffix or '.json'}")
            else:
                run_out = out
            payload = {"summary": summary, "documents": documents, "rows": rows}
            run_out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n[summary] t={temperature} r={rep}")
            for key in ("answer_accuracy", "answered_rate", "recall_at_k", "mrr_at_k",
                        "pii_leak_rate", "citation_support_rate", "citation_coverage",
                        "fallback_rate", "median_e2e_s"):
                print(f"  {key}: {summary[key]}")
            print(f"[eval] wrote {run_out}")

    if len(combined) > 1:
        summary_out = out.with_name(f"{out.stem}_{tag}_summary.json")
        summary_out.write_text(
            json.dumps({"runs": combined}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[sweep] wrote {summary_out}")


if __name__ == "__main__":
    main()
