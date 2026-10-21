"""Evaluation harness for the ANexus RAG pipeline.

Runs the golden dataset(s) through the *real* retrieval stack and reports:

  * retrieval quality  - Precision@K, true Recall@K, HitRate@K, MRR@K, nDCG@K,
                         context precision (average precision) and context recall
                         (fraction of gold answer points present in the context)
  * answer quality     - accuracy vs curated literal answer points (conjunctive
                         ``required_fact_groups`` or OR ``acceptable_answer_points``),
                         per category and per provider; model-emitted citation support
  * answer grounding   - claim-level token-F1 of each answer sentence against the
                         retrieved context (offline stand-in for RAGAS faithfulness)
                         and embedding-cosine answer relevance vs the golden answer
  * safety             - % of final answers that leaked a raw PII token
  * provider behavior  - which provider answered, fallback chains used, latency

Retrieval and generation are also scored *separately*: ``--oracle`` re-answers
every question with the gold pages injected as context (retrieval bypassed via a
harness-local monkeypatch), so the oracle-vs-end-to-end gap attributes error to
retrieval rather than generation. All metrics are computed offline (no LLM judge).

Works offline (``SIMULATE_LLM=1``, the default in CI) and against real providers
when keys are present. ``.env`` is loaded automatically (shell env wins).

Usage:
    python -m evaluation.run_eval \\
        --file Synthetic_Mortgage_Loan_File_TEST_01.pdf \\
        --file Synthetic_Mortgage_Loan_File_TEST_02.pdf \\
        [--k 5] [--out evaluation/results.json] [--simulate] [--oracle]

``--file`` may be repeated to ingest several documents into one corpus (the CI
default is both synthetic mortgage PDFs). Each golden item's ``source_file``
scopes its expected pages to that document; hits are matched by
``(document_id, page_number)`` so equal page numbers in different documents do
not inflate recall.

``--dataset`` may be repeated to override the two default golden files. Both
on-disk shapes are accepted: a plain JSON list, or a dict wrapper with a
``questions`` list (as in ``golden_dataset_02.json``). Question ids are
re-prefixed per source file (``d01_q001`` / ``d02_q001``) so two 50-question
sets can share one corpus.

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

Real-provider cost/rate controls (for paid sweeps):
    * ``--provider groq`` pins generation to one provider (no silent fallback);
    * ``--max-cost-usd 1.00`` stops a pass cleanly once spend reaches the budget;
    * ``--min-interval-s 1.0`` paces questions to stay inside rate limits;
    * ``--resume`` continues an interrupted run from its ``*.partial.json``
      checkpoint instead of re-paying for completed questions;
    * ``--no-preflight`` skips the one-call-per-provider smoke check.

Token usage is read from each provider response and counted for **every** call
(answer + rewrite/expansion/summary) via ``evaluation.cost.CostMeter``; per-pass
totals land in ``summary["cost"]`` (USD defaults to 0 for the free-tier
providers - override with ``ANEXUS_PRICE_<PROVIDER>``).
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
from contextlib import contextmanager
from pathlib import Path

from config import settings
from evaluation import metrics
from evaluation.cost import CostMeter
from src.generation.llm import build_default_handler
from src.pipeline_factory import build_pipeline, reset_pipeline_cache
from src.redaction.presidio_scrubber import scrub_chunks
from src.retrieval.reranker import RankedChunk
from src.routing.quota_tracker import QuotaTracker

# Golden datasets loaded by default (override with --dataset). Maps the dataset
# file to the document each of its questions was authored against, so a merged
# two-document run matches hits per document (same page number in both PDFs).
DEFAULT_DATASETS = {
    "evaluation/golden_dataset_01.json": "Synthetic_Mortgage_Loan_File_TEST_01.pdf",
    "evaluation/golden_dataset_02.json": "Synthetic_Mortgage_Loan_File_TEST_02.pdf",
}

# Dataset-02 used a shorter category vocabulary before consolidation; keep the
# loader tolerant of either spelling so old copies still score.
_CATEGORY_ALIASES = {
    "direct": "direct_fact_lookup",
    "definition": "definition_meaning",
    "numerical": "numerical_financial",
}

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
    """Grade an answer against the golden item's expected points.

    Two item shapes are supported:

    * ``required_fact_groups`` - a list of literal groups where every group must
      match at least one of its OR-variants (conjunctive). This is the format
      used by ``golden_dataset_02.json``: a correct answer must establish *all*
      the required facts, not just one.
    * ``acceptable_answer_points`` / ``expected_answer`` - a flat list of
      variants where *any* one match is enough (disjunctive). Kept for
      ``golden_dataset_01.json`` back-compat.

    Matching is a normalized substring test so formatting differences
    (``$320,000`` vs ``320000``) still count. Negative/unknown items also
    accept an explicit refusal.

    Returns ``(correct, matched_evidence)``.
    """
    text = _norm_text(answer_text)
    if not text:
        return False, None

    groups = list(item.get("required_fact_groups") or [])
    if groups:
        matched: list[str] = []
        for group in groups:
            hit = next((v for v in group if _norm_text(v) and _norm_text(v) in text), None)
            if hit is None:
                return False, None
            matched.append(hit)
        return True, "; ".join(matched)

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


def _context_recall_for_item(item: dict, context: str) -> float:
    """Fraction of the item's gold facts present in the retrieved context.

    For ``required_fact_groups`` an item is covered when *any* variant of each
    group appears (conjunctive over groups, disjunctive within a group), matching
    how ``score_answer`` grades. Otherwise it falls back to the flat
    ``acceptable_answer_points`` / ``expected_answer`` list.
    """
    ctx = _norm_text(context)
    if not ctx:
        return 0.0
    groups = list(item.get("required_fact_groups") or [])
    if groups:
        covered = 0
        for group in groups:
            if any(_norm_text(v) in ctx for v in group if _norm_text(v)):
                covered += 1
        return round(covered / len(groups), 4)
    points = list(item.get("acceptable_answer_points") or [])
    if item.get("expected_answer"):
        points.append(item["expected_answer"])
    if not points:
        return 0.0
    return metrics.context_recall(points, context)


def _oracle_documents(bundle, item: dict) -> list:
    """Gold chunks for an item: every stored chunk of the item's source pages.

    Returns LangChain ``Document`` objects in page order. No retrieval, no
    ranking - this is the "perfect retrieval" baseline the end-to-end pass is
    measured against. Downstream the real reranker and PII scrubber still run,
    so the oracle differs from the live path only in *which* chunks it sees.
    """
    document_id = item.get("source_file")
    wanted = set(item.get("source_pages", []))
    if document_id is None or not wanted:
        return []
    try:
        stored = bundle.vector_db.get_all_documents()
    except Exception:  # pragma: no cover - defensive: oracle degrades to empty
        return []
    gold = []
    for entry in stored:
        meta = dict(entry.metadata or {})
        page = meta.get("page_number")
        if page is None:
            continue
        doc = str(meta.get("document_id") or meta.get("filename") or "")
        if doc != document_id or int(page) not in wanted:
            continue
        gold.append(entry)
    gold.sort(key=lambda d: int((d.metadata or {}).get("page_number") or 0))
    return gold


@contextmanager
def _bypass_retrieval(pipeline, bundle, item: dict):
    """Temporarily swap the pipeline's retrieval for the item's gold chunks.

    Harness-local monkeypatch: production code is untouched and every original
    method is restored on exit. The query expander is disabled so no LLM call
    (and no paraphrase drift) can change which context the generator sees.
    """
    gold = _oracle_documents(bundle, item)

    def _gold_expand(*_args, **_kwargs):
        return []

    def _gold_retrieve(*_args, **_kwargs):
        return list(gold)

    original = (pipeline.expander.expand,
                pipeline.retriever.hybrid_retrieve,
                pipeline.retriever.multi_query_retrieve)
    pipeline.expander.expand = _gold_expand
    pipeline.retriever.hybrid_retrieve = _gold_retrieve
    pipeline.retriever.multi_query_retrieve = _gold_retrieve
    try:
        yield [f"{item['source_file']}#{int((d.metadata or {}).get('page_number'))}"
               for d in gold]
    finally:
        (pipeline.expander.expand,
         pipeline.retriever.hybrid_retrieve,
         pipeline.retriever.multi_query_retrieve) = original


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


def load_datasets(specs: dict[str, str]) -> list[dict]:
    """Load and merge one or more golden dataset files.

    ``specs`` maps a dataset file path to the document its questions were
    authored against (injected as ``source_file``). Accepts both on-disk shapes:
    a plain JSON list (``golden_dataset_01.json``) or a dict wrapper with a
    ``questions`` list (``golden_dataset_02.json``). Ids are re-prefixed with a
    per-file tag (``d01_q001``...) so two 50-question sets can share one corpus,
    and category aliases are normalized to the canonical vocabulary.
    """
    merged: list[dict] = []
    for path_str, source_file in specs.items():
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"Dataset not found: {path_str}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions") if isinstance(payload, dict) else payload
        if not isinstance(questions, list):
            raise SystemExit(f"{path_str}: expected a list of questions or a "
                             f"dict with a 'questions' list")
        tag = path.stem.replace("golden_dataset_", "d")
        for q in questions:
            item = dict(q)
            item["id"] = f"{tag}_{q['id']}"
            item["category"] = _CATEGORY_ALIASES.get(
                str(item.get("category", "")), item.get("category", "unknown")
            )
            item["source_file"] = source_file
            merged.append(item)
    return merged


def _retrieval_prep(dataset: list[dict], bundle, top_k: int, single_document_id):
    """Compute the temperature-independent retrieval facts, once."""
    prep = []
    for item in dataset:
        retrieved: list[RankedChunk] = bundle.retriever.hybrid_retrieve(item["question"])
        reranked = bundle.reranker.rerank(item["question"], retrieved, top_k=top_k)
        expected_pages = set(item.get("source_pages", []))
        expected_doc = item.get("source_file") or single_document_id
        seen, hits = set(), []
        # Rank-ordered binary relevance for nDCG (index 0 = rank 1). A repeated
        # (doc, page) is skipped, so a duplicate of an already-counted page does
        # not inflate the gain.
        is_relevant: list[bool] = [False] * len(reranked)
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
                is_relevant[rank - 1] = True
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

        # Retrieved context the generator actually saw (scrubbed, top-k).
        source_snippets = [
            str(scrubbed)
            for _, scrubbed in scrub_chunks(reranked[:top_k], bundle.scrubber)
        ]
        context_text = "\n".join(source_snippets)

        prep.append({
            # Back-compat alias: the old "recall_at_k" was really a binary hit
            # rate; keep it under its honest name and add true recall alongside.
            "hit_at_k": metrics.hit_at_k(hits, top_k),
            "recall_at_k": metrics.recall_at_k(hits, len(expected_pages), top_k),
            "precision_at_k": metrics.precision_at_k(hits, top_k),
            "ndcg_at_k": metrics.ndcg_at_k(is_relevant, top_k),
            "mrr_at_k": metrics.mrr_at_k(hits, top_k),
            "context_precision": metrics.context_precision(
                hits, len(expected_pages), top_k
            ),
            "context_recall": _context_recall_for_item(item, context_text),
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


def _run_pass(dataset, prep, pipeline, temperature: float,
              bundle=None, oracle: bool = False, meter=None,
              max_cost_usd: float | None = None, pace_s: float = 0.0,
              skip_ids: set[str] | None = None, on_row=None):
    """One scored pass of the dataset through the pipeline.

    ``oracle=True`` re-answers each question with the gold pages injected
    (retrieval bypassed via ``_bypass_retrieval``), isolating generator quality
    from retrieval quality.

    ``meter`` + ``max_cost_usd`` stop the pass cleanly once spend reaches the
    budget; ``pace_s`` throttles between questions; ``skip_ids`` resumes a
    partially completed run; ``on_row`` receives each row as it completes so a
    crashed paid sweep loses at most one question.
    """
    settings.LLM_TEMPERATURE = temperature
    rows: list[dict] = []
    skipped = set(skip_ids or ())
    started = False

    for item, base in zip(dataset, prep):
        if item["id"] in skipped:
            continue
        if meter is not None and meter.exceeds(max_cost_usd):
            print(f"[eval] budget cap ${max_cost_usd:.4f} reached; stopping pass")
            break
        if pace_s and started:
            time.sleep(pace_s)
        started = True

        t0 = time.perf_counter()
        if oracle:
            with _bypass_retrieval(pipeline, bundle, item) as gold_refs:
                answer = pipeline.answer_query(item["question"])
            oracle_refs = gold_refs
        else:
            answer = pipeline.answer_query(item["question"])
            oracle_refs = None
        latency = time.perf_counter() - t0

        row = {"id": item["id"], "category": item["category"], "question": item["question"]}
        for key in ("hit_at_k", "recall_at_k", "precision_at_k", "ndcg_at_k",
                    "mrr_at_k", "context_precision", "context_recall",
                    "topk_refs", "topk_pages", "first_hit_rank",
                    "expected_pages", "expected_doc", "source_snippets"):
            row[key] = base[key]

        row["answer_text"] = answer.text
        row["latency_s"] = round(latency, 3)
        row["provider"] = answer.provider
        row["intent"] = answer.intent
        row["fallbacks_tried"] = list(answer.fallbacks_tried)
        row["citations"] = [c.page_number for c in answer.citations]
        if oracle_refs is not None:
            row["oracle_gold_refs"] = oracle_refs

        is_correct, matched = score_answer(answer.text, item)
        row["answer_correct"] = is_correct
        row["matched_point"] = matched

        context_text = "\n".join(base["source_snippets"])
        ground = metrics.groundedness(answer.text, context_text)
        row["groundedness"] = ground["score"]
        row["ungrounded_fraction"] = ground["ungrounded_fraction"]
        row["answer_claims"] = ground["claims"]

        row["answer_relevance"] = _answer_relevance(answer.text, item.get("expected_answer"))

        cited = parse_citations(answer.text)
        supported = [ref for ref in cited if ref in base["allowed_labels"]]
        row["cited_refs"] = [f"{doc}#{page}" for doc, page in cited]
        row["citation_precision"] = round(len(supported) / len(cited), 4) if cited else None
        row["cited_count"] = len(cited)
        row["cited_ok"] = len(supported)
        row["pii_leaked"] = bool(leak_score(answer.text) > 0)

        mark = "OK  " if is_correct else ("MISS" if base["first_hit_rank"] else "NORET")
        print(
            f"[eval] t={temperature} {item['id']} {mark} "
            f"hit={row['hit_at_k']:.0f} ans={int(is_correct)} "
            f"gnd={row['groundedness']:.2f} prov={answer.provider or '-'} {latency:.1f}s"
        )
        rows.append(row)
        if on_row is not None:
            on_row(row)

    return rows


def _summarize(rows: list[dict], temperature: float, mode: str,
               cost: dict | None = None) -> dict:
    """Aggregate rows into the summary dict (also used after a resume merge)."""
    n = max(1, len(rows))
    correct = sum(1 for r in rows if r.get("answer_correct"))
    no_context = sum(
        1 for r in rows
        if not r.get("answer_text")
        or str(r.get("answer_text", "")).startswith("The answer is not available")
    )
    answered = len(rows) - no_context
    leaked = sum(1 for r in rows if r.get("pii_leaked"))
    total_cites = sum(int(r.get("cited_count", 0)) for r in rows)
    total_cited_ok = sum(int(r.get("cited_ok", 0)) for r in rows)
    cited_rows = sum(1 for r in rows if int(r.get("cited_count", 0)) > 0)
    fallback_rows = sum(1 for r in rows if r.get("fallbacks_tried"))
    latencies = [float(r.get("latency_s") or 0.0) for r in rows]
    grounded_scores = [float(r["groundedness"]) for r in rows if r.get("answer_claims")]
    ungrounded_fracs = [
        float(r["ungrounded_fraction"]) for r in rows if r.get("answer_claims")
    ]
    relevance_scores = [
        float(r["answer_relevance"]) for r in rows if r.get("answer_relevance") is not None
    ]
    nli_rows = [r for r in rows if r.get("nli_claim_count")]
    nli_entailed = [
        float(r["nli_entailed_fraction"]) for r in nli_rows
        if r.get("nli_entailed_fraction") is not None
    ]
    nli_contra = [
        float(r["nli_contradiction_fraction"]) for r in nli_rows
        if r.get("nli_contradiction_fraction") is not None
    ]

    def mean(key: str) -> float:
        return round(sum(float(r.get(key) or 0.0) for r in rows) / n, 4)

    return {
        "temperature": temperature,
        "items": len(rows),
        "mode": mode,
        "answer_accuracy": round(correct / n, 4),
        "answered_rate": round(answered / n, 4),
        "no_context_rate": round(no_context / n, 4),
        # Retrieval quality (temperature-independent; recomputed identically per pass)
        "hit_at_k": mean("hit_at_k"),
        "recall_at_k": mean("recall_at_k"),
        "precision_at_k": mean("precision_at_k"),
        "ndcg_at_k": mean("ndcg_at_k"),
        "mrr_at_k": mean("mrr_at_k"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        # Answer quality / grounding (offline analogues of the RAGAS suite)
        "groundedness": (round(sum(grounded_scores) / len(grounded_scores), 4)
                         if grounded_scores else 0.0),
        "ungrounded_fraction": (round(sum(ungrounded_fracs) / len(ungrounded_fracs), 4)
                                if ungrounded_fracs else 0.0),
        "answer_relevance": (round(sum(relevance_scores) / len(relevance_scores), 4)
                             if relevance_scores else None),
        # Phase D: local NLI claim verification (absent unless --nli was used)
        "nli_entailed_fraction": (round(sum(nli_entailed) / len(nli_entailed), 4)
                                  if nli_entailed else None),
        "nli_contradiction_fraction": (round(sum(nli_contra) / len(nli_contra), 4)
                                       if nli_contra else None),
        "nli_rows_scored": len(nli_rows),
        "pii_leak_rate": round(leaked / n, 4),
        "citation_support_rate": round(total_cited_ok / total_cites, 4) if total_cites else 0.0,
        "citation_coverage": round(cited_rows / max(1, answered), 4),
        "fallback_rate": round(fallback_rows / n, 4),
        "median_e2e_s": round(statistics.median(latencies), 3) if latencies else 0.0,
        "cost": cost or {},
        "by_category": _breakdown(rows, "category"),
        "by_provider": _breakdown(rows, "provider"),
    }


class _RunCheckpoint:
    """Append-only partial-run file so a paid sweep can resume after a crash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict] = []

    def load(self) -> list[dict]:
        """Return the persisted rows as a **fresh list**.

        The copy is load-bearing: callers keep this list as their "prior rows"
        while ``add`` keeps appending to ``self.rows`` throughout the next pass.
        Returning the internal list would make ``prior + new`` count every row
        of that pass twice (the first Phase B run reported 185 rows for 100
        questions, and 200 oracle rows for 100).
        """
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self.rows = list(payload.get("rows") or [])
            except Exception:
                self.rows = []
        return list(self.rows)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"rows": self.rows}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def clear(self) -> None:
        self.rows = []
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _answer_relevance(answer_text: str, expected: str | None) -> float | None:
    """Embedding-cosine relevance of an answer to the golden answer.

    Lazy-imports the BGE embedder (already resident after ingest) and degrades
    to ``None`` if embedding fails, so one bad vector never kills a run.
    """
    if not expected:
        return None
    try:
        from src.ingestion.embedder import embed_texts

        return metrics.answer_relevance(answer_text, str(expected), embed_texts)
    except Exception:  # pragma: no cover - depends on the local model
        return None


def _dataset_specs(raw: list[str] | None) -> dict[str, str]:
    """Resolve --dataset values into ``{dataset_path: source_file}``.

    ``None`` -> both default golden datasets that exist on disk. Each entry may
    be ``PATH`` (the source document is looked up in ``DEFAULT_DATASETS``,
    required otherwise) or ``PATH=SOURCE_FILE`` for an explicit pairing.
    """
    if not raw:
        specs = {path: source for path, source in DEFAULT_DATASETS.items()
                 if Path(path).exists()}
        if not specs:
            raise SystemExit(
                "No default golden datasets found; pass --dataset PATH=SOURCE_FILE"
            )
        return specs
    specs = {}
    for entry in raw:
        if "=" in entry:
            path, source = entry.split("=", 1)
        else:
            path, source = entry, DEFAULT_DATASETS.get(entry)
            if source is None:
                raise SystemExit(
                    f"{path}: no source document known; pass {path}=SOURCE_FILE"
                )
        specs[path] = source
    return specs


class _PinnedRouter:
    """Router proxy that forces generation onto a single provider.

    Used by ``--provider`` for apples-to-apples per-provider comparisons: the
    intent classifier still runs, but its ``provider_order`` is replaced by
    ``[provider]`` so no silent fallback can mix providers within a run.
    """

    def __init__(self, inner, provider: str) -> None:
        self._inner = inner
        self.provider = provider

    def route(self, query: str, available_providers: list[str]):
        decision = self._inner.route(query, available_providers=available_providers)
        decision.provider = self.provider
        decision.provider_order = [self.provider]
        return decision


def _posthoc_nli(results_path: Path) -> bool:
    """Score NLI over a finished run in a *separate* process.

    The NLI cross-encoder (~700 MB) must not be resident beside the embedder,
    presidio and torch: on the 4 GB eval box that combination pages and
    chroma's pool times out on every query (~100x slower rows). A fresh
    process loads only the NLI model, so the two footprints never overlap.
    Failures never invalidate the run - it just keeps its rows without nli_*.
    """
    import subprocess
    import sys

    cmd = [sys.executable, "-u", "-m", "evaluation.nli",
           "--results", str(results_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] --nli post-scorer did not run ({exc}); results unchanged")
        return False
    stream = (proc.stdout or "") + (proc.stderr or "")
    if stream:
        print(stream, end="" if stream.endswith("\n") else "\n")
    if proc.returncode != 0:
        print(f"[eval] --nli post-scorer failed (exit {proc.returncode}); "
              f"results kept without nli_*")
        return False
    return True


def _preflight(handler, provider: str | None) -> None:
    """Fail fast on missing/broken provider keys before a long paid run."""
    configured = handler.available_providers
    if not configured:
        raise SystemExit(
            "[eval] no LLM providers configured; set API keys or pass --simulate"
        )
    if provider is not None and provider not in configured:
        raise SystemExit(
            f"[eval] --provider {provider!r} is not configured "
            f"(configured: {configured})"
        )
    probe = [{"role": "user", "content": "Reply with the single word: ok"}]
    targets = [provider] if provider else list(configured)
    ok = 0
    for name in targets:
        try:
            handler.adapters[name].chat(probe)
            ok += 1
            print(f"[eval] preflight {name}: OK")
        except Exception as exc:  # surface the provider's own error
            print(f"[eval] preflight {name}: FAIL {exc}")
    if provider is not None and ok == 0:
        raise SystemExit(f"[eval] pinned provider {provider!r} failed its smoke call")
    if ok == 0:
        raise SystemExit("[eval] every provider failed its smoke call")


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
        "--dataset", action="append", default=None,
        help="Golden dataset file (repeatable). Accepts PATH or PATH=SOURCE_FILE. "
             "Defaults to golden_dataset_01.json + golden_dataset_02.json.",
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="Also run an oracle pass (gold pages injected, retrieval bypassed) "
             "so generation quality can be attributed separately from retrieval.",
    )
    parser.add_argument(
        "--provider", choices=["nvidia", "gemini_flash", "groq"], default=None,
        help="Pin all generation calls to one provider (strict, no fallback).",
    )
    parser.add_argument(
        "--min-interval-s", type=float, default=0.0,
        help="Sleep this many seconds between questions (pace rate-limited providers).",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=0.0,
        help="Stop a pass once accumulated cost reaches this USD budget (0 = no cap).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a *.partial.json checkpoint, skipping completed questions.",
    )
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="Skip the real-provider smoke check before a run.",
    )
    parser.add_argument(
        "--nli", action="store_true",
        help="After generation, verify answer claims against the retrieved context "
             "with a local NLI cross-encoder (run as a separate post-hoc process; "
             "adds nli_* metrics to end-to-end rows; first use downloads the weights).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only evaluate the first N golden items (smoke-testing; 0 = all)",
    )
    args = parser.parse_args()

    settings.SIMULATE_LLM = bool(args.simulate)
    temperatures = _parse_temperatures(args.temperature)

    dataset = load_datasets(_dataset_specs(args.dataset))
    if args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"[eval] dataset: {len(dataset)} questions "
          f"({len({q['source_file'] for q in dataset})} document(s))")

    meter = CostMeter()
    reset_pipeline_cache()
    # state_path=None: never carry a cooldown across runs/processes.
    tracker = QuotaTracker(state_path=None)
    handler = build_default_handler(tracker, on_response=meter.record)
    bundle = build_pipeline(quota_tracker=tracker, llm_handler=handler)
    pipeline = bundle.pipeline

    if args.provider:
        if args.simulate:
            raise SystemExit("--provider cannot be combined with --simulate")
        pipeline.router = _PinnedRouter(pipeline.router, args.provider)
    if not args.simulate and not args.no_preflight:
        _preflight(handler, args.provider)

    if args.nli:
        # Scored in its own process after generation (see _posthoc_nli): the
        # ~700 MB NLI model must not sit in memory beside the embedder.
        print("[eval] NLI claim verification queued (post-hoc subprocess)")

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
            if multi:
                safe = str(temperature).replace(".", "p")
                run_out = out.with_name(
                    f"{out.stem}_{tag}_t{safe}_r{rep}{out.suffix or '.json'}"
                )
            else:
                run_out = out

            e2e_ckpt = _RunCheckpoint(
                run_out.with_name(f"{run_out.stem}.partial.json")
            )
            oracle_ckpt = _RunCheckpoint(
                run_out.with_name(f"{run_out.stem}.oracle.partial.json")
            )
            if args.resume:
                prior_e2e = e2e_ckpt.load()
                prior_oracle = oracle_ckpt.load()
            else:
                e2e_ckpt.clear()
                oracle_ckpt.clear()
                prior_e2e, prior_oracle = [], []

            before = meter.snapshot()
            rows_new = _run_pass(
                dataset, prep, pipeline, temperature,
                meter=meter, max_cost_usd=args.max_cost_usd or None,
                pace_s=args.min_interval_s,
                skip_ids={r["id"] for r in prior_e2e},
                on_row=e2e_ckpt.add,
            )
            rows = prior_e2e + rows_new
            budget_hit = meter.exceeds(args.max_cost_usd or None)

            summary = _summarize(
                rows, temperature, "end_to_end", meter.delta(before)
            )
            summary["repeat"] = rep
            summary["documents"] = len(documents)
            summary["chunks_total"] = total_chunks
            summary["ingest_s"] = round(ingest_s, 2)
            summary["provider_pin"] = args.provider
            summary["budget_exceeded"] = budget_hit

            oracle_rows: list[dict] = []
            if args.oracle and not budget_hit:
                obefore = meter.snapshot()
                oracle_new = _run_pass(
                    dataset, prep, pipeline, temperature, bundle=bundle, oracle=True,
                    meter=meter, max_cost_usd=args.max_cost_usd or None,
                    pace_s=args.min_interval_s,
                    skip_ids={r["id"] for r in prior_oracle},
                    on_row=oracle_ckpt.add,
                )
                oracle_rows = prior_oracle + oracle_new
                oracle_summary = _summarize(
                    oracle_rows, temperature, "oracle", meter.delta(obefore)
                )
                summary["oracle_answer_accuracy"] = oracle_summary["answer_accuracy"]
                summary["oracle_groundedness"] = oracle_summary["groundedness"]
                summary["retrieval_gap"] = round(
                    oracle_summary["answer_accuracy"] - summary["answer_accuracy"], 4
                )
                summary["oracle_by_category"] = oracle_summary["by_category"]
                summary["oracle_cost"] = oracle_summary["cost"]
                summary["budget_exceeded"] = meter.exceeds(args.max_cost_usd or None)

            combined.append(summary)

            payload = {"summary": summary, "documents": documents, "rows": rows}
            if args.oracle:
                payload["oracle_rows"] = oracle_rows
            run_out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            e2e_ckpt.clear()
            oracle_ckpt.clear()

            if args.nli and rows and _posthoc_nli(run_out):
                try:
                    written = json.loads(run_out.read_text(encoding="utf-8"))
                    if isinstance(written.get("summary"), dict):
                        summary = written["summary"]
                        combined[-1] = summary
                except Exception:  # noqa: BLE001 - keep the pre-NLI summary
                    pass

            print(f"\n[summary] t={temperature} r={rep} ({summary['mode']})")
            for key in ("answer_accuracy", "answered_rate",
                        "hit_at_k", "recall_at_k", "precision_at_k", "ndcg_at_k",
                        "mrr_at_k", "context_precision", "context_recall",
                        "groundedness", "ungrounded_fraction", "answer_relevance",
                        "nli_entailed_fraction", "nli_contradiction_fraction",
                        "pii_leak_rate", "citation_support_rate", "citation_coverage",
                        "fallback_rate", "median_e2e_s"):
                value = summary.get(key)
                print(f"  {key}: {value}")
            cost = summary.get("cost") or {}
            print(f"  cost_usd: {cost.get('cost_usd', 0.0)} "
                  f"calls: {cost.get('calls', 0)} "
                  f"tokens: {cost.get('total_tokens', 0)}")
            if summary.get("budget_exceeded"):
                print("  budget_exceeded: True (pass stopped early)")
            if args.oracle:
                for key in ("oracle_answer_accuracy", "oracle_groundedness", "retrieval_gap"):
                    print(f"  {key}: {summary.get(key)}")
            print(f"[eval] wrote {run_out}")

    if len(combined) > 1:
        summary_out = out.with_name(f"{out.stem}_{tag}_summary.json")
        summary_out.write_text(
            json.dumps({"runs": combined}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[sweep] wrote {summary_out}")


if __name__ == "__main__":
    main()
