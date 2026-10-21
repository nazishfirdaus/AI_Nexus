"""LLM-judge evaluation (RAGAS) over a finished real run - Phase C.

``evaluation/run_eval.py`` scores a run with *offline* metrics only (string
matching, lexical groundedness, embedding cosine) so it stays reproducible and
free. This module is the other half of the picture: it re-scores an existing
``results.json`` with the RAGAS LLM-judge suite, which is the only way to get
an opinion on *semantic* faithfulness/relevance without hand-writing a rubric.

It deliberately evaluates a **finished run** rather than driving the pipeline
itself: rows already contain the question, the answer and the exact scrubbed
context the generator saw, so no retrieval or generation is repeated and this
costs only judge calls.

Because ``ragas==0.4.3`` needs two VertexAI names that langchain-community 0.4.x
removed, the import goes through :mod:`evaluation.ragas_compat`.

Usage::

    # default: all metrics, groq judge, every reference-bearing row
    python -m evaluation.ragas_real --results evaluation/results.json

    # quick look
    python -m evaluation.ragas_real --limit 10 --max-workers 4

    # judge model / metric selection
    python -m evaluation.ragas_real --judge-model openai/gpt-oss-120b \
        --metrics faithfulness,answer_correctness
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

from config import settings

#: metric name -> class name in ragas's legacy ``ragas.metrics`` registry
#: (the only family ``evaluate()``'s isinstance check accepts - see
#: :func:`_resolve_metrics`)
METRIC_CLASSES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "answer_correctness": "AnswerCorrectness",
    "context_precision": "ContextPrecision",
    "context_recall": "ContextRecall",
    "response_groundedness": "ResponseGroundedness",
}
DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
    "response_groundedness",
]
#: default judge: strongest general model this Groq account serves
DEFAULT_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")

#: metrics that need an embedding model (the rest are judge-only)
EMBEDDING_METRICS = {"answer_correctness", "answer_relevancy", "semantic_similarity"}

# The column names ragas 0.4.3's SingleTurnSample schema expects.
_COL_QUESTION = "user_input"
_COL_ANSWER = "response"
_COL_CONTEXTS = "retrieved_contexts"
_COL_REFERENCE = "reference"

#: mirrors evaluation.run_eval.DEFAULT_DATASETS (kept local on purpose: importing
#: run_eval drags in the whole pipeline package, which costs minutes of model
#: imports; tests/test_ragas_real.py asserts the two stay identical).
_DEFAULT_SPECS = {
    "evaluation/golden_dataset_01.json": "Synthetic_Mortgage_Loan_File_TEST_01.pdf",
    "evaluation/golden_dataset_02.json": "Synthetic_Mortgage_Loan_File_TEST_02.pdf",
}


def _dataset_paths(raw: list[str] | None) -> list[str]:
    """``None`` -> both golden datasets; else the ``PATH`` part of each entry.

    Entries may be ``PATH`` or ``PATH=SOURCE``; unlike run_eval's loader this
    module never reads a document, so a missing source is not an error here.
    """
    if not raw:
        return list(_DEFAULT_SPECS)
    return [str(entry).partition("=")[0] for entry in raw]


def _load_references(dataset_args: list[str] | None) -> dict[str, str]:
    """Map golden item id -> expected_answer (``d01_q001`` style ids)."""
    refs: dict[str, str] = {}
    for path_str in _dataset_paths(dataset_args):
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"dataset not found: {path_str}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions") if isinstance(payload, dict) else payload
        if not isinstance(questions, list):
            raise SystemExit(f"{path_str}: expected a list of questions")
        tag = path.stem.replace("golden_dataset_", "d")
        for q in questions:
            expected = q.get("expected_answer")
            if expected:
                refs[f"{tag}_{q['id']}"] = str(expected)
    return refs


def _build_rows(rows: list[dict], references: dict[str, str]) -> list[dict]:
    """Select + rename run rows into the RAGAS single-turn schema."""
    out: list[dict] = []
    for row in rows:
        reference = references.get(row.get("id", ""))
        if not reference:
            continue  # metrics below all want a reference; skip unanswerable joins
        contexts = row.get("source_snippets") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        if not contexts or not row.get("answer_text"):
            continue
        out.append(
            {
                _COL_QUESTION: row["question"],
                _COL_ANSWER: row["answer_text"],
                _COL_CONTEXTS: [str(c) for c in contexts],
                _COL_REFERENCE: reference,
                # kept alongside for reporting only (not a RAGAS column)
                "_id": row["id"],
                "category": row.get("category"),
                "provider": row.get("provider"),
            }
        )
    return out


def _build_judge(provider: str, model: str):
    """A ragas legacy-compatible judge over a Groq/NVIDIA OpenAI endpoint.

    ``llm_factory`` (ragas 0.4.3's recommended entry point) returns an
    ``InstructorLLM``, which the legacy metric classes cannot call at all -
    they reach for ``agenerate_text``/``generate_text`` and crash with
    ``AttributeError``. The only wrappers that *do* implement that interface
    are the deprecated-but-working ``LangchainLLMWrapper`` plus a langchain
    chat model, so build that instead. ``bypass_n=True`` keeps groq from ever
    seeing the ``n`` parameter (it rejects it) by issuing n separate calls.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from src.generation.llm import GroqAdapter, NvidiaAdapter

    key_field = {"groq": "GROQ_API_KEY", "nvidia": "NVIDIA_API_KEY"}[provider]
    base = {"groq": GroqAdapter, "nvidia": NvidiaAdapter}[provider].base_url
    api_key = os.environ.get(key_field)
    if not api_key:
        raise SystemExit(f"--judge {provider}: {key_field} is not set")
    chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        # adapters carry a full request URL; langchain appends the path itself
        base_url=base.rsplit("/chat/completions", 1)[0],
        temperature=0.01,
        timeout=60,
        max_retries=3,
    )
    return LangchainLLMWrapper(langchain_llm=chat, bypass_n=True)


def _build_embeddings(model_name: str):
    """sentence-transformers embeddings for ragas's legacy metric family.

    ``ragas.embeddings.HuggingfaceEmbeddings`` is abstract (no async methods)
    and the modern ``HuggingFaceEmbeddings`` has no ``embed_query``, which is
    what the legacy ``AnswerRelevancy``/``AnswerCorrectness`` reach for - so
    implement the four methods the ABC demands directly. Uses the same
    ``settings.EMBEDDING_MODEL_NAME`` as the app's retriever, so judge-side
    similarity is measured in the run's own embedding space.

    Built as a factory so the class body only executes after
    :func:`evaluation.ragas_compat.ensure_ragas_importable` has run.
    """
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.run_config import RunConfig

    class _LocalEmbeddings(BaseRagasEmbeddings):
        def __init__(self, model_name: str):
            super().__init__()
            from sentence_transformers import SentenceTransformer

            self.model_name = model_name
            self.run_config = RunConfig()
            self._model = SentenceTransformer(model_name)

        def embed_query(self, text: str) -> list[float]:
            return self._model.encode(text).tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(list(texts)).tolist()

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    return _LocalEmbeddings(model_name)


def _resolve_metrics(names: list[str], judge, embeddings=None) -> list:
    """Instantiate each metric with the collaborators its constructor asks for.

    ragas 0.4.3 ships two metric families and only one of them works with
    ``evaluate()``: the modern v2 classes under ``ragas.metrics.collections``
    subclass ``collections.base.BaseMetric``, which ``evaluate()``'s
    ``isinstance(m, ragas.metrics.base.Metric)`` check rejects outright, while
    the legacy classes behind ``ragas.metrics.__getattr__`` do pass it (they
    log a DeprecationWarning).  So resolve from the legacy registry first and
    fail loudly if a name lands outside it.
    """
    import inspect
    import warnings

    import ragas.metrics as registry
    from ragas.metrics.base import Metric

    metrics = []
    for name in names:
        cls_name = METRIC_CLASSES.get(name, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cls = getattr(registry, cls_name, None)
        if not (isinstance(cls, type) and issubclass(cls, Metric)):
            raise SystemExit(
                f"unknown metric {name!r} -> {cls_name!r} is not a metric "
                f"ragas.evaluate() accepts (known: "
                f"{', '.join(sorted(METRIC_CLASSES))})"
            )
        params = inspect.signature(cls).parameters
        kwargs: dict = {}
        if "llm" in params:
            kwargs["llm"] = judge
        if "embeddings" in params:
            if embeddings is None:
                raise SystemExit(
                    f"metric {name!r} needs an embedding model but was not given "
                    f"one (add it to EMBEDDING_METRICS)"
                )
            kwargs["embeddings"] = embeddings
        metrics.append(cls(**kwargs))
    return metrics


def _finite(value):
    """A score, or ``None`` if the job failed.

    ragas assigns ``NaN`` to a sample whose judge call raised (timeout, parse
    failure); ``NaN`` both poisons arithmetic means and serialises as the
    non-standard ``NaN`` literal in JSON, so scrub it everywhere.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _load_scored(path: Path) -> tuple[list[dict], dict]:
    """Prior ``{"rows", "summary"}`` from a previous run, or empty.

    Missing or corrupt files are treated as "nothing to resume" rather than an
    error: a half-written report after a quota kill must not block the next
    pass. Row values are kept verbatim - a prior row may hold two of six
    metrics (the multi-day subset workflow), which the merge layer handles.
    """
    if not path.exists():
        return [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], {}
    if not isinstance(payload, dict):
        return [], {}
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if not isinstance(rows, list) or not isinstance(summary, dict):
        return [], {}
    return rows, summary


def _merge_scored(prior: list[dict], fresh: list[dict]) -> list[dict]:
    """Union prior + fresh rows by ``_id``, fresh values winning per column.

    Day 1 may score faithfulness for rows 1-50; day 2 scores answer_relevancy
    for rows 51-100. Merging keeps all four cells, so a resumed multi-day pass
    accumulates one report instead of overwriting it. Prior rows the fresh run
    did not touch (already fully scored, so skipped) survive as-is.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in prior:
        row_id = str(row.get("_id"))
        merged[row_id] = dict(row)
        order.append(row_id)
    for row in fresh:
        row_id = str(row.get("_id"))
        if row_id not in merged:
            merged[row_id] = {}
            order.append(row_id)
        merged[row_id].update(row)
    return [merged[row_id] for row_id in order]


def _unscored_rows(prepared: list[dict], prior: list[dict],
                   pairs: list[tuple[str, str]]) -> list[dict]:
    """Rows still missing at least one of the metrics this run will compute.

    ``prepared`` entries carry ``_id``. A row counts as done only when every
    requested column is already present and finite; anything less is re-sent,
    which is what lets a new metric be added on a later day without re-paying
    for metrics already scored.
    """
    if not prior:
        return prepared
    by_id = {str(row.get("_id")): row for row in prior}
    pending: list[dict] = []
    for row in prepared:
        done = by_id.get(str(row.get("_id")))
        if done is not None and all(
            _finite(done.get(column)) is not None for _, column in pairs
        ):
            continue
        pending.append(row)
    return pending


def _union_pairs(current: list[tuple[str, str]], prior_summary: dict) -> list[tuple[str, str]]:
    """Current (name, column) pairs plus any columns a prior run recorded.

    Keeps day-1 metrics in the merged summary when day 2 only asks for a
    different metric subset: ``columns`` in the prior summary still names them.
    """
    by_column: dict[str, str] = {column: name for name, column in current}
    for name, column in (prior_summary.get("columns") or {}).items():
        by_column.setdefault(column, name)
    return [(name, column) for column, name in by_column.items()]


def _summarize(scored: list[dict], pairs: list[tuple[str, str]]) -> dict:
    """Mean of each metric, overall and grouped by category / provider.

    ``pairs`` maps the report name (``response_groundedness``) to the column
    ragas actually wrote (``nv_response_groundedness`` - legacy metrics name
    themselves, and a few disagree with their CLI name).
    """
    summary: dict = {"items": len(scored), "columns": dict(pairs)}
    for name, column in pairs:
        values = [v for v in (_finite(r.get(column)) for r in scored) if v is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None

    for key in ("category", "provider"):
        buckets: dict[str, dict] = {}
        for row in scored:
            label = str(row.get(key) or "unknown")
            bucket = buckets.setdefault(label, {"count": 0, "_n": {}})
            bucket["count"] += 1
            for name, column in pairs:
                value = _finite(row.get(column))
                if value is not None:
                    bucket[name] = bucket.get(name, 0.0) + value
                    bucket["_n"][name] = bucket["_n"].get(name, 0) + 1
        summary[f"by_{key}"] = {
            label: {
                "count": bucket["count"],
                **{
                    name: round(bucket[name] / bucket["_n"][name], 4)
                    for name in bucket["_n"]
                },
            }
            for label, bucket in buckets.items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="evaluation/results.json",
                        help="run output written by evaluation.run_eval")
    parser.add_argument("--out", default=None,
                        help="default: <results dir>/ragas_real.json")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help=f"comma-separated subset of: {', '.join(METRIC_CLASSES)}")
    parser.add_argument("--judge", default="groq", choices=["groq", "nvidia"],
                        help="provider serving the judge (must be OpenAI-compatible)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="judge model id (eval-only; need not match the app's)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="parallel judge calls (free tiers tolerate 4)")
    parser.add_argument("--timeout-s", type=int, default=600,
                        help="per-job budget; multi-call metrics (answer_relevancy "
                             "strictness=3, context_precision per retrieved chunk) "
                             "need far more than ragas's 180s default on free tiers")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N rows (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="merge into the existing --out file, skipping rows "
                             "already scored for every --metrics column; use with "
                             "a per-day --metrics subset to spread the judge quota "
                             "across days")
    parser.add_argument("--dataset", action="append", default=None,
                        help="golden dataset(s) for reference answers "
                             "(default: both, same as run_eval)")
    parser.add_argument("--pass", dest="row_pass", default="rows",
                        choices=["rows", "oracle_rows"],
                        help="which rows to score: end-to-end or oracle")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"results not found: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get(args.row_pass) or []
    if not rows:
        raise SystemExit(f"{results_path} has no {args.row_pass}")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    references = _load_references(args.dataset)
    prepared = _build_rows(rows, references)
    skipped = len(rows) - len(prepared)
    if args.limit > 0:
        prepared = prepared[: args.limit]
    if not prepared:
        raise SystemExit("no rows with an answer + reference to evaluate")

    out = Path(args.out) if args.out else results_path.with_name("ragas_real.json")
    prior_rows: list[dict] = []
    prior_summary: dict = {}
    prior_names: list[str] = []
    if args.resume:
        prior_rows, prior_summary = _load_scored(out)
        prior_names = list(prior_summary.get("metrics") or [])
        if prior_rows:
            print(f"[ragas] resume: {len(prior_rows)} rows already in {out}",
                  flush=True)

    # Union the report names so a day-2 subset still summarises day-1 columns.
    summary_pairs = _union_pairs([], prior_summary)
    print(f"[ragas] {len(prepared)} rows "
          f"(skipped {skipped}: no reference or no answer), metrics: "
          f"{', '.join(metric_names)}", flush=True)

    from evaluation.ragas_compat import ensure_ragas_importable

    ensure_ragas_importable()
    print("[ragas] imported", flush=True)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.executor import RunConfig

    judge = _build_judge(args.judge, args.judge_model)
    print(f"[ragas] judge: {args.judge}/{args.judge_model}", flush=True)

    # Only pay for the 2-minute embedder load when a chosen metric needs it.
    embeddings = None
    if set(metric_names) & EMBEDDING_METRICS:
        embeddings = _build_embeddings(settings.EMBEDDING_MODEL_NAME)
        print("[ragas] embeddings ready", flush=True)

    metrics = _resolve_metrics(metric_names, judge, embeddings)
    # legacy metrics name themselves, and a few disagree with the CLI name
    # (ResponseGroundedness reports as "nv_response_groundedness")
    pairs = list(zip(metric_names, [m.name for m in metrics]))
    summary_pairs = _union_pairs(pairs, prior_summary)

    todo = _unscored_rows(prepared, prior_rows, pairs) if args.resume else prepared

    scored: list[dict] = []
    if todo:
        sent = [dict(row) for row in todo]
        ids = [r.pop("_id") for r in sent]
        dataset = EvaluationDataset.from_list(sent)
        print(f"[ragas] evaluating {len(dataset)} rows...", flush=True)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=args.max_workers, timeout=args.timeout_s),
            raise_exceptions=False,
            show_progress=True,
        )

        frame = result.to_pandas()
        records = frame.to_dict(orient="records")
        # ragas evaluates through a thread pool: refuse to attach scores to the
        # wrong rows rather than silently producing a plausible-looking report.
        if len(records) != len(sent):
            raise SystemExit(
                f"ragas returned {len(records)} scored rows for {len(sent)} inputs"
            )
        if _COL_QUESTION in frame.columns:
            drifted = [
                i
                for i, (record, source) in enumerate(zip(records, sent))
                if record.get(_COL_QUESTION) != source[_COL_QUESTION]
            ]
            if drifted:
                raise SystemExit(
                    f"ragas reordered its results ({len(drifted)} rows mismatch, "
                    f"first at index {drifted[0]}); refusing to score"
                )
        for record, row_id, source in zip(records, ids, sent):
            record["_id"] = row_id
            record["category"] = source.get("category")
            record["provider"] = source.get("provider")
            for _, column in pairs:
                record[column] = _finite(record.get(column))
            scored.append(record)
    else:
        print("[ragas] nothing new to score; every row already has these metrics",
              flush=True)

    all_rows = _merge_scored(prior_rows, scored) if args.resume else scored
    summary = _summarize(all_rows, summary_pairs)
    summary.update(
        {
            "pass": args.row_pass,
            "judge": f"{args.judge}/{args.judge_model}",
            "timeout_s": args.timeout_s,
            "max_workers": args.max_workers,
            "rows_in_run": len(rows),
            "rows_skipped_no_reference": skipped,
            # union across resumed days so the report names every metric it holds
            "metrics": sorted(set(metric_names) | set(prior_names)),
            "rows_scored_this_run": len(scored),
        }
    )

    out.write_text(
        json.dumps({"summary": summary, "rows": all_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[ragas] summary ({args.row_pass}, n={summary['items']}):", flush=True)
    for name, column in summary_pairs:
        scored_n = sum(1 for r in all_rows if r.get(column) is not None)
        note = "" if scored_n == summary["items"] else f"  ({scored_n} scored)"
        print(f"  {name}: {summary.get(name)}{note}")
    if args.resume:
        print(f"[ragas] scored {len(scored)} new row(s) this run", flush=True)
    print(f"[ragas] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
