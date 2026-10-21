"""TruthfulQA closed-book benchmark (Phase E).

TruthfulQA asks questions on which a model is tempted to answer confidently
and falsely. Closed-book here means: no documents, no retrieval, no pipeline -
just the bare generator answering multiple-choice questions. It is the
hallucination baseline for the two-document RAG eval: a model that stays
truthful without context is better calibrated when the context does not
contain the answer.

Only the MC1 split is used (one correct choice among incorrect ones), scored
by exact letter selection - deterministic, judge-free, cheap. The HF release
ships the raw generation rows (``best_answer`` + ``incorrect_answers``) rather
than the original ``mc1_targets``; both shapes are accepted and normalised
into one MC1 record at load time.

Data: the first run downloads ``truthfulqa/truthful_qa`` (generation config)
through ``datasets`` and caches it as ``evaluation/data/truthfulqa_generation.json``
(gitignored); later runs and CI read that file. ``--path`` points at a
pre-downloaded copy instead. ``--download-only`` just fetches the cache.

Usage::

    python -m evaluation.benchmarks.truthfulqa --download-only
    python -m evaluation.benchmarks.truthfulqa --limit 50
    python -m evaluation.benchmarks.truthfulqa --simulate --limit 10

Because no pipeline is built (no embedder / chroma / presidio), a run is
light enough for the 4 GB eval box and only spends provider tokens.
``--simulate`` exercises the plumbing; the simulated adapter cannot pick a
letter, so its accuracy is meaningless by design.
"""
from __future__ import annotations

try:  # load .env before settings
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

import argparse
import json
import random
import re
import time
from pathlib import Path

from config import settings
from evaluation.cost import CostMeter
from src.generation.llm import ScrubbedText, build_default_handler
from src.routing.quota_tracker import QuotaTracker

DATA_PATH = Path("evaluation/data/truthfulqa_generation.json")
#: MC1 choices are lettered A..; TruthfulQA has at most a dozen
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")

SYSTEM_PROMPT = ScrubbedText(
    "You answer multiple-choice questions. Reply with only the letter of the "
    "single best answer."
)


def _normalize(raw: dict) -> dict | None:
    """Reduce either dataset shape to one MC1 record (None if unusable).

    The original repo's files carry ``mc1_targets`` (choices + label); the HF
    release carries ``best_answer`` + ``incorrect_answers`` instead. Both end
    up as ``{"question", "category", "type", "mc1_targets"}`` with the correct
    answer at a known index before shuffling.
    """
    question = raw.get("question")
    if not question:
        return None
    base = {"question": question,
            "category": raw.get("category"), "type": raw.get("type")}
    target = raw.get("mc1_targets") or {}
    if target.get("choices"):
        return {**base, "mc1_targets": {
            "choices": list(target["choices"]), "label": int(target["label"]),
        }}
    correct = raw.get("best_answer")
    incorrect = [a for a in (raw.get("incorrect_answers") or [])
                 if isinstance(a, str) and a.strip() and a != correct]
    if not correct or not incorrect:
        return None
    return {**base, "mc1_targets": {"choices": [correct] + incorrect, "label": 0}}


def build_question(item: dict, n_incorrect: int = 0, seed: int = 0) -> tuple[list[str], int]:
    """Return ``(choices, correct_index)`` for one MC1 item.

    The correct answer is shuffled in with the incorrect ones, seeded by the
    question text, so the letter of the correct answer is stable per
    (question, seed) but never always "A". ``n_incorrect=0`` keeps every
    incorrect choice (faithful MC1); a positive value caps them for cheap
    smoke runs.
    """
    target = item["mc1_targets"]
    label_in = int(target["label"])
    correct = target["choices"][label_in]
    incorrect = [c for i, c in enumerate(target["choices"]) if i != label_in]
    rng = random.Random(f"{seed}:{item['question']}")
    rng.shuffle(incorrect)
    kept = [correct] + (incorrect if n_incorrect <= 0 else incorrect[:n_incorrect])
    order = list(range(len(kept)))
    rng.shuffle(order)
    return [kept[i] for i in order], order.index(0)


def build_messages(question: str, choices: list[str]) -> list[dict]:
    lines = [f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices)]
    user = (
        f"Question: {question}\n\nOptions:\n" + "\n".join(lines) +
        "\n\nReply with only the letter of the single best answer."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ScrubbedText(user)},
    ]


def parse_choice(text: str | None, choices: list[str]) -> int | None:
    """Map a raw reply onto a choice index, or ``None`` if unparseable."""
    if not text:
        return None
    stripped = text.strip()
    match = _LETTER_RE.search(stripped)
    if match:
        index = LETTERS.index(match.group(1))
        if index < len(choices):
            return index
    normalized = " ".join(stripped.lower().split())
    for i, choice in enumerate(choices):
        if " ".join(choice.lower().split()) == normalized:
            return i
    return None


def load_items(path: Path | None = None, limit: int = 0, seed: int = 0) -> list[dict]:
    """Raw TruthfulQA records, downloaded once into the local cache."""
    source = Path(path) if path else DATA_PATH
    if not source.exists():
        if path:
            raise SystemExit(f"TruthfulQA file not found: {path}")
        print(f"[tqa] downloading truthfulqa/truthful_qa (generation) -> {source}")
        from datasets import load_dataset

        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps([dict(row) for row in dataset], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"[tqa] cached {len(dataset)} questions")
    raw_items = json.loads(source.read_text(encoding="utf-8"))
    usable = []
    for raw in raw_items:
        item = _normalize(raw)
        if item is not None:
            usable.append(item)
    dropped = len(raw_items) - len(usable)
    if dropped:
        print(f"[tqa] skipped {dropped} record(s) without a usable MC1 target")
    if limit > 0:
        random.Random(seed).shuffle(usable)
        usable = usable[:limit]
    return usable


#: both cooldown message shapes: the tracker's "cooling down for 20s (...)"
#: log line and the handler's "cooling down (20s left)" failure detail
_COOLDOWN_RE = re.compile(r"cooling down[^0-9]*(\d+(?:\.\d+)?)s")
#: never park a row longer than this waiting out a cooldown
_MAX_COOLDOWN_WAIT_S = 60.0


def _cooldown_wait(error: str | None) -> float | None:
    """Seconds to wait out a provider cooldown, or ``None`` for other errors.

    Without this a single quota cooldown would fail every remaining row of a
    long run (the handler has no other provider to fall back to when one is
    pinned), turning a real run into zero data.
    """
    if not error:
        return None
    match = _COOLDOWN_RE.search(error)
    if not match:
        return None
    wait = float(match.group(1)) + 0.5
    return wait if wait <= _MAX_COOLDOWN_WAIT_S else None


def run_closed_book(items, handler, provider_order, temperature: float = 0.0,
                    meter: CostMeter | None = None, pace_s: float = 0.0,
                    max_cost_usd: float | None = None, n_incorrect: int = 0,
                    seed: int = 0) -> list[dict]:
    """Answer each question with the bare generator and score the letter."""
    rows: list[dict] = []
    for i, item in enumerate(items):
        row_id = f"tqa_{i:04d}"
        choices, label = build_question(item, n_incorrect=n_incorrect, seed=seed)
        messages = build_messages(item["question"], choices)
        response = None
        error = None
        latency = 0.0
        wait = 0.0
        for attempt in range(2):
            if wait:
                print(f"[tqa] {row_id} provider cooling down; "
                      f"waiting {wait:.0f}s")
                time.sleep(wait)
            started = time.perf_counter()
            try:
                response = handler.generate_messages(
                    messages, provider_order, temperature=temperature
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
                response = None
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
            wait = _cooldown_wait(error)
            if error is None or wait is None:
                break
        text = getattr(response, "text", None)
        parsed = parse_choice(text, choices)
        row = {
            "id": row_id,
            "category": item.get("category"),
            "type": item.get("type"),
            "question": item["question"],
            "choices": choices,
            "label": label,
            "response_text": text,
            "parsed_index": parsed,
            "correct": (parsed == label) if parsed is not None else None,
            "provider": getattr(response, "provider", None),
            "model": getattr(response, "model", None),
            "latency_s": round(latency, 3),
            "error": error,
        }
        rows.append(row)
        mark = "?" if parsed is None else ("+" if row["correct"] else "-")
        print(f"[tqa] {row['id']} {mark} {row['category']} "
              f"({row['latency_s']:.1f}s)")
        if error:
            print(f"[tqa] {row['id']} error: {error}")
        if pace_s and i + 1 < len(items):
            time.sleep(pace_s)
        if meter is not None and meter.exceeds(max_cost_usd):
            print("[tqa] budget reached; stopping early")
            break
    return rows


def _accuracy(group: list[dict]) -> float | None:
    answered = [r for r in group if r["parsed_index"] is not None]
    if not answered:
        return None
    return round(sum(1 for r in answered if r["correct"]) / len(answered), 4)


def summarize(rows: list[dict], temperature: float = 0.0,
              cost: dict | None = None, pinned: str | None = None) -> dict:
    """Same shape as run_eval's summary so one report script covers both."""
    if not rows:
        return {
            "mode": "truthfulqa_closed_book", "temperature": temperature,
            "items": 0, "answered_rate": None, "accuracy": None,
            "by_category": {}, "by_type": {}, "median_latency_s": None,
            "fallback_rate": None, "cost": cost or {},
        }
    answered = [r for r in rows if r["parsed_index"] is not None]
    by_category: dict[str, list[dict]] = {}
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row.get("category") or "unknown", []).append(row)
        by_type.setdefault(row.get("type") or "unknown", []).append(row)
    latencies = sorted(
        r["latency_s"] for r in rows if r.get("latency_s") is not None
    )
    fallbacks = [
        r for r in rows
        if r.get("provider") and pinned and r["provider"] != pinned
    ]
    return {
        "mode": "truthfulqa_closed_book",
        "temperature": temperature,
        "items": len(rows),
        "answered_rate": round(len(answered) / len(rows), 4),
        "accuracy": _accuracy(rows),
        "by_category": {k: _accuracy(v) for k, v in sorted(by_category.items())},
        "by_type": {k: _accuracy(v) for k, v in sorted(by_type.items())},
        "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
        "fallback_rate": round(len(fallbacks) / len(rows), 4),
        "cost": cost or {},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=50,
                        help="evaluate N questions (0 = all 817)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--provider", default=None,
                        help="pin one provider instead of the fallback chain")
    parser.add_argument("--simulate", action="store_true",
                        help="offline plumbing check (accuracy is meaningless)")
    parser.add_argument("--download-only", action="store_true",
                        help="fetch the dataset cache and exit")
    parser.add_argument("--path", default=None,
                        help="use a pre-downloaded TruthfulQA JSON instead of the cache")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the sampled subset and answer letters")
    parser.add_argument("--n-incorrect", type=int, default=0,
                        help="cap incorrect choices per question (0 = all, faithful MC1)")
    parser.add_argument("--min-interval-s", type=float, default=0.0,
                        help="pause between questions (free-tier rate limiting)")
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="per-call timeout; raise it for reasoning models "
                             "(default: ANEXUS_LLM_TIMEOUT_S, 30)")
    parser.add_argument("--max-cost-usd", type=float, default=None,
                        help="stop the pass cleanly once spend reaches this")
    parser.add_argument("--out", default="evaluation/truthfulqa_closed_book.json")
    args = parser.parse_args(argv)

    if args.download_only:
        items = load_items(path=args.path)
        print(f"[tqa] {len(items)} questions ready "
              f"({DATA_PATH if not args.path else args.path})")
        return

    settings.SIMULATE_LLM = bool(args.simulate)
    if args.timeout_s:  # adapters read the timeout when they are built
        settings.LLM_TIMEOUT_S = float(args.timeout_s)
    meter = CostMeter()
    handler = build_default_handler(QuotaTracker(state_path=None), on_response=meter.record)
    if args.provider:
        if args.simulate:
            raise SystemExit("--provider cannot be combined with --simulate")
        if args.provider not in handler.available_providers:
            raise SystemExit(
                f"--provider {args.provider!r} is not configured "
                f"(configured: {handler.available_providers})"
            )
        provider_order = [args.provider]
    else:
        provider_order = handler.available_providers
    if not provider_order:
        raise SystemExit("no LLM providers configured; set API keys or --simulate")

    items = load_items(path=args.path, limit=args.limit, seed=args.seed)
    print(f"[tqa] closed-book TruthfulQA: {len(items)} questions, "
          f"t={args.temperature}, providers={provider_order}")

    before = meter.snapshot()
    rows = run_closed_book(
        items, handler, provider_order, temperature=args.temperature,
        meter=meter, pace_s=args.min_interval_s, max_cost_usd=args.max_cost_usd,
        n_incorrect=args.n_incorrect, seed=args.seed,
    )
    summary = summarize(
        rows, temperature=args.temperature, cost=meter.delta(before),
        pinned=provider_order[0],
    )
    summary["provider_pin"] = args.provider
    summary["seed"] = args.seed
    summary["n_incorrect"] = args.n_incorrect
    summary["timeout_s"] = settings.LLM_TIMEOUT_S
    summary["budget_exceeded"] = meter.exceeds(args.max_cost_usd)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[summary] {summary['items']} questions")
    print(f"  accuracy: {summary['accuracy']}")
    print(f"  answered_rate: {summary['answered_rate']}")
    print(f"  median_latency_s: {summary['median_latency_s']}")
    print(f"  fallback_rate: {summary['fallback_rate']}")
    cost = summary["cost"]
    print(f"  cost_usd: {cost.get('cost_usd', 0.0)} "
          f"calls: {cost.get('calls', 0)} "
          f"tokens: {cost.get('total_tokens', 0)}")
    if summary.get("budget_exceeded"):
        print("  budget_exceeded: True (pass stopped early)")
    print(f"[tqa] wrote {out}")


if __name__ == "__main__":
    main()
