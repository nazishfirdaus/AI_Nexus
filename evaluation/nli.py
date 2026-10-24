"""Local NLI claim verification (Phase D).

The offline metrics in :mod:`evaluation.metrics` measure *lexical* grounding
(overlap, n-gram support). This module adds a semantic check that needs no
API and no rubric: split the generated answer into claims and ask a local
natural-language-inference cross-encoder whether the retrieved context
**entails** each claim, and whether any context **contradicts** it.

Model choice: ``cross-encoder/nli-deberta-v3-base`` (MNLI-trained, ~700 MB of
fp32 weights, CPU-friendly). The model is loaded once per process and shared,
exactly like :class:`src.retrieval.reranker.CrossEncoderReranker`, because
loading costs tens of seconds. First use downloads the weights into the local
HF cache.

Usage::

    # post-hoc over a finished run (its own process - see main())
    python -m evaluation.nli --results evaluation/results.json

    # or programmatically
    from evaluation.nli import NliScorer, nli_metrics
    print(nli_metrics(answer_text, contexts, NliScorer()))

Why post-hoc and not inline in ``run_eval``
-------------------------------------------
``evaluation/run_eval.py`` offers ``--nli``, which runs *this* CLI in a
subprocess once generation has finished, rather than scoring rows as they
complete. Resident next to the BGE embedder (~1.3 GB), presidio/spacy and
torch, the NLI weights pushed a 4 GB machine into paging, which showed up as
chroma pool timeouts and ~100x slower rows. In a fresh process only the NLI
model is loaded, so the two footprints never overlap; if loading fails,
``run_eval`` warns and keeps the results without ``nli_*`` fields.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import settings

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_LOCK = threading.Lock()

#: sentence boundaries: punctuation followed by whitespace, or hard newlines
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
#: drop degenerate fragments ("N/A.", "-") so they don't dilute the fractions
MIN_CLAIM_WORDS = 2


def _load_model(name: str, device: str):
    key = (name, device)
    with _LOCK:
        if key not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            logger.info("Loading NLI cross-encoder %s on %s", name, device)
            _MODEL_CACHE[key] = CrossEncoder(name, device=device)
        return _MODEL_CACHE[key]


def split_claims(answer_text: str) -> list[str]:
    """Split an answer into claim-sized sentences (order preserved)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(answer_text or "")]
    return [p for p in parts if len(p.split()) >= MIN_CLAIM_WORDS]


def _softmax(logits) -> list[list[float]]:
    """Row-wise softmax; CrossEncoder returns raw logits for classification."""
    import math

    rows: list[list[float]] = []
    for row in logits:
        values = [float(v) for v in row]
        peak = max(values)
        exps = [math.exp(v - peak) for v in values]
        total = sum(exps) or 1.0
        rows.append([e / total for e in exps])
    return rows


@dataclass
class ClaimVerdict:
    claim: str
    entailed: bool
    contradiction: bool
    entailment_prob: float
    contradiction_prob: float
    best_context_index: int | None


class NliScorer:
    """Scores (context, claim) pairs with a local cross-encoder NLI model."""

    def __init__(
        self,
        model_name: str = settings.NLI_MODEL_NAME,
        batch_size: int = settings.NLI_BATCH_SIZE,
        device: str = "cpu",
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.threshold = threshold

    @property
    def model(self):
        return _load_model(self.model_name, self.device)

    def _label_indices(self) -> tuple[int, int]:
        """(entailment, contradiction) column indices for this checkpoint."""
        raw = self.model.config.id2label
        labels = [str(value).lower() for _, value in sorted(
            raw.items(), key=lambda kv: int(kv[0])
        )]
        try:
            return labels.index("entailment"), labels.index("contradiction")
        except ValueError as exc:  # pragma: no cover - model contract
            raise RuntimeError(
                f"{self.model_name} is not an NLI checkpoint: labels are {labels}"
            ) from exc

    def verify(self, answer_text: str, contexts: Sequence[str]) -> list[ClaimVerdict]:
        """Verdict per claim; a claim is judged against every context."""
        claims = split_claims(answer_text)
        contexts = [c for c in contexts if c and c.strip()]
        if not claims or not contexts:
            return []

        # premise = the retrieved context, hypothesis = the claim under test
        pairs = [(context, claim) for claim in claims for context in contexts]
        logits = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        probs = _softmax(logits)
        ent_idx, contra_idx = self._label_indices()

        verdicts: list[ClaimVerdict] = []
        for i, claim in enumerate(claims):
            chunk = probs[i * len(contexts) : (i + 1) * len(contexts)]
            entailed_p = max(row[ent_idx] for row in chunk)
            best_index = max(range(len(chunk)), key=lambda j: chunk[j][ent_idx])
            contra_p = max(row[contra_idx] for row in chunk)
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    entailed=entailed_p >= self.threshold
                    and entailed_p >= contra_p,
                    contradiction=contra_p >= self.threshold,
                    entailment_prob=round(entailed_p, 4),
                    contradiction_prob=round(contra_p, 4),
                    best_context_index=best_index,
                )
            )
        return verdicts


def nli_metrics(answer_text: str, contexts: Sequence[str], scorer: NliScorer) -> dict:
    """Aggregate verdicts into the row-level fractions run_eval reports."""
    verdicts = scorer.verify(answer_text, contexts)
    if not verdicts:
        return {}
    n = len(verdicts)
    return {
        "nli_claim_count": n,
        "nli_entailed_fraction": round(
            sum(v.entailed for v in verdicts) / n, 4
        ),
        "nli_contradiction_fraction": round(
            sum(v.contradiction for v in verdicts) / n, 4
        ),
    }


def score_rows(rows: list[dict], scorer: NliScorer, limit: int = 0) -> int:
    """Add ``nli_*`` fields to every row that has an answer and contexts.

    Idempotent: rows without an answer or without contexts are left untouched,
    and re-scoring simply overwrites the previous nli_* values. Returns how
    many rows were scored.
    """
    targets = rows[:limit] if limit else rows
    scored = 0
    for row in targets:
        answer = row.get("answer_text")
        contexts = row.get("source_snippets") or []
        if not answer or not contexts:
            continue
        try:
            computed = nli_metrics(str(answer), contexts, scorer)
        except Exception as exc:  # noqa: BLE001 - never lose a finished run
            print(f"[nli] scoring failed for {row.get('id')}: {exc}")
            continue
        if computed:
            row.update(computed)
            scored += 1
    return scored


def summarize_nli(rows: list[dict]) -> dict:
    """Summary block for a results file (absent rows simply drop out)."""
    scored = [r for r in rows if r.get("nli_claim_count")]
    entailed = [
        float(r["nli_entailed_fraction"]) for r in scored
        if r.get("nli_entailed_fraction") is not None
    ]
    contra = [
        float(r["nli_contradiction_fraction"]) for r in scored
        if r.get("nli_contradiction_fraction") is not None
    ]
    return {
        "nli_entailed_fraction": round(sum(entailed) / len(entailed), 4)
        if entailed else None,
        "nli_contradiction_fraction": round(sum(contra) / len(contra), 4)
        if contra else None,
        "nli_rows_scored": len(scored),
    }


def main(argv: list[str] | None = None) -> None:
    """Post-hoc CLI: add nli_* to an existing results.json.

    Deliberately a *separate process* from ``evaluation.run_eval``: the NLI
    model (~700 MB) must not sit in memory next to the embedder + presidio,
    which on the 4 GB eval box pushes the machine into paging and makes
    chroma's pool time out on every query.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="evaluation/results.json",
                        help="run output written by evaluation.run_eval")
    parser.add_argument("--out", default=None, help="default: edit --results in place")
    parser.add_argument("--pass", dest="row_pass", default="rows",
                        choices=["rows", "oracle_rows"],
                        help="which rows to score (oracle rows carry retrieved - "
                             "not gold - contexts, so prefer the default)")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N rows (0 = all)")
    parser.add_argument("--model", default=settings.NLI_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability at which a verdict is counted")
    args = parser.parse_args(argv)

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"results not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a run output (expected a JSON object)")
    rows = payload.get(args.row_pass)
    if rows is None:
        raise SystemExit(
            f"no {args.row_pass} in {path} (top-level keys: {', '.join(payload)})"
        )

    scorer = NliScorer(model_name=args.model, threshold=args.threshold)
    print(f"[nli] loading {scorer.model_name}", flush=True)
    scorer.model  # fail before touching the file if the weights are missing
    print(f"[nli] scoring {len(rows)} {args.row_pass}", flush=True)
    scored = score_rows(rows, scorer, limit=args.limit)

    summary_block = summarize_nli(rows)
    if isinstance(payload.get("summary"), dict):
        payload["summary"].update(summary_block)
    out = Path(args.out) if args.out else path
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[nli] scored {scored}/{len(rows)} rows", flush=True)
    for key, value in summary_block.items():
        print(f"  {key}: {value}")
    print(f"[nli] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
