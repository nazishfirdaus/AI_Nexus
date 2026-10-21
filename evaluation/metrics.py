"""Offline RAG metrics: retrieval ranking, context quality, and answer grounding.

Every function here is pure (no network, no LLM, no model download) so the unit
suite can exercise them with tiny fixtures and the eval harness can recompute
them from a saved ``results.json``.

The generation-side metrics are deliberate offline stand-ins for the RAGAS
suite, which needs an LLM judge:

* ``groundedness``  ~ RAGAS *faithfulness*  (claim-level, lexical token-F1)
* ``answer_relevance`` ~ RAGAS *answer_relevancy* (embedding cosine vs the gold
  answer; the embed function is injected so tests never load a model)
* ``context_recall`` / ``context_precision`` ~ RAGAS context metrics, computed
  from the gold answer points and the ranked retrieval hit list.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Sequence

# A claim whose best matching context sentence scores below this is ungrounded.
UNGROUNDED_THRESHOLD = 0.3

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9$%.]+")


# --------------------------------------------------------------------------- #
# Retrieval ranking metrics (inputs: 1-based hit ranks in the reranked list)
# --------------------------------------------------------------------------- #
def hits_at_k(hit_ranks: Sequence[int], k: int) -> int:
    """How many relevant items appear at rank <= k."""
    return sum(1 for r in hit_ranks if 1 <= r <= k)


def hit_at_k(hit_ranks: Sequence[int], k: int) -> float:
    """Binary hit rate: 1.0 if any relevant item is within the top k."""
    return 1.0 if hits_at_k(hit_ranks, k) > 0 else 0.0


def precision_at_k(hit_ranks: Sequence[int], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        return 0.0
    return hits_at_k(hit_ranks, k) / k


def recall_at_k(hit_ranks: Sequence[int], total_relevant: int, k: int) -> float:
    """Fraction of all relevant items recovered within the top k."""
    if total_relevant <= 0:
        return 0.0
    return hits_at_k(hit_ranks, k) / total_relevant


def mrr_at_k(hit_ranks: Sequence[int], k: int) -> float:
    """Reciprocal rank of the first relevant item within the top k."""
    best = min((r for r in hit_ranks if 1 <= r <= k), default=0)
    return 1.0 / best if best else 0.0


def ndcg_at_k(is_relevant: Sequence[bool], k: int) -> float:
    """Normalized discounted cumulative gain with binary gains.

    ``is_relevant`` is the ranked relevance list (index 0 = rank 1).
    """
    if k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, rel in enumerate(is_relevant[:k], start=1) if rel)
    n_relevant = min(k, sum(1 for rel in is_relevant if rel))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(hit_ranks: Sequence[int], total_relevant: int, k: int) -> float:
    """Average precision over the top-k (rank-weighted signal-to-noise).

    Each relevant hit contributes the precision at its rank, averaged over all
    relevant items - high ranks count more, which is what distinguishes this
    from plain Precision@K.
    """
    if total_relevant <= 0 or k <= 0:
        return 0.0
    hits = sorted(r for r in hit_ranks if 1 <= r <= k)
    ap = sum((i + 1) / rank for i, rank in enumerate(hits))
    return ap / total_relevant


# --------------------------------------------------------------------------- #
# Context quality (answer-point coverage of the retrieved context)
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def points_in_context(points: Iterable[str], context: str) -> int:
    """How many of the given answer points appear (normalized) in the context."""
    haystack = normalize_text(context)
    found = 0
    for point in points or []:
        needle = normalize_text(str(point))
        if needle and needle in haystack:
            found += 1
    return found


def context_recall(points: Iterable[str], context: str) -> float:
    """Fraction of gold answer points recoverable from the retrieved context.

    This is the offline stand-in for RAGAS context recall: if the answer cannot
    even be assembled from what was fetched, any later failure is retrieval's
    fault, not the generator's.
    """
    points = list(points or [])
    if not points:
        return 0.0
    return points_in_context(points, context) / len(points)


# --------------------------------------------------------------------------- #
# Answer grounding (offline faithfulness)
# --------------------------------------------------------------------------- #
def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims (decimals like 37.05 kept)."""
    if not answer:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(answer.replace("\n", " "))]
    return [p for p in parts if p]


def split_sentences(text: str) -> list[str]:
    return split_claims(text)


def _tokens(text: str) -> list[str]:
    # Keep "." inside decimals (37.05) but drop sentence-final periods, so
    # "37.05%." and "37.05%" tokenize identically.
    return [t for raw in _TOKEN_RE.findall(normalize_text(text))
            if (t := raw.rstrip("."))]


def token_f1(a: str, b: str) -> float:
    """Token-level F1 between two strings (order-insensitive multiset overlap)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def claim_support(claim: str, context: str) -> float:
    """Best token-F1 between a claim and any single sentence of the context.

    Scoring against context *sentences* (rather than the whole blob) keeps long
    contexts from inflating recall with unrelated tokens.
    """
    best = 0.0
    for sentence in split_sentences(context):
        best = max(best, token_f1(claim, sentence))
    return best


def groundedness(answer: str, context: str,
                 threshold: float = UNGROUNDED_THRESHOLD) -> dict:
    """Claim-level grounding of an answer in its retrieved context.

    Returns ``{"score": mean best-support, "ungrounded_fraction": ..., "claims": n}``.
    ``score`` is the offline analogue of RAGAS faithfulness; the ungrounded
    fraction surfaces outright fabrications (claims with no supporting sentence).
    """
    claims = split_claims(answer)
    if not claims:
        return {"score": 0.0, "ungrounded_fraction": 0.0, "claims": 0}
    supports = [claim_support(claim, context) for claim in claims]
    ungrounded = sum(1 for s in supports if s < threshold)
    return {
        "score": round(sum(supports) / len(supports), 4),
        "ungrounded_fraction": round(ungrounded / len(supports), 4),
        "claims": len(claims),
    }


# --------------------------------------------------------------------------- #
# Answer relevance (offline answer-relevancy)
# --------------------------------------------------------------------------- #
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def answer_relevance(answer: str, expected: str,
                     embed: Callable[[list[str]], list[Sequence[float]]]) -> float:
    """Cosine similarity between the answer and the golden answer embeddings.

    ``embed`` is injected (``src.ingestion.embedder.embed_texts`` in production,
    a stub in tests) so this stays offline and model-free in the unit suite.
    This is the offline analogue of RAGAS answer_relevancy: it catches paraphrases
    that the substring-based accuracy scorer misses.
    """
    if not answer or not expected:
        return 0.0
    vectors = embed([answer, expected])
    if len(vectors) < 2:
        return 0.0
    return round(cosine(vectors[0], vectors[1]), 4)
