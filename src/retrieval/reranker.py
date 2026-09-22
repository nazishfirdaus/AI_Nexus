"""Local cross-encoder reranking. The model is loaded once per process and shared."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_LOCK = threading.Lock()


def _load_model(name: str, device: str):
    key = (name, device)
    with _LOCK:
        if key not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            logger.info("Loading cross-encoder %s on %s", name, device)
            _MODEL_CACHE[key] = CrossEncoder(name, device=device)
        return _MODEL_CACHE[key]


@dataclass
class RankedChunk:
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict:
        return {"content": self.content, "metadata": self.metadata, "score": self.score}


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL_NAME,
        top_k: int = settings.TOP_K_RERANK,
        batch_size: int = settings.RERANK_BATCH_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size  # small batches keep RAM low on 4 GB machines
        self.device = device

    @property
    def model(self):
        return _load_model(self.model_name, self.device)

    def rerank(
        self, query: str, docs: Sequence[Document], top_k: int | None = None
    ) -> list[RankedChunk]:
        """Score (query, chunk) pairs, sort descending, return top K with metadata."""
        k = top_k or self.top_k
        candidates = [d for d in docs if d.page_content and d.page_content.strip()]
        if not query.strip() or not candidates:
            return []

        pairs = [(query, d.page_content) for d in candidates]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        ranked = sorted(
            (
                RankedChunk(
                    content=d.page_content,
                    metadata=dict(d.metadata or {}),
                    score=float(s),
                )
                for d, s in zip(candidates, scores)
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[:k]