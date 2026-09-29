"""Page-aware recursive text chunking.

Takes page-level Documents (with `page_number`, `document_id` metadata) and returns
chunk-level Documents that keep the page metadata intact and add a `chunk_id`.

A self-contained recursive splitter is used instead of ``langchain_text_splitters``:
importing that package pulls in torch/sentence-transformers, spaCy and (uninstalled)
NLTK just to reach ``RecursiveCharacterTextSplitter``, which makes every import slow
and fragile. The algorithm below mirrors the recursive splitter behaviour (splitters
from coarse to fine, chunk-size + overlap windows).
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

from langchain_core.documents import Document

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]


def chunk_id_for(document_id: str, page_number: int, index: int) -> str:
    return f"{document_id}_p{page_number}_c{index}"


class Chunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Sequence[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators or _DEFAULT_SEPARATORS)

# ------------------------------------------------------------- split logic
    @staticmethod
    def _word_carry(pieces: list[str], max_chars: int) -> tuple[list[str], int]:
        """Suffix of `pieces` whose joined length fits in max_chars (kept whole)."""
        carry: list[str] = []
        total = 0
        for piece in reversed(pieces):
            need = len(piece) + (1 if carry else 0)
            if total + need > max_chars:
                break
            carry.insert(0, piece)
            total += need
        return carry, total

    def _merge_parts(self, parts: list[str], separator: str) -> list[str]:
        """Refine oversized parts, then greedily join into <= chunk_size chunks.

        Overlap is carried word-aligned across chunk boundaries as whole pieces, so
        text is never duplicated or lost mid-word.
        """
        stream: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                stream.extend(self.split_text(part))
            else:
                stream.append(part)

        chunks: list[str] = []
        buffer: list[str] = []
        buf_len = 0
        for part in stream:
            extra = len(part) + (1 if buffer else 0)
            if buf_len + extra <= self.chunk_size:
                buffer.append(part)
                buf_len += extra
                continue

            if buffer:
                chunks.append(separator.join(buffer))
            carry, carry_len = self._word_carry(buffer, self.chunk_overlap)
            buffer, buf_len = carry, carry_len
            extra = len(part) + (1 if buffer else 0)
            if buf_len + extra <= self.chunk_size:
                buffer.append(part)
                buf_len += extra
            elif buffer:
                # Overlap carry plus the part cannot fit even without a separator:
                # keep the carry as a short fragment (preserves continuity).
                chunks.append(separator.join(buffer))
                buffer = [part]
                buf_len = len(part)
            else:
                buffer = [part]
                buf_len = len(part)

        if buffer:
            chunks.append(separator.join(buffer))
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Recursively split a single page into <= chunk_size pieces."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        for separator in self.separators:
            if not separator:
                continue
            if separator not in text:
                continue
            parts = [p for p in text.split(separator) if p.strip()]
            if len(parts) <= 1:
                continue
            merged = self._merge_parts(parts, separator)
            out: list[str] = []
            for piece in merged:
                if len(piece) <= self.chunk_size:
                    out.append(piece)
                else:
                    out.extend(self.split_text(piece))
            return out

        # No separator helped (e.g. one giant token): hard split with overlap.
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    # ----------------------------------------------------------- public API
    def split(self, documents: Iterable[Document]) -> list[Document]:
        """Split page Documents into chunk Documents, preserving metadata + chunk_id."""
        pages = list(documents)
        chunks: list[Document] = []
        for page_number, document in enumerate(pages, start=1):
            text = document.page_content or ""
            if not text.strip():
                continue
            metadata = dict(document.metadata or {})
            page = metadata.get("page_number", page_number)
            for index, piece in enumerate(self.split_text(text), start=1):
                chunk_metadata = dict(metadata)
                chunk_metadata["source"] = chunk_metadata.get(
                    "source", chunk_metadata.get("filename", "unknown")
                )
                doc_id = chunk_metadata.get("document_id", "unknown")
                chunk_metadata["chunk_id"] = chunk_id_for(doc_id, page, index)
                chunks.append(Document(page_content=piece, metadata=chunk_metadata))
        logger.info("Chunked %d page(s) into %d chunk(s)", len(pages), len(chunks))
        return chunks