"""Page-aware recursive text chunking.

Takes page-level Documents (with `page_number`, `document_id` metadata) and returns
chunk-level Documents that keep the page metadata intact and add a `chunk_id`.

A self-contained recursive splitter is used instead of ``langchain_text_splitters``:
importing that package pulls in torch/sentence-transformers, spaCy and (uninstalled)
NLTK just to reach ``RecursiveCharacterTextSplitter``, which makes every import slow
and fragile. The algorithm below mirrors the recursive splitter behaviour (splitters
from coarse to fine, chunk-size + overlap windows).

Sizing is measured with an injectable ``length_fn`` (characters by default). Passing
the embedding model's tokenizer lets chunks be sized in model tokens instead, without
this module importing any heavy library itself.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Sequence

from langchain_core.documents import Document

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

# Sentence boundary: whitespace that follows a sentence terminator. The lookbehind
# keeps the terminator with the preceding sentence, so joining sentences back with
# a single space reproduces the text exactly.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_id_for(document_id: str, page_number: int, index: int) -> str:
    return f"{document_id}_p{page_number}_c{index}"


class Chunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Sequence[str] | None = None,
        length_fn: Callable[[str], int] | None = None,
        target_size: int | None = None,
        sentence_aware: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators or _DEFAULT_SEPARATORS)
        # Sentence-aware mode packs whole sentences into chunks (never splitting a
        # sentence that fits) instead of splitting on the first matching separator.
        self.sentence_aware = bool(sentence_aware)
        # ``length_fn`` measures a piece in the same unit as chunk_size/overlap.
        # Default is characters; a tokenizer length enables token-aware sizing.
        self._measure: Callable[[str], int] = length_fn or len
        # Soft fill target: close a chunk once it reaches this many units. Only
        # tighter than chunk_size; equal to it (the default) means "fill the cap".
        target = target_size or chunk_size
        self.target_size = min(max(1, target), chunk_size)

    # ------------------------------------------------------------- split logic
    def _word_carry(self, pieces: list[str], max_units: int) -> tuple[list[str], int]:
        """Suffix of `pieces` whose joined length fits in max_units (kept whole)."""
        carry: list[str] = []
        total = 0
        for piece in reversed(pieces):
            need = self._measure(piece) + (1 if carry else 0)
            if total + need > max_units:
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
            if self._measure(part) > self.chunk_size:
                stream.extend(self._split_recursive(part))
            else:
                stream.append(part)

        chunks: list[str] = []
        buffer: list[str] = []
        buf_len = 0
        for part in stream:
            part_len = self._measure(part)
            extra = part_len + (1 if buffer else 0)
            if buf_len + extra <= self.chunk_size:
                buffer.append(part)
                buf_len += extra
                # Soft target: close the chunk early so sizes stay uniform.
                if self.target_size < self.chunk_size and buf_len >= self.target_size:
                    chunks.append(separator.join(buffer))
                    carry, carry_len = self._word_carry(buffer, self.chunk_overlap)
                    buffer, buf_len = carry, carry_len
                continue

            if buffer:
                chunks.append(separator.join(buffer))
            carry, carry_len = self._word_carry(buffer, self.chunk_overlap)
            buffer, buf_len = carry, carry_len
            extra = part_len + (1 if buffer else 0)
            if buf_len + extra <= self.chunk_size:
                buffer.append(part)
                buf_len += extra
            elif buffer:
                # Overlap carry plus the part cannot fit even without a separator:
                # keep the carry as a short fragment (preserves continuity).
                chunks.append(separator.join(buffer))
                buffer = [part]
                buf_len = part_len
            else:
                buffer = [part]
                buf_len = part_len

        if buffer:
            chunks.append(separator.join(buffer))
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Split text with no usable separator, windows measured in length units."""
        units = self._measure(text)
        if units <= self.chunk_size:
            return [text]
        # Convert a unit window into a character window using the measured density.
        # For the default len() measure the density is exactly 1 char/unit.
        chars_per_unit = len(text) / units if units else 1.0
        window = max(1, int(round(self.chunk_size * chars_per_unit)))
        step = max(1, int(round((self.chunk_size - self.chunk_overlap) * chars_per_unit)))
        return [text[i : i + window] for i in range(0, len(text), step)]

    def split_text(self, text: str) -> list[str]:
        """Recursively split a single page into <= chunk_size pieces."""
        text = text.strip()
        if not text:
            return []
        if self._measure(text) <= self.chunk_size:
            return [text]

        if self.sentence_aware:
            sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
            if len(sentences) > 1:
                return self._merge_sentences(sentences)

        return self._split_recursive(text)

    def _merge_sentences(self, sentences: list[str]) -> list[str]:
        """Pack whole sentences into chunks; oversized sentences split generically."""
        merged = self._merge_parts(sentences, " ")
        out: list[str] = []
        for piece in merged:
            if self._measure(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split_recursive(piece))
        return out

    def _split_recursive(self, text: str) -> list[str]:
        """Generic recursive split (no sentence awareness) used as the fallback."""
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
                if self._measure(piece) <= self.chunk_size:
                    out.append(piece)
                else:
                    out.extend(self._split_recursive(piece))
            return out

        # No separator helped (e.g. one giant token): hard split with overlap.
        return self._hard_split(text)

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
