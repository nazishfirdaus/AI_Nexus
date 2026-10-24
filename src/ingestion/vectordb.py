"""ChromaDB persistent vector store.

Stores BGE embeddings + chunk metadata locally. This is purely the local index; the
retriever builds its BM25 index from `get_all_documents()`.

Interface used elsewhere:
  * add_chunks(documents, embeddings)
  * search(query_embedding, k, where=None) -> list[Document]
  * delete_document(document_id)
  * reset()
  * get_all_documents() -> list[Document]
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from typing import TYPE_CHECKING

from config.settings import (
    CHROMA_ADD_BATCH_SIZE,
    CHROMA_GET_BATCH_SIZE,
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
)

if TYPE_CHECKING:
    from chromadb import PersistentClient
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# Chroma's Rust backend uses a small SQLite connection pool; under heavy machine
# load (e.g. several large models resident) a checkout can transiently exceed the
# pool's acquire timeout even though nothing is logically wrong. Retry a bounded
# number of times before surfacing the error.
_POOL_TIMEOUT_RETRIES = 5
_POOL_TIMEOUT_BACKOFF_S = 2.0


def _is_pool_timeout(exc: BaseException) -> bool:
    return "pool timed out while waiting for an open connection" in str(exc).lower()


def _retry_on_pool_timeout(fn):
    """Call ``fn`` retrying on Chroma's transient pool-acquire timeout."""
    for attempt in range(_POOL_TIMEOUT_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - narrowed by _is_pool_timeout
            if _is_pool_timeout(exc) and attempt < _POOL_TIMEOUT_RETRIES - 1:
                logger.warning(
                    "Chroma pool timeout (attempt %d/%d); retrying",
                    attempt + 1,
                    _POOL_TIMEOUT_RETRIES,
                )
                time.sleep(_POOL_TIMEOUT_BACKOFF_S * (attempt + 1))
                continue
            raise


def _clean_metadata(metadata: dict) -> dict:
    """Chroma stores only JSON-safe scalars as collection metadata values."""
    clean: dict = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, (int, float, str)):
            clean[key] = value
        elif isinstance(value, (list, tuple, set)):
            cast = [str(v) for v in value]
            clean[key] = cast
        else:
            clean[key] = str(value)
    return clean


class VectorDB:
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        client: Optional[PersistentClient] = None,
    ) -> None:
        self._lock = threading.Lock()
        if client is not None:
            self._client = client
            self._own_client = False
        else:
            from chromadb import PersistentClient

            self._client = PersistentClient(path=persist_directory)
            self._own_client = True
        self._collection = self._client.get_or_create_collection(collection_name)
        self._collection_name = collection_name

    @property
    def collection(self) -> Collection:
        return self._collection

    def count(self) -> int:
        return _retry_on_pool_timeout(self._collection.count)

    def add_chunks(
        self,
        documents: Iterable[Document],
        embeddings: Iterable[Iterable[float]],
    ) -> int:
        """Store chunk Documents with their pre-computed embeddings.

        Documents and embeddings are consumed in lock-step. Returns the number added.
        """
        docs = list(documents)
        vectors = list(embeddings)
        if len(docs) != len(vectors):
            raise ValueError(
                f"Cannot match {len(docs)} documents to {len(vectors)} embeddings"
            )
        if not docs:
            return 0

        ids: list[str] = []
        metadatas: list[dict] = []
        texts: list[str] = []
        embeddings_out: list[list[float]] = []

        for doc, vector in zip(docs, vectors):
            meta = doc.metadata or {}
            ids.append(str(meta.get("chunk_id") or f"auto_{len(ids)}"))
            metadatas.append(_clean_metadata(meta))
            texts.append(doc.page_content)
            embeddings_out.append([float(v) for v in vector])

        # Batch the write so a very large document never issues a single giant
        # request or a single in-Chroma buffer.
        for start in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
            end = start + CHROMA_ADD_BATCH_SIZE
            _retry_on_pool_timeout(
                lambda start=start, end=end: self._collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings_out[start:end],
                    metadatas=metadatas[start:end],
                    documents=texts[start:end],
                )
            )
        logger.info("Added %d chunk(s) to Chroma collection %s", len(ids), self._collection_name)
        return len(ids)

    def search(
        self,
        query_embedding: Iterable[float],
        k: int = 10,
        where: Optional[dict] = None,
    ) -> List[Document]:
        """K-nearest-neighbour search; returns Documents with metadata + distance score.

        `where` is an optional Chroma metadata filter (e.g. {"document_id": "x"})
        to restrict the search to a subset of the corpus.
        """
        if k <= 0:
            return []
        try:
            result = _retry_on_pool_timeout(
                lambda: self._collection.query(
                    query_embeddings=[list(query_embedding)],
                    n_results=k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            )
        except Exception as e:
            logger.exception("Chroma query failed")
            raise RuntimeError(f"Vector search failed: {e}") from e

        docs: List[Document] = []
        metadatas = result.get("metadatas") or [[]]
        documents = result.get("documents") or [[]]
        distances = result.get("distances") or [[]]
        for meta, text, distance in zip(metadatas[0], documents[0], distances[0]):
            md = dict(meta or {})
            if md.get("ocr_used") is not None:
                md["ocr_used"] = bool(md["ocr_used"])
            md["distance"] = float(distance)
            docs.append(Document(page_content=text or "", metadata=md))
        return docs

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to one document. Returns chunks removed."""
        try:
            _retry_on_pool_timeout(
                lambda: self._collection.delete(where={"document_id": document_id})
            )
        except Exception:
            logger.exception("Could not delete chunks of document %s", document_id)
            raise
        remaining = _retry_on_pool_timeout(self._collection.count)
        logger.info(
            "Deleted document %s from Chroma collection %s (%d chunk(s) left)",
            document_id,
            self._collection_name,
            remaining,
        )
        return remaining

    def document_ids(self) -> List[str]:
        """Distinct document_ids currently stored (used to reconcile the registry)."""
        try:
            result = _retry_on_pool_timeout(
                lambda: self._collection.get(include=["metadatas"])
            )
        except Exception:
            logger.exception("Could not read document ids from Chroma")
            return []
        ids = {
            str(meta.get("document_id"))
            for meta in (result.get("metadatas") or [])
            if meta and meta.get("document_id") is not None
        }
        return sorted(ids)

    def reset(self) -> None:
        """Drop the current collection and recreate it empty (new-document behavior)."""
        try:
            _retry_on_pool_timeout(
                lambda: self._client.delete_collection(self._collection_name)
            )
        except Exception:
            logger.exception("Could not delete collection %s", self._collection_name)
        self._collection = _retry_on_pool_timeout(
            lambda: self._client.get_or_create_collection(self._collection_name)
        )
        logger.info("Reset Chroma collection %s", self._collection_name)

    def get_all_documents(self) -> List[Document]:
        """Return every stored chunk as a LangChain Document (used to rebuild BM25).

        Reads in pages of ``CHROMA_GET_BATCH_SIZE`` so memory stays bounded even
        for a large corpus.
        """
        total = _retry_on_pool_timeout(self._collection.count)
        if total == 0:
            return []
        docs: List[Document] = []
        offset = 0
        while offset < total:
            result = _retry_on_pool_timeout(
                lambda offset=offset: self._collection.get(
                    include=["documents", "metadatas"],
                    limit=CHROMA_GET_BATCH_SIZE,
                    offset=offset,
                )
            )
            metadatas = result.get("metadatas") or []
            documents = result.get("documents") or []
            for meta, text in zip(metadatas, documents):
                md = dict(meta or {})
                if md.get("ocr_used") is not None:
                    md["ocr_used"] = bool(md["ocr_used"])
                docs.append(Document(page_content=text or "", metadata=md))
            if len(metadatas) < CHROMA_GET_BATCH_SIZE:
                break
            offset += CHROMA_GET_BATCH_SIZE
        return docs

    def close(self) -> None:
        if self._own_client:
            self._client.close()