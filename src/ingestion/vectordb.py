"""Vector store implementations.

* :class:`VectorDB` — legacy local ChromaDB store (opt-in).
* :class:`src.ingestion.qdrant_vectordb.QdrantVectorDB` — Qdrant Cloud (default).

:func:`build_vector_db` returns whichever backend ``ANEXUS_VECTOR_BACKEND``
selects. Both store BGE/embedding vectors + chunk metadata; the retriever
builds its BM25 index from ``get_all_documents()``.

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
from src.observability import hash_id, span

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


class EmbeddingDimensionMismatch(ValueError):
    """Raised when vectors do not match the dimension already stored in Chroma.

    Mixing dimensions inside one collection makes retrieval meaningless, so we
    fail loudly instead of silently writing unusable vectors or wiping data.
    """


def _dimension_error(expected: int, actual: int) -> str:
    return (
        f"Embedding dimension mismatch: the vector index holds {expected}-dim "
        f"vectors but the active embedder produces {actual}-dim vectors. "
        "Ingestion and retrieval must use the same embedding model. "
        "Existing vectors were left untouched; reset the store and re-ingest "
        "your documents (python -m scripts.reset_vector_store --all-users), "
        "or switch back to the model that built the index."
    )


class VectorDB:
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        client: Optional[PersistentClient] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._expected_dim: Optional[int] = None
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

    # ---------------------------------------------------------- dimension guard
    def _expected_dimension(self) -> Optional[int]:
        """Dimension of vectors already in this collection (cached).

        Reads the value stored in collection metadata; for collections created
        before that metadata existed, it infers the dimension from one row.
        """
        if self._expected_dim is not None:
            return self._expected_dim

        metadata = getattr(self._collection, "metadata", None) or {}
        raw = metadata.get("embedding_dim") if isinstance(metadata, dict) else None
        if isinstance(raw, (int, float)) and int(raw) > 0:
            self._expected_dim = int(raw)
            return self._expected_dim

        try:
            result = _retry_on_pool_timeout(
                lambda: self._collection.get(include=["embeddings"], limit=1)
            )
            embeddings = result.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                self._expected_dim = len(embeddings[0])
        except Exception:
            # Empty collection or a backend that cannot return embeddings yet.
            pass
        return self._expected_dim

    def _remember_dimension(self, dimension: int) -> None:
        self._expected_dim = int(dimension)
        try:
            metadata = dict(getattr(self._collection, "metadata", None) or {})
            metadata["embedding_dim"] = int(dimension)
            self._collection.modify(metadata=metadata)
        except Exception:
            # Metadata is an optimisation; row-based inference still works.
            pass

    @property
    def add_batch_size(self) -> int:
        return max(1, int(CHROMA_ADD_BATCH_SIZE))

    @property
    def _collection_hash(self) -> str:
        """Non-reversible collection id for spans/logs (tenant-safe)."""
        return hash_id(self._collection_name)

    @property
    def get_batch_size(self) -> int:
        return max(1, int(CHROMA_GET_BATCH_SIZE))

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
        with span(
            "ingestion.vectordb.add_chunks",
            collection_hash=self._collection_hash,
            document_count=len(docs),
        ):
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

            # Refuse to mix embedding spaces: a dimension change (e.g. Gemini
            # 3072 -> all-mpnet 768) silently corrupts retrieval otherwise.
            incoming_dim = len(embeddings_out[0])
            if incoming_dim <= 0 or any(
                len(vector) != incoming_dim for vector in embeddings_out
            ):
                raise EmbeddingDimensionMismatch(
                    "Every embedding in a batch must share one non-zero dimension."
                )
            expected = self._expected_dimension()
            if expected is not None and expected != incoming_dim:
                raise EmbeddingDimensionMismatch(
                    _dimension_error(expected, incoming_dim)
                )
            if expected is None:
                self._remember_dimension(incoming_dim)

            # Batch the write so a very large document never issues a single giant
            # request or a single in-Chroma buffer.
            batch = self.add_batch_size
            for start in range(0, len(ids), batch):
                end = start + batch
                _retry_on_pool_timeout(
                    lambda start=start, end=end: self._collection.add(
                        ids=ids[start:end],
                        embeddings=embeddings_out[start:end],
                        metadatas=metadatas[start:end],
                        documents=texts[start:end],
                    )
                )
            logger.info(
                "Added %d chunk(s) to Chroma collection %s",
                len(ids),
                self._collection_hash,
            )
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
        query_vector = list(query_embedding)
        expected = self._expected_dimension()
        if expected is not None and len(query_vector) != expected:
            raise EmbeddingDimensionMismatch(
                _dimension_error(expected, len(query_vector))
            )
        with span(
            "ingestion.vectordb.search",
            k=k,
            has_document_filter=where is not None,
            collection_hash=self._collection_hash,
        ):
            try:
                result = _retry_on_pool_timeout(
                    lambda: self._collection.query(
                        query_embeddings=[query_vector],
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
        with span(
            "ingestion.vectordb.delete_document",
            document_id=hash_id(document_id),
            collection_hash=self._collection_hash,
        ):
            try:
                _retry_on_pool_timeout(
                    lambda: self._collection.delete(where={"document_id": document_id})
                )
            except Exception:
                logger.exception("Could not delete chunks of document %s", hash_id(document_id))
                raise
            remaining = _retry_on_pool_timeout(self._collection.count)
            logger.info(
                "Deleted document %s from Chroma collection %s (%d chunk(s) left)",
                hash_id(document_id),
                self._collection_hash,
                remaining,
            )
            return remaining

    def document_ids(self) -> List[str]:
        """Distinct document_ids currently stored (used to reconcile the registry)."""
        with span(
            "ingestion.vectordb.document_ids",
            collection_hash=self._collection_hash,
        ):
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
        with span(
            "ingestion.vectordb.reset", collection_hash=self._collection_hash
        ):
            try:
                _retry_on_pool_timeout(
                    lambda: self._client.delete_collection(self._collection_name)
                )
            except Exception:
                logger.exception("Could not delete collection %s", self._collection_hash)
            self._collection = _retry_on_pool_timeout(
                lambda: self._client.get_or_create_collection(self._collection_name)
            )
            self._expected_dim = None
            logger.info("Reset Chroma collection %s", self._collection_hash)

    def get_all_documents(self) -> List[Document]:
        """Return every stored chunk as a LangChain Document (used to rebuild BM25).

        Reads in pages of ``CHROMA_GET_BATCH_SIZE`` so memory stays bounded even
        for a large corpus.
        """
        with span(
            "ingestion.vectordb.get_all_documents",
            collection_hash=self._collection_hash,
        ):
            total = _retry_on_pool_timeout(self._collection.count)
            if total == 0:
                return []
            docs: List[Document] = []
            batch = self.get_batch_size
            offset = 0
            while offset < total:
                result = _retry_on_pool_timeout(
                    lambda offset=offset: self._collection.get(
                        include=["documents", "metadatas"],
                        limit=batch,
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
                if len(metadatas) < batch:
                    break
                offset += batch
            return docs

    def close(self) -> None:
        if self._own_client:
            self._client.close()


def build_vector_db(
    *,
    collection_name: Optional[str] = None,
    qdrant_client=None,
    chroma_client=None,
):
    """Return the configured vector store (Qdrant Cloud by default).

    ``ANEXUS_VECTOR_BACKEND=chroma`` opts back into the legacy local store.
    Both backends expose the same interface, so the retriever and pipeline do
    not care which one is active.
    """
    from config import settings

    backend = (getattr(settings, "VECTOR_BACKEND", "qdrant") or "qdrant").strip().lower()
    name = collection_name or settings.COLLECTION_NAME

    if backend in ("qdrant", "qdrant-cloud", "qdrant_cloud"):
        from src.ingestion.qdrant_vectordb import QdrantVectorDB

        return QdrantVectorDB(collection_name=name, client=qdrant_client)

    if backend == "chroma":
        return VectorDB(
            collection_name=name,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            client=chroma_client,
        )

    raise ValueError(
        f"Unknown ANEXUS_VECTOR_BACKEND={backend!r}; expected 'qdrant' or 'chroma'"
    )