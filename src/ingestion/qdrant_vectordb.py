"""Qdrant Cloud vector store.

Public interface:

  * ``add_chunks(documents, embeddings) -> int``
  * ``search(query_embedding, k, where=None) -> list[Document]``
  * ``delete_document(document_id) -> int``
  * ``document_ids() -> list[str]``
  * ``reset()``
  * ``get_all_documents() -> list[Document]``
  * ``count() -> int``

Design notes
------------
* One Qdrant **collection per tenant** (see ``src/auth/tenancy.py``): isolation
  is by construction, not by a filter someone has to remember to apply.
* Embeddings live in a single unnamed vector with **cosine** distance. The
  embedding provider can change (Gemini 3072-dim / local 768-dim), so the
  collection's dimension is discovered from the first write and every later
  write is validated against it; a mismatch raises instead of silently
  corrupting the index.
* Qdrant point ids must be UUIDs or unsigned ints. Chunk ids
  (``"<doc>_p<page>_c<n>"``) are mapped deterministically with UUIDv5, so
  re-ingesting the same chunk overwrites it instead of duplicating it.
* Payload layout is ``{"text": <chunk text>, "metadata": {...}}``. Keeping the
  metadata nested means no document field can shadow the stored text.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from typing import Any, Iterable, List, Optional

from langchain_core.documents import Document

from config import settings
from src.observability import hash_id, span

logger = logging.getLogger(__name__)

_TEXT_KEY = "text"
_METADATA_KEY = "metadata"
_DOCUMENT_ID_FIELD = f"{_METADATA_KEY}.document_id"
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://anexus.local/qdrant/chunk")


def point_id_for(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id (Qdrant only accepts UUID/int ids)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, str(chunk_id)))


def _clean_metadata(metadata: dict) -> dict:
    """Keep only values Qdrant can store in a payload (JSON scalars/lists)."""
    clean: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            clean[key] = value
        elif isinstance(value, (list, tuple, set)):
            clean[key] = [str(v) for v in value]
        else:
            clean[key] = str(value)
    return clean


def _vector_size(info: Any) -> Optional[int]:
    """Read the vector dimension from a collection info object, if present."""
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    if size:
        return int(size)
    if isinstance(vectors, dict):  # named-vector config
        for vector_params in vectors.values():
            size = getattr(vector_params, "size", None)
            if size:
                return int(size)
    return None


class EmbeddingDimensionMismatch(ValueError):
    """Raised when vectors do not match the collection's stored dimension.

    Mixing dimensions inside one collection makes retrieval meaningless, so we
    fail loudly instead of silently writing unusable vectors or wiping data.
    """


class QdrantVectorDB:
    """Qdrant Cloud implementation of the pipeline's vector-store contract."""

    def __init__(
        self,
        *,
        collection_name: Optional[str] = None,
        client: Any = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        dim: Optional[int] = None,
        prefer_grpc: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._collection_name = collection_name or settings.COLLECTION_NAME
        self._dim: Optional[int] = int(dim or settings.QDRANT_VECTOR_DIM or 0) or None
        self._ready = False

        if client is not None:
            self._client = client
            self._own_client = False
        else:
            from qdrant_client import QdrantClient

            endpoint = (url or settings.QDRANT_URL or "").strip()
            if not endpoint:
                raise RuntimeError(
                    "Qdrant is the vector backend but no endpoint is set. "
                    "Set ANEXUS_QDRANT_URL (or QDRANT_URL / QDRANT_CLUSTER_ENDPOINT) and "
                    "QDRANT_API_KEY in .env."
                )
            self._client = QdrantClient(
                url=endpoint,
                api_key=(
                    api_key
                    if api_key is not None
                    else (settings.QDRANT_API_KEY or None)
                ),
                prefer_grpc=(
                    settings.QDRANT_PREFER_GRPC if prefer_grpc is None else prefer_grpc
                ),
                timeout=(
                    settings.QDRANT_TIMEOUT_S if timeout is None else timeout
                ),
            )
            self._own_client = True

    # ---------------------------------------------------------------- lifecycle
    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def _collection_hash(self) -> str:
        """Non-reversible collection id for spans/logs (tenant-safe)."""
        return hash_id(self._collection_name)

    @property
    def add_batch_size(self) -> int:
        return max(1, int(getattr(settings, "QDRANT_ADD_BATCH_SIZE", 256)))

    @property
    def get_batch_size(self) -> int:
        return max(1, int(getattr(settings, "QDRANT_GET_BATCH_SIZE", 1000)))

    def _validate_dim(self, dim: Optional[int]) -> None:
        if dim is None or self._dim is None or int(dim) == self._dim:
            return
        raise EmbeddingDimensionMismatch(
            f"Embedding dimension mismatch: collection '{self._collection_name}' "
            f"stores {self._dim}-dim vectors but {int(dim)}-dim vectors were supplied. "
            "The embedding model changed, so the collection must be reset and the "
            "corpus re-ingested (python -m scripts.reset_vector_store --all-users)."
        )

    def _ensure_collection(self, dim: Optional[int] = None) -> bool:
        """Ensure the collection exists.

        Returns False when the collection does not exist yet and no dimension is
        known (nothing has ever been written), so read-only calls can treat the
        store as empty instead of creating it with a guessed size.
        """
        with self._lock:
            if self._ready:
                self._validate_dim(dim)
                return True

            if self._client.collection_exists(self._collection_name):
                info = self._client.get_collection(self._collection_name)
                existing = _vector_size(info)
                if existing is not None:
                    self._validate_dim(existing)
                    self._dim = existing
                self._ensure_payload_index()
                self._ready = True
                return True

            if dim is None and self._dim is None:
                return False

            self._create_collection(int(dim or self._dim))
            return True

    def _create_collection(self, dim: int) -> None:
        from qdrant_client import models

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=int(dim), distance=models.Distance.COSINE
            ),
        )
        self._dim = int(dim)
        self._ready = True
        self._ensure_payload_index()
        logger.info(
            "Created Qdrant collection %s (%d-dim, cosine)",
            self._collection_hash,
            dim,
        )

    def _ensure_payload_index(self) -> None:
        """Index ``metadata.document_id`` so scoped search/deletes stay fast."""
        from qdrant_client import models

        try:
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=_DOCUMENT_ID_FIELD,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # index is an optimization, never a hard requirement
            logger.warning(
                "Could not create a payload index on %s (%s); filtering still works "
                "but full-scans the collection",
                self._collection_hash,
                _DOCUMENT_ID_FIELD,
                exc_info=True,
            )

    # -------------------------------------------------------------------- writes
    def add_chunks(
        self,
        documents: Iterable[Document],
        embeddings: Iterable[Iterable[float]],
    ) -> int:
        """Upsert chunk Documents with their pre-computed embeddings."""
        docs = list(documents)
        vectors = list(embeddings)
        if len(docs) != len(vectors):
            raise ValueError(
                f"Cannot match {len(docs)} documents to {len(vectors)} embeddings"
            )
        if not docs:
            return 0

        dim = len(vectors[0])
        if any(len(vector) != dim for vector in vectors):
            raise EmbeddingDimensionMismatch(
                "Embeddings have inconsistent dimensions within one batch"
            )

        with span(
            "ingestion.qdrant.add_chunks",
            collection_hash=self._collection_hash,
            chunks=len(docs),
            dim=dim,
        ):
            with self._lock:
                self._ensure_collection(dim)
                from qdrant_client import models

                points: list[models.PointStruct] = []
                for index, (doc, vector) in enumerate(zip(docs, vectors)):
                    meta = _clean_metadata(doc.metadata)
                    chunk_id = meta.get("chunk_id")
                    if not chunk_id:
                        digest = hashlib.sha1(
                            (doc.page_content or "").encode("utf-8")
                        ).hexdigest()
                        chunk_id = f"auto_{digest}"
                    points.append(
                        models.PointStruct(
                            id=point_id_for(chunk_id),
                            vector=[float(v) for v in vector],
                            payload={
                                _TEXT_KEY: doc.page_content or "",
                                _METADATA_KEY: meta,
                            },
                        )
                    )

                batch = self.add_batch_size
                for start in range(0, len(points), batch):
                    self._client.upsert(
                        collection_name=self._collection_name,
                        points=points[start : start + batch],
                        wait=True,
                    )

        logger.info(
            "Upserted %d chunk(s) into Qdrant collection %s",
            len(points),
            self._collection_hash,
        )
        return len(points)

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to one document. Returns chunks remaining."""
        with span(
            "ingestion.qdrant.delete_document",
            document_id_hash=hash_id(document_id),
            collection_hash=self._collection_hash,
        ):
            with self._lock:
                if not self._ensure_collection():
                    return 0
                from qdrant_client import models

                try:
                    self._client.delete(
                        collection_name=self._collection_name,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key=_DOCUMENT_ID_FIELD,
                                        match=models.MatchValue(value=str(document_id)),
                                    )
                                ]
                            )
                        ),
                        wait=True,
                    )
                    remaining = int(
                        self._client.count(
                            collection_name=self._collection_name, exact=True
                        ).count
                    )
                except Exception:
                    logger.exception(
                        "Could not delete chunks of document %s", hash_id(document_id)
                    )
                    raise

        logger.info(
            "Deleted document %s from Qdrant collection %s (%d chunk(s) left)",
            hash_id(document_id),
            self._collection_hash,
            remaining,
        )
        return remaining

    # --------------------------------------------------------------------- reads
    def count(self) -> int:
        with self._lock:
            if not self._ensure_collection():
                return 0
            return int(
                self._client.count(
                    collection_name=self._collection_name, exact=True
                ).count
            )

    def search(
        self,
        query_embedding: Iterable[float],
        k: int = 10,
        where: Optional[dict] = None,
    ) -> List[Document]:
        """Cosine k-NN search, optionally restricted by a metadata filter.

        ``where`` maps metadata field names to values or operators, e.g.
        ``{"document_id": "x"}`` or ``{"page_number": {"$gte": 3}}``.
        """
        if k <= 0:
            return []
        query_vector = [float(v) for v in query_embedding]
        with span(
            "ingestion.qdrant.search",
            k=k,
            filter_fields=sorted(where.keys()) if where else [],
            collection_hash=self._collection_hash,
        ):
            with self._lock:
                if not self._ensure_collection():
                    return []
                self._validate_dim(len(query_vector))
                try:
                    response = self._client.query_points(
                        collection_name=self._collection_name,
                        query=query_vector,
                        query_filter=self._build_filter(where),
                        limit=int(k),
                        with_payload=True,
                        with_vectors=False,
                    )
                except Exception as exc:
                    logger.exception("Qdrant query failed")
                    raise RuntimeError(f"Vector search failed: {exc}") from exc

            return [self._to_document(point) for point in response.points]

    def document_ids(self) -> List[str]:
        """Distinct document_ids currently stored (used to reconcile the registry)."""
        with span(
            "ingestion.qdrant.document_ids", collection_hash=self._collection_hash
        ):
            with self._lock:
                if not self._ensure_collection():
                    return []
                found: set[str] = set()
                try:
                    offset = None
                    while True:
                        records, offset = self._client.scroll(
                            collection_name=self._collection_name,
                            limit=self.get_batch_size,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False,
                        )
                        for record in records:
                            meta = (record.payload or {}).get(_METADATA_KEY) or {}
                            doc_id = meta.get("document_id")
                            if doc_id is not None:
                                found.add(str(doc_id))
                        if offset is None:
                            break
                except Exception:
                    logger.exception("Could not read document ids from Qdrant")
                    return []
        return sorted(found)

    def get_all_documents(self) -> List[Document]:
        """Every stored chunk as a LangChain Document (used to rebuild BM25).

        Scrolled in pages of ``QDRANT_GET_BATCH_SIZE`` so memory stays bounded
        even for a large corpus.
        """
        docs: List[Document] = []
        with span(
            "ingestion.qdrant.get_all_documents", collection_hash=self._collection_hash
        ):
            with self._lock:
                if not self._ensure_collection():
                    return []
                offset = None
                while True:
                    records, offset = self._client.scroll(
                        collection_name=self._collection_name,
                        limit=self.get_batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    docs.extend(self._to_document(record) for record in records)
                    if offset is None:
                        break
        return docs

    def reset(self) -> None:
        """Drop the collection (recreated lazily on the next write)."""
        with span(
            "ingestion.qdrant.reset", collection_hash=self._collection_hash
        ):
            with self._lock:
                try:
                    if self._client.collection_exists(self._collection_name):
                        self._client.delete_collection(self._collection_name)
                except Exception:
                    logger.exception(
                        "Could not delete collection %s", self._collection_hash
                    )
                self._ready = False
        logger.info("Reset Qdrant collection %s", self._collection_hash)

    def close(self) -> None:
        if self._own_client:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - client teardown is best effort
                logger.debug("Error closing Qdrant client", exc_info=True)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_document(point: Any) -> Document:
        payload = dict(getattr(point, "payload", None) or {})
        raw_meta = payload.get(_METADATA_KEY) or {}
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        if meta.get("ocr_used") is not None:
            meta["ocr_used"] = bool(meta["ocr_used"])

        score = getattr(point, "score", None)
        if score is not None:
            meta["similarity"] = max(-1.0, min(1.0, float(score)))

        return Document(page_content=payload.get(_TEXT_KEY, "") or "", metadata=meta)

    @staticmethod
    def _build_filter(where: Optional[dict]) -> Any:
        if not where:
            return None
        from qdrant_client import models

        must: list[Any] = []
        must_not: list[Any] = []

        for key, value in where.items():
            field = f"{_METADATA_KEY}.{key}"
            if isinstance(value, dict):
                for operator, operand in value.items():
                    if operator == "$eq":
                        must.append(
                            models.FieldCondition(
                                key=field, match=models.MatchValue(value=operand)
                            )
                        )
                    elif operator == "$ne":
                        must_not.append(
                            models.FieldCondition(
                                key=field, match=models.MatchValue(value=operand)
                            )
                        )
                    elif operator == "$in":
                        must.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchAny(any=[str(v) for v in operand]),
                            )
                        )
                    elif operator == "$nin":
                        must_not.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchAny(any=[str(v) for v in operand]),
                            )
                        )
                    elif operator in ("$gt", "$gte", "$lt", "$lte"):
                        must.append(
                            models.FieldCondition(
                                key=field,
                                range=models.Range(**{operator[1:]: operand}),
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unsupported vector-store filter operator: {operator}"
                        )
            elif isinstance(value, (list, tuple, set)):
                must.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchAny(any=[str(v) for v in value]),
                    )
                )
            elif value is not None:
                must.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=value)
                    )
                )

        return models.Filter(must=must or None, must_not=must_not or None)
