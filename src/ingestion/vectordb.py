"""ChromaDB persistent vector store.

Stores BGE embeddings + chunk metadata locally. This is purely the local index; the
retriever builds its BM25 index from `get_all_documents()`.

Interface used elsewhere:
  * add_chunks(documents, embeddings)
  * search(query_embedding, k) -> list[Document]
  * reset()
  * get_all_documents() -> list[Document]
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from typing import TYPE_CHECKING

from config.settings import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME

if TYPE_CHECKING:
    from chromadb import PersistentClient
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


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
        return self._collection.count()

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

        self._collection.add(
            ids=ids,
            embeddings=embeddings_out,
            metadatas=metadatas,
            documents=texts,
        )
        logger.info("Added %d chunk(s) to Chroma collection %s", len(ids), self._collection_name)
        return len(ids)

    def search(self, query_embedding: Iterable[float], k: int = 10) -> List[Document]:
        """K-nearest-neighbour search; returns Documents with metadata + distance score."""
        if k <= 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=k,
                include=["documents", "metadatas", "distances"],
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

    def reset(self) -> None:
        """Drop the current collection and recreate it empty (new-document behavior)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            logger.exception("Could not delete collection %s", self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)
        logger.info("Reset Chroma collection %s", self._collection_name)

    def get_all_documents(self) -> List[Document]:
        """Return every stored chunk as a LangChain Document (used to rebuild BM25)."""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )
        docs: List[Document] = []
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        for meta, text in zip(metadatas, documents):
            md = dict(meta or {})
            if md.get("ocr_used") is not None:
                md["ocr_used"] = bool(md["ocr_used"])
            docs.append(Document(page_content=text or "", metadata=md))
        return docs

    def close(self) -> None:
        if self._own_client:
            self._client.close()