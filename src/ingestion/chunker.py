# src/ingestion/chunker.py

from typing import List, Dict, Any

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def normalize_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces while preserving line breaks
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    chunk_size:
        Maximum approximate number of characters per chunk.

    chunk_overlap:
        Number of characters repeated between consecutive chunks.
    """

    text = normalize_text(text)

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Chunk parsed PDF documents while preserving page metadata.

    Expected document format:
        {
            "text": "...",
            "page": 1,
            "metadata": {...}
        }
    """

    all_chunks = []

    for document in documents:
        text = document.get("text", "")
        page = document.get("page")

        metadata = document.get("metadata", {}).copy()

        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()

            chunk_metadata.update(
                {
                    "page": page,
                    "chunk_index": chunk_index,
                }
            )

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": chunk_metadata,
                }
            )

    return all_chunks


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for page-level parsed PDF output.
    """

    return chunk_documents(pages)


if __name__ == "__main__":
    sample_text = (
        "This is a sample mortgage document. "
        "The applicant must provide identity proof, "
        "income documents, bank statements, and property documents."
    )

    chunks = chunk_text(sample_text)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(chunk)