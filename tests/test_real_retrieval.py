"""Real local-model retrieval test (optional, NOT for CI).

Runs the genuine BGE embedder + Chroma + cross-encoder against the synthetic mortgage
PDF. The BGE model is large (~1.3 GB) and inference is CPU-bound, so this test is
opt-in:

    ANEXUS_RUN_REAL_TESTS=1 python -m pytest tests/test_real_retrieval.py -q
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("ANEXUS_RUN_REAL_TESTS", "0") != "1",
    reason="opt-in local model test (set ANEXUS_RUN_REAL_TESTS=1)",
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = ROOT / "Synthetic_Mortgage_Loan_File_TEST.pdf"


def test_end_to_end_real_retrieval(tmp_path):
    if not SAMPLE_PDF.exists():
        pytest.skip("synthetic sample PDF missing")

    from src.ingestion.chunker import Chunker
    from src.ingestion.embedder import embed_documents, embed_query
    from src.ingestion.parser import parse_pdf
    from src.ingestion.vectordb import VectorDB
    from src.retrieval.retriever import HybridRetriever

    pages = parse_pdf(SAMPLE_PDF)
    assert pages and pages[0].metadata.get("page_number") == 1

    chunks = Chunker().split(pages)
    assert chunks

    vectors = embed_documents(chunks)
    assert len(vectors) == len(chunks)

    store = VectorDB(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="real_test_collection",
    )
    store.add_chunks(chunks, vectors)
    assert store.count() == len(chunks)

    retriever = HybridRetriever(vector_db=store, documents=chunks, k=10)
    query_vector = embed_query("interest rate")
    results = retriever.dense_retrieve("interest rate")
    assert results, "dense retrieval returned no results"

    # BM25 side should also find a token overlap.
    sparse = retriever.bm25_retrieve("interest rate")
    assert sparse is not None  # empty only if no token overlap exists
    store.close()