"""Tests for the real BGE-large embedder. Loads the actual model (no mocking) since the
point is to verify the real model's output shape and behaviour -- slow, real_models only."""
import math

import pytest
from langchain_core.documents import Document

from src.services.retrieval.embedding import (
    embed_documents,
    embed_query,
    embed_texts,
    get_active_model_info,
    get_embedding_dim,
    get_embedding_model,
    token_length,
    EMBEDDING_MODEL_NAME,
    QUERY_INSTRUCTION,
)

pytestmark = pytest.mark.real_models  # excluded by default; see conftest.py


def test_embed_documents_returns_embeddings():
    documents = [
        Document(page_content="The applicant submitted income proof."),
        Document(page_content="The applicant provided bank statements."),
    ]
    embeddings = embed_documents(documents)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[1]) > 0


def test_embedding_dimension():
    embeddings = embed_documents([Document(page_content="The applicant submitted income proof.")])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == get_embedding_dim()


def test_empty_documents():
    assert embed_documents([]) == []


def test_embed_texts_matches_documents():
    texts = ["The applicant submitted income proof.", "Bank statements were provided."]
    from_texts = embed_texts(texts)
    from_docs = embed_documents([Document(page_content=t) for t in texts])
    assert len(from_texts) == len(from_docs) == 2
    assert all(math.isclose(a, b, rel_tol=1e-6) for row_a, row_b in zip(from_texts, from_docs) for a, b in zip(row_a, row_b))


def test_embed_texts_drops_blank():
    assert embed_texts(["", "   "]) == []


def test_token_length_counts_word_pieces():
    assert token_length("") == 0
    long_text = "loan amount interest rate repayment " * 3
    short_text = "loan"
    assert token_length(long_text) > token_length(short_text) >= 1


def test_embedding_values_are_floats():
    embeddings = embed_documents([Document(page_content="The applicant submitted income proof.")])
    assert isinstance(embeddings[0][0], float)


def test_embeddings_are_normalized():
    embeddings = embed_documents([Document(page_content="The applicant submitted income proof.")])
    magnitude = math.sqrt(sum(v * v for v in embeddings[0]))
    assert abs(magnitude - 1.0) < 0.001


def test_model_name():
    assert isinstance(EMBEDDING_MODEL_NAME, str) and EMBEDDING_MODEL_NAME
    assert isinstance(QUERY_INSTRUCTION, str) and QUERY_INSTRUCTION


def test_model_is_loaded_on_cpu():
    info = get_active_model_info()
    if info["provider"] != "local":
        pytest.skip("local fallback is not active (Gemini is being used)")
    model = get_embedding_model()
    assert model.device.type == "cpu"


def test_model_is_reused():
    assert get_embedding_model() is get_embedding_model()


def test_single_document_embedding():
    documents = [
        Document(
            page_content="The applicant's monthly income is 75000 INR.",
            metadata={"page": 3, "source": "Synthetic_Mortgage_Loan_File_TEST.pdf", "chunk_index": 0},
        )
    ]
    embeddings = embed_documents(documents)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == get_embedding_dim()


def test_multiple_documents():
    documents = [
        Document(page_content="The applicant submitted identity proof."),
        Document(page_content="The applicant submitted income proof."),
        Document(page_content="The applicant submitted bank statements."),
        Document(page_content="The property valuation report was submitted."),
    ]
    embeddings = embed_documents(documents)
    assert len(embeddings) == 4
    dim = get_embedding_dim()
    for embedding in embeddings:
        assert len(embedding) == dim


def test_empty_page_content():
    documents = [Document(page_content=""), Document(page_content="   ")]
    assert embed_documents(documents) == []


# ---------------- additions: things the original suite didn't check ----------------
def test_blank_chunks_are_dropped_not_zero_vectors():
    """A mix of real + blank content must drop the blank, not embed it as zeros --
    otherwise zipping vectors back to chunks would misalign."""
    documents = [Document(page_content="Loan amount is 4,50,000."), Document(page_content="   ")]
    embeddings = embed_documents(documents)
    assert len(embeddings) == 1  # blank dropped, only 1 real chunk embedded


def test_embed_query_matches_active_dimension():
    assert len(embed_query("loan amount")) == get_embedding_dim()


def test_embed_query_empty_string():
    assert embed_query("") == []
    assert embed_query("   ") == []


def test_similar_sentences_are_closer_than_unrelated_ones():
    """Sanity check that the model is actually doing semantic embedding, not returning
    noise: a paraphrase should be closer to the original than an unrelated sentence."""
    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))  # vectors are already unit-length

    base = embed_query("What is the monthly loan repayment amount?")
    similar = embed_documents([Document(page_content="The borrower's EMI is 12,500 per month.")])[0]
    unrelated = embed_documents([Document(page_content="The property was painted blue in 1998.")])[0]
    assert cos(base, similar) > cos(base, unrelated)