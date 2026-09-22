"""Tests for the real BGE-large embedder. Loads the actual model (no mocking) since the
point is to verify the real model's output shape and behaviour -- slow, real_models only."""
import math

import pytest
from langchain_core.documents import Document

from src.ingestion.embedder import (
    embed_documents,
    embed_query,
    get_embedding_model,
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
    assert len(embeddings[0]) == 1024


def test_empty_documents():
    assert embed_documents([]) == []


def test_embedding_values_are_floats():
    embeddings = embed_documents([Document(page_content="The applicant submitted income proof.")])
    assert isinstance(embeddings[0][0], float)


def test_embeddings_are_normalized():
    embeddings = embed_documents([Document(page_content="The applicant submitted income proof.")])
    magnitude = math.sqrt(sum(v * v for v in embeddings[0]))
    assert abs(magnitude - 1.0) < 0.001


def test_model_name():
    assert EMBEDDING_MODEL_NAME == "BAAI/bge-large-en"


def test_model_is_loaded_on_cpu():
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
    assert len(embeddings[0]) == 1024


def test_multiple_documents():
    documents = [
        Document(page_content="The applicant submitted identity proof."),
        Document(page_content="The applicant submitted income proof."),
        Document(page_content="The applicant submitted bank statements."),
        Document(page_content="The property valuation report was submitted."),
    ]
    embeddings = embed_documents(documents)
    assert len(embeddings) == 4
    for embedding in embeddings:
        assert len(embedding) == 1024


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


def test_embed_query_adds_instruction_prefix():
    # We can't inspect the exact text sent to the model from the output vector alone,
    # so this checks the documented contract: embed_query must differ from
    # embedding the same text as a passage (the prefix changes the input).
    query_vec = embed_query("What is the loan amount?")
    passage_vec = embed_documents([Document(page_content="What is the loan amount?")])[0]
    assert query_vec != passage_vec


def test_embed_query_dimension():
    assert len(embed_query("loan amount")) == 1024


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