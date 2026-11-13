"""HTTP API for the React frontend.

This package is deliberately independent of Streamlit: it imports the RAG
pipeline, the per-tenant namespace helpers and the SQLite chat store directly.
Run it with ``uvicorn src.api.main:app`` (or ``python -m src.api``).
"""
