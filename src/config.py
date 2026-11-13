"""Agent-backend settings.

The main application reads ``config/settings.py``. Shared values (Gemini + Qdrant)
are imported from there so the two configs cannot drift; only the agent-only
Portkey/Groq/LangSmith keys live here.
"""
import os

from dotenv import load_dotenv

from config import settings as app_settings

# config/settings.py already loads .env; this keeps direct imports safe too.
load_dotenv()


class Settings:
    # --- Gemini embeddings (shared with the main app) ---
    GEMINI_API_KEY = app_settings.GEMINI_API_KEY

    # --- Vector DB (shared endpoint/key/collection naming) ---
    QDRANT_URL = app_settings.QDRANT_URL
    QDRANT_API_KEY = app_settings.QDRANT_API_KEY
    QDRANT_COLLECTION = app_settings.QDRANT_COLLECTION
    COLLECTION_NAME = app_settings.COLLECTION_NAME

    # --- Reasoning engine (agent-only) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- LLM gateway (agent-only) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("ANEXUS_GROQ_SLUG", "rag")
    GROQ_SLUG_2 = os.getenv("ANEXUS_GROQ_SLUG_2", "brag")

    # --- Observability (agent-only) ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )


settings = Settings()

# Agent-only side effect: keep LangChain tracing out of the Streamlit app.
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGSMITH_TRACING
os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
