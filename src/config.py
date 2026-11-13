"""Agent-backend settings.

The main application reads ``config/settings.py``. Shared values (Gemini + Qdrant)
are imported from there so the two configs cannot drift; only the agent-only
Groq/LangSmith keys live here.
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

    # --- LLM gateway (agent-only; used when OPENROUTER_API_KEY is set) ---
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
    OPENROUTER_FALLBACK_MODEL = os.getenv(
        "OPENROUTER_FALLBACK_MODEL", "openai/gpt-oss-20b"
    )
    OPENROUTER_GUARD_MODEL = os.getenv(
        "OPENROUTER_GUARD_MODEL", "openai/gpt-oss-20b"
    )
    OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "ANexus")
    OPENROUTER_CACHE_ENABLED = os.getenv(
        "OPENROUTER_CACHE_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    try:
        OPENROUTER_CACHE_TTL = int(os.getenv("OPENROUTER_CACHE_TTL", "300"))
    except ValueError:
        OPENROUTER_CACHE_TTL = 300

    # --- Reasoning engine (direct Groq fallback when no gateway is configured) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
    GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL", "openai/gpt-oss-20b")

    # --- Observability (agent-only) ---
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING", "true" if LANGSMITH_API_KEY else "false"
    )
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
