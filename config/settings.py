import os

# ---- Chunking / ingestion (unchanged) --------------------------------------
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_TEXT_LENGTH_FOR_OCR = 100
OCR_DPI = 300

# ---- Retrieval / reranking (unchanged) -------------------------------------
BASE_K_RETRIEVAL = 10
TOP_K_RERANK = 5

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "anexus_rag_collection"

# ---- Providers (unchanged) --------------------------------------------------
MODEL_SETTINGS = {
    "GROQ": "groq",
    "GEMINI": "gemini_flash",
    "NVIDIA": "nvidia",
}
FALLBACK_CHAIN = ["nvidia", "gemini_flash", "groq"]

# ============================================================================
# NEW: values used by retriever / reranker / router / quota tracker / llm
# ============================================================================
SIMULATE_LLM = os.getenv("SIMULATE_LLM", "0").strip().lower() in ("1", "true", "yes")  # CI sets "true"

# Hybrid retrieval: each retriever returns BASE_K_RETRIEVAL, fused pool is up to 2x
DENSE_K = BASE_K_RETRIEVAL
BM25_K = BASE_K_RETRIEVAL
HYBRID_CANDIDATES = BASE_K_RETRIEVAL * 2
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5
RRF_K = 60
RERANK_BATCH_SIZE = 8  # small batches for the 4 GB RAM machine

# LLM calls
LLM_TIMEOUT_S = 30
LLM_MAX_TOKENS = 700
LLM_TEMPERATURE = 0.1

# Model id sent to each provider's API (override via env)
PROVIDER_MODELS = {
    "nvidia": os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
    "gemini_flash": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
}

# Router: the intent only picks the FIRST provider; the rest of the chain always
# follows FALLBACK_CHAIN order.
INTENT_PREFERRED_PROVIDER = {
    "lookup":   "groq",          # short factual extraction: fastest first
    "summary":  "gemini_flash",  # long context
    "analysis": "nvidia",        # reasoning / comparison
    "general":  FALLBACK_CHAIN[0],
}

# BGE v1 works best when *queries* (not passages) carry this instruction.
# Applied to the dense query only. Make sure embedder.embed_query does NOT add it too.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# PII redaction (Presidio)
SPACY_MODEL = "en_core_web_sm"   # small model: fits 4 GB RAM / Streamlit Cloud
PII_SCORE_THRESHOLD = 0.4
PII_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE",
    "IP_ADDRESS", "US_SSN", "US_BANK_NUMBER", "AADHAAR_NUMBER", "PAN_NUMBER",
]
# DATE_TIME and LOCATION are deliberately NOT redacted: loan answers depend on dates.

# Quota tracker
QUOTA_STATE_PATH = os.path.join(os.path.dirname(__file__), "streamlit_quota.json")
RATE_LIMIT_COOLDOWN_S = 60
ERROR_COOLDOWN_S = 20
MAX_COOLDOWN_S = 900