import os

# ---- Chunking / ingestion (unchanged) --------------------------------------
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_TEXT_LENGTH_FOR_OCR = 100
OCR_DPI = 300

# ---- Upload limits / supported formats -------------------------------------
MAX_UPLOAD_MB = 20            # per-file cap enforced by the UI
MAX_PDF_PAGES = 100           # hard cap on PDF page count
TEXT_PAGE_TARGET_CHARS = 2000 # target size of a pseudo-page for text-like formats
TABLE_ROWS_PER_PAGE = 50      # data rows per pseudo-page for CSV/XLSX
MAX_TABLE_ROWS = 20000        # safety cap per table/sheet

# The single source of truth for what the uploader accepts and what
# src.ingestion.parsers.parse_document can dispatch on.
SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".csv",
    ".xlsx",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

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
# NOTE (2026-09): the former defaults were retired -- groq llama-3.3-70b-versatile
# and NVIDIA meta/llama-3.1-70b-instruct are gone; Gemini requires gemini-3.8-flash.
PROVIDER_MODELS = {
    "nvidia": os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
    "gemini_flash": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
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
# Chunks overlap, so a label ("A/c No.") and the value it introduces can end up in
# different chunks. This many trailing characters of each chunk are carried into the
# next one for *matching only*, so a label-anchored pattern still sees its value.
PII_CONTEXT_CARRY_CHARS = 80
PII_ENTITIES = [
    # Presidio built-ins
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IP_ADDRESS",
    "US_SSN", "US_BANK_NUMBER", "IBAN_CODE",
    # Indian identity
    "PAN_NUMBER", "AADHAAR_NUMBER", "IN_PERSON_NAME", "IN_ADDRESS", "IN_PINCODE",
    "VOTER_ID", "DL_NO", "PASSPORT_NO",
    # Indian financial
    "IN_ACCOUNT_NO", "IN_LOAN_NO", "IFSC_CODE", "MICR_CODE", "GSTIN", "UPI_ID",
    "CHEQUE_NO", "POLICY_NO", "TAN_NUMBER", "APPLICATION_ID",
]
# Deliberately NOT redacted:
#   DATE_TIME - loan answers are about due dates, tenures and disbursement schedules.
#   LOCATION - property and borrower city are facts the user asked for. The precise
#               parts of an address are covered by IN_ADDRESS instead, so a bare
#               city name stays readable while "S/o X, H.No. 12, Lucknow - 226010"
#               does not.

# Quota tracker
QUOTA_STATE_PATH = os.path.join(os.path.dirname(__file__), "streamlit_quota.json")
RATE_LIMIT_COOLDOWN_S = 60
ERROR_COOLDOWN_S = 20
MAX_COOLDOWN_S = 900

# Chat memory
CHAT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_db", "chats.sqlite3")
HISTORY_MAX_TURNS = 6              # most recent messages sent to the LLM verbatim
HISTORY_MAX_MESSAGE_CHARS = 1000   # per-message cap inside the history window
HISTORY_SUMMARY_TRIGGER = 12       # messages before a rolling summary is maintained
QUERY_REWRITE_ENABLED = True       # heuristic-gated follow-up resolution
REWRITE_MAX_WORDS = 6              # queries this short are treated as follow-ups