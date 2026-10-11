import os

# ---- Chunking / ingestion (unchanged) --------------------------------------
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_TEXT_LENGTH_FOR_OCR = 100
OCR_DPI = 300

# ---- Optional higher-accuracy fallback engines ------------------------------
# The baseline (PyMuPDF plain text + Tesseract OCR) is always the primary. The
# engines below engage ONLY when the primary fails or yields too little text,
# and only if their optional package is installed (see requirements-ocr.txt).
#
# PDF text: when a page's plain PyMuPDF text is insufficient, retry that page
# with pymupdf4llm (layout-aware Markdown) before falling back to OCR.
PDF_TEXT_ENGINE = os.getenv("ANEXUS_PDF_TEXT_ENGINE", "pymupdf").strip().lower()
# OCR: when Tesseract is unavailable or returns nothing, retry with RapidOCR.
OCR_ENGINE = os.getenv("ANEXUS_OCR_ENGINE", "tesseract").strip().lower()

# ---- Text cleaning / preprocessing -----------------------------------------
# NFKC + control/zero-width cleanup in parsers/common.normalize_text is always on.
# De-hyphenation joins a word split across a line break ("loan ac-\ncount").
NORMALIZE_DEHYPHENATE = os.getenv(
    "ANEXUS_NORMALIZE_DEHYPHENATE", "1"
).strip().lower() in ("1", "true", "yes")
# Markdown artifacts from the pymupdf4llm fallback (headers, pipes, **bold**):
# "auto" strips them only for that extractor; "true"/"false" force on/off everywhere.
STRIP_MARKDOWN_ARTIFACTS = os.getenv("ANEXUS_STRIP_MARKDOWN", "auto").strip().lower()
# A text/OCR page whose alphanumeric ratio (over non-space chars) is below this is
# treated as garbled and falls through to the next engine (pymupdf4llm) or OCR.
OCR_GIBBERISH_MIN_ALNUM_RATIO = float(
    os.getenv("ANEXUS_OCR_GIBBERISH_MIN_ALNUM_RATIO", "0.5")
)

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

# ============================================================================
# Auth / multi-tenancy (login + full per-user isolation)
# ============================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Registry of accounts and their encrypted credentials.
AUTH_DB_PATH = os.path.join(_PROJECT_ROOT, "auth_db", "auth.sqlite3")
# One directory per user holding that user's documents + chats SQLite file.
USER_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "users")

# Passphrase for the per-user credential vault (Fernet). Any string works; it is
# stretched with SHA-256 into a 32-byte key. Only needed to save credentials.
SECRET_KEY = os.getenv("ANEXUS_SECRET_KEY", "").strip()

# The first login from this email is auto-provisioned as an admin. Empty = off.
BOOTSTRAP_ADMIN_EMAIL = os.getenv("ANEXUS_BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()

# Local development escape hatch: when true, skip OIDC and act as DEV_USER_EMAIL
# (auto-provisioned as an admin). NEVER enable this in a deployed environment.
AUTH_DISABLED = os.getenv("ANEXUS_AUTH_DISABLED", "0").strip().lower() in ("1", "true", "yes")
DEV_USER_EMAIL = os.getenv("ANEXUS_DEV_EMAIL", "dev@localhost").strip().lower()

# Friendly provider name shown on the login screen (Streamlit reads the real
# OIDC client config from the [auth] block in secrets.toml).
AUTH_PROVIDER_NAME = os.getenv("ANEXUS_AUTH_PROVIDER_NAME", "Google").strip()

ROLE_ADMIN = "admin"
ROLE_USER = "user"
VALID_ROLES = (ROLE_ADMIN, ROLE_USER)

STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"
VALID_STATUSES = (STATUS_ACTIVE, STATUS_DISABLED)

# ---- Web source ------------------------------------------------------------
WEB_FETCH_TIMEOUT_S = 15
WEB_MAX_PAGES = 20                 # hard cap on pages per crawl
WEB_MAX_BYTES = 2_000_000          # per-page download cap (2 MB)
WEB_USER_AGENT = "ANexusBot/1.0 (+https://example.invalid/anexus)"
WEB_CRAWL_DELAY_S = 1.0            # politeness delay between page fetches

# ---- Cloud source (S3-compatible) ------------------------------------------
CLOUD_MAX_FILES = 50               # hard cap on objects imported per run
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip()  # empty = real AWS
S3_REGION = os.getenv("S3_REGION", "").strip()