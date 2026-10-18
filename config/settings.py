import os

# ---- Chunking / ingestion (unchanged) --------------------------------------
CHUNK_SIZE = int(os.getenv("ANEXUS_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("ANEXUS_CHUNK_OVERLAP", "80"))
# Token-aware sizing (optional). When ANEXUS_CHUNK_TOKEN_AWARE=1 the chunker
# measures text with the embedder's own tokenizer instead of characters, so
# chunks are sized against the model's real window. This needs re-embedding the
# whole corpus to take effect; it is OFF by default (character sizing) until an
# evaluation shows a retrieval win.
CHUNK_TOKEN_AWARE = os.getenv(
    "ANEXUS_CHUNK_TOKEN_AWARE", "0"
).strip().lower() in ("1", "true", "yes")
# Hard cap on chunk length in tokens (must stay below the embedder's max window).
CHUNK_MAX_TOKENS = int(os.getenv("ANEXUS_CHUNK_MAX_TOKENS", "500"))
# Soft fill target: chunks are flushed once they reach this many tokens, giving
# more uniform sizes. Must be <= CHUNK_MAX_TOKENS.
CHUNK_TARGET_TOKENS = int(os.getenv("ANEXUS_CHUNK_TARGET_TOKENS", "384"))
# Token overlap carried between consecutive chunks.
CHUNK_OVERLAP_TOKENS = int(os.getenv("ANEXUS_CHUNK_OVERLAP_TOKENS", "64"))
# Contextual chunk headers (optional). "meta" prepends a short "title · page N"
# header to the text that is *embedded* (improving retrieval of context-poor
# chunks), while the stored chunk text stays clean for citations and the LLM.
# "off" (default) embeds the chunk text unchanged. Needs re-embedding.
CHUNK_CONTEXT_HEADER = os.getenv("ANEXUS_CHUNK_CONTEXT_HEADER", "off").strip().lower()
# Sentence-aware chunking (optional): pack whole sentences so a sentence that fits
# is never split across two chunks. Changes chunk boundaries, so needs re-embedding.
CHUNK_SENTENCE_AWARE = os.getenv("ANEXUS_CHUNK_SENTENCE_AWARE", "0").strip().lower() in ("1", "true", "yes", "on")
MIN_TEXT_LENGTH_FOR_OCR = 100
OCR_DPI = 300
# Chroma writes/reads are batched so a single huge upload or corpus never issues
# one giant call (keeps peak RAM and Chroma request size bounded).
CHROMA_ADD_BATCH_SIZE = int(os.getenv("ANEXUS_CHROMA_ADD_BATCH_SIZE", "512"))
CHROMA_GET_BATCH_SIZE = int(os.getenv("ANEXUS_CHROMA_GET_BATCH_SIZE", "1000"))

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

# When a page's plain text and pymupdf4llm output are both empty/thin, the
# optional pdfplumber table engine is tried before OCR unless turned off.
# "auto" (default) engages it only on pages whose extracted text looks tabular.
PDF_TABLE_EXTRACTION = (
    os.getenv("ANEXUS_PDF_TABLE_EXTRACTION", "auto").strip().lower()
)

# ---- Dedup / data quality ---------------------------------------------------
# When a file's content hash matches an existing document that was uploaded under
# a *different* name: "skip" informs the user and ingests nothing (safe default);
# "replace" removes the existing document first and re-adds it under the new name.
DEDUP_ACTION = os.getenv("ANEXUS_DEDUP_ACTION", "skip").strip().lower()

# ---- Over-cleaning safeguards ----------------------------------------------
# "conservative" (default) never drops content on its own; it only *flags* pages
# where cleaning removed a large share of the raw text. Aggressive profiles are
# reserved for a future opt-in and behave the same today.
CLEANING_PROFILE = os.getenv("ANEXUS_CLEANING_PROFILE", "conservative").strip().lower()
# A page whose normalized length falls below this fraction of its raw length is
# counted as "over-cleaned" (reported, never dropped - important text may be dense).
MIN_PAGE_CONTENT_RATIO = float(os.getenv("ANEXUS_MIN_PAGE_CONTENT_RATIO", "0.15"))
# Chunks shorter than this are dropped at ingest as too small to retrieve. The
# default is 0 (disabled): chunker tail fragments can be legitimately short, so
# dropping them by default would lose real content. Set a positive value to opt
# in. Regardless of this setting, chunks that contain no alphanumeric character
# at all (pure punctuation/whitespace noise) are always dropped.
MIN_CHUNK_CHARS = int(os.getenv("ANEXUS_MIN_CHUNK_CHARS", "0"))

# ---- Repeated-line / boilerplate removal ------------------------------------
# "auto" (default) drops lines that repeat across most pages of a multi-page
# document (running headers/footers/page stamps). "off" disables it entirely.
BOILERPLATE_REMOVAL = os.getenv("ANEXUS_BOILERPLATE_REMOVAL", "auto").strip().lower()
# Only documents with at least this many pages are checked (a 1-2 page doc has no
# meaningful notion of "repeated").
BOILERPLATE_MIN_PAGES = int(os.getenv("ANEXUS_BOILERPLATE_MIN_PAGES", "3"))
# A line must appear on at least this fraction of pages to count as boilerplate.
BOILERPLATE_MIN_PAGE_FRACTION = float(
    os.getenv("ANEXUS_BOILERPLATE_MIN_PAGE_FRACTION", "0.6")
)
# Very short repeated lines (page numbers, "Page 4") are only dropped when they
# also meet the character threshold; this avoids nuking short section labels.
BOILERPLATE_MIN_LINE_CHARS = int(os.getenv("ANEXUS_BOILERPLATE_MIN_LINE_CHARS", "12"))

# ---- Embedded metadata ------------------------------------------------------
# Extract document properties (title/author/created) from PDF/DOCX/XLSX/PPTX and
# attach them to page metadata. Harmless when absent.
EXTRACT_EMBEDDED_METADATA = os.getenv(
    "ANEXUS_EXTRACT_EMBEDDED_METADATA", "1"
).strip().lower() in ("1", "true", "yes")

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

# ---- Retrieval / reranking --------------------------------------------------
# Env-overridable so retrieval depth can be tuned without code changes. `BASE_K`
# is fetched per channel (dense and BM25) before fusion; `TOP_K_RERANK` caps the
# contexts handed to the LLM after cross-encoder reranking.
BASE_K_RETRIEVAL = int(os.getenv("ANEXUS_BASE_K_RETRIEVAL", "10"))
TOP_K_RERANK = int(os.getenv("ANEXUS_TOP_K_RERANK", "5"))

# Env-overridable so a multilingual embedder (e.g. bge-m3) can be opted into
# without code changes. Default is English-tuned BGE-large; switching model
# requires re-embedding the corpus.
EMBEDDING_MODEL_NAME = os.getenv("ANEXUS_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en")
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
# Weighted RRF: each source contributes weight / (RRF_K + rank). Equal weights
# reproduce classic RRF; bias toward dense or BM25 without code changes.
DENSE_WEIGHT = float(os.getenv("ANEXUS_DENSE_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("ANEXUS_BM25_WEIGHT", "0.5"))
RRF_K = int(os.getenv("ANEXUS_RRF_K", "60"))
RERANK_BATCH_SIZE = 8  # small batches for the 4 GB RAM machine


def _optional_float(name: str) -> "float | None":
    """Read an optional float env var; unset/blank means "disabled" (None)."""
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


# Optional relevance thresholds (OFF by default: None = no filtering).
# Retrieval threshold is a *cosine similarity* floor on dense hits. Chroma stores
# unit-length vectors in its default L2 space, where squared-L2 distance
# d = 2 - 2*cos, so cos = 1 - d/2 (exact for normalized embeddings).
RETRIEVAL_MIN_SIMILARITY = _optional_float("ANEXUS_RETRIEVAL_MIN_SIMILARITY")
# Cross-encoder score floor (raw logit from ms-marco MiniLM; higher = more relevant).
RERANK_MIN_SCORE = _optional_float("ANEXUS_RERANK_MIN_SCORE")

# LLM calls
LLM_TIMEOUT_S = 30
LLM_MAX_TOKENS = 700
# Per-task sampling temperatures. Lower = more faithful/deterministic. QA stays
# the most conservative; rewrite/summary default to the same value until tuned.
# All three are env-overridable.
LLM_TEMPERATURE = float(os.getenv("ANEXUS_LLM_TEMPERATURE", "0.1"))
LLM_TEMPERATURE_REWRITE = float(
    os.getenv("ANEXUS_LLM_TEMPERATURE_REWRITE", str(LLM_TEMPERATURE))
)
LLM_TEMPERATURE_SUMMARY = float(
    os.getenv("ANEXUS_LLM_TEMPERATURE_SUMMARY", str(LLM_TEMPERATURE))
)

# Character budget for the retrieved context handed to the LLM (0 = unbounded).
# Highest-ranked chunks are kept until the budget is reached; the top chunk is
# always kept even if it alone exceeds the budget. Guards cost/latency as the
# retrieval-depth knobs (BASE_K_RETRIEVAL / TOP_K_RERANK) are turned up.
CONTEXT_MAX_CHARS = int(os.getenv("ANEXUS_CONTEXT_MAX_CHARS", "6000"))

# Model id sent to each provider's API (override via env)
# NOTE (2026-10): former defaults keep retiring -- NVIDIA meta/llama-3.1-70b
# and meta/llama-3.3-70b-instruct are gone (HTTP 410); groq
# llama-3.3-70b-versatile is gone. Current defaults verified live:
# nvidia/nemotron-3-super-120b-a12b, gemini-3.8-flash, openai/gpt-oss-120b.
PROVIDER_MODELS = {
    "nvidia": os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
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

# Local development escape hatch: when true, skip OIDC and act as DEV_USER_EMAIL
# (auto-provisioned as a normal user). NEVER enable this in a deployed environment.
AUTH_DISABLED = os.getenv("ANEXUS_AUTH_DISABLED", "0").strip().lower() in ("1", "true", "yes")
DEV_USER_EMAIL = os.getenv("ANEXUS_DEV_EMAIL", "dev@localhost").strip().lower()

# Friendly provider name shown on the login screen (Streamlit reads the real
# OIDC client config from the [auth] block in secrets.toml).
AUTH_PROVIDER_NAME = os.getenv("ANEXUS_AUTH_PROVIDER_NAME", "Google").strip()

# Any authenticated OIDC identity is auto-provisioned as a normal user.
ROLE_USER = "user"

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