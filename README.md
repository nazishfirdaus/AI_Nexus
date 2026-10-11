# ANexus — Mortgage Document RAG Chatbot

Ask questions about your mortgage documents — uploaded files (PDF, Word, Excel,
CSV, PowerPoint, text/Markdown, or scanned images), an imported **website**, or
files from **S3-compatible cloud storage**. The pipeline **ingests → chunks →
embeds → hybrid-retrieves → reranks → scrubs PII → routes → generates with
citations**, keeping raw personally-identifying information (PII) out of every
language-model call.

The app is **multi-user**: each person signs in, brings their own LLM API keys
(BYOK), and sees only their own documents and chats.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm # needed by the Presidio PII scrubber

# Configure login + the credential vault (see below), then:
streamlit run app.py                    # open the browser tab shown in the terminal
```

To run **without** an identity provider for local development, set
`ANEXUS_AUTH_DISABLED=true` (and `ANEXUS_SECRET_KEY`); you are signed in as a
local admin and the Settings/Admin pages work as normal.

> **Python version:** developed on **Python 3.13** (project venv). Streamlit Cloud
> targets **Python 3.12**; the code is 3.10+-compatible and does not pin to 3.13-only
> syntax, so both work. Runtime overhead from model loading is expected on low-RAM
> machines (target: 4 GB RAM):

| Startup cost           | Approx. time |
|------------------------|--------------|
| Presidio + spaCy       | ~35 s (once)  |
| RAG pipeline cold import | ~2 min (once) |

Model downloads (BGE embedder ~1.3 GB, MiniLM reranker ~90 MB) happen on first use.

## Authentication, users & isolation

- **Login** uses Streamlit's native OIDC auth (`st.login` / `st.user`). Copy
  `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (gitignored) and
  fill in an `[auth]` block (OAuth client id/secret, `cookie_secret`,
  `server_metadata_url`) plus `ANEXUS_SECRET_KEY` and `ANEXUS_BOOTSTRAP_ADMIN_EMAIL`.
- **Provisioning is admin-driven.** The first login from `ANEXUS_BOOTSTRAP_ADMIN_EMAIL`
  becomes an admin; an admin provisions everyone else (and can enable/disable
  accounts or change roles) on the **Admin** page. Unknown emails are denied.
- **Bring your own keys (BYOK).** Each user stores their own LLM provider keys and
  S3 credentials on the **Settings** page. They are encrypted at rest (Fernet) and
  never displayed again. A user's keys take precedence over any deployment-wide
  env keys; with no keys saved, the deployment defaults (or the simulated LLM) are
  used.
- **Full per-user isolation.** Every account has its own Chroma collection and its
  own SQLite database (`data/users/<id>/app.sqlite3`), so documents and chats can
  never leak across accounts. Heavy models (BGE embedder, cross-encoder reranker,
  Presidio/spaCy) stay process-shared to fit the RAM budget.

## Supported formats

| Format | Extensions | How text is extracted |
|--------|-----------|------------------------|
| PDF | `.pdf` | PyMuPDF per page, Tesseract OCR fallback for scanned pages (≤ 100 pages, ≤ 20 MB) |
| Plain text / Markdown | `.txt`, `.md` | decoded (UTF-8 → Latin-1), split on blank lines |
| Word | `.docx` | python-docx; headings start a new citation page, tables rendered row-per-record |
| Spreadsheet | `.xlsx`, `.csv` | openpyxl / csv; `Header: value` rows, 50 rows per citation page, one page series per sheet |
| PowerPoint | `.pptx` | python-pptx; one citation page per slide (incl. tables and speaker notes) |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff` | Tesseract OCR (single page) |

Non-PDF formats have no real pages, so logical units (heading sections, slide
numbers, sheet row blocks) act as pages — citations like "Page 3" mean slide 3
or the third row block. A document's identity is its **full filename**, so
`notes.pdf` and `notes.txt` coexist in one corpus.

## Other import sources

- **Website** — give a start URL and the app crawls the same domain breadth-first
  (honoring `robots.txt`, bounded by `WEB_MAX_PAGES`, per-page size and a polite
  delay), extracts readable article text with **trafilatura**, and ingests the
  whole crawl as **one document** (each page becomes a citation page). A guard
  blocks non-HTTP schemes and private/loopback/link-local hosts (SSRF protection).
- **Cloud storage (S3-compatible)** — connect any S3-compatible bucket (AWS S3,
  MinIO, Cloudflare R2) from the Settings page, then import every supported file
  under a prefix. Objects are streamed with a size cap; the same format parsers
  used for uploads handle the bytes.

## Features

- **Multiple documents** — upload several documents at once, in any mix of the
  formats above, import a website or an S3 bucket, and they form a single corpus
  that every question is searched against. The sidebar lists each document with its
  page/chunk/vector stats, lets you remove individual documents (the rest stay
  searchable) or clear everything, and the whole set persists across restarts —
  re-ingesting the same document replaces it instead of duplicating it.
- **Bring your own keys (BYOK)** — per-user LLM and S3 credentials, encrypted at
  rest, editable on the Settings page. No shared key is ever charged for a user who
  supplied their own.
- **Accounts & roles** — OIDC login with an admin-managed allowlist; admins
  provision users and manage roles/status from the Admin page. Each account's
  documents, chats, and credentials are fully isolated.
- **Hybrid retrieval** — dense (BGE-large) + BM25 fused with Reciprocal Rank Fusion, then a local cross-encoder rereank.
- **PII security gate** — a Presidio scrubber with custom Indian recognizers removes identifying tokens before any LLM call: names (including honorific-matched Indian names), addresses, phone, email, PAN, Aadhaar, voter ID, driving licence, passport, TAN, plus bank and loan account numbers, IFSC, MICR, GSTIN, UPI/VPA, cheque, policy and application numbers. Dates and bare city names are kept — loan and property answers depend on them — while the pinpoint parts of an address are not. The same scrubber also cleans the answer text and the page citation snippets, which are built from the unredacted stored chunks.
- **Label-anchored account numbers** — an Indian account number is 8–20 bare digits, indistinguishable from a loan amount, so those patterns require a label ("A/c no.", "Loan A/c", "Cheque no.") next to the value. Because chunk boundaries can split a label from its value, each chunk is scrubbed with the tail of the previous one as matching context.
- **Semantic routing** — the query intent picks the first provider; the rest of the chain follows the configured fallback order, so a provider outage degrades gracefully instead of failing.
- **Citations** — every answer quotes the source document and page it was built from; the same page number in two different documents stays a distinct citation, and retrieved context is labeled `[Doc: file.pdf | Page N]` so the model attributes facts to the right document.
- **Quota / rate-limit tracking** — cooldowns stop runaway billing and abuse.
- **Simulated LLM** — `SIMULATE_LLM=true` runs the whole pipeline (including tests and CI) with zero API keys.
- **Multi-turn chat memory** - recent conversation history is windowed (last 6
  messages, per-message cap) and fed to the model; once a chat passes 12
  messages a rolling summary folds in the older turns. History and summaries
  pass through the PII gate before any provider call.
- **Follow-up resolution** - short or reference-heavy follow-ups ("What about
  the second one?") are rewritten into standalone questions before retrieval,
  with an automatic fallback to the raw query when the rewrite finds nothing.
- **Persistent multiple conversations** - chats (titles, messages, citations,
  summaries) are stored per user in SQLite under `data/users/<id>/app.sqlite3`.
  Conversations are not bound to a document: any chat can ask about any uploaded
  document. The sidebar lists, renames and deletes conversations; they survive
  restarts and the uploaded documents are re-attached from the registry/Chroma
  without re-uploading.

## Tests

Unit tests use fakes so they run offline, fast, and never download models:

```bash
python -m pytest -q          # 358 passing (default: real-model tests skipped)
```

Real-model tests (BGE embedder, live retrieval) can be run locally with:

```bash
$env:ANEXUS_RUN_REAL_TESTS=1
python -m pytest -q tests/test_embedder.py tests/test_real_retrieval.py
```

## Evaluation

A 50-question golden dataset (`evaluation/golden_dataset.json`) is annotated with
source pages per category (fact lookup, definitions, multi-hop, comparison,
numerical, negative/unknown, compliance). Score the retrieval + answer stack:

```bash
python -m evaluation.run_eval --file Synthetic_Mortgage_Loan_File_TEST.pdf
# optional: RAGAS-style overlap report from the saved results
python -m evaluation.ragas_eval --input evaluation/results.json
```

Metrics reported: **Recall@K**, **MRR@K**, top-k-empty rate, no-context rate,
PII-leak rate, and latency percentiles.

### CI

`.github/workflows/rag-eval.yml` runs the full suite with `SIMULATE_LLM=true` on
push to `main`/`ashfaque-development` and on PRs, then runs the golden-set eval
and uploads `evaluation/results.json` as an artifact.

## Project layout

```
src/
  auth/          secrets vault (Fernet), user store (SQLite), tenant naming, OIDC claim resolution
  ingestion/     parsers/ (format dispatcher: PDF+OCR, DOCX, CSV/XLSX, PPTX, text, images), sources/ (website crawler, S3 client), chunker (recursive, self-contained), embedder (BGE), vectordb (Chroma), document store (SQLite registry)
  retrieval/     hybrid retriever (dense+BM25 RRF), cross-encoder reranker
  redaction/     Presidio scrubber + build_llm_inputs security gate
  routing/       semantic router + quota tracker
  generation/    LLM handler w/ fallback, aggregator (citations)
  memory/        SQLite chat store, history windows, follow-up rewriter
  rag_pipeline.py       end-to-end orchestrator
  pipeline_factory.py   dependency wiring (test doubles supported)
config/          prompts, settings, quota state
.streamlit/      secrets.toml (gitignored) + secrets.toml.example (OIDC config)
auth_db/         user accounts + encrypted credentials (created at runtime, gitignored)
data/            per-user SQLite databases (created at runtime, gitignored)
chat_db/         legacy single-tenant SQLite registry (gitignored)
evaluation/      golden dataset + harness
tests/           pytest suite (fakes; real-model tests gated by ANEXUS_RUN_REAL_TESTS)
app.py           Streamlit UI
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'nltk'` / slow `import langchain_text_splitters`** — not required; the chunker is self-contained. Do not add `langchain_text_splitters` to hot import paths.
- **First query is slow** — the embedder/reranker/spaCy models load lazily once per process; subsequent turns are fast.
- **Chroma lock errors on re-ingest** — only one process should write a user's
  collection at a time; re-ingesting a document replaces that document's chunks,
  and the sidebar's "Clear all documents" resets the store.
- **Missing spaCy model** — run `python -m spacy download en_core_web_sm`.

## License

MIT (see `LICENSE`). The bundled sample PDF and golden dataset are licensed for
evaluation/demo use only.