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
  `server_metadata_url`) plus `ANEXUS_SECRET_KEY`.
- **Provisioning is automatic.** Any identity that authenticates via OIDC is
  auto-provisioned as a normal user on first login and immediately gets its own
  isolated area. There are no roles or admin page. Accounts start active; an
  account can still be disabled in the registry if you need to revoke access.
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
| PDF | `.pdf` | PyMuPDF per page, Tesseract OCR fallback for scanned pages (≤ 100 pages, ≤ 20 MB); optional pymupdf4llm fallback for weak pages and pdfplumber table extraction for tabular pages |
| Plain text / Markdown | `.txt`, `.md` | decoded (UTF-8 → Latin-1), split on blank lines |
| Word | `.docx` | python-docx; headings start a new citation page, tables rendered row-per-record |
| Spreadsheet | `.xlsx`, `.csv` | openpyxl / csv; `Header: value` rows, 50 rows per citation page, one page series per sheet |
| PowerPoint | `.pptx` | python-pptx; one citation page per slide (incl. tables and speaker notes) |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff` | Tesseract OCR (single page); optional RapidOCR fallback |

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

## Optional higher-accuracy fallback engines

The baseline ingestion stack (**PyMuPDF** plain text + **Tesseract** OCR) runs
exactly as before and needs no extra setup. Optional engines can be added as
automatic fallbacks that engage **only when the baseline fails or returns too
little text**:

- **pymupdf4llm** — layout-aware Markdown. When a PDF page's plain text is too
  short, that page is retried with pymupdf4llm before OCR.
- **pdfplumber** — table extraction. When a page's layout text is thin/jumbled
  and the page looks tabular, the page is retried with pdfplumber, whose spans
  become ordinary `Column: value` lines that dense + BM25 retrieval can match.
  Control with `ANEXUS_PDF_TABLE_EXTRACTION` (`auto`|`on`|`off`, default `auto`).
- **RapidOCR** — when Tesseract is unavailable or returns no text, RapidOCR is
  tried instead.

Install the extras (adds `opencv-python-headless` and the self-contained layout,
OCR and table models):

```bash
pip install -r requirements-ocr.txt
```

All engines are optional: without them the app degrades back to the baseline.
Select the *primary* engine (the other becomes the fallback) with
`ANEXUS_PDF_TEXT_ENGINE` (`pymupdf`|`pymupdf4llm`|`tables`), `ANEXUS_OCR_ENGINE`
(`tesseract`|`rapidocr`); each extracted page is tagged with an `extractor` and,
for OCR, an `ocr_engine` field.

## Text cleaning / preprocessing

Every extracted page passes through `parsers/common.normalize_text`, which cleans
the text before it is chunked and embedded:

- **Unicode NFKC** — ligatures (`ﬁ`→`fi`), full-width forms and non-breaking
  spaces become their plain ASCII equivalents.
- **Invisible characters removed** — soft hyphens, zero-width joiners/spaces and
  BOM/control characters, so they can no longer split or corrupt tokens.
- **De-hyphenation** — a word split across a line break (`loan ac-\ncount`) is
  rejoined. Disable with `ANEXUS_NORMALIZE_DEHYPHENATE=false`.
- **Whitespace tidy** — per-line trimming, collapsed runs of spaces/tabs and a cap
  on consecutive blank lines, preserving meaningful line structure.

Two optional guards improve extraction quality without adding dependencies:

- **Markdown-artifact stripping** — the layout-aware `pymupdf4llm` fallback emits
  Markdown (`#` headers, `|` tables, `**bold**`). In the default `auto` mode those
  relics are stripped only for that extractor, so uploaded `.md` files keep their
  authoring syntax. Override with `ANEXUS_STRIP_MARKDOWN` (`auto`|`true`|`false`).
- **OCR gibberish guard** — a page whose alphanumeric ratio is too low is treated
  as noise and retried with the next engine (pymupdf4llm) or OCR, instead of being
  accepted as text. Tune with `ANEXUS_OCR_GIBBERISH_MIN_ALNUM_RATIO` (default `0.5`).

### Repeated-line / boilerplate removal

Running headers, footers and page stamps survive extraction and would otherwise be
chunked into every page. On ingest, lines that repeat across most pages of a
multi-page document are dropped (default `ANEXUS_BOILERPLATE_REMOVAL=auto`). It is
deliberately conservative:

- only documents with at least `ANEXUS_BOILERPLATE_MIN_PAGES` pages (default 3) are checked;
- a line must appear on at least `ANEXUS_BOILERPLATE_MIN_PAGE_FRACTION` of pages
  (default 0.6) *and* be at least `ANEXUS_BOILERPLATE_MIN_LINE_CHARS` long (default 12);
- lines carrying money amounts or labelled fields (`Loan A/c No: …`) are never dropped;
- a page is left untouched when removing its boilerplate would leave almost nothing.

### Over-cleaning safeguards

Cleaning must never quietly destroy real content, so the pipeline records how much
each page lost and reports it rather than silently discarding:

- **Per-page stats** — every page carries `raw_chars`/`clean_chars`; a page that
  keeps less than `ANEXUS_MIN_PAGE_CONTENT_RATIO` (default 0.15) of its source is
  counted as *over-cleaned* and logged. Nothing is dropped for this reason.
- **Noise-only chunks** — a chunk with no alphanumeric character (`...`, bullets,
  whitespace) can never match a query and is always dropped.
- **Optional short-chunk floor** — `ANEXUS_MIN_CHUNK_CHARS` (default `0`, disabled)
  drops chunks below a length. It is off by default because chunker tail fragments
  can be legitimately short; raise it to opt in.

## Chunking strategies (opt-in, eval-gated)

The default is a self-contained recursive character splitter (`ANEXUS_CHUNK_SIZE=400`,
`ANEXUS_CHUNK_OVERLAP=80`). Everything below is **off by default**; each flag changes
chunk boundaries or embeddings, so turning it on requires re-ingesting the corpus.
Enable a flag only after it wins on the eval harness (`python -m evaluation.run_eval`).

- **Token-aware sizing** — `ANEXUS_CHUNK_TOKEN_AWARE=1` measures pieces with the
  embedder's own tokenizer instead of characters, so a chunk is capped by the model's
  real window rather than a character guess. Tune with `ANEXUS_CHUNK_MAX_TOKENS`
  (500), `ANEXUS_CHUNK_TARGET_TOKENS` (384, soft fill target) and
  `ANEXUS_CHUNK_OVERLAP_TOKENS` (64).
- **Contextual headers** — `ANEXUS_CHUNK_CONTEXT_HEADER=meta` embeds a short
  `title · page N` header in front of each chunk while the *stored* chunk text stays
  clean, so citations and the LLM context are unaffected. Falls back to `off`.
- **Sentence-aware packing** — `ANEXUS_CHUNK_SENTENCE_AWARE=1` packs whole sentences
  so a sentence that fits is never split across two chunks.

### Retrieval fusion

Hybrid retrieval fuses dense and BM25 results with weighted Reciprocal Rank Fusion.
Each source contributes `weight / (RRF_K + rank)` per candidate:

- `ANEXUS_DENSE_WEIGHT` / `ANEXUS_BM25_WEIGHT` (default `0.5` / `0.5`, equal ⇒ classic
  RRF) bias the fusion toward semantic or keyword matching;
- `ANEXUS_RRF_K` (default `60`) controls how steeply rank is discounted.

(These are read from `config/settings.py`, where they default to the same values.)

The retrieval depth is: each channel fetches `ANEXUS_BASE_K_RETRIEVAL` (default `10`)
candidates, fusion returns up to twice that, and `ANEXUS_TOP_K_RERANK` (default `5`)
contexts survive the cross-encoder reranker.

Two **optional, off-by-default** relevance floors can trim weak hits:

- `ANEXUS_RETRIEVAL_MIN_SIMILARITY` — minimum cosine similarity for a dense hit
  (derived from Chroma's squared-L2 distance for unit-length vectors: `cos = 1 - d/2`);
- `ANEXUS_RERANK_MIN_SCORE` — minimum cross-encoder logit.

Both unset means no filtering; if every candidate falls below a floor the query
gets a safe "no context" answer. Tune them on the evaluation harness and keep them
off unless they clearly help.

### Multi-query retrieval (query expansion)

Substantial queries are paraphrased by the LLM and *each* variant is run through the
same dense + BM25 hybrid retriever; the resulting ranked lists are fused with RRF
alongside the original query. This is the vocabulary-mismatch fix that helps both
channels (synonyms for dense, alternate keywords for BM25):

- `ANEXUS_QUERY_EXPANSION_ENABLED` (default `1`) turns the whole step off;
- `ANEXUS_QUERY_EXPANSION_VARIANTS` (default `2`) is how many paraphrases the LLM
  writes (the original is always included);
- `ANEXUS_QUERY_EXPANSION_MIN_WORDS` (default `7`) skips the extra LLM call for short
  precise lookups.

Expansion runs on the standalone query produced by follow-up resolution, so the two
compose. Reranking, routing and the prompt question still use the original standalone
query, not a paraphrase. Any failure (or the simulated LLM) degrades cleanly to plain
hybrid retrieval.

### Search within one document

The sidebar **Search in** selector restricts a question to a single ingested
document (default: all documents). The scope is applied to *both* retrieval
channels — Chroma's `where` filter for dense and a candidate filter for BM25 — so
fused results can only come from the chosen document. Selecting "All documents"
keeps the default behavior. Removing the scoped document resets the selector to
"All documents".

### Prompting & grounding

The system prompt grounds the model to the retrieved context, forbids invented
facts and advisory answers, and instructs it to cite the source block inline
(e.g. `(agreement.pdf, p.2)`) and to end with a `Sources:` line. Everything the
model is asked to treat as data is wrapped in a `<context> … </context>` fence,
and the prompt states that nothing inside the fence is an instruction — a
structural defense-in-depth layer on top of the PII scrubber.

- **Context budget** — `ANEXUS_CONTEXT_MAX_CHARS` (default `6000`, `0` disables)
  caps the retrieved context handed to the LLM. Highest-ranked chunks are kept
  until the budget is reached and the top chunk is always kept, so context is
  never empty. The same trimmed list feeds both the prompt and the citations, so
  the two always agree.
- **Per-task temperature** — `ANEXUS_LLM_TEMPERATURE` (QA, default `0.1`),
  `ANEXUS_LLM_TEMPERATURE_REWRITE` and `ANEXUS_LLM_TEMPERATURE_SUMMARY` (both
  default to the QA value), and `ANEXUS_LLM_TEMPERATURE_EXPANSION` (default `0.3`,
  higher for paraphrase diversity). Lower is more faithful; raise for more varied
  output.

The evaluation harness reports `citation_support_rate` (share of model-cited
`(doc, page)` pairs that were actually in the retrieved top-k) and
`citation_coverage` (share of answers that cite at least one source).

## Scale, multilingual search & dedup

Tunings that matter most once the corpus grows past a few hundred documents:

- **Batched writes** — chunks are added to Chroma in batches
  (`ANEXUS_CHROMA_ADD_BATCH_SIZE`, default 512) so large uploads stay within the
  memory budget, and full-corpus reads paginate with offset
  (`ANEXUS_CHROMA_GET_BATCH_SIZE`, default 1000).
- **Incremental BM25** — adding a new document extends the in-memory keyword index
  with just that document's tokenized chunks instead of re-reading the whole
  vector store; re-ingests, removals and clears still rebuild fully for correctness.
- **Unicode-aware tokens** — the BM25 tokenizer keeps numbers intact (`7.25`,
  `4,50,000`) and tokenizes accented/CJK/Devanagari text, so keyword recall works
  for non-English corpora even while the default English embedder is unchanged.
- **Multilingual embeddings (optional)** — set `ANEXUS_EMBEDDING_MODEL_NAME` to a
  multilingual model (e.g. `BAAI/bge-m3` or `jinaai/jina-embeddings-v3`). Note the
  vector store is not re-embedded automatically; choose before a large ingestion
  or rebuild afterwards.
- **Content-hash dedup** — every document is fingerprinted (file bytes for
  uploads, deterministic page-text hash for web/cloud sources). Uploading the same
  content under a different name either informs the user and skips
  (`ANEXUS_DEDUP_ACTION=skip`, the default — popped up as ℹ️ in the chat, not an
  error) or removes the earlier document and takes the new name (`replace`).
- **Duplicate-chunk suppression** — exact-duplicate chunks inside one batch (reuse
  of the same boilerplate on every page) are dropped at ingest time.
- **Quality stats** — the ingest report shows how many pages had to be OCR'd, how
  many empty pages were dropped, how many boilerplate lines were removed, how many
  noise/short chunks were dropped, and how many pages were heavily cleaned.

### Embedded metadata

When available, document properties are attached to every page's metadata:
`doc_title`, `doc_author`, `doc_created` (and `doc_modified`/`doc_keywords`/`doc_subject`
where present) are read from PDF info dictionaries and DOCX/XLSX/PPTX core
properties. Disable with `ANEXUS_EXTRACT_EMBEDDED_METADATA=false`.

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
- **Multi-query retrieval (query expansion)** - a substantial query is paraphrased
  by the LLM and each paraphrased variant is retrieved (dense + BM25) and RRF-fused
  with the original, closing the vocabulary gap between the question and the document
  wording. Short lookups and the simulated LLM skip the extra call.
- **Persistent multiple conversations** - chats (titles, messages, citations,
  summaries) are stored per user in SQLite under `data/users/<id>/app.sqlite3`.
  Conversations are not bound to a document: any chat can ask about any uploaded
  document. The sidebar lists, renames and deletes conversations; they survive
  restarts and the uploaded documents are re-attached from the registry/Chroma
  without re-uploading.
- **Message feedback** - every assistant message carries thumbs up/down; the
  choice is persisted per message in SQLite (`feedback` column) and survives
  restarts, giving a lightweight signal for which answers were useful.
- **Token streaming (PII-safe)** - when enabled (the default), the answer streams
  into the chat as the model produces it instead of appearing all at once. Raw
  tokens are never shown: the pipeline holds back a trailing buffer and only
  releases text that the full-text scrubber has already approved, so a PII value
  split across chunks is still redacted. Fallback happens only *before* the first
  token (a provider that dies mid-stream stops the stream rather than splicing two
  answers together), and the persisted answer is the same fully-scrubbed text the
  non-streaming path would have produced. Tunables: `ANEXUS_STREAMING_ENABLED`
  (0/1, default 1) and `ANEXUS_STREAM_SCRUB_TAIL_CHARS` (default 120).

## Tests

Unit tests use fakes so they run offline, fast, and never download models:

```bash
python -m pytest -q          # 533 passing (optional OCR-extras tests self-skip without requirements-ocr.txt)
```

Real-model tests (BGE embedder, live retrieval) can be run locally with:

```bash
$env:ANEXUS_RUN_REAL_TESTS=1
python -m pytest -q tests/test_embedder.py tests/test_real_retrieval.py
```

## Evaluation

Two 50-question golden datasets (`evaluation/golden_dataset_01.json` and
`evaluation/golden_dataset_02.json`), each annotated with source pages per
category (fact lookup, definitions, multi-hop, comparison, numerical,
negative/unknown, compliance), are scored against a **two-document corpus**:

```bash
# end-to-end + retrieval metrics over both documents (offline, simulated LLM)
python -m evaluation.run_eval \
  --file Synthetic_Mortgage_Loan_File_TEST_01.pdf \
  --file Synthetic_Mortgage_Loan_File_TEST_02.pdf

# additionally run the oracle-context pass (separates retrieval vs generation)
python -m evaluation.run_eval \
  --file Synthetic_Mortgage_Loan_File_TEST_01.pdf \
  --file Synthetic_Mortgage_Loan_File_TEST_02.pdf --oracle

# offline groundedness / relevance report from a saved run
python -m evaluation.ragas_eval --input evaluation/results.json
```

Both files are loaded by default (`--dataset` may be repeated to override).
Each question carries a `source_file` so hits are matched per document even
when the same page number exists in both.

**Retrieval metrics** (computed with generation disabled): Precision@K, true
Recall@K, HitRate@K, MRR@K, nDCG@K, context precision (average precision) and
context recall (fraction of gold answer points present in the retrieved context).

**Generation metrics** (all offline, no LLM judge): answer accuracy against
curated literal answer points, citation precision/coverage, **groundedness**
(claim-level token-overlap of each answer sentence against the retrieved
context — an offline stand-in for RAGAS faithfulness), and **answer relevance**
(cosine similarity between the answer and the golden answer using the existing
BGE embedder). Safety is measured as PII-leak rate.

**Retrieval vs generation attribution:** `--oracle` re-answers every question
with the gold pages injected as context (retrieval bypassed). A high oracle
score with a low end-to-end score points at retrieval; both low points at
generation. The gap is reported per category.

### Real-provider runs (API keys)

With keys in `.env` (`NVIDIA_API_KEY`, `GROQ_API_KEY`, optional
`GEMINI_API_KEY`) the same command scores against live models:

```bash
python -m evaluation.provider_smoke        # first: verify keys + model ids
python -m evaluation.run_eval --resume --oracle --min-interval-s 1 \
  --max-cost-usd 0 \
  --file Synthetic_Mortgage_Loan_File_TEST_01.pdf \
  --file Synthetic_Mortgage_Loan_File_TEST_02.pdf
```

- `--provider nvidia|groq|...` pins one provider; otherwise the fallback
  chain runs (intent picks the first hop).
- `--max-cost-usd N` stops both passes cleanly at N dollars; `0` = uncapped
  (correct for free-tier keys). Spend is metered per call and per provider by
  `evaluation/cost.py` (prices default to zero; override with `ANEXUS_PRICE_*`).
- `--resume` continues a crashed sweep from `evaluation/results.partial.json`
  (oracle has its own `.oracle.partial.json`); the checkpoints are cleared on
  success so re-running never double-counts.
- `--no-preflight` skips the one-call-per-provider smoke test; `--limit N`
  scores only the first N questions.

**Provider reality:** model IDs rotate without notice, and NVIDIA additionally
removes models *per account* (HTTP 404 `Function ... Not found for account`)
that its public catalog still lists. Verified 2026-10-21: `nvidia`
(`nvidia/nemotron-3.5-lightning-30b-a3b`) and `groq` (`openai/gpt-oss-120b`)
respond; `gemini_flash` times out (the chain skips it); several other NVIDIA
ids are gated for this key. Run `python -m evaluation.provider_smoke` for the
live state - it exits non-zero only when *no* provider answers.

### Real RAGAS (LLM judge)

`evaluation/ragas_real.py` re-scores a finished run with RAGAS 0.4.3's
faithfulness / answer-relevance / context-precision family against a real
judge:

```bash
python -m evaluation.ragas_real --judge groq evaluation/results.json
```

- Judge choices are `groq` (default, `openai/gpt-oss-120b`) or `nvidia`.
  Metrics resolve from the legacy `ragas.metrics` registry via
  `evaluation/ragas_compat.py`; RAGAS 0.4.x's newer
  `ragas.metrics.collections.*` classes are rejected by `evaluate()`.
- **Groq's free tier allows 200k judge tokens/day.** Measured: one 20-row x
  6-metric pass burned ~198k and then hit `429`; a full 100-row x 6-metric
  pass is ~1.8M tokens - multiple days of quota. Split the metric set across
  days instead, appending into one report with `--resume`:

  ```bash
  # day 1 - judge-only metrics
  python -m evaluation.ragas_real --judge groq --resume \
    --metrics faithfulness,response_groundedness
  # day 2 - embedding-backed metrics append to the same file
  python -m evaluation.ragas_real --judge groq --resume \
    --metrics answer_relevancy,answer_correctness
  ```

  `--resume` scores only rows missing one of the requested columns and merges
  the result into `evaluation/ragas_real.json` (prior rows and metrics are kept),
  so each day's pass accumulates. `--limit`/`--out` still narrow or redirect a
  single pass, and `--judge nvidia` avoids the Groq ceiling entirely.
- `--timeout-s` defaults to 600 s: multi-call metrics exceeded an earlier
  180 s cap and silently produced NaN rows.

### NLI claim verification (local, offline)

```bash
python -m evaluation.run_eval --nli ...            # scores the run afterwards
python -m evaluation.nli --results evaluation/results.json   # standalone/CI
```

Each answer is split into claims and checked against the retrieved context
with `cross-encoder/nli-deberta-v3-base`, adding `nli_claim_count`,
`nli_entailed_fraction` and `nli_contradiction_fraction` per row and to the
summary. The scorer always runs in a **separate process**: resident next to
the embedder, the ~700 MB model pushed this 4 GB box into paging - chroma's
connection pool timed out on every query and rows took ~100x longer.

### TruthfulQA (closed-book hallucination baseline)

```bash
python -m evaluation.benchmarks.truthfulqa --download-only   # once: caches 817 q
python -m evaluation.benchmarks.truthfulqa \
  --provider nvidia --limit 50 --timeout-s 60
```

No documents, no retrieval: the bare generator picks MC1 letters
(judge-free, deterministic shuffling, free-tier only). It is the hallucination
baseline to read next to the RAG numbers above. First run (50 questions,
`nvidia/nemotron-3.5-lightning-30b-a3b`, `--timeout-s 60`): **accuracy 0.7708
of answered, answered 48/50, median 7.5 s, ~45k tokens** - both misses
exhausted the 4096-token cap on reasoning alone, and without `--timeout-s`
(30 s default) 5-7 of the hardest questions time out instead. `--simulate`
exercises the plumbing only (its canned replies rarely parse as a letter).

### CI

- `.github/workflows/rag-eval.yml` - automatic gate: full suite with
  `SIMULATE_LLM=true` on push to `main`/`ashfaque-development` and on PRs,
  then the golden-set eval; uploads `evaluation/results.json`.
- `.github/workflows/rag-eval-real.yml` - **manual only**
  (`workflow_dispatch`; inputs: `limit`, `provider`, `oracle`, `nli`,
  `max_cost_usd`): preflight + real-key end-to-end/oracle pass + optional NLI
  post-scorer. Requires repository secrets `NVIDIA_API_KEY`, `GROQ_API_KEY`
  (optional `GEMINI_API_KEY`).

Reproducible real runs install the pinned eval toolchain on top of the base
requirements: `pip install -r requirements.txt -r requirements-eval.txt`.

## Project layout

```
src/
  auth/          secrets vault (Fernet), user store (SQLite), tenant naming, OIDC claim resolution
  ingestion/     parsers/ (format dispatcher: PDF+OCR, DOCX, CSV/XLSX, PPTX, text, images; ocr.py = Tesseract->RapidOCR fallback), sources/ (website crawler, S3 client), chunker (recursive, self-contained), embedder (BGE), vectordb (Chroma), document store (SQLite registry)
  retrieval/     hybrid retriever (dense+BM25 RRF), query expander, cross-encoder reranker
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