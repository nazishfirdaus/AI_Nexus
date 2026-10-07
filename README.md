# ANexus — Mortgage Document RAG Chatbot

Ask questions about one or more uploaded mortgage PDFs. The pipeline **ingests → chunks →
embeds → hybrid-retrieves → reranks → scrubs PII → routes → generates with citations**,
keeping raw personally-identifying information (PII) out of every language-model call.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm # needed by the Presidio PII scrubber

streamlit run app.py                    # open the browser tab shown in the terminal
```

To answer with real LLMs, export `NVIDIA_API_KEY`, `GEMINI_API_KEY`, or
`GROQ_API_KEY` (or set them in a `.env`, see `.env.example`).

> **Python version:** developed on **Python 3.13** (project venv). Streamlit Cloud
> targets **Python 3.12**; the code is 3.10+-compatible and does not pin to 3.13-only
> syntax, so both work. Runtime overhead from model loading is expected on low-RAM
> machines (target: 4 GB RAM):

| Startup cost           | Approx. time |
|------------------------|--------------|
| Presidio + spaCy       | ~35 s (once)  |
| RAG pipeline cold import | ~2 min (once) |

Model downloads (BGE embedder ~1.3 GB, MiniLM reranker ~90 MB) happen on first use.

## Features

- **Multiple documents** — upload several PDFs at once; they form a single corpus
  that every question is searched against. The sidebar lists each document with its
  page/chunk/vector stats, lets you remove individual documents (the rest stay
  searchable) or clear everything, and the whole set persists across restarts —
  re-ingesting the same filename replaces that document instead of duplicating it.
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
  summaries) are stored in SQLite under `chat_db/`. Conversations are not bound
  to a document: any chat can ask about any uploaded document. The sidebar
  lists, renames and deletes conversations; they survive restarts and the
  uploaded documents are re-attached from the registry/Chroma without
  re-uploading.

## Tests

Unit tests use fakes so they run offline, fast, and never download models:

```bash
python -m pytest -q          # 275 passing (default: real-model tests skipped)
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
python -m evaluation.run_eval --pdf Synthetic_Mortgage_Loan_File_TEST.pdf
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
  ingestion/     parser (PyMuPDF+OCR), chunker (recursive, self-contained), embedder (BGE), vectordb (Chroma), document store (SQLite registry)
  retrieval/     hybrid retriever (dense+BM25 RRF), cross-encoder reranker
  redaction/     Presidio scrubber + build_llm_inputs security gate
  routing/       semantic router + quota tracker
  generation/    LLM handler w/ fallback, aggregator (citations)
  memory/        SQLite chat store, history windows, follow-up rewriter
  rag_pipeline.py       end-to-end orchestrator
  pipeline_factory.py   dependency wiring (test doubles supported)
config/          prompts, settings, quota state
chat_db/         SQLite conversations + messages + document registry (created at runtime, gitignored)
evaluation/      golden dataset + harness
tests/           pytest suite (fakes; real-model tests gated by ANEXUS_RUN_REAL_TESTS)
app.py           Streamlit UI
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'nltk'` / slow `import langchain_text_splitters`** — not required; the chunker is self-contained. Do not add `langchain_text_splitters` to hot import paths.
- **First query is slow** — the embedder/reranker/spaCy models load lazily once per process; subsequent turns are fast.
- **Chroma lock errors on re-ingest** — only one process should write `chroma_db/` at a time; re-ingesting a filename replaces that document's chunks, and the sidebar's "Clear all documents" resets the store.
- **Missing spaCy model** — run `python -m spacy download en_core_web_sm`.

## License

MIT (see `LICENSE`). The bundled sample PDF and golden dataset are licensed for
evaluation/demo use only.