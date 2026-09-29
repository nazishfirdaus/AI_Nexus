# ANexus — Mortgage Document RAG Chatbot

Ask questions about uploaded mortgage PDFs. The pipeline **ingests → chunks →
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

- **Hybrid retrieval** — dense (BGE-large) + BM25 fused with Reciprocal Rank Fusion, then a local cross-encoder rereank.
- **PII security gate** — Presidio scrubber removes PERSON / AADHAAR / PAN / IBAN / SSN / phone / email / card / IP tokens before any LLM call (dates and locations are kept — loan answers depend on them).
- **Semantic routing** — the query intent picks the first provider; the rest of the chain follows the configured fallback order, so a provider outage degrades gracefully instead of failing.
- **Citations** — every answer quotes the source pages it was built from.
- **Quota / rate-limit tracking** — cooldowns stop runaway billing and abuse.
- **Simulated LLM** — `SIMULATE_LLM=true` runs the whole pipeline (including tests and CI) with zero API keys.

## Tests

Unit tests use fakes so they run offline, fast, and never download models:

```bash
python -m pytest -q          # 157 passing (default: real-model tests skipped)
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
  ingestion/     parser (PyMuPDF+OCR), chunker (recursive, self-contained), embedder (BGE), vectordb (Chroma)
  retrieval/     hybrid retriever (dense+BM25 RRF), cross-encoder reranker
  redaction/     Presidio scrubber + build_llm_inputs security gate
  routing/       semantic router + quota tracker
  generation/    LLM handler w/ fallback, aggregator (citations)
  rag_pipeline.py       end-to-end orchestrator
  pipeline_factory.py   dependency wiring (test doubles supported)
config/          prompts, settings, quota state
evaluation/      golden dataset + harness
tests/           pytest suite (fakes; real-model tests gated by ANEXUS_RUN_REAL_TESTS)
app.py           Streamlit UI
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'nltk'` / slow `import langchain_text_splitters`** — not required; the chunker is self-contained. Do not add `langchain_text_splitters` to hot import paths.
- **First query is slow** — the embedder/reranker/spaCy models load lazily once per process; subsequent turns are fast.
- **Chroma lock errors on re-ingest** — only one process should write `chroma_db/` at a time; the pipeline resets the store on each new PDF.
- **Missing spaCy model** — run `python -m spacy download en_core_web_sm`.

## License

MIT (see `LICENSE`). The bundled sample PDF and golden dataset are licensed for
evaluation/demo use only.