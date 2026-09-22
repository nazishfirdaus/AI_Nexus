"""ANexus — Streamlit entry point.

Owns the UI only: session state, PDF upload, chat view, document status and errors.
All RAG logic happens in src.rag_pipeline via src.pipeline_factory.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import streamlit as st

from src.pipeline_factory import get_pipeline

logger = logging.getLogger(__name__)

APP_TITLE = "ANexus"
SAMPLE_PDF = Path(__file__).resolve().parent / "Synthetic_Mortgage_Loan_File_TEST.pdf"
MAX_PDF_MB = 20
MAX_PDF_PAGES = 100

DISCLAIMER = (
    "Demonstration / educational system. Not affiliated with any bank or lender. "
    "Output is not financial, legal, lending, or compliance advice."
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "messages": [],            # [{"role": "user"|"assistant", "content": str, "citations": list|None}]
        "doc_id": None,
        "doc_name": None,
        "doc_state": "empty",      # empty | processing | ready | error
        "ingest_stats": None,
        "pipeline_cache_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _add_message(role: str, content: str, citations: list | None = None) -> None:
    st.session_state["messages"].append(
        {"role": role, "content": content, "citations": citations}
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_pdf(bytes_data: bytes, filename: str) -> None:
    """Save upload to a temp file and run the pipeline ingestion on it."""
    st.session_state["doc_state"] = "processing"
    st.session_state["doc_name"] = filename
    tmp = Path(os.environ.get("TEMP", ".")) / f"anexus_upload_{int(time.time())}.pdf"
    try:
        tmp.write_bytes(bytes_data)
        bundle = get_pipeline()
        result = bundle.pipeline.ingest_document(tmp)
        st.session_state["doc_id"] = result.document_id
        st.session_state["doc_name"] = result.filename
        st.session_state["ingest_stats"] = result
        st.session_state["doc_state"] = "ready"
        # A new document replaces the conversation.
        st.session_state["messages"] = []
        _add_message(
            "assistant",
            f"Successfully ingested '{result.filename}' — {result.pages} page(s) "
            f"({result.chunks} chunks, {result.vectors_added} vectors). "
            "Ask a question below.",
        )
    except Exception as exc:  # defensively show upload/ingestion errors in the UI
        logger.exception("Ingestion failed")
        st.session_state["doc_state"] = "error"
        _add_message(
            "assistant",
            f"⚠️ Could not process the PDF: {exc}",
        )
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_landing() -> None:
    st.title(APP_TITLE)
    st.caption(DISCLAIMER)
    st.markdown(
        "Upload a mortgage **PDF** and ask questions about it. ANexus extracts the text "
        "(with OCR fallback), embeds it locally, and answers **only** from the supplied "
        "document — with page citations and PII redaction before any cloud call."
    )

    st.markdown("#### Get started")
    uploaded = st.file_uploader("Choose a mortgage document (PDF)", type=["pdf"])
    use_sample = st.button("Use the sample mortgage document", type="secondary")
    return uploaded, use_sample


def handle_source(uploaded, use_sample: bool) -> None:
    """Process the chosen source exactly once per widget interaction."""
    if uploaded is not None and st.session_state.get("doc_name") != uploaded.name:
        ingest_pdf(uploaded.getvalue(), uploaded.name)
    elif use_sample and st.session_state.get("doc_name") != SAMPLE_PDF.name:
        ingest_pdf(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)


def render_chat() -> None:
    st.markdown("#### Ask")
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            citations = msg.get("citations") or []
            if msg["role"] == "assistant" and citations:
                label = ", ".join(f"p.{c['page_number']}" for c in citations)
                with st.expander(f"Sources: {label}"):
                    for c in citations:
                        st.markdown(f"**Page {c['page_number']}** — {c['snippet']}...")
            elif msg["role"] == "assistant" and not citations:
                st.caption("No page citations (answer not tied to a retrieved page).")

    prompt = st.chat_input(
        "Ask a question about the document…",
        disabled=st.session_state["doc_state"] != "ready",
    )
    if prompt is None:
        return

    _add_message("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Searching the document…"):
        try:
            bundle = get_pipeline()
            answer = bundle.pipeline.answer_query(prompt)
        except Exception as exc:  # never let an unexpected failure kill the UI
            logger.exception("answer_query failed")
            answer = None
            error_text = f"❌ Something went wrong while answering: {exc}"

    with st.chat_message("assistant"):
        if answer is None:
            st.write(error_text)
            _add_message("assistant", error_text)
        else:
            st.write(answer.text)
            if answer.citations:
                label = ", ".join(f"p.{c.page_number}" for c in answer.citations)
                with st.expander(f"Sources: {label}"):
                    for c in answer.citations:
                        st.markdown(f"**Page {c.page_number}** — {c.snippet}...")
            else:
                st.caption("No page citations.")
            _add_message(
                "assistant",
                answer.text,
                [c.to_dict() for c in answer.citations],
            )


def render_sidebar() -> None:
    with st.sidebar:
        st.header(APP_TITLE)
        st.caption(DISCLAIMER)
        st.markdown("### Document")
        state = st.session_state["doc_state"]
        if state == "ready":
            st.success(f"Ready: **{st.session_state['doc_name']}**")
            stats = st.session_state.get("ingest_stats")
            if stats:
                st.markdown(
                    f"- {stats.pages} pages\n- {stats.chunks} chunks\n"
                    f"- {stats.vectors_added} vectors\n- {stats.duration_s:.1f}s"
                )
            if st.button("🔄 Upload a new document"):
                bundle = get_pipeline()
                bundle.pipeline.reset()
                st.session_state.update(
                    messages=[], doc_id=None, doc_name=None, doc_state="empty",
                    ingest_stats=None,
                )
                st.rerun()
        elif state == "processing":
            st.info("Processing the PDF…")
        elif state == "error":
            st.error("The uploaded PDF could not be processed.")
        else:
            st.markdown("No document uploaded yet.")

        st.markdown("### Chat")
        if st.session_state["messages"] and st.button("🧹 Clear conversation"):
            st.session_state["messages"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point (import-safe so `streamlit run app.py` and pytest both work)
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
    _init_state()
    render_sidebar()
    uploaded, use_sample = render_landing()
    handle_source(uploaded, use_sample)

    if st.session_state["doc_state"] in ("ready", "error") and st.session_state["messages"]:
        render_chat()


if __name__ == "__main__":
    main()