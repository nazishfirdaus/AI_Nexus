"""ANexus — Streamlit entry point.

Owns the UI only: session state, PDF upload, chat view, document status and errors.
All RAG logic happens in src.rag_pipeline via src.pipeline_factory; chat
persistence (conversations, messages, summaries) lives in src.memory.chat_store.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import streamlit as st

from src.memory.chat_store import ChatStore
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
        "messages_synced": 0,      # how many messages are already in the store
        "conversation_id": None,   # active SQLite conversation (None = new chat)
        "conversation_summary": None,
        "doc_id": None,
        "doc_name": None,
        "doc_state": "empty",      # empty | processing | ready | error
        "ingest_stats": None,
        "session_restored": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _add_message(role: str, content: str, citations: list | None = None) -> None:
    st.session_state["messages"].append(
        {"role": role, "content": content, "citations": citations}
    )


@st.cache_resource
def _get_store() -> ChatStore:
    """One process-wide store; connections are opened per operation."""
    return ChatStore()


@st.cache_resource
def _get_cached_pipeline():
    from src.pipeline_factory import get_pipeline

    return get_pipeline()


# ---------------------------------------------------------------------------
# Conversation lifecycle
# ---------------------------------------------------------------------------
def _sync_messages() -> None:
    """Persist session messages not yet in the store.

    The conversation row is created lazily on the first user message, titled
    from that message and bound to the active document. Assistant-only
    leftovers (welcome/error text) are not worth a conversation of their own.
    """
    conv_id = st.session_state["conversation_id"]
    synced = st.session_state["messages_synced"]
    pending = st.session_state["messages"][synced:]
    if not pending:
        return
    if conv_id is None:
        first_user = next((m for m in pending if m["role"] == "user"), None)
        if first_user is None:
            return
        store = _get_store()
        conv_id = store.create_conversation(
            title=first_user["content"][:40],
            document_id=st.session_state["doc_id"],
            document_name=st.session_state["doc_name"],
        )
        st.session_state["conversation_id"] = conv_id
    store = _get_store()
    for msg in pending:
        store.append_message(
            conv_id, msg["role"], msg["content"], msg.get("citations")
        )
    st.session_state["messages_synced"] = len(st.session_state["messages"])


def _discard_current_chat() -> None:
    st.session_state["conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["messages_synced"] = 0
    st.session_state["conversation_summary"] = None


def _start_new_chat() -> None:
    _sync_messages()
    _discard_current_chat()


def _load_conversation(conv_id: str) -> None:
    store = _get_store()
    conv = store.get_conversation(conv_id)
    if conv is None:
        return
    messages = store.get_messages(conv_id)
    st.session_state["conversation_id"] = conv_id
    st.session_state["messages"] = messages
    st.session_state["messages_synced"] = len(messages)
    st.session_state["conversation_summary"] = conv.get("summary") or None


def _maybe_update_summary() -> None:
    """Refresh the rolling summary once the conversation passes the trigger."""
    conv_id = st.session_state["conversation_id"]
    if not conv_id:
        return
    try:
        bundle = _get_cached_pipeline()
        new_summary = bundle.pipeline.summarize_history(
            st.session_state["messages"],
            st.session_state["conversation_summary"],
        )
    except Exception:
        logger.exception("Conversation summarization failed")
        return
    if new_summary:
        st.session_state["conversation_summary"] = new_summary
        _get_store().set_summary(conv_id, new_summary)


def _restore_document_state() -> None:
    """Re-attach the active document after a restart (Chroma persists on disk)."""
    from config import settings

    if st.session_state["doc_state"] != "empty":
        return
    if not Path(settings.CHROMA_PERSIST_DIRECTORY).exists():
        return
    try:
        pipeline = _get_cached_pipeline().pipeline
        if pipeline.restore_active_document():
            st.session_state["doc_id"] = pipeline.active_document_id
            st.session_state["doc_name"] = pipeline.active_filename
            st.session_state["doc_state"] = "ready"
    except Exception:
        logger.exception("Could not restore the previously ingested document")


def _restore_session() -> None:
    """Once per browser session: restore the document and the latest chat."""
    if st.session_state["session_restored"]:
        return
    st.session_state["session_restored"] = True
    _restore_document_state()
    try:
        conversations = _get_store().list_conversations(limit=1)
    except Exception:
        logger.exception("Could not load saved conversations")
        return
    if conversations:
        _load_conversation(conversations[0]["id"])


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
        # A new document starts a fresh conversation; older chats stay saved.
        _start_new_chat()
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
        _sync_messages()


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


def _render_context_banner() -> None:
    """Explain when the open chat and the active document do not line up."""
    conv_id = st.session_state["conversation_id"]
    doc_state = st.session_state["doc_state"]
    active_name = st.session_state.get("doc_name")
    if conv_id:
        conv = _get_store().get_conversation(conv_id)
        conv_doc = (conv or {}).get("document_name")
        if conv_doc and doc_state == "ready" and active_name and conv_doc != active_name:
            st.info(
                f"This conversation is about **{conv_doc}**; the active document is "
                f"**{active_name}**. Upload **{conv_doc}** to ask new questions about it."
            )
            return
    if conv_id and doc_state != "ready":
        st.info("Upload a document to ask new questions in this conversation.")


def render_chat() -> None:
    st.markdown("#### Ask")
    _render_context_banner()
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
            bundle = _get_cached_pipeline()
            answer = bundle.pipeline.answer_query(
                prompt,
                chat_history=list(st.session_state["messages"][:-1]),
                conversation_summary=st.session_state["conversation_summary"],
            )
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
    _sync_messages()
    _maybe_update_summary()


def _conversation_label(conv: dict) -> str:
    title = (conv.get("title") or "Untitled").strip() or "Untitled"
    count = conv.get("message_count", 0)
    stamp = (conv.get("updated_at") or "")[:16].replace("T", " ")
    return f"{title} · {count} msgs · {stamp}"


def _render_document_section() -> None:
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
            bundle = _get_cached_pipeline()
            bundle.pipeline.reset()
            st.session_state.update(
                doc_id=None, doc_name=None, doc_state="empty", ingest_stats=None,
            )
            _start_new_chat()
            st.rerun()
    elif state == "processing":
        st.info("Processing the PDF…")
    elif state == "error":
        st.error("The uploaded PDF could not be processed.")
    else:
        st.markdown("No document uploaded yet.")


def _render_chat_section() -> None:
    st.markdown("### Chats")
    store = _get_store()

    if st.button("➕ New chat"):
        _start_new_chat()
        st.rerun()

    conversations = store.list_conversations()
    if not conversations:
        st.caption("No saved conversations yet.")
        return

    options = [c["id"] for c in conversations]
    labels = {c["id"]: _conversation_label(c) for c in conversations}
    current = st.session_state["conversation_id"]
    index = options.index(current) if current in options else None
    chosen = st.selectbox(
        "Conversation",
        options,
        index=index,
        format_func=lambda conv_id: labels.get(conv_id, conv_id),
        placeholder="Select a conversation",
        label_visibility="collapsed",
    )
    if chosen and chosen != current:
        _sync_messages()
        _load_conversation(chosen)
        st.rerun()

    with st.expander("Manage this chat"):
        current_conv = store.get_conversation(current) if current else None
        new_title = st.text_input(
            "Title",
            value=(current_conv or {}).get("title") or "",
            key=f"title_{current}",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Rename", disabled=not current):
                if new_title.strip():
                    store.rename_conversation(current, new_title.strip())
                    # keep the widget in sync: `value` is ignored once the key exists
                    st.session_state[f"title_{current}"] = new_title.strip()
                    st.rerun()
        with col_b:
            if st.button("🗑 Delete", disabled=not current):
                _sync_messages()
                to_delete = st.session_state["conversation_id"]
                _discard_current_chat()
                if to_delete:
                    store.delete_conversation(to_delete)
                st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.header(APP_TITLE)
        st.caption(DISCLAIMER)
        _render_document_section()
        _render_chat_section()


# ---------------------------------------------------------------------------
# Entry point (import-safe so `streamlit run app.py` and pytest both work)
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
    _init_state()
    _restore_session()
    render_sidebar()
    uploaded, use_sample = render_landing()
    handle_source(uploaded, use_sample)

    if st.session_state["doc_state"] == "ready" or st.session_state["messages"]:
        render_chat()


if __name__ == "__main__":
    main()
