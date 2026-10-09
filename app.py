"""ANexus — Streamlit entry point.

Owns the UI only: session state, document upload, chat view, document status and errors.
All RAG logic happens in src.rag_pipeline via src.pipeline_factory; chat
persistence (conversations, messages, summaries) lives in src.memory.chat_store.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import streamlit as st

from config.settings import MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS
from src.memory.chat_store import ChatStore

logger = logging.getLogger(__name__)

APP_TITLE = "ANexus"
SAMPLE_PDF = Path(__file__).resolve().parent / "Synthetic_Mortgage_Loan_File_TEST.pdf"

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
        "documents": [],           # registry records: [{id, filename, pages, chunks, ...}]
        "doc_state": "empty",      # empty | processing | ready | error
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
    from that message. Conversations are not bound to a document: any chat can
    ask about any uploaded document. Assistant-only leftovers (welcome/error
    text) are not worth a conversation of their own.
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
        conv_id = store.create_conversation(title=first_user["content"][:40])
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


def _restore_documents_state() -> None:
    """Re-attach all stored documents after a restart (Chroma persists on disk)."""
    from config import settings

    if st.session_state["doc_state"] != "empty":
        return
    if not Path(settings.CHROMA_PERSIST_DIRECTORY).exists():
        return
    try:
        pipeline = _get_cached_pipeline().pipeline
        documents = pipeline.restore_documents()
        if documents:
            st.session_state["documents"] = documents
            st.session_state["doc_state"] = "ready"
    except Exception:
        logger.exception("Could not restore the previously ingested documents")


def _restore_session() -> None:
    """Once per browser session: restore the documents and the latest chat."""
    if st.session_state["session_restored"]:
        return
    st.session_state["session_restored"] = True
    _restore_documents_state()
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
def _refresh_documents() -> list[dict]:
    """Reload the document registry snapshot from the pipeline."""
    documents = _get_cached_pipeline().pipeline.list_documents()
    st.session_state["documents"] = documents
    st.session_state["doc_state"] = "ready" if documents else "empty"
    return documents


def _already_ingested(filename: str) -> bool:
    return any(d["filename"] == filename for d in st.session_state["documents"])


def ingest_documents(items: list[tuple[bytes, str]]) -> None:
    """Save each upload to a temp file and append it to the corpus.

    `items` is [(bytes, original_filename), ...]. Files already in the registry
    are skipped by the caller, so everything here is new. One failing document
    does not abort the rest of the batch. The temp file keeps the original
    extension so the parser can dispatch on it.
    """
    if not items:
        return
    st.session_state["doc_state"] = "processing"
    bundle = _get_cached_pipeline()
    results, errors = [], []
    for bytes_data, filename in items:
        suffix = Path(filename).suffix.lower()
        tmp = Path(os.environ.get("TEMP", ".")) / f"anexus_upload_{int(time.time())}{suffix}"
        try:
            tmp.write_bytes(bytes_data)
            result = bundle.pipeline.ingest_document(tmp, source_name=filename)
            results.append(result)
            if len(results) == 1:
                # A fresh batch starts a fresh conversation; older chats stay saved.
                _start_new_chat()
        except Exception as exc:  # defensively show per-file errors, keep going
            logger.exception("Ingestion failed for %s", filename)
            errors.append((filename, exc))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    _refresh_documents()
    if results:
        names = ", ".join(f"'{r.filename}'" for r in results)
        total_pages = sum(r.pages for r in results)
        total_chunks = sum(r.chunks for r in results)
        total_vectors = sum(r.vectors_added for r in results)
        _add_message(
            "assistant",
            f"Successfully ingested {len(results)} document(s): {names} — "
            f"{total_pages} page(s), {total_chunks} chunks, {total_vectors} vectors "
            "total. You can now ask questions across all uploaded documents.",
        )
    if errors and not results:
        st.session_state["doc_state"] = "error"
    for filename, exc in errors:
        _add_message("assistant", f"⚠️ Could not process '{filename}': {exc}")
    _sync_messages()
    if results:
        # Re-render so the sidebar document list reflects the new corpus.
        st.rerun()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_landing() -> None:
    st.title(APP_TITLE)
    st.caption(DISCLAIMER)
    st.markdown(
        "Upload one or more mortgage documents — **PDF, Word, Excel, CSV, "
        "PowerPoint, text/Markdown** or **images** (scanned via OCR) — and ask "
        "questions across all of them. ANexus extracts the text (with OCR "
        "fallback for PDFs and images), embeds it locally, and answers "
        "**only** from the supplied documents — with per-document page citations and "
        "PII redaction before any cloud call."
    )

    st.markdown("#### Get started")
    uploaded = st.file_uploader(
        "Choose mortgage documents",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
    use_sample = st.button("Use the sample mortgage document", type="secondary")
    return uploaded, use_sample


def handle_source(uploaded, use_sample: bool) -> None:
    """Process new sources exactly once per widget interaction.

    The uploader re-emits every selected file on each rerun, so the registry
    (not the widget value) decides what counts as new. Oversized files are
    skipped with a visible warning.
    """
    items: list[tuple[bytes, str]] = []
    for file in uploaded or []:
        data = file.getvalue()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(
                f"'{file.name}' is larger than {MAX_UPLOAD_MB} MB and was skipped."
            )
            continue
        if Path(file.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            st.warning(f"'{file.name}' has an unsupported format and was skipped.")
            continue
        if _already_ingested(file.name):
            continue
        items.append((data, file.name))
    if use_sample and not _already_ingested(SAMPLE_PDF.name):
        items.append((SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name))
    if items:
        ingest_documents(items)


def _render_context_banner() -> None:
    """Hint when questions cannot be asked yet (no documents uploaded)."""
    if st.session_state["doc_state"] != "ready":
        st.info("Upload one or more documents to ask questions.")


def _render_citations(citations: list[dict]) -> None:
    """Render stored citation dicts with their source document and page."""
    labels = ", ".join(
        f"{c.get('filename') or 'document'} p.{c.get('page_number')}"
        for c in citations
    )
    with st.expander(f"Sources: {labels}"):
        for c in citations:
            name = c.get("filename") or c.get("document_id") or "Document"
            st.markdown(f"**{name} — Page {c.get('page_number')}** — {c.get('snippet')}...")


def render_chat() -> None:
    st.markdown("#### Ask")
    _render_context_banner()
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            citations = msg.get("citations") or []
            if msg["role"] == "assistant" and citations:
                _render_citations(citations)
            elif msg["role"] == "assistant" and not citations:
                st.caption("No page citations (answer not tied to a retrieved page).")

    prompt = st.chat_input(
        "Ask a question about the documents…",
        disabled=st.session_state["doc_state"] != "ready",
    )
    if prompt is None:
        return

    _add_message("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Searching the documents…"):
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
            citations = [c.to_dict() for c in answer.citations]
            if citations:
                _render_citations(citations)
            else:
                st.caption("No page citations.")
            _add_message("assistant", answer.text, citations)
    _sync_messages()
    _maybe_update_summary()


def _conversation_label(conv: dict) -> str:
    title = (conv.get("title") or "Untitled").strip() or "Untitled"
    count = conv.get("message_count", 0)
    stamp = (conv.get("updated_at") or "")[:16].replace("T", " ")
    return f"{title} · {count} msgs · {stamp}"


def _remove_document(document_id: str) -> None:
    """Delete one document from the corpus; the others stay searchable."""
    try:
        bundle = _get_cached_pipeline()
        bundle.pipeline.remove_document(document_id)
        _refresh_documents()
    except Exception:
        logger.exception("Could not remove document %s", document_id)
        st.error("Could not remove that document.")
    st.rerun()


def _clear_documents() -> None:
    """Drop every uploaded document and start over."""
    try:
        bundle = _get_cached_pipeline()
        bundle.pipeline.reset()
        _refresh_documents()
    except Exception:
        logger.exception("Could not clear documents")
        st.error("Could not clear the documents.")
    _start_new_chat()
    st.rerun()


def _render_document_section() -> None:
    st.markdown("### Documents")
    state = st.session_state["doc_state"]
    documents = st.session_state["documents"]

    if state == "processing":
        st.info("Processing the documents…")
    if state == "error" and not documents:
        st.error("The uploaded documents could not be processed.")
    if not documents:
        if state != "processing":
            st.markdown("No documents uploaded yet.")
        return

    st.success(f"**{len(documents)}** document(s) ready — searched together")
    for doc in documents:
        name_col, action_col = st.columns([4, 1])
        with name_col:
            st.markdown(f"**{doc['filename']}**")
            st.caption(
                f"{doc['pages']} pages · {doc['chunks']} chunks · "
                f"{doc['vectors']} vectors"
            )
        with action_col:
            if st.button("🗑", key=f"remove_doc_{doc['id']}",
                         help=f"Remove {doc['filename']}"):
                _remove_document(doc["id"])

    if st.button("🧹 Clear all documents"):
        _clear_documents()


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
