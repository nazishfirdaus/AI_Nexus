"""ANexus — Streamlit entry point.

Owns the UI only: session state, document upload, chat view, document status and errors.
All RAG logic happens in src.rag_pipeline via src.pipeline_factory; chat
persistence (conversations, messages, summaries) lives in src.memory.chat_store.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from config import settings
from config.settings import MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS
from src.auth.identity import resolve_user
from src.auth.secrets_vault import SecretsVault, merge_nonempty
from src.auth.tenancy import user_db_path
from src.auth.user_store import UserStore
from src.memory.chat_store import ChatStore
from src.pipeline_factory import PipelineBundle, build_user_pipeline
from src.rag_pipeline import DuplicateDocumentError

logger = logging.getLogger(__name__)

APP_TITLE = "ANexus"
SAMPLE_PDF = Path(__file__).resolve().parent / "Synthetic_Mortgage_Loan_File_TEST.pdf"

DISCLAIMER = (
    "Demonstration / educational system. Not affiliated with any bank or lender. "
    "Output is not financial, legal, lending, or compliance advice."
)

# Credential form fields: (stored key, label, is_secret).
_LLM_KEY_FIELDS = (
    ("groq_api_key", "Groq API key", True),
    ("gemini_api_key", "Google Gemini API key", True),
    ("nvidia_api_key", "NVIDIA API key", True),
)
_S3_FIELDS = (
    ("s3_access_key_id", "S3 access key ID", False),
    ("s3_secret_access_key", "S3 secret access key", True),
    ("s3_region", "S3 region (e.g. us-east-1)", False),
    ("s3_endpoint_url", "S3 endpoint URL (for S3-compatible stores; blank = AWS)", False),
    ("s3_bucket", "Default bucket", False),
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


# ---------------------------------------------------------------------------
# Identity / per-user context
# ---------------------------------------------------------------------------
# Session-state keys that belong to one user; dropped when the identity changes.
_USER_SCOPED_KEYS = (
    "messages",
    "messages_synced",
    "conversation_id",
    "conversation_summary",
    "documents",
    "doc_state",
    "session_restored",
    "user_ctx",
)


@dataclass
class UserContext:
    """Everything bound to one logged-in user for this browser session."""

    user: dict
    bundle: PipelineBundle
    chat_store: ChatStore


@st.cache_resource
def _get_user_store() -> UserStore:
    """Process-wide account registry; connections open per operation."""
    return UserStore()


def _dev_user() -> dict:
    """Local-development identity (ANEXUS_AUTH_DISABLED), auto-provisioned user."""
    store = _get_user_store()
    email = settings.DEV_USER_EMAIL
    return store.get_by_email(email) or store.provision(
        email, display_name="Local dev"
    )


def _current_user() -> Optional[dict]:
    """Resolve the current identity, or None if not logged in / not allowed."""
    if settings.AUTH_DISABLED:
        return _dev_user()
    try:
        st_user = st.user
    except Exception:
        return None
    return resolve_user(_get_user_store(), st_user)


def _ensure_user_state(user: dict) -> None:
    """Start a clean slate whenever the logged-in identity changes."""
    if st.session_state.get("user_id") != user["id"]:
        for key in _USER_SCOPED_KEYS:
            st.session_state.pop(key, None)
        st.session_state["user_id"] = user["id"]


def _load_credentials(user: dict) -> dict:
    """Decrypt the user's stored provider keys; empty when none are saved."""
    ciphertext = _get_user_store().get_secrets(user["id"])
    if not ciphertext:
        return {}
    try:
        return SecretsVault().decrypt(ciphertext)
    except Exception:
        logger.exception("Could not load stored credentials for %s", user["email"])
        return {}


def _get_user_context(user: dict) -> UserContext:
    """Build (once per session) the isolated pipeline + chat store for `user`."""
    ctx = st.session_state.get("user_ctx")
    if ctx is not None and ctx.user["id"] == user["id"]:
        return ctx
    bundle = build_user_pipeline(user["id"], credentials=_load_credentials(user))
    ctx = UserContext(
        user=user,
        bundle=bundle,
        chat_store=ChatStore(db_path=user_db_path(user["id"])),
    )
    st.session_state["user_ctx"] = ctx
    return ctx


def _chat_store() -> ChatStore:
    return st.session_state["user_ctx"].chat_store


def _bundle() -> PipelineBundle:
    return st.session_state["user_ctx"].bundle


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
        store = _chat_store()
        conv_id = store.create_conversation(title=first_user["content"][:40])
        st.session_state["conversation_id"] = conv_id
    store = _chat_store()
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
    store = _chat_store()
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
        bundle = _bundle()
        new_summary = bundle.pipeline.summarize_history(
            st.session_state["messages"],
            st.session_state["conversation_summary"],
        )
    except Exception:
        logger.exception("Conversation summarization failed")
        return
    if new_summary:
        st.session_state["conversation_summary"] = new_summary
        _chat_store().set_summary(conv_id, new_summary)


def _restore_documents_state() -> None:
    """Re-attach all stored documents after a restart (Chroma persists on disk)."""
    from config import settings

    if st.session_state["doc_state"] != "empty":
        return
    if not Path(settings.CHROMA_PERSIST_DIRECTORY).exists():
        return
    try:
        pipeline = _bundle().pipeline
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
        conversations = _chat_store().list_conversations(limit=1)
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
    documents = _bundle().pipeline.list_documents()
    st.session_state["documents"] = documents
    st.session_state["doc_state"] = "ready" if documents else "empty"
    return documents


def _already_ingested(filename: str) -> bool:
    return any(d["filename"] == filename for d in st.session_state["documents"])


def _current_credentials() -> dict:
    ctx = st.session_state.get("user_ctx")
    return _load_credentials(ctx.user) if ctx else {}


def handle_cloud_import(bucket: str, prefix: str) -> None:
    """List supported objects under bucket/prefix and ingest the new ones."""
    from src.ingestion.sources.cloud import (
        CloudError,
        build_s3_client,
        collect_supported_objects,
    )

    creds = _current_credentials()
    if not (creds.get("s3_access_key_id") and creds.get("s3_secret_access_key")):
        st.warning("Add your S3 credentials on the Settings page first.")
        return
    if not bucket.strip():
        st.warning("Enter a bucket name.")
        return

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    try:
        client = build_s3_client(creds)
        with st.spinner("Listing and downloading from cloud storage…"):
            items = collect_supported_objects(
                client, bucket.strip(), prefix.strip(), max_bytes=max_bytes
            )
    except CloudError as exc:
        st.error(f"Cloud import failed: {exc}")
        return
    except Exception as exc:
        logger.exception("Cloud import failed")
        st.error(f"Cloud import failed: {exc}")
        return

    items = [(data, name) for data, name in items if not _already_ingested(name)]
    if not items:
        st.info("No new supported files found to import.")
        return
    ingest_documents(items)


def _document_exists(document_id: str) -> bool:
    return any(d["id"] == document_id for d in st.session_state["documents"])


def handle_web_import(url: str, max_pages: int) -> None:
    """Crawl a website (same domain) and ingest it as a single document."""
    from src.ingestion.sources.web import WebSourceError, crawl_documents

    if not url.strip():
        st.warning("Enter a website URL.")
        return
    try:
        with st.spinner("Crawling the website…"):
            site = crawl_documents(url.strip(), max_pages=int(max_pages))
    except WebSourceError as exc:
        st.error(f"Website import failed: {exc}")
        return
    except Exception as exc:
        logger.exception("Website import failed")
        st.error(f"Website import failed: {exc}")
        return

    if not site.documents:
        st.info("No readable pages were found at that URL.")
        return
    if _document_exists(site.document_id):
        st.info(f"'{site.title}' is already imported.")
        return

    bundle = _bundle()
    st.session_state["doc_state"] = "processing"
    try:
        result = bundle.pipeline.ingest_parsed_documents(
            site.documents, filename=site.title, document_id=site.document_id
        )
    except DuplicateDocumentError as exc:
        st.session_state["doc_state"] = "ready"
        st.info(str(exc))
        return
    except Exception as exc:
        logger.exception("Website ingestion failed")
        st.session_state["doc_state"] = "error"
        st.error(f"Could not process '{site.title}': {exc}")
        return

    _refresh_documents()
    _start_new_chat()
    _add_message(
        "assistant",
        f"Successfully imported '{result.filename}' — {result.pages} page(s), "
        f"{result.chunks} chunks, {result.vectors_added} vectors crawled from "
        f"{url.strip()}.",
    )
    _sync_messages()
    st.rerun()


def ingest_documents(items: list[tuple[bytes, str]]) -> None:
    """Save each upload to a temp file and append it to the corpus.

    `items` is [(bytes, original_filename), ...]. Files already in the registry
    (by name or by content hash) are skipped by the caller or reported here, so
    everything else is new. One failing document does not abort the rest of the
    batch. The temp file keeps the original extension so the parser can dispatch
    on it.
    """
    if not items:
        return
    st.session_state["doc_state"] = "processing"
    bundle = _bundle()
    results, errors, duplicates = [], [], []
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
        except DuplicateDocumentError as exc:
            duplicates.append(str(exc))
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
        total_ocr = sum(r.pages_ocr for r in results)
        total_boilerplate = sum(r.boilerplate_lines_removed for r in results)
        total_short = sum(r.chunks_dropped_short for r in results)
        total_overcleaned = sum(r.pages_overcleaned for r in results)
        details = []
        if total_ocr:
            details.append(f"{total_ocr} OCR page(s)")
        if total_boilerplate:
            details.append(f"{total_boilerplate} boilerplate line(s) removed")
        if total_short:
            details.append(f"{total_short} short chunk(s) dropped")
        if total_overcleaned:
            details.append(f"{total_overcleaned} page(s) heavily cleaned")
        suffix = f" ({', '.join(details)})" if details else ""
        _add_message(
            "assistant",
            f"Successfully ingested {len(results)} document(s): {names} — "
            f"{total_pages} page(s), {total_chunks} chunks, {total_vectors} vectors{suffix}. "
            "You can now ask questions across all uploaded documents.",
        )
    if errors and not results:
        st.session_state["doc_state"] = "error"
    for filename, exc in errors:
        _add_message("assistant", f"⚠️ Could not process '{filename}': {exc}")
    for message in duplicates:
        _add_message("assistant", f"ℹ️ {message}")
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

    creds = _current_credentials()
    has_s3 = bool(creds.get("s3_access_key_id") and creds.get("s3_secret_access_key"))
    with st.expander("Import from cloud storage (S3-compatible)"):
        if not has_s3:
            st.caption("Add your S3 credentials on the **Settings** page to enable this.")
        bucket = st.text_input("Bucket", value=creds.get("s3_bucket", ""), key="cloud_bucket")
        prefix = st.text_input("Prefix / folder (optional)", value="", key="cloud_prefix")
        if st.button("Import supported files", disabled=not has_s3):
            handle_cloud_import(bucket, prefix)

    with st.expander("Import from a website"):
        url = st.text_input(
            "Start URL",
            value="",
            key="web_url",
            placeholder="https://example.com/loan-terms",
        )
        max_pages = st.number_input(
            "Max pages to crawl",
            min_value=1,
            max_value=settings.WEB_MAX_PAGES,
            value=min(5, settings.WEB_MAX_PAGES),
            key="web_pages",
        )
        if st.button("Crawl and import"):
            handle_web_import(url, int(max_pages))

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
            bundle = _bundle()
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
        bundle = _bundle()
        bundle.pipeline.remove_document(document_id)
        _refresh_documents()
    except Exception:
        logger.exception("Could not remove document %s", document_id)
        st.error("Could not remove that document.")
    st.rerun()


def _clear_documents() -> None:
    """Drop every uploaded document and start over."""
    try:
        bundle = _bundle()
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
    store = _chat_store()

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
        ctx = st.session_state.get("user_ctx")
        if ctx is not None:
            user = ctx.user
            st.markdown(f"**{user.get('display_name') or user.get('email')}**")
            st.caption(user["email"])
            st.radio(
                "Navigation",
                ["Chat", "Settings"],
                key="nav",
                label_visibility="collapsed",
            )
            if not settings.AUTH_DISABLED and st.button("Sign out", use_container_width=True):
                st.logout()
            st.divider()
        _render_document_section()
        _render_chat_section()


def render_login() -> None:
    """Login screen shown before any user data is touched."""
    st.title(APP_TITLE)
    st.caption(DISCLAIMER)
    try:
        logged_in = bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        st.error(
            "Authentication is not configured. Add an `[auth]` block to "
            "`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`)."
        )
        return
    if logged_in:
        st.error("Your account is disabled. Contact the operator.")
        if st.button("Sign out"):
            st.logout()
        return
    st.markdown("Sign in to upload documents and ask questions.")
    if st.button(f"Sign in with {settings.AUTH_PROVIDER_NAME}", type="primary"):
        st.login()


# ---------------------------------------------------------------------------
# Settings (BYOK)
# ---------------------------------------------------------------------------
def _rebuild_context() -> None:
    """Force the per-user pipeline to rebuild on the next rerun (after key changes)."""
    st.session_state.pop("user_ctx", None)


def render_settings(user: dict) -> None:
    st.title("Settings")
    st.caption(
        "Your API keys and cloud credentials are encrypted and stored on the server "
        "for your account only. They are never shown again after saving."
    )
    existing = _load_credentials(user)
    vault_ready = bool(settings.SECRET_KEY)
    if not vault_ready:
        st.warning(
            "Saving credentials is disabled until `ANEXUS_SECRET_KEY` is set in the "
            "environment or secrets."
        )

    with st.form("settings_form"):
        st.subheader("Language model API keys")
        st.caption("Leave a field blank to keep its current value.")
        entered: dict[str, str] = {}
        for field, label, secret in _LLM_KEY_FIELDS:
            entered[field] = st.text_input(
                label,
                value="",
                type="password" if secret else "default",
                placeholder="set" if existing.get(field) else "",
                key=f"cred_{field}",
            )

        st.subheader("Cloud storage (S3-compatible)")
        for field, label, secret in _S3_FIELDS:
            entered[field] = st.text_input(
                label,
                value="",
                type="password" if secret else "default",
                placeholder="set" if existing.get(field) else "",
                key=f"cred_{field}",
            )

        col_save, col_clear = st.columns(2)
        save = col_save.form_submit_button(
            "Save credentials", type="primary", disabled=not vault_ready
        )
        clear = col_clear.form_submit_button("Clear all credentials", disabled=not existing)

    if save:
        merged = merge_nonempty(existing, entered)
        try:
            _get_user_store().set_secrets(user["id"], SecretsVault().encrypt(merged))
        except Exception as exc:
            st.error(f"Could not save credentials: {exc}")
        else:
            _rebuild_context()
            st.success("Saved. Your credentials are now in use.")
            st.rerun()

    if clear:
        _get_user_store().clear_secrets(user["id"])
        _rebuild_context()
        st.success("Cleared all stored credentials.")
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point (import-safe so `streamlit run app.py` and pytest both work)
# ---------------------------------------------------------------------------
def _apply_streamlit_secrets() -> None:
    """Bridge a few Streamlit secrets into the environment / settings module.

    `config.settings` reads plain environment variables, and Streamlit secrets
    live in `st.secrets`, so copy the ones the app needs across at startup.
    """
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("ANEXUS_SECRET_KEY",):
        try:
            if key in secrets and not os.environ.get(key):
                os.environ[key] = str(secrets[key])
        except Exception:
            continue
    settings.SECRET_KEY = os.environ.get("ANEXUS_SECRET_KEY", "").strip()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
    _apply_streamlit_secrets()
    user = _current_user()
    if user is None:
        render_login()
        return
    _ensure_user_state(user)
    _init_state()
    _get_user_context(user)
    _restore_session()
    render_sidebar()

    page = st.session_state.get("nav", "Chat")
    if page == "Settings":
        render_settings(user)
        return

    uploaded, use_sample = render_landing()
    handle_source(uploaded, use_sample)
    if st.session_state["doc_state"] == "ready" or st.session_state["messages"]:
        render_chat()


if __name__ == "__main__":
    main()
