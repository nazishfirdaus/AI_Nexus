import { useEffect, useRef, useState } from "react"
import { api, type Conversation } from "../utils/api"
import { Icon } from "./Icon"

interface Props {
  conversations: Conversation[]
  activeId: string | null
  onSelect: (id: string) => void
  onCreated: (conversation: Conversation) => void
  onDeleted: (id: string) => void
  onRenamed: (conversation: Conversation) => void
  onError: (message: string) => void
}

export function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onCreated,
  onDeleted,
  onRenamed,
  onError,
}: Props) {
  const [query, setQuery] = useState("")
  const [renaming, setRenaming] = useState<string | null>(null)
  const [draftTitle, setDraftTitle] = useState("")
  const [busy, setBusy] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (renaming) inputRef.current?.select()
  }, [renaming])

  const visible = conversations.filter((c) =>
    (c.title || "").toLowerCase().includes(query.trim().toLowerCase())
  )

  async function create() {
    if (busy) return
    setBusy(true)
    try {
      const created = await api.createConversation({})
      onCreated(created)
      onSelect(created.id)
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not create conversation.")
    } finally {
      setBusy(false)
    }
  }

  async function commitRename(id: string) {
    const title = draftTitle.trim()
    setRenaming(null)
    if (!title) return
    try {
      const updated = await api.renameConversation(id, title)
      onRenamed(updated)
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not rename conversation.")
    }
  }

  async function remove(id: string) {
    if (!window.confirm("Delete this conversation and its messages?")) return
    try {
      await api.deleteConversation(id)
      onDeleted(id)
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not delete conversation.")
    }
  }

  return (
    <aside className="rail" aria-label="Workspace">
      <div className="rail-top">
        <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
          <span className="brand-mark" aria-hidden="true">
            A
          </span>
          <span className="brand-words">
            <span className="brand-name">ANexus</span>
            <span className="brand-sub">Mortgage intelligence</span>
          </span>
        </a>
      </div>

      <button className="btn btn--new" type="button" onClick={create} disabled={busy}>
        <Icon name="plus" />
        <span>{busy ? "Creating…" : "New conversation"}</span>
      </button>

      <div className="rail-search">
        <Icon name="search" size={14} />
        <input
          type="search"
          value={query}
          placeholder="Search conversations"
          aria-label="Search conversations"
          autoComplete="off"
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <div className="rail-scroll">
        <div className="rail-label">
          <span>Recent</span>
          <span className="mono">{conversations.length}</span>
        </div>

        <nav className="convo-list" aria-label="Conversations">
          {visible.map((c) => {
            const active = c.id === activeId
            if (renaming === c.id) {
              return (
                <div key={c.id} className="convo is-active">
                  <input
                    ref={inputRef}
                    className="convo-rename"
                    value={draftTitle}
                    aria-label="Rename conversation"
                    onChange={(e) => setDraftTitle(e.target.value)}
                    onBlur={() => commitRename(c.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault()
                        commitRename(c.id)
                      }
                      if (e.key === "Escape") setRenaming(null)
                    }}
                  />
                </div>
              )
            }
            return (
              <div key={c.id} className={`convo${active ? " is-active" : ""}`}>
                <button
                  type="button"
                  className="convo-open"
                  onClick={() => onSelect(c.id)}
                  title={c.title || "Untitled"}
                >
                  <span className="convo-title">{c.title || "Untitled"}</span>
                  <span className="convo-meta mono">
                    {c.message_count} {c.message_count === 1 ? "message" : "messages"}
                    {c.document_name ? ` · ${c.document_name}` : ""}
                  </span>
                </button>
                <button
                  type="button"
                  className="convo-more"
                  aria-label={`Actions for ${c.title || "conversation"}`}
                  onClick={() => {
                    setRenaming(c.id)
                    setDraftTitle(c.title || "")
                  }}
                >
                  <Icon name="more" size={14} />
                </button>
                <button
                  type="button"
                  className="convo-more convo-more--delete"
                  aria-label={`Delete ${c.title || "conversation"}`}
                  onClick={() => remove(c.id)}
                >
                  <Icon name="trash" size={14} />
                </button>
              </div>
            )
          })}

          {conversations.length === 0 && (
            <p className="rail-empty">
              No conversations yet. Ask a question to create the first one.
            </p>
          )}
          {conversations.length > 0 && visible.length === 0 && (
            <p className="rail-empty">No conversation matches that search.</p>
          )}
        </nav>
      </div>

      <div className="rail-footer">
        <p className="rail-legal">
          <Icon name="lock" size={11} />
          Tenant-scoped storage · Qdrant + SQLite
        </p>
      </div>
    </aside>
  )
}