import { useEffect, useRef, useState } from "react"
import type { ChatResponse, FeedbackValue, Message } from "../utils/api"
import type { CorpusStats, DocView } from "../utils/documents"
import { formatDuration, formatNumber } from "../utils/documents"
import { Icon } from "./Icon"

interface Props {
  messages: Message[]
  streaming: boolean
  draft: string
  documents: DocView[]
  stats: CorpusStats
  activeTitle: string
  documentId: string | null
  lastAnswer: ChatResponse | null
  onScope: (documentId: string | null) => void
  onSend: (text: string) => void
  onFeedback: (messageId: number, value: FeedbackValue) => void
  onStop: () => void
}

function paragraphs(text: string): string[] {
  return text
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
}

export function ChatView({
  messages,
  streaming,
  draft,
  documents,
  stats,
  activeTitle,
  documentId,
  lastAnswer,
  onScope,
  onSend,
  onFeedback,
  onStop,
}: Props) {
  const [text, setText] = useState("")
  const [scopeOpen, setScopeOpen] = useState(false)
  const threadRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const scopeName =
    documents.find((doc) => doc.id === documentId)?.filename ?? "All documents"

  useEffect(() => {
    const node = threadRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [messages, draft, streaming])

  function grow() {
    const node = textareaRef.current
    if (!node) return
    node.style.height = "auto"
    node.style.height = `${Math.min(node.scrollHeight, 150)}px`
  }

  function submit() {
    const value = text.trim()
    if (!value || streaming) return
    onSend(value)
    setText("")
    requestAnimationFrame(grow)
  }

  const suggestions = documents.slice(0, 3).map((doc) => ({
    label: `Summarise ${doc.filename}`,
    prompt: `Summarise ${doc.filename}`,
  }))

  return (
    <section className="view view--chat is-active" aria-label="Chat">
      <div className="thread" ref={threadRef} aria-live="polite">
        <div className="thread-inner">
          {messages.length === 0 && !streaming && (
            <div className="empty">
              <span className="empty-mark" aria-hidden="true">
                A
              </span>
              <span className="eyebrow">Grounded answers</span>
              <h2>Ask your mortgage documents.</h2>
              <p>
                Retrieved from Qdrant, cited by page. Answers are generated only
                from the indexed corpus.
              </p>

              {suggestions.length > 0 ? (
                <div className="empty-suggestions">
                  {suggestions.map((item) => (
                    <button
                      key={item.prompt}
                      className="empty-suggestion"
                      type="button"
                      onClick={() => onSend(item.prompt)}
                    >
                      <Icon name="file" size={14} />
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="empty-note">
                  No documents indexed for this tenant yet — upload one from the
                  library.
                </p>
              )}

              <p className="empty-note">
                {formatNumber(stats.documents)} documents ·{" "}
                {formatNumber(stats.chunks)} chunks ·{" "}
                {formatNumber(stats.pages)} pages indexed
              </p>
            </div>
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <div key={`u-${message.id}`} className="msg msg-user">
                <div className="msg-user-bubble">{message.content}</div>
                <span className="msg-time mono">{shortTime(message.created_at)}</span>
              </div>
            ) : (
              <article key={`a-${message.id}`} className="msg msg-ai">
                <header className="msg-ai-head">
                  <span className="msg-ai-rule" aria-hidden="true" />
                  <span className="msg-ai-name">ANexus</span>
                  <span className="msg-ai-time mono">
                    {shortTime(message.created_at)}
                  </span>
                </header>

                <div className="answer">
                  {paragraphs(message.content).map((part, index) => (
                    <p key={index}>{part}</p>
                  ))}
                </div>

                {message.citations && message.citations.length > 0 && (
                  <div className="answer-sources">
                    {message.citations.map((citation, index) => (
                      <figure key={`${citation.document_id}-${citation.page_number}-${index}`} className="src-card">
                        <span className="src-card-top">
                          <span className="src-card-index">{index + 1}</span>
                          <span className="src-card-type">
                            Page {citation.page_number}
                          </span>
                          <span className="src-card-score">
                            {citation.score.toFixed(2)}
                          </span>
                        </span>
                        <figcaption className="src-card-doc">
                          {citation.filename || citation.document_id}
                        </figcaption>
                        {citation.snippet && (
                          <span className="src-card-snippet">{citation.snippet}</span>
                        )}
                      </figure>
                    ))}
                  </div>
                )}

                <footer className="answer-foot">
                  <button
                    type="button"
                    className={`icon-btn${message.feedback === "up" ? " is-on" : ""}`}
                    aria-label="Mark answer helpful"
                    aria-pressed={message.feedback === "up"}
                    onClick={() =>
                      onFeedback(message.id, message.feedback === "up" ? null : "up")
                    }
                  >
                    <Icon name="thumb-up" size={14} />
                  </button>
                  <button
                    type="button"
                    className={`icon-btn${message.feedback === "down" ? " is-on" : ""}`}
                    aria-label="Mark answer unhelpful"
                    aria-pressed={message.feedback === "down"}
                    onClick={() =>
                      onFeedback(message.id, message.feedback === "down" ? null : "down")
                    }
                  >
                    <Icon name="thumb-down" size={14} />
                  </button>
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label="Copy answer"
                    onClick={() =>
                      navigator.clipboard?.writeText(message.content).catch(() => undefined)
                    }
                  >
                    <Icon name="copy" size={14} />
                  </button>
                  <span className="foot-divider" aria-hidden="true" />
                  <span className="foot-label">
                    {message.citations?.length ?? 0}{" "}
                    {message.citations?.length === 1 ? "citation" : "citations"}
                  </span>
                </footer>
              </article>
            )
          )}

          {streaming && (
            <article className="msg msg-ai">
              <header className="msg-ai-head">
                <span className="msg-ai-rule" aria-hidden="true" />
                <span className="msg-ai-name">ANexus</span>
                <span className="msg-ai-time mono">streaming</span>
              </header>
              <div className="answer">
                {draft ? (
                  paragraphs(draft).map((part, index) => <p key={index}>{part}</p>)
                ) : (
                  <div className="thinking-row">
                    <span className="dot-pulse" aria-hidden="true">
                      <i />
                      <i />
                      <i />
                    </span>
                    retrieving · reranking · generating
                  </div>
                )}
                <span className="stream-caret" aria-hidden="true" />
              </div>
            </article>
          )}
        </div>
      </div>

      <div className="composer-wrap">
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault()
            submit()
          }}
        >
          <label className="sr-only" htmlFor="prompt">
            Ask a question about your documents
          </label>
          <textarea
            id="prompt"
            ref={textareaRef}
            rows={1}
            value={text}
            placeholder="Ask across your documents…"
            onChange={(event) => {
              setText(event.target.value)
              grow()
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault()
                submit()
              }
            }}
          />

          <div className="composer-bar">
            <div className="composer-tools">
              <div className="menu-wrap">
                <button
                  className="scope-chip"
                  type="button"
                  aria-expanded={scopeOpen}
                  aria-haspopup="listbox"
                  onClick={() => setScopeOpen((open) => !open)}
                >
                  <Icon name="book" size={13} />
                  <span>{scopeName}</span>
                  <Icon name="chevron-down" size={11} className="chev" />
                </button>

                {scopeOpen && (
                  <div className="menu menu--scope" role="listbox" aria-label="Search scope">
                    <button
                      className={`menu-item${documentId === null ? " is-selected" : ""}`}
                      type="button"
                      role="option"
                      aria-selected={documentId === null}
                      onClick={() => {
                        onScope(null)
                        setScopeOpen(false)
                      }}
                    >
                      <span>All documents</span>
                      <span className="mono menu-meta">{documents.length}</span>
                    </button>
                    {documents.length > 0 && <div className="menu-sep" />}
                    {documents.map((doc) => (
                      <button
                        key={doc.id}
                        className={`menu-item${documentId === doc.id ? " is-selected" : ""}`}
                        type="button"
                        role="option"
                        aria-selected={documentId === doc.id}
                        onClick={() => {
                          onScope(doc.id)
                          setScopeOpen(false)
                        }}
                      >
                        <span>{doc.filename}</span>
                        <span className="mono menu-meta">{doc.pages} p</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <span className="tool-divider" aria-hidden="true" />
              <span className="model-chip" title="Model reported by the API">
                <span className="pip pip--live" />
                {lastAnswer?.model || "model assigned on send"}
              </span>
            </div>

            {streaming ? (
              <button
                className="send send--stop"
                type="button"
                aria-label="Stop generating"
                onClick={onStop}
              >
                <Icon name="x" />
              </button>
            ) : (
              <button
                className="send"
                type="submit"
                aria-label="Send question"
                disabled={text.trim().length === 0}
              >
                <Icon name="arrow-up" />
              </button>
            )}
          </div>
        </form>

        <p className="composer-note">
          <Icon name="shield" size={11} />
          <span>
            {lastAnswer
              ? `Answered by ${lastAnswer.provider || "provider"} · ${
                  lastAnswer.model || "model"
                } · ${formatDuration(lastAnswer.latency_s)} · ${lastAnswer.status}`
              : "Answers are generated only from the indexed corpus."}
          </span>
        </p>
      </div>

      <span className="sr-only" aria-live="polite">
        {activeTitle}
      </span>
    </section>
  )
}

function shortTime(iso: string): string {
  if (!iso) return ""
  const parsed = new Date(iso)
  if (Number.isNaN(parsed.getTime())) return iso
  return parsed.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
}