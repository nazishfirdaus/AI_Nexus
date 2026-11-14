import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  api,
  setTenant,
  type AgentResult,
  type ChatResponse,
  type Conversation,
  type DocumentRecord,
  type FeedbackValue,
  type Health,
  type Message,
} from "./utils/api"
import { corpusStats, formatNumber, toView } from "./utils/documents"
import { useAuth } from "./context/AuthContext"
import { ConversationSidebar } from "./components/ConversationSidebar"
import { ChatView } from "./components/ChatView"
import { LibraryView } from "./components/LibraryView"
import { AgentView } from "./components/AgentView"
import { EvalView } from "./components/EvalView"
import { SettingsView, type ThemeMode } from "./components/SettingsView"
import { EvidencePanel } from "./components/EvidencePanel"
import { Icon } from "./components/Icon"
import "./styles/main.css"

type View = "chat" | "library" | "agent" | "eval" | "settings"

interface Toast {
  id: number
  text: string
  tone: "info" | "error"
}

const THEME_KEY = "anexus-theme"

const VIEW_LABEL: Record<View, string> = {
  chat: "Chat",
  library: "Library",
  agent: "Agent",
  eval: "Eval",
  settings: "Settings",
}

const VIEW_CRUMB: Record<View, string> = {
  chat: "Chat",
  library: "Corpus",
  agent: "Agent",
  eval: "Evaluation",
  settings: "Workspace",
}

function resolveTheme(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light"
  }
  return mode
}

export function App() {
  const { user, loading, login } = useAuth()

  const [view, setView] = useState<View>("chat")
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [lastAnswer, setLastAnswer] = useState<ChatResponse | null>(null)

  const [streaming, setStreaming] = useState(false)
  const [draft, setDraft] = useState("")
  const [scopeDocumentId, setScopeDocumentId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [evidenceOpen, setEvidenceOpen] = useState(true)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [theme, setTheme] = useState<ThemeMode>("system")

  const [agentQuery, setAgentQuery] = useState("")
  const [agentThread, setAgentThread] = useState("default_user")
  const [agentRunning, setAgentRunning] = useState(false)
  const [agentResult, setAgentResult] = useState<AgentResult | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const toastId = useRef(0)

  const pushToast = useCallback((text: string, tone: "info" | "error" = "info") => {
    const id = (toastId.current += 1)
    setToasts((current) => [...current.slice(-2), { id, text, tone }])
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id))
    }, 4000)
  }, [])

  /* ── theme ─────────────────────────────────────────────────────────── */
  useEffect(() => {
    let stored: string | null = null
    try {
      stored = window.localStorage.getItem(THEME_KEY)
    } catch {
      // storage may be blocked
    }
    const mode: ThemeMode =
      stored === "light" || stored === "dark" || stored === "system"
        ? stored
        : "system"
    setTheme(mode)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = resolveTheme(theme)
    try {
      window.localStorage.setItem(THEME_KEY, theme)
    } catch {
      // storage may be blocked
    }
  }, [theme])

  /* ── tenant binding ────────────────────────────────────────────────── */
  useEffect(() => {
    setTenant(user)
  }, [user])

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.listConversations())
    } catch (err) {
      pushToast(err instanceof Error ? err.message : "Could not load conversations.", "error")
    }
  }, [pushToast])

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments())
    } catch (err) {
      pushToast(
        err instanceof Error
          ? `Could not load documents: ${err.message}`
          : "Could not load documents.",
        "error"
      )
    }
  }, [pushToast])

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health())
    } catch {
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    if (loading || !user) return
    void refreshHealth()
    void refreshConversations()
    void refreshDocuments()
  }, [loading, user, refreshHealth, refreshConversations, refreshDocuments])

  /* ── conversation loading ──────────────────────────────────────────── */
  const openConversation = useCallback(
    async (id: string) => {
      setActiveId(id)
      setView("chat")
      setLastAnswer(null)
      try {
        const detail = await api.getConversation(id)
        setMessages(detail.messages)
      } catch (err) {
        pushToast(
          err instanceof Error ? err.message : "Could not load the conversation.",
          "error"
        )
      }
    },
    [pushToast]
  )

  const startNewConversation = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setStreaming(false)
    setDraft("")
    setActiveId(null)
    setMessages([])
    setLastAnswer(null)
    setView("chat")
  }, [])

  /* ── chat ──────────────────────────────────────────────────────────── */
  const send = useCallback(
    async (text: string) => {
      const message = text.trim()
      if (!message || streaming) return

      setView("chat")
      setStreaming(true)
      setDraft("")
      setLastAnswer(null)

      const optimisticUser: Message = {
        id: -Date.now(),
        role: "user",
        content: message,
        citations: null,
        feedback: null,
        created_at: new Date().toISOString(),
      }
      setMessages((current) => {
        if (activeId === null) return [optimisticUser]
        return [...current, optimisticUser]
      })

      const controller = new AbortController()
      abortRef.current = controller
      let streamed = ""
      // The server assigns the conversation id, so track it from the stream
      // rather than from the (possibly stale) render-time value.
      let conversationId = activeId

      try {
        await api.streamChat(
          {
            message,
            conversation_id: activeId,
            document_id: scopeDocumentId,
          },
          {
            onMeta: (meta) => {
              conversationId = meta.conversation_id
            },
            onDelta: (chunk) => {
              streamed += chunk
              setDraft(streamed)
            },
            onDone: (payload) => {
              conversationId = payload.conversation_id
              setLastAnswer(payload)
            },
            onError: (detail) => pushToast(detail, "error"),
          },
          controller.signal
        )

        // Reload from the server so message ids (needed for feedback) are real.
        if (conversationId) {
          const detail = await api.getConversation(conversationId)
          setMessages(detail.messages)
          setActiveId(conversationId)
        }
        void refreshConversations()
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          pushToast(err instanceof Error ? err.message : "The request failed.", "error")
        }
      } finally {
        abortRef.current = null
        setStreaming(false)
        setDraft("")
      }
    },
    [activeId, pushToast, refreshConversations, scopeDocumentId, streaming]
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setStreaming(false)
    setDraft("")
  }, [])

  const feedback = useCallback(
    async (messageId: number, value: FeedbackValue) => {
      if (messageId < 0) return
      try {
        const updated = await api.setFeedback(messageId, value)
        setMessages((current) =>
          current.map((item) => (item.id === updated.id ? updated : item))
        )
      } catch (err) {
        pushToast(err instanceof Error ? err.message : "Could not save feedback.", "error")
      }
    },
    [pushToast]
  )

  /* ── documents ─────────────────────────────────────────────────────── */
  const upload = useCallback(
    async (files: File[]) => {
      setUploading(true)
      let added = 0
      let chunks = 0
      try {
        for (const file of files) {
          const result = await api.uploadDocument(file)
          added += 1
          chunks += result.chunks
        }
        pushToast(
          `Ingested ${added} file${added === 1 ? "" : "s"} · ${formatNumber(chunks)} chunks`,
          "info"
        )
        await refreshDocuments()
      } catch (err) {
        pushToast(err instanceof Error ? err.message : "Ingestion failed.", "error")
      } finally {
        setUploading(false)
      }
    },
    [pushToast, refreshDocuments]
  )

  const removeDocument = useCallback(
    async (document: { id: string; filename: string }) => {
      if (!window.confirm(`Remove ${document.filename} from the corpus?`)) return
      try {
        await api.deleteDocument(document.id)
        pushToast(`Removed ${document.filename}`)
        await refreshDocuments()
        void refreshHealth()
      } catch (err) {
        pushToast(
          err instanceof Error ? err.message : "Could not delete the document.",
          "error"
        )
      }
    },
    [pushToast, refreshDocuments, refreshHealth]
  )

  /* ── agent ─────────────────────────────────────────────────────────── */
  const runAgent = useCallback(async () => {
    const question = agentQuery.trim()
    if (!question || agentRunning) return
    setAgentRunning(true)
    setAgentResult(null)
    try {
      const result = await api.runAgent(question, agentThread.trim() || "default_user")
      setAgentResult(result)
      if (result.guardrails === "blocked") {
        pushToast("Blocked by guardrails.", "error")
      }
    } catch (err) {
      pushToast(err instanceof Error ? err.message : "The agent run failed.", "error")
    } finally {
      setAgentRunning(false)
    }
  }, [agentQuery, agentRunning, agentThread, pushToast])

  /* ── derived ───────────────────────────────────────────────────────── */
  const views = useMemo(() => documents.map(toView), [documents])
  const stats = useMemo(() => corpusStats(views), [views])
  const activeTitle =
    conversations.find((item) => item.id === activeId)?.title ?? "New conversation"

  if (loading) {
    return (
      <div className="boot">
        <span className="boot-mark">A</span>
        <p>Loading ANexus…</p>
      </div>
    )
  }

  return (
    <div className={`app${view === "chat" ? "" : " app--wide"}`}>
      <ConversationSidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={openConversation}
        onCreated={(conversation) =>
          setConversations((current) => [conversation, ...current])
        }
        onDeleted={(id) => {
          setConversations((current) => current.filter((item) => item.id !== id))
          if (activeId === id) startNewConversation()
        }}
        onRenamed={(conversation) =>
          setConversations((current) =>
            current.map((item) => (item.id === conversation.id ? conversation : item))
          )
        }
        onError={(message) => pushToast(message, "error")}
      />

      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            <div className="topbar-title">
              <span className="eyebrow">{VIEW_CRUMB[view]}</span>
              <h1>
                {view === "chat"
                  ? activeTitle
                  : view === "library"
                    ? "Document library"
                    : view === "agent"
                      ? "LangGraph agent"
                      : view === "eval"
                        ? "Evaluation harness"
                        : "Settings"}
              </h1>
              <p>
                {view === "chat"
                  ? `${formatNumber(stats.documents)} documents · ${formatNumber(
                      stats.chunks
                    )} chunks`
                  : view === "library"
                    ? `${formatNumber(stats.documents)} documents · ${formatNumber(
                        stats.pages
                      )} pages`
                    : view === "agent"
                      ? "planner · retriever · responder"
                      : view === "eval"
                        ? "results*.json on disk"
                        : `${health?.user_id ?? "tenant"} · ${
                            health?.environment ?? "api offline"
                          }`}
              </p>
            </div>

            <nav className="view-tabs" aria-label="Views">
              {(["chat", "library", "agent", "eval", "settings"] as View[]).map(
                (item) => (
                  <button
                    key={item}
                    type="button"
                    className={`view-tab${view === item ? " is-active" : ""}`}
                    onClick={() => setView(item)}
                  >
                    {VIEW_LABEL[item]}
                  </button>
                )
              )}
            </nav>
          </div>

          <div className="topbar-right">
            <button
              className="icon-btn theme-btn"
              type="button"
              aria-label="Toggle colour theme"
              onClick={() =>
                setTheme((current) =>
                  resolveTheme(current) === "dark" ? "light" : "dark"
                )
              }
            >
              <Icon name="sun" className="ico-sun" />
              <Icon name="moon" className="ico-moon" />
            </button>

            {view === "chat" && (
              <button
                className="btn btn--ghost btn--sm evidence-toggle"
                type="button"
                aria-expanded={evidenceOpen}
                aria-controls="evidence"
                onClick={() => setEvidenceOpen((open) => !open)}
              >
                <Icon name="layers" />
                <span>Trace</span>
              </button>
            )}
          </div>
        </header>

        {view === "chat" && (
          <ChatView
            messages={messages}
            streaming={streaming}
            draft={draft}
            documents={views}
            stats={stats}
            activeTitle={activeTitle}
            documentId={scopeDocumentId}
            lastAnswer={lastAnswer}
            onScope={setScopeDocumentId}
            onSend={send}
            onFeedback={feedback}
            onStop={stop}
          />
        )}

        {view === "library" && (
          <LibraryView
            documents={views}
            stats={stats}
            uploading={uploading}
            onUpload={upload}
            onDelete={removeDocument}
            onRefresh={() => {
              void refreshDocuments()
              void refreshHealth()
            }}
          />
        )}

        {view === "agent" && (
          <AgentView
            query={agentQuery}
            onQuery={setAgentQuery}
            threadId={agentThread}
            onThreadId={setAgentThread}
            running={agentRunning}
            result={agentResult}
            onRun={runAgent}
            onError={(message) => pushToast(message, "error")}
          />
        )}

        {view === "eval" && (
          <EvalView onError={(message) => pushToast(message, "error")} />
        )}

        {view === "settings" && (
          <SettingsView
            health={health}
            tenant={user ?? ""}
            theme={theme}
            onTheme={setTheme}
            onTenant={(next) => {
              if (!/^[A-Za-z0-9_-]{1,64}$/.test(next)) {
                pushToast(
                  "Tenant id must be 1-64 characters of letters, digits, '-' or '_'.",
                  "error"
                )
                return
              }
              login(next)
              startNewConversation()
              pushToast(`Switched to tenant ${next}`)
            }}
          />
        )}
      </main>

      {view === "chat" && evidenceOpen && (
        <EvidencePanel
          answer={lastAnswer}
          stats={stats}
          health={health}
          onClose={() => setEvidenceOpen(false)}
        />
      )}

      <div className="toast-stack" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`toast${toast.tone === "error" ? " toast--error" : ""}`}
          >
            <Icon name={toast.tone === "error" ? "alert" : "check"} size={14} />
            <span>{toast.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}