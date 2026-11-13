/**
 * Typed client for the ANexus HTTP API.
 *
 * Request/response shapes mirror `src/api/schemas.py` exactly. No field is
 * invented here: every value the UI renders comes from a server response.
 */

export interface Citation {
  page_number: number
  score: number
  snippet: string
  document_id: string
  filename: string
}

export type ChatStatus = "answered" | "no_context"

export interface ChatResponse {
  conversation_id: string
  user_message_id: number
  assistant_message_id: number
  answer: string
  citations: Citation[]
  provider: string
  model: string
  latency_s: number
  intent: string
  status: ChatStatus
}

export interface Conversation {
  id: string
  title: string
  document_id: string | null
  document_name: string | null
  summary: string
  created_at: string
  updated_at: string
  message_count: number
}

export type FeedbackValue = "up" | "down" | null

export interface Message {
  id: number
  role: string
  content: string
  citations: Citation[] | null
  feedback: FeedbackValue
  created_at: string
}

export interface ConversationDetail extends Conversation {
  messages: Message[]
}

export interface DocumentRecord {
  id: string
  filename: string
  pages: number
  chunks: number
  vectors: number
  duration_s: number
  content_hash: string | null
  created_at: string
}

export interface IngestResult {
  document_id: string
  filename: string
  pages: number
  chunks: number
  duration_s: number
  vectors_added: number
  content_hash: string | null
  pages_empty_dropped: number
  pages_ocr: number
  pages_overcleaned: number
  chunks_dropped_short: number
  boilerplate_lines_removed: number
  chunks_mean_chars: number
  chunks_median_chars: number
  chunks_max_chars: number
}

export interface Health {
  status: string
  service: string
  environment: string
  user_id: string
  collection: string
  qdrant_configured: boolean
}

export interface StreamHandlers {
  onMeta?: (meta: { conversation_id: string; user_message_id: number }) => void
  onDelta?: (text: string) => void
  onDone?: (payload: ChatResponse) => void
  onError?: (detail: string) => void
}

/* ── Agent (LangGraph planner / retriever / responder) ─────────────────── */

export interface AgentEdge {
  from_node: string
  to_node: string
}

export interface AgentInfo {
  nodes: string[]
  edges: AgentEdge[]
  mermaid: string
  entry_point: string
  checkpointer: string
  guardrails_ready: boolean
  corpus: string
}

export type GuardrailState = "passed" | "blocked" | "unavailable"

export interface AgentResult {
  question: string
  answer: string
  thought_process: string[]
  status: string
  sources: string[]
  guardrails: GuardrailState
}

/* ── Evaluation harness ────────────────────────────────────────────────── */

export interface EvalRunSummary {
  name: string
  modified_at: string
  items: number
  summary: Record<string, unknown>
}

export interface EvalRun extends EvalRunSummary {
  rows: Record<string, unknown>[]
}

const API_KEY = import.meta.env.VITE_API_KEY ?? ""

let tenant: string | null = null

/** Set the tenant sent as ``X-User-Id``; null lets the API pick its default. */
export function setTenant(user: string | null): void {
  tenant = user
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const merged: Record<string, string> = { ...extra }
  if (tenant) merged["X-User-Id"] = tenant
  if (API_KEY) merged["X-API-Key"] = API_KEY
  return merged
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === "string") detail = body.detail
    } catch {
      // Non-JSON error body; keep the status text.
    }
    throw new Error(detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  health(): Promise<Health> {
    return fetch("/api/health", { headers: headers() }).then(json<Health>)
  },

  listConversations(limit = 50): Promise<Conversation[]> {
    return fetch(`/api/conversations?limit=${limit}`, {
      headers: headers(),
    }).then(json<Conversation[]>)
  },

  getConversation(id: string): Promise<ConversationDetail> {
    return fetch(`/api/conversations/${encodeURIComponent(id)}`, {
      headers: headers(),
    }).then(json<ConversationDetail>)
  },

  createConversation(input: {
    title?: string
    document_id?: string | null
    document_name?: string | null
  }): Promise<Conversation> {
    return fetch("/api/conversations", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        title: input.title ?? "",
        document_id: input.document_id ?? null,
        document_name: input.document_name ?? null,
      }),
    }).then(json<Conversation>)
  },

  renameConversation(id: string, title: string): Promise<Conversation> {
    return fetch(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ title }),
    }).then(json<Conversation>)
  },

  deleteConversation(id: string): Promise<void> {
    return fetch(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers: headers(),
    }).then(() => undefined)
  },

  setFeedback(messageId: number, value: FeedbackValue): Promise<Message> {
    return fetch(`/api/messages/${messageId}/feedback`, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ value }),
    }).then(json<Message>)
  },

  listDocuments(): Promise<DocumentRecord[]> {
    return fetch("/api/documents", { headers: headers() }).then(
      json<DocumentRecord[]>
    )
  },

  uploadDocument(file: File): Promise<IngestResult> {
    const form = new FormData()
    form.append("file", file)
    return fetch("/api/documents", {
      method: "POST",
      headers: headers(),
      body: form,
    }).then(json<IngestResult>)
  },

  deleteDocument(id: string): Promise<void> {
    return fetch(`/api/documents/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers: headers(),
    }).then(() => undefined)
  },

  sendChat(input: {
    message: string
    conversation_id?: string | null
    document_id?: string | null
  }): Promise<ChatResponse> {
    return fetch("/api/chat", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        message: input.message,
        conversation_id: input.conversation_id ?? null,
        document_id: input.document_id ?? null,
      }),
    }).then(json<ChatResponse>)
  },

  /**
   * Stream an answer over SSE. The endpoint is ``POST`` (it needs a body), so
   * this cannot use ``EventSource``; the frames are parsed by hand.
   */
  async streamChat(
    input: {
      message: string
      conversation_id?: string | null
      document_id?: string | null
    },
    handlers: StreamHandlers,
    signal?: AbortSignal
  ): Promise<void> {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify({
        message: input.message,
        conversation_id: input.conversation_id ?? null,
        document_id: input.document_id ?? null,
      }),
      signal,
    })

    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const body = (await res.json()) as { detail?: unknown }
        if (typeof body.detail === "string") detail = body.detail
      } catch {
        // keep status text
      }
      handlers.onError?.(detail)
      return
    }
    if (!res.body) {
      handlers.onError?.("The server returned an empty stream.")
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let boundary = buffer.indexOf("\n\n")
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        dispatchFrame(frame, handlers)
        boundary = buffer.indexOf("\n\n")
      }
    }

    buffer += decoder.decode()
    if (buffer.trim()) dispatchFrame(buffer, handlers)
  },

  /* ── agent ─────────────────────────────────────────────────────────── */
  agentInfo(): Promise<AgentInfo> {
    return fetch("/api/agent", { headers: headers() }).then(json<AgentInfo>)
  },

  runAgent(q: string, threadId: string): Promise<AgentResult> {
    return fetch("/api/agent/query", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ q, thread_id: threadId }),
    }).then(json<AgentResult>)
  },

  /* ── evaluation ────────────────────────────────────────────────────── */
  evalRuns(): Promise<EvalRunSummary[]> {
    return fetch("/api/eval/runs", { headers: headers() }).then(
      json<EvalRunSummary[]>
    )
  },

  evalRun(name: string): Promise<EvalRun> {
    return fetch(`/api/eval/runs/${encodeURIComponent(name)}`, {
      headers: headers(),
    }).then(json<EvalRun>)
  },
}

function dispatchFrame(frame: string, handlers: StreamHandlers): void {
  let event = "message"
  const dataLines: string[] = []

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim()
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return

  const raw = dataLines.join("\n")
  let payload: Record<string, unknown>
  try {
    payload = JSON.parse(raw) as Record<string, unknown>
  } catch {
    return
  }

  switch (event) {
    case "meta":
      handlers.onMeta?.({
        conversation_id: String(payload.conversation_id ?? ""),
        user_message_id: Number(payload.user_message_id ?? 0),
      })
      break
    case "delta":
      if (typeof payload.text === "string") handlers.onDelta?.(payload.text)
      break
    case "done":
      handlers.onDone?.(payload as unknown as ChatResponse)
      break
    case "error":
      handlers.onError?.(
        typeof payload.detail === "string"
          ? payload.detail
          : "The response stream failed."
      )
      break
    default:
      break
  }
}