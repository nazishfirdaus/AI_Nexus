import { useCallback, useEffect, useRef } from "react"
import { api, type AgentResult } from "../utils/api"
import { Icon } from "./Icon"

interface Props {
  query: string
  onQuery: (value: string) => void
  threadId: string
  onThreadId: (value: string) => void
  running: boolean
  result: AgentResult | null
  onRun: () => void
  onError: (message: string) => void
}

export function AgentView({
  query,
  onQuery,
  threadId,
  onThreadId,
  running,
  result,
  onRun,
  onError,
}: Props) {
  const errorRef = useRef(onError)
  errorRef.current = onError

  const drawGraph = useCallback(async (source: string) => {
    const mermaid = (await import("mermaid")).default
    mermaid.initialize({
      startOnLoad: false,
      theme: "neutral",
      securityLevel: "strict",
    })
    const { svg } = await mermaid.render(`agent-graph-${Date.now()}`, source)
    return svg
  }, [])

  // Draw the graph once per mount; `errorRef` keeps the callback out of deps.
  useEffect(() => {
    let cancelled = false
    const host = document.getElementById("agent-graph")
    if (!host) return

    api
      .agentInfo()
      .then(async (info) => {
        const svg = await drawGraph(info.mermaid)
        if (!cancelled) host.innerHTML = svg
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          errorRef.current(
            err instanceof Error ? err.message : "Could not load the graph."
          )
        }
      })

    return () => {
      cancelled = true
    }
  }, [drawGraph])

  return (
    <section className="view view--agent is-active" aria-label="Agent">
      <div className="page">
        <div className="page-head">
          <div>
            <span className="eyebrow">Agent</span>
            <h2 className="display">LangGraph planner</h2>
            <p className="page-sub">
              Plan → retrieve → answer, gated by NeMo guardrails. This is a
              separate stack from the RAG pipeline behind chat: it has its own
              collection, its own memory and its own model calls.
            </p>
          </div>
          <button className="btn btn--gold" type="button" onClick={onRun} disabled={running}>
            <Icon name="arrow-up" />
            <span>{running ? "Running…" : "Run agent"}</span>
          </button>
        </div>

        <div className="agent-grid">
          <div className="card agent-graph-card">
            <div className="card-head">
              <div>
                <h3>Graph</h3>
                <p>Rendered from the compiled graph at request time.</p>
              </div>
            </div>
            <div id="agent-graph" className="agent-graph" />
          </div>

          <div className="card">
            <div className="card-head">
              <div>
                <h3>Run</h3>
                <p>Guardrails run first; the graph runs only if they pass.</p>
              </div>
            </div>

            <div className="field-grid">
              <label className="field field--wide">
                <span className="field-label">Question</span>
                <textarea
                  className="input agent-question"
                  rows={3}
                  value={query}
                  placeholder="Ask the agent something about the corpus…"
                  onChange={(event) => onQuery(event.target.value)}
                />
              </label>
              <label className="field">
                <span className="field-label">Thread id</span>
                <input
                  className="input input--mono"
                  value={threadId}
                  onChange={(event) => onThreadId(event.target.value)}
                />
              </label>
              <div className="field">
                <span className="field-label">Status</span>
                <span className={`status ${running ? "status--warn" : "status--muted"}`}>
                  <span className="pip" />
                  {running ? "running" : result ? result.status || "idle" : "idle"}
                </span>
              </div>
            </div>

            {result && (
              <div className="agent-result">
                <div className="agent-result-head">
                  <span className="eyebrow">Answer</span>
                  <span
                    className={`badge ${
                      result.guardrails === "blocked"
                        ? "badge--danger"
                        : result.guardrails === "unavailable"
                          ? "badge--muted"
                          : "badge--gold"
                    }`}
                  >
                    guardrails: {result.guardrails}
                  </span>
                </div>

                <div className="answer">
                  {result.answer
                    .split(/\n{2,}/)
                    .filter(Boolean)
                    .map((part, index) => (
                      <p key={index}>{part}</p>
                    ))}
                </div>

                {result.thought_process.length > 0 && (
                  <>
                    <div className="panel-label agent-sub-label">Plan</div>
                    <ol className="pipeline">
                      {result.thought_process.map((step, index) => (
                        <li key={index} className="pipe-step is-done">
                          <span className="pipe-dot" />
                          <div className="pipe-copy">
                            <span className="pipe-title">{step}</span>
                          </div>
                          <span className="pipe-value mono">{index + 1}</span>
                        </li>
                      ))}
                    </ol>
                  </>
                )}

                {result.sources.length > 0 && (
                  <>
                    <div className="panel-label agent-sub-label">
                      Sources ({result.sources.length})
                    </div>
                    <div className="source-list">
                      {result.sources.map((source, index) => (
                        <div key={index} className="source-item">
                          <span className="source-doc">{source.slice(0, 220)}</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}