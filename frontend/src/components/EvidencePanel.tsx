import type { ChatResponse, Health } from "../utils/api"
import type { CorpusStats } from "../utils/documents"
import { formatDuration, formatNumber } from "../utils/documents"
import { Icon } from "./Icon"

interface Props {
  answer: ChatResponse | null
  stats: CorpusStats
  health: Health | null
  onClose: () => void
}

export function EvidencePanel({ answer, stats, health, onClose }: Props) {
  // Scores are raw retrieval scores and can be negative, so bars are scaled
  // against the strongest citation in this answer rather than a 0..1 clamp.
  const peak = answer
    ? answer.citations.reduce(
        (max, citation) => Math.max(max, Math.abs(citation.score)),
        0
      )
    : 0

  return (
    <aside className="evidence" aria-label="Run trace">
      <div className="evidence-head">
        <span className="eyebrow">Run trace</span>
        <button
          type="button"
          className="icon-btn evidence-close"
          aria-label="Close trace"
          onClick={onClose}
        >
          <Icon name="x" />
        </button>
      </div>

      <div className="panel-block">
        <div className="panel-label">Last answer</div>

        {answer ? (
          <ol className="pipeline">
            <li className="pipe-step is-done">
              <span className="pipe-dot" />
              <div className="pipe-copy">
                <span className="pipe-title">Provider</span>
                <span className="pipe-sub">{answer.provider || "unreported"}</span>
              </div>
              <span className="pipe-value mono">{answer.status}</span>
            </li>
            <li className="pipe-step is-done">
              <span className="pipe-dot" />
              <div className="pipe-copy">
                <span className="pipe-title">Model</span>
                <span className="pipe-sub">{answer.model || "unreported"}</span>
              </div>
              <span className="pipe-value mono">
                {formatDuration(answer.latency_s)}
              </span>
            </li>
            <li className="pipe-step is-done">
              <span className="pipe-dot" />
              <div className="pipe-copy">
                <span className="pipe-title">Citations</span>
                <span className="pipe-sub">
                  {answer.citations.length} passage
                  {answer.citations.length === 1 ? "" : "s"} returned
                </span>
              </div>
              <span className="pipe-value mono">{answer.citations.length}</span>
            </li>
            {answer.intent && (
              <li className="pipe-step is-done">
                <span className="pipe-dot" />
                <div className="pipe-copy">
                  <span className="pipe-title">Intent</span>
                  <span className="pipe-sub">classifier label</span>
                </div>
                <span className="pipe-value mono">{answer.intent}</span>
              </li>
            )}
          </ol>
        ) : (
          <p className="source-empty">
            No answer yet. Send a question to populate the trace.
          </p>
        )}
      </div>

      <div className="panel-block">
        <div className="panel-label">Search scope</div>
        <div className="scope-card scope-card--static">
          <span className="scope-card-icon">
            <Icon name="book" size={13} />
          </span>
          <span className="scope-card-copy">
            <span>Tenant corpus</span>
            <span className="mono">
              {formatNumber(stats.documents)} documents ·{" "}
              {formatNumber(stats.chunks)} chunks
            </span>
          </span>
        </div>
      </div>

      <div className="panel-block panel-block--sources">
        <div className="panel-heading">
          <span className="panel-label">Sources</span>
          <span className="mono panel-count">{answer?.citations.length ?? 0}</span>
        </div>

        <div className="source-list">
          {answer && answer.citations.length > 0 ? (
            answer.citations.map((citation, index) => (
              <article
                key={`${citation.document_id}-${citation.page_number}-${index}`}
                className="source-item"
              >
                <span className="source-top">
                  <span className="source-index">{index + 1}</span>
                  <span className="source-type">page {citation.page_number}</span>
                  <span className="source-score">{citation.score.toFixed(2)}</span>
                </span>
                <span className="source-doc">
                  {citation.filename || citation.document_id}
                </span>
                {citation.snippet && (
                  <span className="source-snippet">{citation.snippet}</span>
                )}
                <span className="source-bar" aria-hidden="true">
                  <span
                    style={{
                      width: `${
                        peak > 0 ? (Math.abs(citation.score) / peak) * 100 : 0
                      }%`,
                    }}
                  />
                </span>
              </article>
            ))
          ) : (
            <p className="source-empty">
              Sources appear here once an answer has been generated.
            </p>
          )}
        </div>
      </div>

      <div className="panel-foot mono">
        <span>
          {health
            ? `collection ${health.collection}`
            : "collection unavailable"}
        </span>
        <span>
          {health
            ? `${health.environment} · qdrant ${
                health.qdrant_configured ? "configured" : "missing"
              }`
            : "tenant unknown"}
        </span>
      </div>
    </aside>
  )
}