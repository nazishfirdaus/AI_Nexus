import { useMemo, useRef, useState } from "react"
import type { DocKind, DocView, CorpusStats } from "../utils/documents"
import { formatDate, formatDuration, formatNumber, matchesQuery } from "../utils/documents"
import { Icon } from "./Icon"

interface Props {
  documents: DocView[]
  stats: CorpusStats
  uploading: boolean
  onUpload: (files: File[]) => void
  onDelete: (document: DocView) => void
  onRefresh: () => void
}

const FILTERS: { key: DocKind | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "pdf", label: "PDF" },
  { key: "sheet", label: "Sheets" },
  { key: "doc", label: "Docs" },
  { key: "image", label: "Images" },
  { key: "web", label: "Web" },
]

export function LibraryView({
  documents,
  stats,
  uploading,
  onUpload,
  onDelete,
  onRefresh,
}: Props) {
  const [filter, setFilter] = useState<DocKind | "all">("all")
  const [query, setQuery] = useState("")
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const rows = useMemo(
    () =>
      documents.filter(
        (doc) =>
          (filter === "all" || doc.kind === filter) && matchesQuery(doc, query)
      ),
    [documents, filter, query]
  )

  function handleFiles(list: FileList | null) {
    if (!list || list.length === 0) return
    onUpload(Array.from(list))
    if (inputRef.current) inputRef.current.value = ""
  }

  return (
    <section className="view view--library is-active" aria-label="Document library">
      <div className="page">
        <div className="page-head">
          <div>
            <span className="eyebrow">Corpus</span>
            <h2 className="display">Document library</h2>
            <p className="page-sub">
              Documents are parsed, chunked and embedded into this tenant's Qdrant
              collection. Counts below come from the store.
            </p>
          </div>
          <div className="page-actions">
            <button className="btn btn--ghost" type="button" onClick={onRefresh}>
              <Icon name="refresh" />
              <span>Refresh</span>
            </button>
            <button
              className="btn btn--gold"
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
            >
              <Icon name="plus" />
              <span>{uploading ? "Ingesting…" : "Add documents"}</span>
            </button>
          </div>
        </div>

        <div className="stat-strip">
          <div className="stat">
            <span className="stat-label">Documents</span>
            <span className="stat-value mono">{formatNumber(stats.documents)}</span>
            <span className="stat-foot">
              {stats.formats} {stats.formats === 1 ? "format" : "formats"}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Pages</span>
            <span className="stat-value mono">{formatNumber(stats.pages)}</span>
            <span className="stat-foot">parsed</span>
          </div>
          <div className="stat">
            <span className="stat-label">Chunks</span>
            <span className="stat-value mono">{formatNumber(stats.chunks)}</span>
            <span className="stat-foot">embedded</span>
          </div>
          <div className="stat">
            <span className="stat-label">Vectors</span>
            <span className="stat-value mono">{formatNumber(stats.vectors)}</span>
            <span className="stat-foot">in Qdrant</span>
          </div>
        </div>

        <div className="import-grid">
          <label
            className={`dropzone${dragging ? " is-drag" : ""}`}
            tabIndex={0}
            onDragOver={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              handleFiles(event.dataTransfer.files)
            }}
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              hidden
              onChange={(event) => handleFiles(event.target.files)}
            />
            <span className="drop-icon">
              <Icon name="upload" size={15} />
            </span>
            <span className="drop-title">
              {uploading ? "Ingesting…" : "Drop files to ingest"}
            </span>
            <span className="drop-sub">
              PDF, Word, Excel, CSV, PowerPoint, HTML, text or scanned images
            </span>
            <span className="drop-link">Browse files</span>
          </label>

          <div className="import-side">
            <div className="import-card import-card--static">
              <span className="corpus-icon">
                <Icon name="database" size={14} />
              </span>
              <span>
                <span className="import-title">Storage</span>
                <span className="import-sub">
                  Qdrant vectors + SQLite document registry
                </span>
              </span>
            </div>
            <div className="import-card import-card--static">
              <span className="corpus-icon">
                <Icon name="layers" size={14} />
              </span>
              <span>
                <span className="import-title">Retrieval</span>
                <span className="import-sub">Dense + BM25, fused with RRF</span>
              </span>
            </div>
          </div>
        </div>

        <div className="library-toolbar">
          <div className="filter-chips" role="group" aria-label="Filter by format">
            {FILTERS.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`chip${filter === item.key ? " is-active" : ""}`}
                onClick={() => setFilter(item.key)}
              >
                {item.label}
                <span className="mono">{stats.byKind[item.key]}</span>
              </button>
            ))}
          </div>
          <div className="table-search">
            <Icon name="search" size={13} />
            <input
              type="search"
              value={query}
              placeholder="Filter documents"
              aria-label="Filter documents"
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
        </div>

        <div className="table-shell">
          <table className="doc-table">
            <thead>
              <tr>
                <th scope="col">Document</th>
                <th scope="col">Format</th>
                <th scope="col" className="num">
                  Pages
                </th>
                <th scope="col" className="num">
                  Chunks
                </th>
                <th scope="col" className="num">
                  Vectors
                </th>
                <th scope="col">Ingested</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((doc) => (
                <tr key={doc.id}>
                  <td>
                    <div className="doc-cell">
                      <span className="doc-tile" aria-hidden="true">
                        <Icon name={doc.icon} size={14} />
                      </span>
                      <span className="doc-copy">
                        <span className="doc-name" title={doc.filename}>
                          {doc.filename}
                        </span>
                        <span className="doc-meta mono">
                          ingest {formatDuration(doc.duration_s)}
                        </span>
                      </span>
                    </div>
                  </td>
                  <td>
                    <span className="format-badge">{doc.format}</span>
                  </td>
                  <td className="num">{formatNumber(doc.pages)}</td>
                  <td className="num">{formatNumber(doc.chunks)}</td>
                  <td className="num">{formatNumber(doc.vectors)}</td>
                  <td className="doc-date">{formatDate(doc.created_at)}</td>
                  <td className="cell-actions">
                    <button
                      type="button"
                      className="icon-btn row-menu"
                      aria-label={`Delete ${doc.filename}`}
                      onClick={() => onDelete(doc)}
                    >
                      <Icon name="trash" size={14} />
                    </button>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr className="empty-row">
                  <td colSpan={7}>
                    {documents.length === 0
                      ? "No documents indexed for this tenant yet."
                      : "No document matches this filter."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          <div className="table-foot">
            <span>
              Showing {rows.length} of {documents.length} documents
            </span>
            <span className="mono">Dense + BM25 · RRF fusion</span>
          </div>
        </div>
      </div>
    </section>
  )
}