import type { DocumentRecord } from "./api"

/**
 * Presentation metadata derived from what the API actually returns.
 *
 * `DocumentOut` carries only id/filename/pages/chunks/vectors/duration/hash/
 * created_at, so format, kind and icon are inferred from the file extension.
 * Nothing here is fabricated: an unknown extension falls back to "other".
 */
export type DocKind = "pdf" | "sheet" | "doc" | "image" | "web" | "other"

export interface DocView extends DocumentRecord {
  ext: string
  format: string
  kind: DocKind
  icon: string
}

const SHEETS = new Set(["xlsx", "xls", "csv", "tsv"])
const DOCS = new Set(["docx", "doc", "pptx", "ppt", "md", "txt", "rtf", "odt"])
const IMAGES = new Set(["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"])
const WEB = new Set(["html", "htm"])

function extensionOf(filename: string): string {
  const parts = filename.split(".")
  return parts.length > 1 ? (parts.pop() as string).toLowerCase() : ""
}

export function toView(doc: DocumentRecord): DocView {
  const ext = extensionOf(doc.filename)
  const kind: DocKind = SHEETS.has(ext)
    ? "sheet"
    : DOCS.has(ext)
      ? "doc"
      : IMAGES.has(ext)
        ? "image"
        : WEB.has(ext)
          ? "web"
          : ext === "pdf"
            ? "pdf"
            : "other"
  const icon =
    kind === "sheet"
      ? "table"
      : kind === "image"
        ? "image"
        : kind === "web"
          ? "globe"
          : ext === "pptx" || ext === "ppt"
            ? "presentation"
            : "file"
  return {
    ...doc,
    ext,
    kind,
    icon,
    format: ext ? ext.toUpperCase() : "FILE",
  }
}

export interface CorpusStats {
  documents: number
  pages: number
  chunks: number
  vectors: number
  formats: number
  byKind: Record<DocKind | "all", number>
}

export function corpusStats(views: DocView[]): CorpusStats {
  const byKind: Record<DocKind | "all", number> = {
    all: views.length,
    pdf: 0,
    sheet: 0,
    doc: 0,
    image: 0,
    web: 0,
    other: 0,
  }
  for (const doc of views) byKind[doc.kind] += 1

  return {
    documents: views.length,
    pages: views.reduce((sum, doc) => sum + (doc.pages || 0), 0),
    chunks: views.reduce((sum, doc) => sum + (doc.chunks || 0), 0),
    vectors: views.reduce((sum, doc) => sum + (doc.vectors || 0), 0),
    formats: new Set(views.map((doc) => doc.format)).size,
    byKind,
  }
}

export function matchesQuery(doc: DocView, query: string): boolean {
  if (!query) return true
  return doc.filename.toLowerCase().includes(query.toLowerCase())
}

/** `2026-10-12T09:31:00+00:00` -> `12 Oct 2026`, or the raw value if unparsable. */
export function formatDate(value: string): string {
  if (!value) return "—"
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  })
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—"
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  return `${seconds.toFixed(1)} s`
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value || 0)
}