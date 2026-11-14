import type { ReactNode } from "react"

/**
 * Minimal, dependency-free Markdown renderer for model answers.
 *
 * Only constructs the assistant actually emits are supported: paragraphs,
 * `**bold**`, `*italic*`, inline code, fenced code, headings, blockquotes and
 * bullet/numbered lists. Everything is rendered as React nodes, so raw HTML in
 * a model response is displayed as text rather than injected into the page.
 *
 * Incomplete markers (for example `**Loa` mid-stream) are left literal until
 * the closing marker arrives.
 */

type Block =
  | { kind: "p"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "heading"; level: number; text: string }
  | { kind: "quote"; text: string }
  | { kind: "code"; text: string }
  | { kind: "hr" }

const INLINE_RE =
  /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|\[[^\]\n]+\]\((?:https?:\/\/|mailto:)[^)\s]+\))/g

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  INLINE_RE.lastIndex = 0

  while ((match = INLINE_RE.exec(text))) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    const token = match[0]
    const key = `${keyPrefix}-${match.index}`

    if (token.startsWith("`")) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>)
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith("*")) {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>)
    } else {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token)
      if (link) {
        nodes.push(
          <a key={key} href={link[2]} target="_blank" rel="noreferrer noopener">
            {link[1]}
          </a>
        )
      } else {
        nodes.push(token)
      }
    }

    last = match.index + token.length
  }

  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

function isBlockStart(line: string): boolean {
  const trimmed = line.trim()
  return (
    /^#{1,6}\s+/.test(trimmed) ||
    trimmed.startsWith("```") ||
    /^[-*+]\s+/.test(trimmed) ||
    /^\d+[.)]\s+/.test(trimmed) ||
    /^>\s?/.test(trimmed) ||
    /^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)
  )
}

function parseBlocks(text: string): Block[] {
  const lines = text.replace(/\r\n?/g, "\n").split("\n")
  const blocks: Block[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index]
    const trimmed = line.trim()
    if (!trimmed) {
      index += 1
      continue
    }

    if (trimmed.startsWith("```")) {
      index += 1
      const code: string[] = []
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        code.push(lines[index])
        index += 1
      }
      if (index < lines.length) index += 1
      blocks.push({ kind: "code", text: code.join("\n") })
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed)
    if (heading) {
      blocks.push({
        kind: "heading",
        level: heading[1].length,
        text: heading[2],
      })
      index += 1
      continue
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push({ kind: "hr" })
      index += 1
      continue
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      const items: string[] = []
      while (index < lines.length && /^[-*+]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*+]\s+/, ""))
        index += 1
      }
      blocks.push({ kind: "list", ordered: false, items })
      continue
    }

    if (/^\d+[.)]\s+/.test(trimmed)) {
      const items: string[] = []
      while (index < lines.length && /^\d+[.)]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+[.)]\s+/, ""))
        index += 1
      }
      blocks.push({ kind: "list", ordered: true, items })
      continue
    }

    if (/^>\s?/.test(trimmed)) {
      const quote: string[] = []
      while (index < lines.length && /^>\s?/.test(lines[index].trim())) {
        quote.push(lines[index].trim().replace(/^>\s?/, ""))
        index += 1
      }
      blocks.push({ kind: "quote", text: quote.join(" ") })
      continue
    }

    const paragraph: string[] = [trimmed]
    index += 1
    while (
      index < lines.length &&
      lines[index].trim() &&
      !isBlockStart(lines[index])
    ) {
      paragraph.push(lines[index].trim())
      index += 1
    }
    blocks.push({ kind: "p", text: paragraph.join(" ") })
  }

  return blocks
}

export function Markdown({ text }: { text: string }) {
  return (
    <>
      {parseBlocks(text).map((block, blockIndex) => {
        switch (block.kind) {
          case "p":
            return <p key={blockIndex}>{renderInline(block.text, `p-${blockIndex}`)}</p>
          case "list": {
            const items = block.items.map((item, itemIndex) => (
              <li key={itemIndex}>
                {renderInline(item, `li-${blockIndex}-${itemIndex}`)}
              </li>
            ))
            return block.ordered ? (
              <ol key={blockIndex}>{items}</ol>
            ) : (
              <ul key={blockIndex}>{items}</ul>
            )
          }
          case "heading": {
            const Tag = block.level <= 2 ? "h3" : "h4"
            return <Tag key={blockIndex}>{renderInline(block.text, `h-${blockIndex}`)}</Tag>
          }
          case "quote":
            return (
              <blockquote key={blockIndex}>
                {renderInline(block.text, `q-${blockIndex}`)}
              </blockquote>
            )
          case "code":
            return (
              <pre key={blockIndex}>
                <code>{block.text}</code>
              </pre>
            )
          case "hr":
            return <hr key={blockIndex} />
          default:
            return null
        }
      })}
    </>
  )
}
