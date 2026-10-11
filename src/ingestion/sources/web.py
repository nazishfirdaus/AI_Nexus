"""Fetch and crawl web pages into ingestable documents.

Two layers:

  * `fetch_page` retrieves a single URL (size-capped) and extracts the readable
    main text with trafilatura.
  * `crawl_documents` does a breadth-first, same-domain crawl from a start URL,
    honouring robots.txt, a politeness delay and a page cap.

Security: every URL is validated before a request - scheme must be http/https and
the host must resolve to a public address, so the crawler cannot be pointed at
localhost, link-local metadata endpoints or other private ranges (SSRF guard).

One crawl becomes ONE document: each page is a "page" of that document
(`page_number` = fetch order), and the `document_id` is derived from the start
URL so re-crawling the same site replaces the previous copy.

The HTTP getter, resolver, sleep and robots check are all injectable so the module
is fully testable without network access.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_CONTENT_TYPES = ("html", "xml", "text/plain")


class WebSourceError(Exception):
    """Raised for an invalid/unsafe URL or an unusable page response."""


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    html: bytes


@dataclass
class CrawledSite:
    documents: list[Document] = field(default_factory=list)
    document_id: str = ""
    title: str = ""


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------
def _default_resolver(host: str) -> bool:
    """True if every address `host` resolves to is public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def validate_url(url: str, *, resolver: Optional[Callable[[str], bool]] = None) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise WebSourceError(f"Unsupported URL scheme: {parsed.scheme or '(none)'!r}")
    if not parsed.hostname:
        raise WebSourceError("URL has no host")
    check = resolver or _default_resolver
    if not check(parsed.hostname):
        raise WebSourceError(f"Refusing to fetch non-public host: {parsed.hostname}")


# ---------------------------------------------------------------------------
# Single page fetch
# ---------------------------------------------------------------------------
def _default_get(url, **kwargs):
    return requests.get(url, **kwargs)


def extract_text(html: bytes | str, url: str) -> str:
    """Readable main text via trafilatura; falls back to raw text for plain files."""
    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True
        )
    except Exception:
        logger.debug("trafilatura extraction failed for %s", url, exc_info=True)
        text = None
    if text and text.strip():
        return text.strip()
    if isinstance(html, bytes):
        return html.decode("utf-8", errors="replace").strip()
    return str(html).strip()


def _html_title(html: bytes, url: str) -> str:
    try:
        from lxml import html as lxml_html

        tree = lxml_html.fromstring(html)
        titles = tree.xpath("//title/text()")
        if titles and titles[0].strip():
            return titles[0].strip()
    except Exception:
        logger.debug("Could not read <title> of %s", url, exc_info=True)
    return url


def fetch_page(
    url: str,
    *,
    get: Optional[Callable] = None,
    timeout: float = settings.WEB_FETCH_TIMEOUT_S,
    max_bytes: int = settings.WEB_MAX_BYTES,
    resolver: Optional[Callable[[str], bool]] = None,
) -> WebPage:
    """Download one page and extract its text. Raises WebSourceError on failure."""
    validate_url(url, resolver=resolver)
    get = get or _default_get
    try:
        response = get(url, headers={"User-Agent": settings.WEB_USER_AGENT}, timeout=timeout)
        response.raise_for_status()
    except WebSourceError:
        raise
    except Exception as exc:
        raise WebSourceError(f"Could not fetch {url}: {exc}") from exc

    raw = response.content or b""
    if max_bytes is not None and len(raw) > max_bytes:
        raw = raw[:max_bytes]

    content_type = (response.headers or {}).get("Content-Type", "")
    if content_type and not any(t in content_type.lower() for t in _CONTENT_TYPES):
        raise WebSourceError(f"Unsupported content type at {url}: {content_type}")

    text = extract_text(raw, url)
    if not text:
        raise WebSourceError(f"No readable text at {url}")
    return WebPage(url=url, title=_html_title(raw, url), text=text, html=raw)


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------
class RobotsGate:
    """Caches one robots.txt parser per origin; fails open if it cannot be read."""

    def __init__(self, get: Callable, user_agent: str) -> None:
        self._get = get
        self._ua = user_agent
        self._parsers: dict[str, object] = {}

    def _load(self, origin: str):
        from urllib.robotparser import RobotFileParser

        parser = RobotFileParser()
        try:
            resp = self._get(origin + "/robots.txt", headers={"User-Agent": self._ua}, timeout=5)
            if getattr(resp, "status_code", 200) >= 400:
                return None
            parser.parse((resp.text or "").splitlines())
        except Exception:
            return None
        return parser

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._parsers:
            self._parsers[origin] = self._load(origin)
        parser = self._parsers[origin]
        if parser is None:
            return True
        return bool(parser.can_fetch(self._ua, url))


# ---------------------------------------------------------------------------
# Link discovery + crawl
# ---------------------------------------------------------------------------
def _extract_links(html: bytes, base_url: str) -> list[str]:
    try:
        from lxml import html as lxml_html

        tree = lxml_html.fromstring(html)
    except Exception:
        return []
    links: list[str] = []
    for href in tree.xpath("//a/@href"):
        if not isinstance(href, str) or not href.strip():
            continue
        absolute = urljoin(base_url, href.strip())
        parsed = urlparse(absolute)
        if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.hostname:
            continue
        links.append(urlunparse(parsed._replace(fragment="")))
    return links


def site_document_id(url: str) -> str:
    """Deterministic id for a site, based on its normalised start URL."""
    parsed = urlparse(url)
    normalized = urlunparse(
        parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(), fragment="")
    )
    return "web_" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def crawl_documents(
    start_url: str,
    *,
    get: Optional[Callable] = None,
    max_pages: int = settings.WEB_MAX_PAGES,
    same_domain: bool = True,
    delay: float = settings.WEB_CRAWL_DELAY_S,
    timeout: float = settings.WEB_FETCH_TIMEOUT_S,
    max_bytes: int = settings.WEB_MAX_BYTES,
    resolver: Optional[Callable[[str], bool]] = None,
    sleep: Optional[Callable[[float], None]] = None,
    robots_check: Optional[Callable[[str], bool]] = None,
) -> CrawledSite:
    """Breadth-first crawl of `start_url`; returns one multi-page document."""
    get = get or _default_get
    sleep = sleep or time.sleep
    max_pages = max_pages or settings.WEB_MAX_PAGES
    validate_url(start_url, resolver=resolver)
    robots = robots_check or RobotsGate(get, settings.WEB_USER_AGENT).can_fetch

    start_host = urlparse(start_url).hostname
    document_id = site_document_id(start_url)
    seen = {start_url}
    queue: deque[str] = deque([start_url])
    documents: list[Document] = []
    site_title: Optional[str] = None

    while queue and len(documents) < max_pages:
        url = queue.popleft()
        if not robots(url):
            continue
        try:
            page = fetch_page(
                url, get=get, timeout=timeout, max_bytes=max_bytes, resolver=resolver
            )
        except WebSourceError as exc:
            logger.info("Skipping %s: %s", url, exc)
            continue

        if site_title is None:
            site_title = page.title
        documents.append(
            Document(
                page_content=page.text,
                metadata={
                    "document_id": document_id,
                    "filename": site_title,
                    "source": url,
                    "url": url,
                    "page_number": len(documents) + 1,
                },
            )
        )
        for link in _extract_links(page.html, url):
            if link in seen:
                continue
            if same_domain and urlparse(link).hostname != start_host:
                continue
            seen.add(link)
            queue.append(link)

        if queue and len(documents) < max_pages:
            sleep(delay)

    return CrawledSite(
        documents=documents,
        document_id=document_id,
        title=site_title or (start_host or start_url),
    )
