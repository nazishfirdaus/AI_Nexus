"""Tests for the website source: SSRF guard, extraction, crawl (no network)."""
import pytest

from src.ingestion.sources import web


# ------------------------------------------------------------------ fakes
class FakeResponse:
    def __init__(self, content=b"", content_type="text/html", status=200):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status

    @property
    def text(self):
        return self.content.decode("utf-8", "replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def html_page(title, body, links=()):
    anchors = "".join(f'<a href="{h}">link</a>' for h in links)
    return (
        f"<html><head><title>{title}</title></head><body><article>"
        f"<h1>{title}</h1><p>{body}</p>{anchors}</article></body></html>"
    ).encode("utf-8")


def make_get(pages):
    def _get(url, headers=None, timeout=None):
        if url not in pages:
            raise RuntimeError("connection refused")
        return pages[url]

    return _get


ALLOW = lambda host: True  # noqa: E731


# ------------------------------------------------------------------ SSRF guard
@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "javascript:alert(1)"])
def test_validate_url_rejects_bad_scheme(url):
    with pytest.raises(web.WebSourceError):
        web.validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.5/",
        "http://192.168.1.10/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ],
)
def test_validate_url_rejects_private_hosts(url):
    with pytest.raises(web.WebSourceError):
        web.validate_url(url)


def test_validate_url_allows_public_host():
    web.validate_url("https://example.com/page", resolver=ALLOW)


# ------------------------------------------------------------------ fetch
def test_fetch_page_extracts_text_and_title():
    pages = {"https://example.com/": FakeResponse(html_page("Loan Terms", "Interest rate is 7.25 percent."))}
    page = web.fetch_page("https://example.com/", get=make_get(pages), resolver=ALLOW)
    assert page.title == "Loan Terms"
    assert "7.25" in page.text


def test_fetch_page_rejects_unsupported_content_type():
    pages = {"https://example.com/a.pdf": FakeResponse(b"%PDF", content_type="application/pdf")}
    with pytest.raises(web.WebSourceError):
        web.fetch_page("https://example.com/a.pdf", get=make_get(pages), resolver=ALLOW)


def test_fetch_page_caps_download_size():
    big = html_page("Big", "x" * 5000)
    pages = {"https://example.com/": FakeResponse(big)}
    page = web.fetch_page("https://example.com/", get=make_get(pages), max_bytes=100, resolver=ALLOW)
    assert len(page.html) <= 100


def test_fetch_page_wraps_network_errors():
    with pytest.raises(web.WebSourceError):
        web.fetch_page("https://example.com/", get=make_get({}), resolver=ALLOW)


# ------------------------------------------------------------------ crawl
def test_site_document_id_is_stable_across_cosmetic_differences():
    a = web.site_document_id("https://Example.com/docs?x=1#frag")
    b = web.site_document_id("https://example.com/docs?x=1")
    assert a == b
    assert a.startswith("web_")


def test_crawl_returns_one_document_per_site():
    pages = {
        "https://example.com/": FakeResponse(
            html_page("Home", "Welcome to the mortgage site.", links=["https://example.com/a"])
        ),
        "https://example.com/a": FakeResponse(html_page("Rates", "Rates page content here.")),
    }
    site = web.crawl_documents(
        "https://example.com/",
        get=make_get(pages),
        resolver=ALLOW,
        sleep=lambda _s: None,
        robots_check=lambda _u: True,
        max_pages=5,
    )
    assert site.title == "Home"
    assert len(site.documents) == 2
    assert {d.metadata["document_id"] for d in site.documents} == {site.document_id}
    assert [d.metadata["page_number"] for d in site.documents] == [1, 2]
    assert site.documents[0].metadata["source"] == "https://example.com/"


def test_crawl_stays_on_domain_and_respects_cap():
    pages = {
        "https://example.com/": FakeResponse(
            html_page(
                "Home",
                "Root page text.",
                links=["https://example.com/a", "https://other.com/b"],
            )
        ),
        "https://example.com/a": FakeResponse(html_page("A", "Page A text.")),
        "https://other.com/b": FakeResponse(html_page("B", "Offsite text.")),
    }
    site = web.crawl_documents(
        "https://example.com/",
        get=make_get(pages),
        resolver=ALLOW,
        sleep=lambda _s: None,
        robots_check=lambda _u: True,
        max_pages=1,
    )
    assert len(site.documents) == 1  # cap respected
    assert all(d.metadata["source"].startswith("https://example.com") for d in site.documents)


def test_crawl_honours_robots():
    pages = {
        "https://example.com/": FakeResponse(
            html_page("Home", "Root text.", links=["https://example.com/blocked"])
        ),
        "https://example.com/blocked": FakeResponse(html_page("Blocked", "Secret.")),
    }
    site = web.crawl_documents(
        "https://example.com/",
        get=make_get(pages),
        resolver=ALLOW,
        sleep=lambda _s: None,
        robots_check=lambda url: "/blocked" not in url,
    )
    assert [d.metadata["source"] for d in site.documents] == ["https://example.com/"]


def test_crawl_skips_unreachable_pages():
    pages = {
        "https://example.com/": FakeResponse(
            html_page("Home", "Root text.", links=["https://example.com/gone"])
        ),
    }
    site = web.crawl_documents(
        "https://example.com/",
        get=make_get(pages),
        resolver=ALLOW,
        sleep=lambda _s: None,
        robots_check=lambda _u: True,
    )
    assert len(site.documents) == 1
