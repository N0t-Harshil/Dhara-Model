from __future__ import annotations

import gzip
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; DocBuilder/1.0; +https://github.com/anomalyco/specialized-coding-model)"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(total=MAX_RETRIES, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _clean_text(html: str) -> str:
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'\s+', ' ', html).strip()
    return html


def _extract_section_text(html: str, section_tag: str = "main", fallback: str = "body") -> str:
    m = re.search(f'<{section_tag}[^>]*>(.*?)</{section_tag}>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return _clean_text(m.group(1))
    if fallback:
        m = re.search(f'<{fallback}[^>]*>(.*?)</{fallback}>', html, re.DOTALL | re.IGNORECASE)
        if m:
            return _clean_text(m.group(1))
    return _clean_text(html)


def _find_client_redirect(html: str, base_url: str) -> Optional[str]:
    """Follow <meta refresh> and <script>location.replace|href client-side redirects."""
    m = re.search(
        r'<meta\s+http-equiv=["\']refresh["\'][^>]*content=["\']\d+;\s*url=([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if m:
        return urljoin(base_url, m.group(1))
    m = re.search(r'location\.(?:replace|assign|href)\s*(?:=\s*)?\(?\s*["\']([^"\']+)["\']', html)
    if m:
        return urljoin(base_url, m.group(1))
    return None


_TEMPLATE_RE = re.compile(r'\{\{.*?\}\}|{%[-+]?.*?[-+]?%}|\$\{.*?\}|<%.*?%>', re.DOTALL)


def _has_template_syntax(text: str) -> bool:
    return bool(_TEMPLATE_RE.search(text))


def _is_invalid_href(href: str) -> bool:
    stripped = href.strip()
    if not stripped or stripped == '#':
        return True
    if stripped.lower().startswith('javascript:'):
        return True
    if re.match(r'void\s*\(\s*0?\s*\)', stripped, re.IGNORECASE):
        return True
    return False


def _fetch(url: str, session: Optional[requests.Session] = None, timeout: Optional[int] = None) -> Optional[Tuple[str, str]]:
    s = session or _session()
    current_url = url
    for attempt in range(4):
        try:
            r = s.get(current_url, timeout=timeout or REQUEST_TIMEOUT)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                logger.warning("Client error %d fetching %s, not retrying", r.status_code, current_url)
                return None
            r.raise_for_status()
            html = r.text
            redirect_url = _find_client_redirect(html, r.url)
            if redirect_url and redirect_url != r.url:
                logger.debug("Client redirect %s -> %s", r.url, redirect_url)
                current_url = redirect_url
                continue
            return (html, r.url)
        except requests.exceptions.Timeout as e:
            logger.warning("Timeout fetching %s (attempt %d/4): %s", current_url, attempt + 1, e)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500 and status != 429:
                logger.warning("Client error %d fetching %s, not retrying", status, current_url)
                return None
            logger.warning("HTTP error fetching %s (attempt %d/4): %s", current_url, attempt + 1, e)
        except requests.exceptions.ConnectionError as e:
            logger.warning("Connection error fetching %s (attempt %d/4): %s", current_url, attempt + 1, e)
        except Exception as e:
            logger.warning("Error fetching %s (attempt %d/4): %s: %s", current_url, attempt + 1, type(e).__name__, e)
        current_url = url
    logger.error("All 4 attempts failed for %s", url)
    return None


@dataclass
class DocPage:
    url: str
    title: str
    text: str
    sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceConfig:
    name: str
    base_url: str
    language: str = "text"
    category: str = "docs"
    url_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    content_selector: str = "main"
    content_fallback: str = "body"
    max_pages: int = 5000
    min_text_length: int = 200
    max_text_length: int = 100000
    request_timeout: int = 30
    retry_count: int = 4


class DocScraper(ABC):
    SOURCE_NAME: str = ""
    BASE_URL: str = ""
    LANGUAGE: str = "text"
    CATEGORY: str = "docs"
    MAX_PAGES: int = 5000
    MIN_TEXT_LENGTH: int = 200
    MAX_TEXT_LENGTH: int = 100000

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) / self.SOURCE_NAME if output_dir else None
        self._session = _session()
        self._visited: Set[str] = set()
        self._failed: Set[str] = set()
        self._discovered_count: int = 0
        self._duplicate_count: int = 0
        self._failed_count: int = 0
        self._filtered_count: int = 0

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.split("#")[0]
        parsed = urlparse(url)
        if parsed.path.endswith("/") and not parsed.path.endswith(".html/"):
            url = url.rstrip("/")
        return url

    @abstractmethod
    def discover_urls(self) -> Generator[str, None, None]:
        ...

    @abstractmethod
    def extract_text(self, html: str, url: str) -> Optional[str]:
        ...

    @classmethod
    def get_config(cls) -> SourceConfig:
        return SourceConfig(
            name=cls.SOURCE_NAME,
            base_url=cls.BASE_URL,
            language=cls.LANGUAGE,
            category=cls.CATEGORY,
            max_pages=cls.MAX_PAGES,
            min_text_length=cls.MIN_TEXT_LENGTH,
            max_text_length=cls.MAX_TEXT_LENGTH,
        )

    def scrape(self) -> Generator[DocPage, None, None]:
        count = 0
        seen: Set[str] = set()
        for raw_url in self.discover_urls():
            url = self._normalize_url(raw_url)
            if url in seen:
                self._duplicate_count += 1
                continue
            seen.add(url)
            if url in self._visited or url in self._failed:
                self._duplicate_count += 1
                continue
            if _has_template_syntax(url) or _is_invalid_href(url):
                self._filtered_count += 1
                logger.info("Filtered template URL: %s", url)
                continue
            self._discovered_count += 1
            if count >= self.MAX_PAGES:
                break
            self._visited.add(url)
            result = _fetch(url, self._session)
            if not result:
                self._failed.add(url)
                self._failed_count += 1
                continue
            html, _ = result
            text = self.extract_text(html, url)
            if not text or len(text) < self.MIN_TEXT_LENGTH:
                continue
            if len(text) > self.MAX_TEXT_LENGTH:
                text = text[:self.MAX_TEXT_LENGTH]
            title = self._extract_title(html) or url
            count += 1
            yield DocPage(url=url, title=title, text=text)

    def _extract_title(self, html: str) -> Optional[str]:
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
        if m:
            return _clean_text(m.group(1))
        return None

    def _summary(self, count: int) -> str:
        parts = [f"{count} pages"]
        if self._discovered_count:
            parts.append(f"discovered={self._discovered_count}")
        if self._duplicate_count:
            parts.append(f"dups={self._duplicate_count}")
        if self._filtered_count:
            parts.append(f"filtered={self._filtered_count}")
        if self._failed_count:
            parts.append(f"failed={self._failed_count}")
        return " | ".join(parts)

    def scrape_to_jsonl(self, output_path: Path) -> int:
        count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8") as f:
            for page in self.scrape():
                rec = {
                    "source": self.SOURCE_NAME,
                    "language": self.LANGUAGE,
                    "category": self.CATEGORY,
                    "title": page.title,
                    "url": page.url,
                    "text": page.text,
                    "text_length": len(page.text),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        logger.info("  %s done: %s", self.SOURCE_NAME, self._summary(count))
        return count


class SphinxScraper(DocScraper):
    """Base for Sphinx-generated docs with predictable URL patterns."""

    URL_PATTERNS: List[str] = []
    EXCLUDE_PATTERNS: List[str] = []
    CONTENT_SELECTOR: str = "main"
    CONTENT_FALLBACK: str = "body"

    @classmethod
    def get_config(cls) -> SourceConfig:
        config = super().get_config()
        config.url_patterns = cls.URL_PATTERNS
        config.exclude_patterns = cls.EXCLUDE_PATTERNS
        config.content_selector = cls.CONTENT_SELECTOR
        config.content_fallback = cls.CONTENT_FALLBACK
        return config

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        seeds = [self.BASE_URL] + [urljoin(self.BASE_URL, p) for p in self.URL_PATTERNS]
        for seed in seeds:
            norm = self._normalize_url(seed)
            if norm not in discovered and norm not in self._failed:
                discovered.add(norm)
                yield norm
        for seed in seeds:
            result = _fetch(seed, self._session)
            if not result:
                continue
            html, final_url = result
            base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
            for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
                href = match.group(1)
                if href.startswith("#") or href.startswith("mailto:"):
                    continue
                if any(ex in href for ex in self.EXCLUDE_PATTERNS):
                    continue
                full = self._normalize_url(urljoin(final_url, href))
                if not full.startswith(self.BASE_URL) and not full.startswith(base):
                    continue
                if _has_template_syntax(full) or _is_invalid_href(href):
                    self._filtered_count += 1
                    logger.info("Filtered template URL: %s", full)
                    continue
                if full in discovered or full in self._failed:
                    continue
                discovered.add(full)
                yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        text = _extract_section_text(html, self.CONTENT_SELECTOR, self.CONTENT_FALLBACK)
        if len(text) < self.MIN_TEXT_LENGTH:
            return None
        return text


class PythonDocScraper(SphinxScraper):
    SOURCE_NAME = "python-docs"
    BASE_URL = "https://docs.python.org/3/"
    LANGUAGE = "python"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "tutorial/", "library/", "reference/", "howto/",
        "using/", "whatsnew/", "faq/",
    ]
    EXCLUDE_PATTERNS = [".pdf", ".txt", "_sources", "genindex", "modindex", "search.html", "glossary.html"]


class PyTorchDocScraper(SphinxScraper):
    SOURCE_NAME = "pytorch-docs"
    BASE_URL = "https://pytorch.org/docs/stable/"
    LANGUAGE = "python"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "tensors.html", "torch.html", "nn.html", "optim.html", "data.html",
        "autograd.html", "onnx.html", "distributed.html", "jit.html",
        "amp.html", "quantization.html", "sparse.html", "cuda.html",
        "index.html",
    ]
    EXCLUDE_PATTERNS = [".pdf", ".txt", "_sources", "genindex", "modindex", "search.html"]

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        for pattern in [self.BASE_URL] + self.URL_PATTERNS:
            url = self._normalize_url(urljoin(self.BASE_URL, pattern))
            if url not in discovered:
                discovered.add(url)
                yield url
        genindex = urljoin(self.BASE_URL, "genindex.html")
        result = _fetch(genindex, self._session)
        if result:
            html, final_url = result
            base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
            for m in re.finditer(r'href=["\']([^"\']+\.html)["\']', html):
                href = m.group(1)
                if any(ex in href for ex in self.EXCLUDE_PATTERNS):
                    continue
                full = self._normalize_url(urljoin(final_url, href))
                if not full.startswith(self.BASE_URL) and not full.startswith(base):
                    continue
                if _has_template_syntax(full) or _is_invalid_href(href):
                    self._filtered_count += 1
                    logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                    continue
                if full in discovered or full in self._failed:
                    continue
                discovered.add(full)
                yield full


class NumPyDocScraper(SphinxScraper):
    SOURCE_NAME = "numpy-docs"
    BASE_URL = "https://numpy.org/doc/stable/"
    LANGUAGE = "python"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "user/", "reference/", "dev/",
    ]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html", "glossary.html"]


class FastAPIDocScraper(SphinxScraper):
    SOURCE_NAME = "fastapi-docs"
    BASE_URL = "https://fastapi.tiangolo.com/"
    LANGUAGE = "python"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "tutorial/", "advanced/", "reference/", "deployment/",
        "how-to/", "project-generation/", "alternatives/",
    ]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html", "overrides"]


class OpenCVDocScraper(SphinxScraper):
    SOURCE_NAME = "opencv-docs"
    BASE_URL = "https://docs.opencv.org/4.x/"
    LANGUAGE = "cpp"
    CATEGORY = "docs"
    URL_PATTERNS: List[str] = []
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html"]


class ONNXDocScraper(SphinxScraper):
    SOURCE_NAME = "onnx-docs"
    BASE_URL = "https://onnx.ai/onnx/"
    LANGUAGE = "python"
    CATEGORY = "docs"
    URL_PATTERNS = ["intro/", "api/", "operators/"]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html"]


class DockerDocScraper(SphinxScraper):
    SOURCE_NAME = "docker-docs"
    BASE_URL = "https://docs.docker.com/"
    LANGUAGE = "shell"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "get-started/", "build/", "compose/", "engine/", "network/",
        "storage/", "config/", "admin/", "desktop/", "docker-hub/",
        "reference/",
    ]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html", "search/"]


class KubernetesDocScraper(SphinxScraper):
    SOURCE_NAME = "kubernetes-docs"
    BASE_URL = "https://kubernetes.io/docs/"
    LANGUAGE = "shell"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "setup/", "concepts/", "tasks/", "tutorials/", "reference/",
    ]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html"]


class PostgreSQLDocScraper(SphinxScraper):
    SOURCE_NAME = "postgresql-docs"
    BASE_URL = "https://www.postgresql.org/docs/current/"
    LANGUAGE = "sql"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "tutorial-", "sql-", "functions-", "runtime-", "admin-",
        "monitoring-", "app-", "reference-",
    ]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html"]

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        norm = self._normalize_url(self.BASE_URL)
        if norm not in discovered and norm not in self._failed:
            discovered.add(norm)
            yield norm
        result = _fetch(self.BASE_URL, self._session)
        if result:
            html, final_url = result
            base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
            for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                href = m.group(1)
                if not href.endswith(".html") and not href.endswith("/"):
                    continue
                if not re.search(r'/docs/(current|\d+)/', href):
                    continue
                if any(ex in href for ex in self.EXCLUDE_PATTERNS):
                    continue
                full = self._normalize_url(urljoin(final_url, href))
                if not full.startswith("https://www.postgresql.org/docs/"):
                    continue
                if _has_template_syntax(full) or _is_invalid_href(href):
                    self._filtered_count += 1
                    logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                    continue
                if full not in discovered and full not in self._failed:
                    discovered.add(full)
                    yield full


class SQLiteDocScraper(DocScraper):
    SOURCE_NAME = "sqlite-docs"
    BASE_URL = "https://sqlite.org/docs.html"
    LANGUAGE = "sql"
    CATEGORY = "docs"

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        url = self._normalize_url(self.BASE_URL)
        discovered.add(url)
        yield url
        result = _fetch(self.BASE_URL, self._session)
        if result:
            html, final_url = result
            for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                href = m.group(1)
                full = self._normalize_url(urljoin(final_url, href))
                if _has_template_syntax(full) or _is_invalid_href(href):
                    self._filtered_count += 1
                    logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                    continue
                if not full.startswith("https://sqlite.org/"):
                    continue
                if any(ex in full for ex in [".pdf", ".txt", ".zip", ".tar"]):
                    continue
                if full in discovered or full in self._failed:
                    continue
                discovered.add(full)
                yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        return _extract_section_text(html, "body")


class MDNDocScraper(DocScraper):
    SOURCE_NAME = "mdn-docs"
    BASE_URL = "https://developer.mozilla.org/en-US/docs/Web"
    LANGUAGE = "javascript"
    CATEGORY = "docs"

    # Hardcoded core pages ensures minimum yield even if JS nav fails
    CORE_TOPICS: List[str] = [
        "HTML", "CSS", "JavaScript", "HTTP", "Web/API", "Web/Guide",
        "Web/SVG", "Web/MathML", "Web/Web_Components", "Web/Events",
        "Web/Performance", "Web/Security", "Web/Accessibility",
    ]

    DEPRECATED_PREFIXES: List[str] = [
        "Web/API/SVGPathSeg", "Web/API/SVGAnimatedPathData",
        "Web/API/GestureEvent", "Web/API/Microsoft",
        "Web/API/EXT_disjoint_timer_query_webgl2",
        "Web/API/SVGViewSpec", "Web/API/SVGUseElementShadowRoot",
        "Web/API/ShadowAnimation", "Web/API/SVGSVGElement/",
    ]

    def _mdn_url(self, topic: str) -> str:
        return f"https://developer.mozilla.org/en-US/docs/{topic}"

    def _is_deprecated(self, path: str) -> bool:
        for prefix in self.DEPRECATED_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        for topic in self.CORE_TOPICS:
            url = self._mdn_url(topic)
            if url not in discovered:
                discovered.add(url)
                yield url

        result = _fetch(self.BASE_URL, self._session)
        if result:
            html, _ = result
            for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                href = m.group(1)
                if not href.startswith("/en-US/docs/Web/"):
                    continue
                full = self._normalize_url(f"https://developer.mozilla.org{href}")
                if full in discovered or href == "/en-US/docs/Web":
                    continue
                if self._is_deprecated(href.replace("/en-US/docs/", "")):
                    continue
                if full in self._failed:
                    continue
                discovered.add(full)
                yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        return _extract_section_text(html, "article", "main")


class RustBookScraper(DocScraper):
    SOURCE_NAME = "rust-book"
    BASE_URL = "https://doc.rust-lang.org/book/"
    LANGUAGE = "rust"
    CATEGORY = "docs"

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        url = self._normalize_url(self.BASE_URL)
        discovered.add(url)
        yield url
        result = _fetch(self.BASE_URL, self._session)
        if result:
            html, final_url = result
            base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
            for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                href = m.group(1)
                if not href.endswith(".html"):
                    continue
                full = self._normalize_url(urljoin(final_url, href))
                if not full.startswith(self.BASE_URL) and not full.startswith(base):
                    continue
                if _has_template_syntax(full) or _is_invalid_href(href):
                    self._filtered_count += 1
                    logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                    continue
                if full in discovered or full in self._failed:
                    continue
                discovered.add(full)
                yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        return _extract_section_text(html, "main", "body")


class GoDocScraper(SphinxScraper):
    SOURCE_NAME = "go-docs"
    BASE_URL = "https://go.dev/doc/"
    LANGUAGE = "go"
    CATEGORY = "docs"
    URL_PATTERNS = [
        "tutorial/", "effective_go", "faq", "wiki/",
    ]
    EXCLUDE_PATTERNS = [".pdf", "_sources", "genindex", "modindex", "search.html"]


class RFCDocScraper(DocScraper):
    """Scrape IETF RFCs (technical specifications for long-context training)."""
    SOURCE_NAME = "rfcs"
    BASE_URL = "https://www.rfc-editor.org/rfc/"
    LANGUAGE = "text"
    CATEGORY = "long_context"
    MIN_TEXT_LENGTH = 5000
    MAX_TEXT_LENGTH = 500000

    def discover_urls(self) -> Generator[str, None, None]:
        result = _fetch("https://www.ietf.org/rfc/rfc-index.txt", self._session)
        if result:
            txt, _ = result
            for i, m in enumerate(re.finditer(r'^(\d{4,5})\s+', txt, re.MULTILINE)):
                if i >= self.MAX_PAGES:
                    break
                rfc_num = m.group(1).lstrip("0") or "0"
                yield f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"

    def extract_text(self, html: str, url: str) -> Optional[str]:
        # RFC text files are plain text, not HTML — no tag cleaning needed
        text = html.strip()
        if len(text) < self.MIN_TEXT_LENGTH:
            return None
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
        return text


class LinuxKernelDocScraper(DocScraper):
    """Scrape Linux kernel documentation."""
    SOURCE_NAME = "linux-kernel-docs"
    BASE_URL = "https://www.kernel.org/doc/html/latest/"
    LANGUAGE = "text"
    CATEGORY = "long_context"
    MIN_TEXT_LENGTH = 500
    MAX_TEXT_LENGTH = 200000

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        url = self._normalize_url(self.BASE_URL)
        discovered.add(url)
        yield url
        result = _fetch(self.BASE_URL, self._session)
        if result:
            html, final_url = result
            base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
            for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                href = m.group(1)
                if href.startswith("#") or ".pdf" in href:
                    continue
                full = self._normalize_url(urljoin(final_url, href))
                if not full.startswith(self.BASE_URL) and not full.startswith(base):
                    continue
                if _has_template_syntax(full) or _is_invalid_href(href):
                    self._filtered_count += 1
                    logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                    continue
                if full in discovered or full in self._failed:
                    continue
                discovered.add(full)
                yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        return _extract_section_text(html, "div.document", "body")


class LangSpecScraper(DocScraper):
    """Scrape programming language specifications."""
    SOURCE_NAME = "lang-specs"
    BASE_URL = "https://docs.python.org/3/reference/"
    LANGUAGE = "text"
    CATEGORY = "long_context"
    MIN_TEXT_LENGTH = 3000

    def discover_urls(self) -> Generator[str, None, None]:
        python_ref = [
            "https://docs.python.org/3/reference/index.html",
            "https://docs.python.org/3/reference/lexical_analysis.html",
            "https://docs.python.org/3/reference/datamodel.html",
            "https://docs.python.org/3/reference/executionmodel.html",
            "https://docs.python.org/3/reference/expressions.html",
            "https://docs.python.org/3/reference/simple_stmts.html",
            "https://docs.python.org/3/reference/compound_stmts.html",
            "https://docs.python.org/3/reference/import.html",
            "https://docs.python.org/3/reference/grammar.html",
        ]
        for url in python_ref:
            yield url
        rust_ref = [
            "https://doc.rust-lang.org/reference/introduction.html",
            "https://doc.rust-lang.org/reference/notation.html",
            "https://doc.rust-lang.org/reference/lexical-structure.html",
            "https://doc.rust-lang.org/reference/types.html",
            "https://doc.rust-lang.org/reference/expressions.html",
            "https://doc.rust-lang.org/reference/statements.html",
            "https://doc.rust-lang.org/reference/items.html",
            "https://doc.rust-lang.org/reference/attributes.html",
            "https://doc.rust-lang.org/reference/macros.html",
        ]
        for url in rust_ref:
            yield url

    def extract_text(self, html: str, url: str) -> Optional[str]:
        return _extract_section_text(html, "main", "body")


CUDA_DOC_URLS = [
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
    "https://docs.nvidia.com/cuda/cuda-runtime-api/index.html",
    "https://docs.nvidia.com/cuda/cuda-driver-api/index.html",
    "https://docs.nvidia.com/cuda/cuda-math-api/index.html",
    "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
    "https://docs.nvidia.com/cuda/cublas/index.html",
    "https://docs.nvidia.com/cuda/curand/index.html",
    "https://docs.nvidia.com/cuda/cusolver/index.html",
    "https://docs.nvidia.com/cuda/cusparse/index.html",
    "https://docs.nvidia.com/cuda/cufft/index.html",
    "https://docs.nvidia.com/cuda/nvrtc/index.html",
    "https://docs.nvidia.com/cuda/nvml-api/index.html",
    "https://docs.nvidia.com/cuda/thrust/index.html",
    # Additional seed pages for broader coverage
    "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html",
    "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html",
    "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html",
    "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
    "https://docs.nvidia.com/cuda/cuda-c++-programming-guide/index.html",
]


class CUDADocScraper(DocScraper):
    SOURCE_NAME = "cuda-docs"
    BASE_URL = "https://docs.nvidia.com/cuda/"
    LANGUAGE = "cpp"
    CATEGORY = "docs"
    CUDA_PAGES = CUDA_DOC_URLS

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        for url in self.CUDA_PAGES:
            norm = self._normalize_url(url)
            if norm not in discovered and norm not in self._failed:
                discovered.add(norm)
                yield norm
            result = _fetch(norm, self._session)
            if result:
                html, final_url = result
                base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
                for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                    href = m.group(1)
                    if not href.endswith(".html"):
                        continue
                    full = self._normalize_url(urljoin(final_url, href))
                    if not full.startswith(self.BASE_URL) and not full.startswith(base):
                        continue
                    if _has_template_syntax(full) or _is_invalid_href(href):
                        self._filtered_count += 1
                        logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                        continue
                    if full in discovered or full in self._failed:
                        continue
                    discovered.add(full)
                    yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        return _extract_section_text(html, "main", "body")


class CuDNNDocScraper(DocScraper):
    SOURCE_NAME = "cudnn-docs"
    BASE_URL = "https://docs.nvidia.com/deeplearning/cudnn/latest/"
    LANGUAGE = "cpp"
    CATEGORY = "docs"
    CUDNN_PAGES = [
        "https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html",
        "https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-graph-library.html",
        "https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-ops-library.html",
        "https://docs.nvidia.com/deeplearning/cudnn/latest/developer-guide/index.html",
        "https://docs.nvidia.com/deeplearning/cudnn/latest/release-notes/index.html",
        "https://docs.nvidia.com/deeplearning/cudnn/latest/installation-guide/index.html",
        "https://docs.nvidia.com/deeplearning/cudnn/latest/index.html",
    ]

    def discover_urls(self) -> Generator[str, None, None]:
        discovered: Set[str] = set()
        for url in self.CUDNN_PAGES:
            norm = self._normalize_url(url)
            if norm not in discovered and norm not in self._failed:
                discovered.add(norm)
                yield norm
            result = _fetch(norm, self._session)
            if result:
                html, final_url = result
                base = final_url if final_url.endswith('/') else final_url.rsplit('/', 1)[0] + '/'
                for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
                    href = m.group(1)
                    if not href.endswith(".html"):
                        continue
                    full = self._normalize_url(urljoin(final_url, href))
                    if not full.startswith("https://docs.nvidia.com/deeplearning/cudnn/") and not full.startswith(base):
                        continue
                    if _has_template_syntax(full) or _is_invalid_href(href):
                        self._filtered_count += 1
                        logger.debug("Filtered template/invalid href: %s -> %s", href, full)
                        continue
                    if full in discovered or full in self._failed:
                        continue
                    discovered.add(full)
                    yield full

    def extract_text(self, html: str, url: str) -> Optional[str]:
        text = _extract_section_text(html, "article", "body")
        if len(text) < self.MIN_TEXT_LENGTH:
            text = _extract_section_text(html, "main", "body")
        return text if len(text) >= self.MIN_TEXT_LENGTH else None


SCRAPERS: Dict[str, type[DocScraper]] = {
    "python": PythonDocScraper,
    "pytorch": PyTorchDocScraper,
    "numpy": NumPyDocScraper,
    "fastapi": FastAPIDocScraper,
    "opencv": OpenCVDocScraper,
    "onnx": ONNXDocScraper,
    "docker": DockerDocScraper,
    "kubernetes": KubernetesDocScraper,
    "postgresql": PostgreSQLDocScraper,
    "sqlite": SQLiteDocScraper,
    "mdn": MDNDocScraper,
    "rust-book": RustBookScraper,
    "go-docs": GoDocScraper,
    "cuda": CUDADocScraper,
    "cudnn": CuDNNDocScraper,
    "rfcs": RFCDocScraper,
    "linux-kernel": LinuxKernelDocScraper,
    "lang-specs": LangSpecScraper,
}


def scrape_all(output_dir: str, sources: Optional[List[str]] = None, max_per_source: int = 2000,
               source_timeout: int = 600) -> Dict[str, int]:
    import threading
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: Dict[str, int] = {}
    names = sources or list(SCRAPERS.keys())
    for name in names:
        cls = SCRAPERS.get(name)
        if cls is None:
            logger.warning("Unknown source: %s (available: %s)", name, list(SCRAPERS.keys()))
            continue
        logger.info("Scraping %s ...", name)
        try:
            scraper = cls(output_dir=str(output_path))
            scraper.MAX_PAGES = max_per_source
            jsons_path = output_path / scraper.SOURCE_NAME / "documents.jsonl"
            count = 0
            exc_info: List[Optional[Exception]] = [None]
            def _run():
                nonlocal count
                try:
                    count = scraper.scrape_to_jsonl(jsons_path)
                except Exception as e:
                    exc_info[0] = e
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=source_timeout)
            if t.is_alive():
                logger.error("  %s: TIMEOUT after %ds (discovered=%d, failed=%d)",
                             name, source_timeout, scraper._discovered_count, scraper._failed_count)
                results[name] = -1
            elif exc_info[0]:
                raise exc_info[0]
            else:
                results[name] = count
                if count == 0:
                    logger.warning("  %s: 0 pages scraped (discovered=%d, failed=%d, dups=%d)",
                                   name, scraper._discovered_count, scraper._failed_count, scraper._duplicate_count)
        except Exception as e:
            logger.error("Failed to scrape %s: %s", name, e)
            results[name] = 0
    return results


def build_doc_registry(
    base_dir: str,
    weights: Optional[Dict[str, float]] = None,
    total_docs_weight: float = 0.15,
) -> dict:
    from src.data.registry import DatasetInfo

    default_weights = {
        "python": 0.18, "pytorch": 0.13, "numpy": 0.06, "fastapi": 0.03,
        "opencv": 0.02, "onnx": 0.02, "docker": 0.05, "kubernetes": 0.05,
        "postgresql": 0.04, "sqlite": 0.04, "mdn": 0.15, "rust-book": 0.10,
        "go-docs": 0.08, "cuda": 0.03, "cudnn": 0.02,
    }
    weights = weights or default_weights
    total_w = sum(weights.values())
    entries = []
    for name, rel_weight in weights.items():
        jsonl_path = Path(base_dir) / name / "documents.jsonl"
        if not jsonl_path.exists():
            logger.warning("  %s: no cached data at %s — skipping", name, jsonl_path)
            continue
        w = (rel_weight / total_w) * total_docs_weight
        entries.append(DatasetInfo(
            path="json",
            name=name,
            data_dir=str(jsonl_path.parent),
            category="docs",
            weight=round(w, 4),
            quality_score=0.95,
            language=SCRAPERS.get(name, PythonDocScraper).LANGUAGE,  # type: ignore
            domain="backend",
            text_fields=["text"],
            license="various",
            streaming=True,
        ))
    return {"json": entries}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Build documentation datasets")
    parser.add_argument("--output-dir", default="data/docs", help="Output directory for JSONL files")
    parser.add_argument("--sources", nargs="*", help="Specific sources to scrape (default: all)")
    parser.add_argument("--max-per-source", type=int, default=2000, help="Max pages per source")
    args = parser.parse_args()
    results = scrape_all(args.output_dir, args.sources, args.max_per_source)
    total = sum(results.values())
    logger.info("Scraped %d total pages across %d sources", total, len(results))
    for name, count in sorted(results.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d pages", name, count)
