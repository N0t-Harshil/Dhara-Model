from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TEXT_FIELD_CANDIDATES = ["text", "content", "body", "code", "document", "article", "source",
                         "output", "problem", "solution", "abstract", "section"]

TEXT_FIELD_PRIORITY = [
    "text", "content", "body", "code", "document", "article", "source",
    "output", "problem", "solution", "abstract", "section",
    "func_code_string", "whole_func_string", "documentation",
    "doc_string", "func_documentation_string",
]


def detect_text_fields(sample: dict, known_fields=None) -> list:
    if known_fields:
        for field in known_fields:
            val = sample.get(field)
            if val and isinstance(val, str) and len(val) > 20:
                return [field]
    for field in TEXT_FIELD_PRIORITY:
        val = sample.get(field)
        if val and isinstance(val, str) and len(val) > 20:
            return [field]
    for key, val in sample.items():
        if isinstance(val, str) and len(val) > 50:
            return [key]
    return ["text"]


def extract_text(sample: dict, fields=None) -> str:
    candidates = fields or TEXT_FIELD_PRIORITY
    for field in candidates:
        val = sample.get(field)
        if val and isinstance(val, str):
            return val.strip()
    for field in TEXT_FIELD_PRIORITY:
        val = sample.get(field)
        if val and isinstance(val, str):
            return val.strip()
    texts = []
    for key, val in sample.items():
        if isinstance(val, str) and len(val) > 100:
            texts.append(val)
    if texts:
        return "\n\n".join(texts).strip()
    return ""


@dataclass
class DatasetInfo:
    path: str
    category: str
    weight: float
    quality_score: float
    name: Optional[str] = None
    split: str = "train"
    data_dir: Optional[str] = None
    language: Optional[str] = None
    domain: str = "general"
    fallbacks: List[str] = field(default_factory=list)
    text_fields: List[str] = field(default_factory=lambda: list(TEXT_FIELD_CANDIDATES))
    license: str = "unknown"
    max_samples: Optional[int] = None
    function_sampling: bool = False
    streaming: bool = True
    priority: int = 10
    long_context: bool = False
    estimated_tokens: Optional[int] = None
    doc_type: str = "general"

    def load_kwargs(self) -> dict:
        kwargs: dict = {"path": self.path, "split": self.split, "streaming": self.streaming}
        if self.name:
            kwargs["name"] = self.name
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir
        return kwargs


class DatasetRegistry:
    def __init__(self, exclude: Optional[List[str]] = None) -> None:
        self._entries: Dict[str, DatasetInfo] = {}
        self._fallback_entries: Dict[str, DatasetInfo] = {}
        self._used_fallbacks: Dict[str, str] = {}
        self._counter: int = 0
        # Dataset path prefixes to drop from the registry entirely (primary
        # AND fallback). Used e.g. to blacklist the-stack-v2-dedup, whose
        # rows carry metadata but no `content` column.
        self._exclude: List[str] = list(exclude or [])

    def _is_excluded(self, path: str) -> bool:
        return any(path.startswith(p) for p in self._exclude)

    def register(self, info: DatasetInfo, fallback_only: bool = False) -> None:
        """Add a dataset entry. With fallback_only=True the entry is only
        resolvable as a fallback chain member (never streamed as a primary
        dataset by all_entries()/by_category()) — used for the fallback
        datasets referenced by primary entries."""
        if self._is_excluded(info.path):
            logger.info("Registry exclude %r — skipping %s entry %r",
                        self._exclude, "fallback-only" if fallback_only else "primary",
                        f"{info.path}/{info.name or 'default'}")
            return
        key = f"{info.path}/{info.name or 'default'}/{info.category}/{self._counter}"
        if fallback_only:
            self._fallback_entries[key] = info
        else:
            self._entries[key] = info
        self._counter += 1

    def get(self, key: str) -> Optional[DatasetInfo]:
        return self._entries.get(key)

    def get_by_path_category(self, path: str, category: str) -> Optional[DatasetInfo]:
        for info in itertools.chain(self._entries.values(), self._fallback_entries.values()):
            if info.path == path and info.category == category:
                return info
        return None

    def all_entries(self) -> List[DatasetInfo]:
        return list(self._entries.values())

    def by_category(self, cat: str) -> List[DatasetInfo]:
        return [e for e in self.all_entries() if e.category == cat]

    def log_fallback(self, primary: str, fallback: str) -> None:
        self._used_fallbacks[primary] = fallback
        logger.warning("Fallback: %s -> %s", primary, fallback)

    def normalize_weights(self, category_targets: Dict[str, float] = None) -> float:
        from collections import defaultdict
        cat_entries = defaultdict(list)
        for e in self.all_entries():
            cat_entries[e.category].append(e)
        targets = category_targets or {}
        for cat, entries in cat_entries.items():
            target = targets.get(cat)
            if target is None or not entries:
                continue
            current = sum(e.weight for e in entries)
            if current > 0:
                scale = target / current
                for e in entries:
                    e.weight = round(e.weight * scale, 6)
                new_sum = sum(e.weight for e in entries)
                diff = target - new_sum
                if abs(diff) > 1e-10:
                    entries[-1].weight = round(entries[-1].weight + diff, 6)
        return sum(e.weight for e in self.all_entries())

    def summary(self) -> dict:
        return {
            "total_registered": len(self._entries),
            "total_weight": sum(e.weight for e in self._entries.values()),
            "fallbacks_used": dict(self._used_fallbacks),
        }


# ── Constants ─────────────────────────────────────────────────────

STACK_V2_LANGUAGES: Dict[str, str] = {
    "Python": "python", "C++": "cpp", "JavaScript": "javascript",
    "TypeScript": "typescript", "Java": "java", "Rust": "rust",
    "Go": "go", "SQL": "sql", "Shell": "shell",
    "PHP": "php", "Kotlin": "kotlin", "Swift": "swift", "R": "r",
    "Julia": "julia", "Scala": "scala", "Lua": "lua", "C": "c",
    "Objective-C": "objective-c",
}

CODE_LANG_TARGETS: Dict[str, float] = {
    "python": 0.25, "cpp": 0.15, "javascript": 0.10, "typescript": 0.08,
    "java": 0.08, "rust": 0.07, "go": 0.06, "sql": 0.05,
    "shell": 0.04, "csharp": 0.04, "php": 0.03, "c": 0.02,
    "kotlin": 0.01, "swift": 0.01, "scala": 0.01, "r": 0.01,
    "julia": 0.01, "lua": 0.01, "objective-c": 0.01,
}

WEB_FALLBACKS = [
    "allenai/dolma", "togethercomputer/RedPajama-Data-1T", "tiiuae/falcon-refinedweb",
]

DOC_SOURCES: Dict[str, Dict] = {
    "python-docs":       {"lang": "python",     "qs": 0.99, "w": 0.020, "domain": "backend"},
    "pytorch-docs":      {"lang": "python",     "qs": 0.99, "w": 0.020, "domain": "ml"},
    "numpy-docs":        {"lang": "python",     "qs": 0.98, "w": 0.010, "domain": "ml"},
    "rust-book":         {"lang": "rust",       "qs": 0.98, "w": 0.010, "domain": "backend"},
    "go-docs":           {"lang": "go",         "qs": 0.97, "w": 0.010, "domain": "backend"},
    "mdn-docs":          {"lang": "javascript", "qs": 0.95, "w": 0.010, "domain": "web"},
    "fastapi-docs":      {"lang": "python",     "qs": 0.96, "w": 0.005, "domain": "backend"},
    "cuda-docs":         {"lang": "cpp",        "qs": 0.95, "w": 0.005, "domain": "ml"},
    "linux-kernel-docs":  {"lang": "text",       "qs": 0.94, "w": 0.005, "domain": "os"},
    "opencv-docs":       {"lang": "cpp",        "qs": 0.93, "w": 0.005, "domain": "cv"},
    "kubernetes-docs":   {"lang": "shell",      "qs": 0.92, "w": 0.005, "domain": "cloud"},
    "docker-docs":       {"lang": "shell",      "qs": 0.91, "w": 0.005, "domain": "cloud"},
    "postgresql-docs":   {"lang": "sql",        "qs": 0.92, "w": 0.003, "domain": "backend"},
    "sqlite-docs":       {"lang": "sql",        "qs": 0.91, "w": 0.003, "domain": "backend"},
    "cudnn-docs":        {"lang": "cpp",        "qs": 0.92, "w": 0.003, "domain": "ml"},
    "onnx-docs":         {"lang": "python",     "qs": 0.91, "w": 0.003, "domain": "ml"},
    "rfcs":              {"lang": "text",       "qs": 0.90, "w": 0.003, "domain": "protocols"},
    "lang-specs":        {"lang": "text",       "qs": 0.90, "w": 0.003, "domain": "languages"},
}

# Token weights per category (must sum to 1.0)
CATEGORY_WEIGHTS = {
    "code": 0.30,
    "web_text": 0.20,
    "docs": 0.15,
    "wiki": 0.10,
    "math": 0.10,
    "science": 0.05,
    "books": 0.05,
    "structured_knowledge": 0.05,
}


# ── Registration helpers ──────────────────────────────────────────

def _register_code(registry: DatasetRegistry) -> float:
    code_total = CATEGORY_WEIGHTS["code"]
    acc = 0.0

    # opc-fineweb-code-corpus 15% of code (replaces removed FineWeb-Code)
    opc = code_total * 0.15
    registry.register(DatasetInfo(
        path="OpenCoder-LLM/opc-fineweb-code-corpus",
        category="code", weight=round(opc, 4), quality_score=0.90,
        domain="algorithms", text_fields=["text"], license="MIT",
        fallbacks=["code-search-net/code_search_net", "codeparrot/codeparrot-clean"],
        priority=10,
    ))
    acc += opc

    # Stack v2 30% of code (was 35% — make room for opc)
    sv2 = code_total * 0.30
    lang_sum = sum(CODE_LANG_TARGETS.values())
    for lang_name, lang_key in STACK_V2_LANGUAGES.items():
        target = CODE_LANG_TARGETS.get(lang_key, 0.01) / lang_sum
        w = round(sv2 * target, 4)
        if w < 0.0005:
            continue
        fb = ["code-search-net/code_search_net", "codeparrot/codeparrot-clean"]
        if lang_key == "sql":
            fb = ["b-mc2/sql-create-context", "code-search-net/code_search_net"]
        registry.register(DatasetInfo(
            path="bigcode/the-stack-v2-dedup",
            name=lang_name, category="code", weight=w, quality_score=0.88,
            language=lang_key, domain="backend",
            text_fields=["content", "text"], license="various",
            priority=5, function_sampling=True, fallbacks=fb,
        ))
        acc += w

    # CodeSearchNet 20% of code (was 25% — make room for opc)
    # Namespaced id: newer datasets/huggingface_hub reject the legacy
    # namespace-less "code_search_net" id in hf:// URIs (same repo, same data).
    csn = code_total * 0.20
    registry.register(DatasetInfo(
        path="code-search-net/code_search_net", category="code",
        weight=round(csn, 4), quality_score=0.92,
        domain="backend", text_fields=["code", "func_code_string", "whole_func_string", "text"],
        license="various", priority=10, function_sampling=True,
    ))
    acc += csn

    # CodeParrot 20% of code (was 25% — make room for opc)
    cp = code_total * 0.20
    registry.register(DatasetInfo(
        path="codeparrot/codeparrot-clean", category="code",
        weight=round(cp, 4), quality_score=0.84,
        language="python", domain="backend",
        text_fields=["content", "text"], license="MIT",
        priority=10, function_sampling=True,
    ))
    acc += cp

    # CodeContests 15% of code (was 10%)
    cc = code_total * 0.15
    registry.register(DatasetInfo(
        path="deepmind/code_contests", category="code",
        weight=round(cc, 4), quality_score=0.90,
        domain="algorithms", text_fields=["description", "solutions", "text"],
        license="Apache-2.0",
        fallbacks=["codeparrot/codeparrot-clean"],
        priority=20,
    ))
    acc += cc

    return round(acc, 4)


def _register_docs(registry: DatasetRegistry) -> float:
    doc_total = CATEGORY_WEIGHTS["docs"]
    acc = 0.0

    # FineWeb-Edu as general doc proxy
    edu = round(doc_total * 0.03, 4)
    if edu > 0:
        registry.register(DatasetInfo(
            path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
            category="docs", weight=edu, quality_score=0.92,
            domain="general", text_fields=["text"], license="ODC-BY", priority=10,
        ))
        acc += edu

    # Official doc sources (point to doc_builder output; fall back to FineWeb-Edu)
    for doc_name, cfg in DOC_SOURCES.items():
        w = round(doc_total * cfg["w"] / sum(d["w"] for d in DOC_SOURCES.values()), 4)
        if w < 0.001:
            continue
        registry.register(DatasetInfo(
            path="json", name=doc_name,
            data_dir=str(Path(f"data/docs/{doc_name}")),
            category="docs", weight=w, quality_score=cfg["qs"],
            language=cfg["lang"], domain=cfg["domain"],
            text_fields=["text"], license="various",
            fallbacks=["HuggingFaceFW/fineweb-edu"],
            priority=5,
        ))
        acc += w

    return round(acc, 4)


def _register_wiki(registry: DatasetRegistry) -> float:
    wiki_total = CATEGORY_WEIGHTS["wiki"]
    registry.register(DatasetInfo(
        path="wikimedia/wikipedia", name="20231101.en",
        category="wiki", weight=round(wiki_total, 4), quality_score=0.95,
        domain="web", text_fields=["text"], license="CC-BY-SA", priority=1,
    ))
    return round(wiki_total, 4)


def _register_math(registry: DatasetRegistry) -> float:
    math_total = CATEGORY_WEIGHTS["math"]
    acc = 0.0

    entries = [
        ("open-web-math/open-web-math", None, 0.92, 0.42, "science", []),
        ("AI-MO/NuminaMath-CoT", None, 0.90, 0.20, "science", ["AI-MO/NuminaMath-1.5"]),
        ("AI-MO/NuminaMath-1.5", None, 0.91, 0.15, "science", ["AI-MO/NuminaMath-CoT"]),
        ("GAIR/MathPile", None, 0.93, 0.10, "science", ["open-web-math/open-web-math"]),
        ("HuggingFaceFW/fineweb-edu", "sample-100BT", 0.85, 0.10, "science",
         ["open-web-math/open-web-math"]),
        ("akjadhav/leandojo-lean4-formal-informal-strings-split", None, 0.93, 0.03, "science",
         ["open-web-math/open-web-math"]),
    ]
    for path, name, qs, frac, domain, fallbacks in entries:
        w = round(math_total * frac, 4)
        if w < 0.001:
            continue
        registry.register(DatasetInfo(
            path=path, name=name, category="math", weight=w, quality_score=qs,
            domain=domain, text_fields=["text", "problem", "solution"], license="various",
            fallbacks=fallbacks, priority=10,
        ))
        acc += w
    return round(acc, 4)


def _register_science(registry: DatasetRegistry) -> float:
    sci_total = CATEGORY_WEIGHTS["science"]
    acc = 0.0

    # s2orc_full (schema is metadata struct, not text), pubmed (script-based, fails datasets 2.20+),
    # openalex/arxiv/acl_anthology (not real HF datasets) — all removed.
    # Science corpus is entirely FineWeb-Edu until a suitable parquet-based scientific dataset is found.
    entries = [
        ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-10", 0.85, 1.0, "science", []),
    ]
    for path, name, qs, frac, domain, fallbacks in entries:
        w = round(sci_total * frac, 4)
        if w < 0.001:
            continue
        registry.register(DatasetInfo(
            path=path, name=name, category="science", weight=w, quality_score=qs,
            domain=domain, text_fields=["text", "title", "abstract"], license="various",
            fallbacks=fallbacks, priority=10,
        ))
        acc += w
    return round(acc, 4)


def _register_books(registry: DatasetRegistry) -> float:
    book_total = CATEGORY_WEIGHTS["books"]
    acc = 0.0

    # pg19 (script-based, fails datasets 2.20+), gutenberg/openstax/libretexts (not real) — all removed.
    # Books corpus from FineWeb-Edu + FineWeb.
    entries = [
        ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2023-50", 0.88, 0.50, "web", []),
        ("HuggingFaceFW/fineweb", "CC-MAIN-2021-10", 0.85, 0.30, "web", []),
    ]
    for path, name, qs, frac, domain, fallbacks in entries:
        w = round(book_total * frac, 4)
        if w < 0.001:
            continue
        registry.register(DatasetInfo(
            path=path, name=name, category="books", weight=w, quality_score=qs,
            domain=domain, text_fields=["text", "content"], license="various",
            fallbacks=fallbacks, priority=10, long_context=True,
        ))
        acc += w
    return round(acc, 4)


def _register_structured(registry: DatasetRegistry) -> float:
    sk_total = CATEGORY_WEIGHTS["structured_knowledge"]
    acc = 0.0

    # WIT (script-based, fails datasets 2.20+), wikidata/dbpedia/conceptnet/wordnet (not real) — all removed.
    # Structured knowledge from FineWeb-Edu + FineWeb.
    entries = [
        ("HuggingFaceFW/fineweb-edu", "sample-350BT", 0.85, 0.50, "web", []),
        ("HuggingFaceFW/fineweb", "sample-100BT", 0.82, 0.50, "web", []),
    ]
    for path, name, qs, frac, domain, fallbacks in entries:
        w = round(sk_total * frac, 4)
        if w < 0.001:
            continue
        registry.register(DatasetInfo(
            path=path, name=name, category="structured_knowledge",
            weight=w, quality_score=qs,
            domain=domain, text_fields=["text", "description", "title"], license="various",
            fallbacks=fallbacks, priority=15,
        ))
        acc += w
    return round(acc, 4)


# ── Registry builder ──────────────────────────────────────────────

def build_registry(exclude: Optional[List[str]] = None) -> DatasetRegistry:
    if exclude:
        logger.info("Building registry with excluded paths: %s", exclude)
    registry = DatasetRegistry(exclude=exclude)

    total = 0.0

    total += _register_code(registry)
    logger.info("  code:         %.4f", total)

    # Fallback-only: sql-create-context is referenced as an SQL fallback by
    # the-stack-v2-dedup entries — make the chain resolvable without ever
    # streaming it as a primary dataset.
    registry.register(DatasetInfo(
        path="b-mc2/sql-create-context", category="code", weight=0.0,
        quality_score=0.80, language="sql", domain="backend",
        text_fields=["question", "answer", "context"], priority=1,
    ), fallback_only=True)

    # Web text (SlimPajama removed — 404 on HF)
    web_total = CATEGORY_WEIGHTS["web_text"]
    registry.register(DatasetInfo(
        path="HuggingFaceFW/fineweb", category="web_text",
        weight=round(web_total, 4), quality_score=0.93,
        domain="web", text_fields=["text"], license="ODC-BY",
        fallbacks=WEB_FALLBACKS, priority=1,
    ))
    for fb_path in WEB_FALLBACKS:
        registry.register(DatasetInfo(
            path=fb_path, category="web_text", weight=0.0, quality_score=0.85,
            domain="web", text_fields=["text"], priority=1,
        ), fallback_only=True)
    total += web_total
    logger.info("  web_text:     %.4f", total)

    # Docs
    total += _register_docs(registry)
    logger.info("  docs:         %.4f", total)

    # Wiki
    total += _register_wiki(registry)
    logger.info("  wiki:         %.4f", total)

    # Math
    total += _register_math(registry)
    logger.info("  math:         %.4f", total)

    # Science
    total += _register_science(registry)
    logger.info("  science:      %.4f", total)

    # Books
    total += _register_books(registry)
    logger.info("  books:        %.4f", total)

    # Structured knowledge
    total += _register_structured(registry)
    logger.info("      structured:   %.4f", total)

    total = round(total, 4)
    norm_total = registry.normalize_weights(CATEGORY_WEIGHTS)
    logger.info("  TOTAL:        %.4f  (normalized to %.6f)", total, norm_total)
    return registry
