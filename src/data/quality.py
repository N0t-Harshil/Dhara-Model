from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import struct
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CONTAMINATION_BENCHMARK_PATTERNS: Dict[str, List[str]] = {
    "human_eval": [
        r"def\s+(check\w*|candidate|human_eval)", r"def test_", r"assert.*==",
    ],
    "mbpp": [
        r"\"\"\".*>>>.*", r"def\s+check\w+",
    ],
    "mmlu": [
        r"(?:Answer|answer):\s*[A-D]", r"Question \d+:", r"\([A-D]\)\s",
    ],
    "gsm8k": [
        r"####\s+\-?\d+", r"Let's think step by step", r"<<.*=.*>>",
    ],
    "arc": [
        r"grid\s*=\s*\[\[", r"A\.\s*\[", r"output_grid",
    ],
}

# ── Language Detection ────────────────────────────────────────────

_LANG_SIGNATURES: Dict[str, List[str]] = {
    "python": [r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r"\bclass\s+\w+", r"\bprint\s*\(", r"\bif\s+__name__"],
    "javascript": [r"\bfunction\s+\w+\s*\(", r"\bconst\s+\w+\s*=", r"\blet\s+\w+\s*=", r"\bvar\s+\w+\s*=", r"=>\s*{"],
    "typescript": [r":\s*(string|number|boolean|any|void)\b", r"\binterface\s+\w+", r"\btype\s+\w+\s*="],
    "java": [r"\bpublic\s+(class|void|static)", r"\bprivate\s+\w+", r"\bprotected\s+\w+", r"\bimport\s+java\."],
    "cpp": [r"#include\s*[<\"]", r"\bauto\s+\w+\s*=", r"\btemplate\s*<", r"\bnamespace\s+\w+"],
    "c": [r"#include\s*[<\"](stdlib|stdio|string|math)\.h", r"\bint\s+main\s*\(", r"\bstruct\s+\w+\s*{"],
    "rust": [r"\bfn\s+\w+\s*\(", r"\blet\s+mut\s+", r"\bimpl\s+\w+", r"\buse\s+\w+::"],
    "go": [r"\bfunc\s+\w+\s*\(", r"\bpackage\s+\w+", r"\bimport\s+\(", r"\bdefer\s+"],
    "csharp": [r"\busing\s+System", r"\bnamespace\s+\w+", r"\bclass\s+\w+\s*:"],
    "php": [r"<\?php", r"\$\w+\s*=", r"\bfunction\s+\w+\s*\("],
    "ruby": [r"\bdef\s+\w+", r"\bend\b", r"\brequire\s+", r"@\w+"],
    "shell": [r"#!/bin/(bash|sh|zsh)", r"\bexport\s+\w+=", r"\becho\s+\"", r"\bif\s+\[\["],
    "sql": [r"\bSELECT\s", r"\bFROM\s+\w+", r"\bWHERE\s+\w+", r"\bJOIN\s+\w+", r"\bCREATE\s+TABLE"],
    "text": [r"[A-Z][a-z]+ [A-Z][a-z]+", r"\.\s+[A-Z]", r"[,;:] [a-z]+"],
}


def detect_language(text: str) -> str:
    lang, _ = _detect_language_with_counts(text)
    return lang


def _detect_language_with_counts(text: str) -> Tuple[str, Dict[str, int]]:
    """Exact language detection; identical winner to ``detect_language``.

    Counts matching signature patterns per language with early-exit pruning:
    - a language reaching 5 matching patterns wins immediately (maximum possible);
    - remaining patterns of a language are skipped once they can no longer beat
      the current best (a later tie loses to an earlier language, matching
      ``max(scores, key=scores.get)`` over dict insertion order).
    """
    scores: Dict[str, int] = {}
    best = 0
    for lang, patterns in _LANG_SIGNATURES.items():
        if len(patterns) <= best:
            continue
        count = 0
        remaining = len(patterns)
        for p in patterns:
            if re.search(p, text, re.MULTILINE):
                count += 1
                if count == 5:
                    return lang, scores
            remaining -= 1
            if count + remaining <= best:
                break
        if count > 0:
            scores[lang] = count
            if count > best:
                best = count
    if scores:
        return max(scores, key=scores.get), scores
    return "text", scores


# ── Quality Scoring Engine ───────────────────────────────────────

def _estimate_perplexity(text: str) -> float:
    words = text.split()
    if len(words) < 10:
        return 100.0
    char_counts = [len(w) for w in words if w]
    if not char_counts:
        return 100.0
    avg_len = sum(char_counts) / len(char_counts)
    variance = sum((c - avg_len) ** 2 for c in char_counts) / len(char_counts)
    variety = len(set(w.lower() for w in words)) / max(len(words), 1)
    score = avg_len * 0.3 + math.sqrt(variance) * 0.2 + (1 - variety) * 50
    return max(5.0, min(500.0, score))


def _language_confidence(text: str) -> float:
    lang, counts = _detect_language_with_counts(text)
    signatures = _LANG_SIGNATURES.get(lang, _LANG_SIGNATURES["text"])
    hits = counts.get(lang)
    if hits is None:
        hits = sum(1 for p in signatures if re.search(p, text, re.MULTILINE))
    expected = len(signatures)
    if expected == 0:
        return 0.5
    return min(1.0, hits / expected * 2.0)


def _formatting_quality(text: str) -> float:
    score = 0.0
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0
    avg_len = sum(len(l) for l in non_empty) / len(non_empty)
    if 15 < avg_len < 120:
        score += 0.25
    elif avg_len > 300:
        score -= 0.15
    var_len = np.std([len(l) for l in non_empty]) if len(non_empty) > 1 else 0
    if 10 < var_len < 80:
        score += 0.15
    elif var_len < 5:
        score -= 0.10
    blank_ratio = sum(1 for l in lines if not l.strip()) / max(len(lines), 1)
    if 0.05 < blank_ratio < 0.40:
        score += 0.10
    elif blank_ratio > 0.60:
        score -= 0.10
    indent_chars = sum(1 for c in text if c in " \t")
    indent_ratio = indent_chars / max(len(text), 1)
    if 0.05 < indent_ratio < 0.30:
        score += 0.10
    unique_lines = len(set(non_empty)) / max(len(non_empty), 1)
    if unique_lines > 0.70:
        score += 0.15
    elif unique_lines < 0.25:
        score -= 0.15
    punct_count = sum(1 for c in text if c in ".!?,;:")
    punct_ratio = punct_count / max(len(text), 1)
    if 0.02 < punct_ratio < 0.15:
        score += 0.10
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if 0.01 < caps_ratio < 0.30:
        score += 0.10
    elif caps_ratio > 0.50:
        score -= 0.10
    return max(-0.5, min(1.0, score))


def _code_quality(text: str, language: str) -> float:
    score = 0.0
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0
    if language == "python":
        try:
            compile(text, "<string>", "exec")
            score += 0.25
        except SyntaxError:
            score -= 0.10
        if re.search(r'"""', text):
            score += 0.10
        if re.search(r':\s*(int|str|float|bool|List|Dict|Tuple|Optional|Any)\b', text):
            score += 0.10
        if re.search(r'\b(test|assert|unittest|pytest)\b', text, re.IGNORECASE):
            score += 0.10
    elif language in ("javascript", "typescript"):
        try:
            if text.count("{") > 0 and text.count("}") > 0:
                score += 0.15
        except Exception:
            pass
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text)
    unique_ids = len(set(identifiers))
    id_diversity = unique_ids / max(len(identifiers), 1)
    if id_diversity > 0.50:
        score += 0.15
    elif id_diversity < 0.20:
        score -= 0.10
    comment_markers = ["#", "//", "/*", "'''", '"""', "<!--", "//! ", "/// "]
    comment_lines = sum(1 for l in non_empty if any(l.strip().startswith(m) for m in comment_markers))
    comment_ratio = comment_lines / max(len(non_empty), 1)
    if 0.02 < comment_ratio < 0.40:
        score += 0.10
    elif comment_ratio > 0.60:
        score -= 0.10
    func_count = len(re.findall(r'\b(def|function|fn|func|sub)\s+\w+\s*\(', text, re.MULTILINE))
    if func_count > 3:
        score += 0.10 * min(1.0, func_count / 20)
    return max(-0.5, min(1.0, score))


def _doc_completeness(text: str) -> float:
    score = 0.0
    if re.search(r'(Overview|Introduction|Getting Started|Installation|Tutorial|API Reference)', text, re.IGNORECASE):
        score += 0.15
    if re.search(r'(Parameters|Arguments|Args|Returns|Raises|Example|Note|Warning|See Also)', text, re.IGNORECASE):
        score += 0.15
    code_blocks = len(re.findall(r'```', text))
    if code_blocks >= 2:
        score += 0.10
    section_headers = len(re.findall(r'^#{1,6}\s|\n#{1,6}\s', text, re.MULTILINE))
    if section_headers > 2:
        score += 0.10 * min(1.0, section_headers / 15)
    if re.search(r'https?://\S+', text):
        score += 0.05
    if re.search(r'[A-Z]\w+\.md|`\w+`', text):
        score += 0.05
    word_count = len(text.split())
    if 100 < word_count < 10000:
        score += 0.10
    return min(1.0, score)


_TOXIC_WORDS = ["fuck", "shit", "asshole", "bastard", "bitch", "crap", "damn", "dick",
                "douche", "dumbass", "prick", "slut", "whore", "motherfucker"]
_TOXIC_RE = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in _TOXIC_WORDS) + r')\b', re.IGNORECASE)


def _toxicity_score(text: str) -> float:
    found = set(_TOXIC_RE.findall(text.lower()))
    return min(1.0, len(found) * 0.25)


def _duplicate_score(text: str, seen: Optional[Set[str]] = None) -> float:
    sample = text[:500].lower().strip()
    sample = re.sub(r'\s+', ' ', sample)
    if seen is None:
        return 0.0
    h = hashlib.md5(sample.encode()).hexdigest()[:16]
    if h in seen:
        return 1.0
    seen.add(h)
    return 0.0


def document_quality_score(
    text: str,
    category: str = "web_text",
    language: str = "text",
    seen_hashes: Optional[Set[str]] = None,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    result["length"] = min(1.0, len(text) / 50000) if len(text) > 0 else 0.0
    result["perplexity"] = max(0.0, 1.0 - _estimate_perplexity(text) / 200)
    result["language_conf"] = _language_confidence(text)
    result["formatting"] = _formatting_quality(text)
    result["code_quality"] = _code_quality(text, language) if category == "code" else 0.0
    result["doc_completeness"] = _doc_completeness(text) if category in ("docs", "wiki") else 0.0
    result["toxicity"] = 1.0 - _toxicity_score(text)
    if seen_hashes is not None:
        result["exact_duplicate"] = 1.0 - _duplicate_score(text, seen_hashes)
    base_components = ["length", "language_conf", "formatting", "toxicity"]
    if category == "code":
        base_components.append("code_quality")
    if category in ("docs", "wiki"):
        base_components.append("doc_completeness")
    weights_map = {
        "code": {"length": 0.10, "language_conf": 0.15, "formatting": 0.15,
                 "code_quality": 0.35, "toxicity": 0.10, "perplexity": 0.15},
        "docs": {"length": 0.10, "language_conf": 0.10, "formatting": 0.20,
                 "doc_completeness": 0.35, "toxicity": 0.10, "perplexity": 0.15},
        "wiki": {"length": 0.10, "language_conf": 0.10, "formatting": 0.15,
                 "doc_completeness": 0.35, "toxicity": 0.10, "perplexity": 0.20},
        "math": {"length": 0.10, "language_conf": 0.10, "formatting": 0.15,
                 "doc_completeness": 0.15, "toxicity": 0.10, "perplexity": 0.40},
        "science": {"length": 0.10, "language_conf": 0.10, "formatting": 0.15,
                    "doc_completeness": 0.20, "toxicity": 0.10, "perplexity": 0.35},
        "books": {"length": 0.15, "language_conf": 0.15, "formatting": 0.20,
                  "toxicity": 0.10, "perplexity": 0.40},
        "web_text": {"length": 0.10, "language_conf": 0.20, "formatting": 0.25,
                     "toxicity": 0.15, "perplexity": 0.30},
        "structured_knowledge": {"length": 0.10, "language_conf": 0.20, "formatting": 0.25,
                                 "toxicity": 0.15, "perplexity": 0.30},
        "long_context": {"length": 0.30, "language_conf": 0.10, "formatting": 0.15,
                         "toxicity": 0.10, "perplexity": 0.35},
    }
    weights = weights_map.get(category, weights_map["web_text"])
    final = sum(weights.get(k, 0) * max(0.0, min(1.0, v)) for k, v in result.items() if k in weights)
    result["final"] = max(0.0, min(1.0, final))
    return result


# ── Deduplication ────────────────────────────────────────────────

class ExactDeduplicator:
    def __init__(self) -> None:
        self.seen_hashes: Set[str] = set()
        self.duplicate_count: int = 0

    def is_duplicate(self, text: str) -> bool:
        sample = text[:1000] if len(text) > 1000 else text
        normalized = " ".join(sample.lower().split())
        h = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            self.duplicate_count += 1
            return True
        self.seen_hashes.add(h)
        return False

    def reset(self) -> None:
        self.seen_hashes.clear()
        self.duplicate_count = 0

    def stats(self) -> str:
        return f"Removed {self.duplicate_count} exact duplicates"


class MinHashDeduplicator:
    def __init__(self, threshold: float = 0.85, num_hashes: int = 128) -> None:
        self.threshold = threshold
        self.num_hashes = num_hashes
        self._signatures: List[Tuple[int, ...]] = []

    def _shingles(self, text: str, k: int = 5) -> List[str]:
        return [text[i:i + k] for i in range(max(1, len(text) - k + 1))]

    def _signature(self, text: str) -> Tuple[int, ...]:
        shingles = self._shingles(text)
        sig = []
        for seed in range(1, self.num_hashes + 1):
            min_hash = min(hashlib.sha256((str(seed) + s).encode()).hexdigest() for s in shingles)
            sig.append(hash(min_hash))
        return tuple(sig)

    def is_duplicate(self, content: str) -> bool:
        sig = self._signature(content)
        for other in self._signatures:
            matches = sum(1 for a, b in zip(sig, other) if a == b)
            similarity = matches / self.num_hashes
            if similarity >= self.threshold:
                return True
        self._signatures.append(sig)
        return False


def _compute_simhash_fp(text: str, hash_bits: int = 64) -> int:
    """SimHash fingerprint; identical output to SimHashDeduplicator._fingerprint.

    Module-level (picklable) so fingerprints can be computed in a process pool.
    Counts set bits per position (majority voting) instead of maintaining a
    signed weight vector — same result, ~2x faster.
    """
    cnt = [0] * hash_bits
    n = 0
    for token in re.findall(r"\w+", text.lower()):
        h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "little")
        n += 1
        bits = h
        while bits:
            lsb = bits & -bits
            cnt[lsb.bit_length() - 1] += 1
            bits ^= lsb
    fp = 0
    for i in range(hash_bits):
        if cnt[i] * 2 > n:
            fp |= 1 << i
    return fp


def _pool_simhash_fp(text: str) -> int:
    return _compute_simhash_fp(text)


def _pool_quality_score(text: str, category: str, language: str) -> float:
    return document_quality_score(text, category, language)["final"]


def _popcount_swar(x: np.ndarray) -> np.ndarray:
    """Per-element popcount (uint64) via SWAR, fully vectorized — works on any numpy.

    Each round packs 2-bit counts into 4-bit, then bytes, then 16/32-bit words.
    No value exceeds 64, so no uint64 overflow occurs at any stage.
    """
    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)
    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)
    x = (x & 0x00000000FFFFFFFF) + ((x >> 32) & 0x00000000FFFFFFFF)
    return x


class SimHashDeduplicator:
    def __init__(self, threshold: float = 0.85, hash_bits: int = 64) -> None:
        self.threshold = threshold
        self.hash_bits = hash_bits
        self._allowed_dist = int((1.0 - threshold) * hash_bits)
        self._arr = np.empty(65536, dtype=np.uint64)
        self._count: int = 0
        self._fingerprints: List[int] = []
        self.duplicate_count: int = 0

    def _ensure_capacity(self, extra: int) -> None:
        if self._count + extra > len(self._arr):
            cap = max(len(self._arr) * 2, self._count + extra)
            new = np.empty(cap, dtype=np.uint64)
            new[: self._count] = self._arr[: self._count]
            self._arr = new

    def _fingerprint(self, text: str) -> int:
        return _compute_simhash_fp(text, self.hash_bits)

    def _hamming_distance(self, a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _similarity(self, a: int, b: int) -> float:
        return 1.0 - self._hamming_distance(a, b) / self.hash_bits

    def is_duplicate(self, content: str) -> bool:
        return self._is_duplicate_fp(self._fingerprint(content))

    def _is_duplicate_fp(self, fp: int) -> bool:
        return self.process_batch([fp])[0]

    def process_batch(self, fps: List[int], chunk_size: int = 256) -> List[bool]:
        """Sequential-equivalent dedup for a batch of fingerprints.

        Decisions are EXACTLY the same as processing one-by-one in order:
        each fingerprint is compared (hamming <= allowed_dist) against every
        fingerprint inserted before it. Comparison is vectorized per chunk
        (chunk vs stored array + strict-upper-triangle within the chunk), so
        the cost is the same O(N^2) work as the original linear scan, but
        executed in C at ~ns/element instead of a Python loop.
        """
        n = len(fps)
        if n == 0:
            return []
        b = np.asarray(fps, dtype=np.uint64)
        out = np.zeros(n, dtype=bool)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = b[start:end]
            m = end - start
            hits = np.zeros(m, dtype=bool)
            if self._count:
                x = chunk[:, None] ^ self._arr[None, : self._count]
                hits = (_popcount_swar(x) <= self._allowed_dist).any(axis=1)
            if m > 1:
                x2 = chunk[:, None] ^ chunk[None, :]
                tri = _popcount_swar(x2) <= self._allowed_dist
                for k in range(1, m):
                    if (tri[k, :k] & ~hits[:k]).any():
                        hits[k] = True
            out[start:end] = hits
            keep = np.nonzero(~hits)[0]
            if keep.size:
                new_fps = chunk[keep]
                self._ensure_capacity(len(new_fps))
                self._arr[self._count:self._count + len(new_fps)] = new_fps
                self._count += len(new_fps)
                self._fingerprints.extend(int(x) for x in new_fps)
            self.duplicate_count += int(hits.sum())
        return [bool(x) for x in out]

    def reset(self) -> None:
        self._arr = np.empty(65536, dtype=np.uint64)
        self._count = 0
        self._fingerprints.clear()
        self.duplicate_count = 0

    def stats(self) -> str:
        return f"SimHash: {self.duplicate_count} duplicates, {self._count} unique fingerprints"


class SemanticDeduplicator:
    """Embedding-based semantic deduplication using sentence-transformers.

    Uses cosine similarity on document embeddings.
    For efficiency, batches documents and only compares against a representative set.
    """

    def __init__(self, threshold: float = 0.92, model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 128, max_centroids: int = 10000) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_centroids = max_centroids
        self._model = None
        self._centroids: List[np.ndarray] = []
        self.duplicate_count: int = 0
        self._document_count: int = 0

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("SemanticDeduplicator: loaded %s", self.model_name)
            except ImportError:
                logger.warning("SemanticDeduplicator: sentence-transformers not installed. "
                               "Install with: pip install sentence-transformers")
                raise
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        truncated = [t[:512] for t in texts]
        return model.encode(truncated, batch_size=self.batch_size, show_progress_bar=False,
                            normalize_embeddings=True)

    def is_duplicate(self, text: str) -> bool:
        emb = self._embed([text])[0]
        if len(self._centroids) > 0:
            centroid_arr = np.array(self._centroids)
            sims = np.dot(centroid_arr, emb)
            max_sim = float(sims.max())
            if max_sim >= self.threshold:
                self.duplicate_count += 1
                return True
        # Randomly sample centroids to avoid unbounded growth
        if len(self._centroids) < self.max_centroids:
            self._centroids.append(emb)
        else:
            idx = hash(text) % self.max_centroids
            self._centroids[idx] = emb
        self._document_count += 1
        return False

    def filter_batch(self, texts: List[str]) -> Tuple[List[str], List[bool]]:
        if not texts:
            return [], []
        embs = self._embed(texts)
        keep: List[str] = []
        flags: List[bool] = []
        for i, (text, emb) in enumerate(zip(texts, embs)):
            if len(self._centroids) > 0:
                centroid_arr = np.array(self._centroids)
                sims = np.dot(centroid_arr, emb)
                max_sim = float(sims.max())
                if max_sim >= self.threshold:
                    self.duplicate_count += 1
                    flags.append(False)
                    continue
            if len(self._centroids) < self.max_centroids:
                self._centroids.append(emb)
            else:
                idx = hash(text) % self.max_centroids
                self._centroids[idx] = emb
            keep.append(text)
            flags.append(True)
            self._document_count += 1
        return keep, flags

    def reset(self) -> None:
        self._centroids.clear()
        self.duplicate_count = 0
        self._document_count = 0

    def stats(self) -> str:
        return (f"Semantic: {self.duplicate_count} duplicates, "
                f"{len(self._centroids)} centroids, {self._document_count} total docs")


# ── Legacy Compatibility ─────────────────────────────────────────

class QualityFilter:
    LOW_QUALITY_MARKERS = [
        "lorem ipsum", "todo: add code", "your code here",
        "coming soon", "placeholder", "fixme", "not implemented",
    ]

    @staticmethod
    def check_length(content: str, min_len: int = 50, max_len: int = 250000) -> bool:
        return min_len <= len(content) <= max_len

    @staticmethod
    def is_high_quality_content(content: str) -> bool:
        lowered = content.lower()
        return not any(m in lowered for m in QualityFilter.LOW_QUALITY_MARKERS)

    @staticmethod
    def check_language_markers(content: str, language: str) -> bool:
        sigs = _LANG_SIGNATURES.get(language, [r"\s+"])
        return any(re.search(s, content) for s in sigs)


class QualityScorer:
    @staticmethod
    def heuristic_score(content: str) -> float:
        return document_quality_score(content)["final"]

    @staticmethod
    def advanced_score(content: str, category: str = "web_text", language: str = "text") -> float:
        return document_quality_score(content, category, language)["final"]


class ContaminationFilter:
    def __init__(self, benchmarks: Optional[List[str]] = None) -> None:
        self.benchmarks = benchmarks or list(_CONTAMINATION_BENCHMARK_PATTERNS.keys())
        self._patterns: List[re.Pattern] = []
        for bm in self.benchmarks:
            for pat in _CONTAMINATION_BENCHMARK_PATTERNS.get(bm, []):
                self._patterns.append(re.compile(pat, re.IGNORECASE))

    def is_contaminated(self, content: str) -> bool:
        return any(p.search(content) for p in self._patterns)

    def filter_items(self, items: List[Dict]) -> List[Dict]:
        clean = []
        for item in items:
            text = " ".join(str(v) for v in item.values())
            if not self.is_contaminated(text):
                clean.append(item)
        return clean
