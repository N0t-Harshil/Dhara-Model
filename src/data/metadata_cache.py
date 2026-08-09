from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Xet-backed repos (the-stack-v2-dedup, OpenCoder, ...) stall forever on the
# xet backend of newer huggingface_hub before the first byte of any file. The
# plain HTTPS resolve/ path (fsspec/pyarrow range reads) is always available
# and never stalls, so disable xet by default unless explicitly enabled.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# The "Resolving data files: N/N" tqdm bars come from datasets' internal repo
# listing during first resolution — silence them (the count of files is logged
# by the driver anyway).
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

SCHEMA_VERSION = 1


# ── Fingerprinting ────────────────────────────────────────────────────


def compute_fingerprint(
    repo: str,
    name: Optional[str],
    split: str,
    data_dir: Optional[str],
    files: Optional[List[str]],
    revision: Optional[str],
    preprocess_sig: str,
    token_sig: str,
    fingerprint_version: int = SCHEMA_VERSION,
) -> str:
    """Stable fingerprint of a resolved dataset plus the processing context.

    Any change in repository id, subset, split, resolved file list, revision,
    preprocessing signature or tokenizer signature invalidates the cache."""
    parts = [
        "mc", str(fingerprint_version),
        repo, name or "", split, data_dir or "",
        "|".join(sorted(files or [])),
        revision or "",
        preprocess_sig,
        token_sig,
    ]
    raw = "\x1f".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_source(src: Any) -> str:
    """Normalize a data-source object (may be str or file-spec) to a string."""
    if isinstance(src, str):
        return src
    if isinstance(src, (list, tuple)):
        return "|".join(normalize_source(x) for x in src)
    for attr in ("path", "file", "uri", "url", "fname", "path_info"):
        if hasattr(src, attr):
            raw = getattr(src, attr)
            if raw is not None:
                if isinstance(raw, (list, tuple)):
                    return "|".join(normalize_source(x) for x in raw)
                return str(raw)
    return str(src)


def extract_data_sources(ex) -> List[str]:
    """Best-effort extraction of the resolved file list from a streaming
    dataset's underlying iterable (Arrow/Parquet/JSON/CSV/Text examples
    iterables). Returns an empty list if the dataset is not file-based."""
    candidates = ("shard_data_sources", "data_sources", "parquet_files",
                  "filepaths", "data_files", "fnames", "files")
    for attr in candidates:
        raw = getattr(ex, attr, None)
        if raw is None:
            continue
        if callable(raw) and not isinstance(raw, type):
            try:
                raw = raw()
            except Exception:
                continue
        try:
            if isinstance(raw, dict):
                raw = [v for v in raw.values() if isinstance(v, list)]
            if not isinstance(raw, (list, tuple)):
                continue
            out = [s for s in (normalize_source(x) for x in raw) if s]
            if out:
                return out
        except Exception:
            continue
    return []


def rewrite_hf_url(src: str) -> str:
    """Rewrite hf://datasets/... source URLs to https resolve URLs so the
    fast path can stream remote files without touching the HF api.

    Handles modern two-component ids (org/repo), optional @revision pins,
    legacy single-component ids (e.g. code_search_net — hub datasets that
    predate namespaces) and already-resolved https URLs."""
    if src.startswith("http://") or src.startswith("https://"):
        return src
    m = re.match(r"^hf(?:s)?://datasets/(.+?)/resolve/(.+)$", src)
    if m:
        return f"https://huggingface.co/datasets/{m.group(1)}/resolve/{m.group(2)}"
    m = re.match(r"^hf://datasets/([^/@]+)/([^/@]+)@([^/]+)/(.*)$", src)
    if m:
        org, repo, revision, path = m.groups()
        return f"https://huggingface.co/datasets/{org}/{repo}/resolve/{revision}/{path}"
    m = re.match(r"^hf://datasets/([^/@]+)@([^/]+)/(.*)$", src)
    if m:
        repo, revision, path = m.groups()
        return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
    m = re.match(r"^hf://datasets/([^/]+)/([^/]+)/(.*)$", src)
    if m:
        return f"https://huggingface.co/datasets/{m.group(1)}/{m.group(2)}/resolve/{m.group(3)}"
    m = re.match(r"^hf://datasets/([^/]+)/(.*)$", src)
    if m:
        return f"https://huggingface.co/datasets/{m.group(1)}/resolve/{m.group(2)}"
    return src


def _loader_for_files(files: List[str]) -> Optional[str]:
    exts = [Path(f.split("?")[0]).suffix.lower() for f in files]
    exts = [e for e in exts if e]
    if not exts:
        return None
    if all(e == ".parquet" for e in exts):
        return "parquet"
    if all(e in (".json", ".jsonl", ".json.gz", ".jsonl.gz") for e in exts):
        return "json"
    if all(e == ".csv" for e in exts):
        return "csv"
    if all(e == ".arrow" for e in exts):
        return "arrow"
    if all(e == ".txt" for e in exts):
        return "text"
    return None


def parse_loader(kind: str) -> Optional[str]:
    """Map a source kind hint to the built-in datasets loader name."""
    return kind if kind in ("parquet", "json", "csv", "arrow", "text") else None


# ── Load-error diagnostics (gated datasets etc.) ───────────────────


GATED_DATASET_HELP = (
    "To request access:\n"
    "  1. Go to https://huggingface.co/{path}\n"
    "  2. Click 'Agree and access repository'\n"
    "  3. Set your HF token via:\n"
    "       data.hf_token: hf_xxxx\n"
    "     or env var: $env:HF_TOKEN='hf_xxxx'\n"
)


def _is_gated_error(e: Exception) -> bool:
    err = str(e).lower()
    return any(kw in err for kw in ["gated", "access", "permission", "401", "403", "cannot access"])


def _handle_load_error(path: str, e: Exception) -> None:
    if _is_gated_error(e):
        token = os.environ.get("HF_TOKEN", "")
        if token:
            logger.exception("Gated dataset %s – token present but access denied. "
                             "Request access at https://huggingface.co/%s", path, path)
        else:
            logger.exception("Gated dataset %s – no HF_TOKEN set. "
                             "Set data.hf_token in config or export HF_TOKEN.", path)
            logger.info(GATED_DATASET_HELP.format(path=path))
    else:
        logger.exception("Failed to load %s", path)


# ── Cache store ─────────────────────────────────────────────────────


class DatasetMetadataCache:
    """Persistent cache of resolved HuggingFace dataset metadata.

    Layout (mirrors the documented cache/):

        <dir>/<safe_name>/
            dataset_info.json   — repo id, name, split, data_dir
            fingerprint.json    — computed fingerprint + version
            revision.json       — upstream revision/commit captured on resolution
            split.json          — split name + shard count
            cache_location.json — file sources (local paths or remote URLs)
            record.json         — full record (single source of truth)
    """

    def __init__(
        self,
        root: str | Path,
        enabled: bool = True,
        fingerprint_version: int = SCHEMA_VERSION,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.fingerprint_version = fingerprint_version

    # -- keys / paths -------------------------------------------------

    @staticmethod
    def safe_dir_name(repo: str, name: Optional[str]) -> str:
        parts = []
        for part in [repo, name or "default"]:
            part = part.replace("/", "___")
            part = re.sub(r"[^A-Za-z0-9_.-]+", "_", part)
            parts.append(part)
        return "__".join(parts)

    def record_dir(self, info) -> Path:
        return self.root / self.safe_dir_name(info.path, info.name)

    # -- record construction ------------------------------------------

    def _expected_fingerprint(
        self,
        info,
        files: Optional[List[str]],
        revision: Optional[str],
        preprocess_sig: str,
        token_sig: str,
    ) -> str:
        return compute_fingerprint(
            repo=info.path,
            name=info.name,
            split=info.split,
            data_dir=info.data_dir,
            files=files,
            revision=revision,
            preprocess_sig=preprocess_sig,
            token_sig=token_sig,
            fingerprint_version=self.fingerprint_version,
        )

    def build_record(
        self,
        info,
        files: Optional[List[str]],
        revision: Optional[str],
        preprocess_sig: str,
        token_sig: str,
        loader: Optional[str] = None,
    ) -> Dict[str, Any]:
        fp = self._expected_fingerprint(info, files, revision, preprocess_sig, token_sig)
        return {
            "schema_version": SCHEMA_VERSION,
            "fingerprint_version": self.fingerprint_version,
            "repo": info.path,
            "name": info.name,
            "split": info.split,
            "data_dir": info.data_dir,
            "revision": revision,
            "files": files or [],
            "num_shards": len(files or []),
            "loader": loader,
            "fingerprint": fp,
            "preprocess_sig": preprocess_sig,
            "token_sig": token_sig,
        }

    def compute_fingerprint_for(
        self,
        info,
        files: Optional[List[str]],
        revision: Optional[str],
        preprocess_sig: str,
        token_sig: str,
    ) -> str:
        return self._expected_fingerprint(info, files, revision, preprocess_sig, token_sig)

    # -- IO ----------------------------------------------------------

    def get(self, repo: str, name: Optional[str] = None, split: str = "train") -> Optional[Dict[str, Any]]:
        """Load a cached record without verification. Returns None if absent."""
        rec_dir = self.root / self.safe_dir_name(repo, name)
        rec_path = rec_dir / "record.json"
        try:
            if rec_path.exists():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                if rec.get("schema_version") != SCHEMA_VERSION:
                    return None
                if rec.get("split") != split:
                    return None
                return rec
            # fall back to the split-out layout (older/manual caches)
            finger = rec_dir / "fingerprint.json"
            if not finger.exists():
                return None
            fp = json.loads(finger.read_text(encoding="utf-8"))
            rec = {"schema_version": SCHEMA_VERSION, "fingerprint_version": fp.get("version", SCHEMA_VERSION),
                   "fingerprint": fp.get("fingerprint", ""), "revision": None,
                   "files": [], "num_shards": 0, "loader": None, "split": split}
            loc = rec_dir / "cache_location.json"
            if loc.exists():
                d = json.loads(loc.read_text(encoding="utf-8"))
                rec["files"] = d.get("files", [])
                rec["num_shards"] = d.get("num_shards", len(rec["files"]))
            return rec
        except Exception as e:
            logger.debug("Metadata cache read failed for %s/%s: %s", repo, name, e)
            return None

    def verify(
        self,
        info,
        preprocess_sig: str,
        token_sig: str,
        repo: str | None = None,
        name: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the cached record if its fingerprint still matches the
        current repo/name/split + processing context, else None."""
        if not self.enabled:
            return None
        repo = repo or info.path
        name = name if name is not None else info.name
        split = split if split is not None else info.split
        rec = self.get(repo, name, split)
        if rec is None:
            return None
        expected = self._expected_fingerprint(info, rec.get("files"), rec.get("revision"), preprocess_sig, token_sig)
        if rec.get("fingerprint") != expected:
            logger.debug("Metadata cache fingerprint mismatch for %s/%s", repo, name)
            return None
        if rec.get("files") is None:
            return None
        return rec

    def invalidate(self, repo: str, name: Optional[str] = None, split: str = "train") -> None:
        """Remove a stale record so the next run re-resolves the dataset."""
        rec_dir = self.root / self.safe_dir_name(repo, name)
        try:
            if rec_dir.exists():
                import shutil
                shutil.rmtree(rec_dir)
                logger.info("Metadata cache invalidated for %s/%s", repo, name or "default")
        except Exception as e:
            logger.warning("Metadata cache invalidation failed for %s/%s (%s)", repo, name, e)

    def save(self, rec: Dict[str, Any], info) -> bool:
        """Persist a record (atomically). Returns True on success."""
        if not self.enabled:
            return False
        try:
            rec_dir = self.record_dir(info)
            rec_dir.mkdir(parents=True, exist_ok=True)
            import uuid

            def _atomic_write(path: Path, content) -> None:
                tmp = rec_dir / f".tmp-{uuid.uuid4().hex}"
                tmp.write_text(json.dumps(content, indent=2, default=str, ensure_ascii=False),
                               encoding="utf-8")
                try:
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)

            files = {
                "dataset_info.json": {
                    "repo": rec.get("repo"), "name": rec.get("name"), "split": rec.get("split"),
                    "data_dir": rec.get("data_dir"),
                },
                "fingerprint.json": {
                    "fingerprint": rec.get("fingerprint"), "version": rec.get("fingerprint_version"),
                    "schema_version": rec.get("schema_version"),
                },
                "revision.json": {"revision": rec.get("revision")},
                "split.json": {"split": rec.get("split"), "num_shards": rec.get("num_shards"),
                               "loader": rec.get("loader")},
                "cache_location.json": {
                    "kind": "files",
                    "files": rec.get("files", []),
                    "num_shards": rec.get("num_shards", len(rec.get("files", []))),
                },
                "record.json": rec,
            }
            for fname, content in files.items():
                _atomic_write(rec_dir / fname, content)
            return True
        except Exception as e:
            logger.warning("Metadata cache save failed for %s/%s (%s)", info.path, info.name, e)
            return False


# ── Fast-path streaming ──────────────────────────────────────────────


def stream_from_record(
    rec: Dict[str, Any],
    limit: Optional[int] = None,
    token: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Stream rows directly from a cached file list, skipping all HuggingFace
    resolution (repo, split, shard enumeration). Uses the built-in datasets
    loaders so no module is downloaded. Raises on failure so the caller can
    fall back to a normal (resolution-based) load and refresh the cache."""
    files = rec.get("files") or []
    loader = rec.get("loader") or _loader_for_files(files)
    if not loader or not files:
        raise RuntimeError(f"No file sources in record for {rec.get('repo')}")
    urls = [rewrite_hf_url(f) for f in files]

    from datasets import load_dataset

    split = rec.get("split", "train")
    ds = load_dataset(
        loader,
        data_files={split: urls},
        split=split,
        streaming=True,
        token=token,
    )
    if isinstance(ds, dict):
        ds = ds.get(split) or list(ds.values())[0]
    count = 0
    for sample in ds:
        yield sample
        count += 1
        if limit and count >= limit:
            return