from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from datasets import DatasetDict, IterableDatasetDict, load_dataset

from src.data.metadata_cache import (
    DatasetMetadataCache,
    _handle_load_error,
    _loader_for_files,
    extract_data_sources,
    normalize_source,
    stream_from_record,
)
from src.data.registry import DatasetInfo, detect_text_fields, extract_text

logger = logging.getLogger(__name__)

BUILDER_SCHEMA_VERSION = 2

DRIVER_KIND_FILE = "file"
DRIVER_KIND_SCRIPT = "script"
DRIVER_KIND_LOCAL = "local"
DRIVER_KIND_STREAMING = "streaming"


def compute_builder_fingerprint(
    repo: str,
    name: Optional[str],
    split: str,
    data_dir: Optional[str],
    revision: Optional[str],
    script_revision: Optional[str],
    builder_class: Optional[str],
    preprocess_sig: str,
    token_sig: str,
    fingerprint_version: int = BUILDER_SCHEMA_VERSION,
) -> str:
    """Stable fingerprint of a script dataset's builder identity plus the
    processing context. Any change in repository id, subset, split, revision
    pin, downloaded script revision, builder class, preprocessing signature or
    tokenizer signature invalidates the cached builder record."""
    parts = [
        "bc", str(fingerprint_version),
        repo, name or "", split, data_dir or "",
        revision or "", script_revision or "", builder_class or "",
        preprocess_sig, token_sig,
    ]
    raw = "\x1f".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def inspect_builder(builder, split: str) -> Tuple[List[str], Optional[str]]:
    """Best-effort (files, loader) from a builder — metadata only, no data
    download. An empty file list means the dataset is script-generated (no
    shard files to cache)."""
    # 1) Resolved file list carried by the builder config — auto-converted
    #    parquet/arrow repos (Parquet<Repo> builders) and data_files-based
    #    scripts expose their shards here without touching the network.
    files: List[str] = []
    try:
        df = getattr(getattr(builder, "config", None), "data_files", None)
        if isinstance(df, dict):
            raw = df.get(split) if split in df else [v for v in df.values() if isinstance(v, list)]
        elif df is not None:
            raw = df
        else:
            raw = None
        if raw is not None:
            files = [f for f in (normalize_source(x) for x in raw) if f]
    except Exception as e:
        logger.debug("Builder data_files inspection failed for %s: %s",
                     getattr(builder, "name", "?"), e)
    # 2) Fall back to the streaming iterable's exposed shard sources
    #    (script modules that stream from parquet/arrow file lists).
    if not files:
        try:
            it = builder.as_streaming_dataset(split)
            ex = getattr(it, "_ex_iterable", None)
            files = extract_data_sources(ex) if ex is not None else []
        except Exception as e:
            logger.debug("Builder inspection failed for %s: %s",
                         getattr(builder, "name", "?"), e)
    loader = _loader_for_files(files) if files else None
    return files, loader


def script_identity(builder) -> Tuple[str, Optional[str]]:
    """Best-effort (builder class name, downloaded script revision) — used as
    part of the builder-cache fingerprint so script updates invalidate."""
    bclass = builder.__class__.__name__
    rev = (getattr(builder, "_dataset_revision", None)
           or getattr(builder, "revision", None)
           or getattr(builder, "_revision", None))
    return bclass, (rev or None)


def local_file_list(path: str) -> Tuple[List[str], Optional[str]]:
    """(files, loader) for a dataset rooted on the local filesystem."""
    local = Path(path)
    if local.is_file():
        files = [str(local)]
    else:
        files = sorted(str(p) for p in local.rglob("*") if p.is_file())
    loader = _loader_for_files(files) if files else None
    if loader is None and files:
        # Mixed-extension directory (e.g. data_dir with .jsonl + .md
        # sidecar docs): filter to the dominant loadable extension so the
        # dataset still classifies as local instead of falling back to HF.
        exts = [
            Path(f.split("?")[0]).suffix.lower()
            for f in files if Path(f.split("?")[0]).suffix.lower()
        ]
        if exts:
            dominant = max(set(exts), key=exts.count)
            filtered = [f for f in files
                        if Path(f.split("?")[0]).suffix.lower() == dominant]
            sub_loader = _loader_for_files(filtered)
            if sub_loader is not None:
                files, loader = filtered, sub_loader
    return files, loader


# Seconds allowed for a script stream to produce its first row before the
# stall is surfaced as an error with guidance (Xet-backed repos, dead
# connections, or a blocked network can otherwise hang forever). 0 disables.
_FIRST_ROW_TIMEOUT = float(os.environ.get("DATA_SCRIPT_FIRST_ROW_TIMEOUT", "600"))


def _bounded_next(it, timeout: float, what: str):
    """Pull one row from `it`, failing with a diagnostic TimeoutError if the
    first row takes longer than `timeout` seconds (a stalled pyarrow /
    HfFileSystem fetch would otherwise hang the pipeline silently). The stuck
    fetch keeps running in a daemon thread — the caller is notified via
    TimeoutError — so the byte stream is not consumed or corrupted."""
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return next(it, None)
    from queue import SimpleQueue

    q: SimpleQueue = SimpleQueue()

    def pull():
        try:
            q.put(("ok", next(it, None)))
        except BaseException as e:  # noqa: BLE001 - must not lose the error
            q.put(("err", e))

    th = threading.Thread(target=pull, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(
            f"First row of {what} not available within {timeout:.0f}s — "
            "network/Xet stall. Check HF_TOKEN and connectivity, or set "
            "HF_HUB_DISABLE_XET=1 (Xet-backed repos may materialize whole "
            "shards before streaming).")
    status, val = q.get()
    if status == "err":
        raise val
    return val


class DatasetDriver:
    """One dataset-loading family, owning a single dataset.

    A driver can:
      * resolve()  — make sure a usable record exists (identity metadata)
      * fingerprint() — stable identity incl. processing context
      * stream()  — yield gated rows ({**sample, "text": text, ...})
      * resume()  — describe where an interrupted stream should restart
      * cache() / invalidate() — persist / drop the record
      * warmup() / prefetch() — resolve on the current / a background thread

    Implementations must never download row data in resolve()/warmup() —
    only stream() touches data. Records never hold content; they hold file
    lists (file family) or builder identity + iterator state (script family).
    """

    kind: str = DRIVER_KIND_STREAMING

    def __init__(
        self,
        record: Optional[Dict[str, Any]] = None,
        meta_cache: Optional[DatasetMetadataCache] = None,
        builder_cache: Optional["BuilderCache"] = None,
    ) -> None:
        self.record = record
        self.meta_cache = meta_cache
        self.builder_cache = builder_cache

    def _info_like(self) -> SimpleNamespace:
        rec = self.record or {}
        return SimpleNamespace(
            path=rec.get("repo"), name=rec.get("name"),
            split=rec.get("split", "train"), data_dir=rec.get("data_dir"),
        )

    def fingerprint(self) -> str:
        return (self.record or {}).get("fingerprint", "")

    def resolve(self) -> bool:
        return self.record is not None

    def cache(self) -> bool:
        return False

    def invalidate(self) -> None:
        return None

    def resume(self) -> Dict[str, Any]:
        return {"shard": 0, "offset": 0}

    def warmup(self) -> bool:
        return self.resolve()

    def prefetch(self) -> None:
        def _bg() -> None:
            try:
                self.warmup()
            except Exception as e:
                logger.debug("Driver prefetch failed (%s)", e)

        threading.Thread(target=_bg, daemon=True,
                         name=f"driver-prefetch-{self.kind}").start()

    def stream(
        self,
        limit: Optional[int] = None,
        text_fields: Optional[List[str]] = None,
        token: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        raise NotImplementedError


class FileDatasetDriver(DatasetDriver):
    """File-backed datasets (Arrow/Parquet/CSV/JSON/Text): a metadata record
    caches the resolved file list, so streaming reuses the Arrow iterables
    directly — zero repository resolution, zero shard enumeration.

    The pipeline layers shard-parallel streaming (ShardCoordinator) on top of
    this driver's record; ``stream()`` is the sequential fast path."""

    kind = DRIVER_KIND_FILE

    def cache(self) -> bool:
        if self.record is None or self.meta_cache is None or not self.meta_cache.enabled:
            return False
        return self.meta_cache.save(self.record, self._info_like())

    def invalidate(self) -> None:
        if self.record is not None and self.meta_cache is not None:
            self.meta_cache.invalidate(
                self.record["repo"], self.record.get("name"),
                self.record.get("split", "train"))

    def stream(
        self,
        limit: Optional[int] = None,
        text_fields: Optional[List[str]] = None,
        token: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        rec = self.record
        if rec is None:
            logger.warning("File driver has no record — nothing to stream")
            return
        detected = None
        count = 0
        try:
            for sample in stream_from_record(
                rec, limit=limit, token=token or os.environ.get("HF_TOKEN"),
            ):
                if detected is None:
                    detected = detect_text_fields(sample, text_fields)
                text = extract_text(sample, detected)
                if text:
                    yield {**sample, "text": text}
                    count += 1
                    if limit and count >= limit:
                        return
        except Exception as e:
            logger.warning("Fast-path streaming failed for %s (%s) — "
                           "re-resolving dataset metadata", rec.get("repo"), e)
            if self.meta_cache is not None:
                self.meta_cache.invalidate(
                    rec.get("repo"), rec.get("name"), rec.get("split", "train"))
            raise


class LocalDatasetDriver(FileDatasetDriver):
    """Datasets rooted on the local filesystem: the file list is enumerated
    from disk with zero HuggingFace involvement, then treated exactly like the
    file family (metadata record + shard-parallel streaming)."""

    kind = DRIVER_KIND_LOCAL


class ScriptDatasetDriver(DatasetDriver):
    """Script / IterableDataset datasets (The Stack V2, OpenCoder, CodeParrot,
    CodeSearchNet, Code Contests, ...): no shard files exist, so the record
    NEVER holds a file list — it holds the builder identity (fingerprint,
    builder class, script revision, HF revision, split/subset) plus the
    raw-row offset the last stream reached.

    A warm start verifies the builder record (no Hub call), reuses the
    builder, and resumes the iterator with islice(offset) — the raw offset is
    persisted as the stream progresses and reset when the dataset is
    exhausted naturally."""

    kind = DRIVER_KIND_SCRIPT

    def __init__(
        self,
        record: Optional[Dict[str, Any]],
        builder_cache: Optional["BuilderCache"] = None,
        load_builder: Optional[Callable] = None,
    ) -> None:
        super().__init__(record=record, builder_cache=builder_cache)
        self._load_builder = load_builder
        self._offset = int(((record or {}).get("resume") or {}).get("offset", 0) or 0)
        self._raw_consumed = self._offset
        self._natural_end = False
        self._streaming = False

    @property
    def resumed(self) -> bool:
        return self._offset > 0

    def cache(self) -> bool:
        if self.record is None or self.builder_cache is None or not self.builder_cache.enabled:
            return False
        return self.builder_cache.save(self.record, self._info_like())

    def invalidate(self) -> None:
        if self.record is not None and self.builder_cache is not None:
            self.builder_cache.invalidate(
                self.record["repo"], self.record.get("name"),
                self.record.get("split", "train"))

    def resume(self) -> Dict[str, Any]:
        return {"shard": 0, "offset": self._raw_consumed}

    def raw_consumed(self) -> int:
        return self._raw_consumed

    def natural_end(self) -> bool:
        return self._natural_end

    def save_resume(self) -> None:
        if self.record is not None and self.builder_cache is not None:
            self.builder_cache.save_resume(self.record, self._raw_consumed)

    def reset_resume(self) -> None:
        if self.record is not None and self.builder_cache is not None:
            self.builder_cache.reset_resume(self.record)
            self._offset = 0
        # In-memory counters must match the persisted reset, or resume()/
        # raw_consumed() report a stale "next position" for the run that
        # exhausted the dataset naturally.
        self._raw_consumed = 0

    def stream(
        self,
        limit: Optional[int] = None,
        text_fields: Optional[List[str]] = None,
        token: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        rec = self.record
        if rec is None:
            logger.warning("Script driver has no record — nothing to stream")
            return
        lb = self._load_builder or load_dataset_builder  # module attr → injectable
        token = token if token is not None else os.environ.get("HF_TOKEN")
        try:
            builder = lb(
                rec["repo"], name=rec.get("name") or None,
                data_dir=rec.get("data_dir") or None,
                revision=rec.get("revision") or None, token=token,
            )
            it = builder.as_streaming_dataset(rec.get("split", "train"))
        except Exception as e:
            _handle_load_error(rec["repo"], e)
            return
        offset = self._offset
        if offset > 0:
            it = itertools.islice(it, offset, None)
        detected = None
        raw = 0
        gated = 0
        self._streaming = True
        self._natural_end = False
        try:
            it_iter = iter(it)
            first = True
            while True:
                if first:
                    first = False
                    sample = _bounded_next(it_iter, _FIRST_ROW_TIMEOUT, rec["repo"])
                else:
                    sample = next(it_iter, None)
                if sample is None:
                    break
                raw += 1
                if detected is None:
                    detected = detect_text_fields(sample, text_fields)
                text = extract_text(sample, detected)
                if not text:
                    continue
                gated += 1
                self._raw_consumed = offset + raw
                yield {**sample, "text": text, "_shard": 0, "_raw_seq": offset + raw}
                if limit and gated >= limit:
                    return
            self._natural_end = True
        finally:
            self._raw_consumed = offset + raw
            self._streaming = False


class StreamingDatasetDriver(DatasetDriver):
    """Direct-resolution streaming with no persistent record — used when a
    dataset cannot be classified as file- or script-backed (offline, gated,
    missing repo, caches disabled). Mirrors the original cold path exactly,
    still capturing file metadata opportunistically mid-stream."""

    kind = DRIVER_KIND_STREAMING

    def __init__(
        self,
        info: Optional[DatasetInfo] = None,
        meta_cache: Optional[DatasetMetadataCache] = None,
        preprocess_sig: str = "",
        token_sig: str = "",
    ) -> None:
        super().__init__(record=None, meta_cache=meta_cache)
        self.info = info
        self.preprocess_sig = preprocess_sig
        self.token_sig = token_sig

    def stream(
        self,
        limit: Optional[int] = None,
        text_fields: Optional[List[str]] = None,
        token: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        info = self.info
        if info is None:
            return
        kwargs: dict = {"path": info.path, "split": info.split, "streaming": True}
        if info.name:
            kwargs["name"] = info.name
        if info.data_dir:
            kwargs["data_dir"] = info.data_dir

        detected = None
        count = 0
        try:
            ds = load_dataset(**kwargs)
        except Exception as e:
            _handle_load_error(info.path, e)
            return

        if ds is None:
            logger.error("load_dataset returned None for %s", info.path)
            return

        if isinstance(ds, (DatasetDict, IterableDatasetDict)):
            try:
                ds = ds[info.split]
            except KeyError:
                logger.exception("Split '%s' not found in %s (available: %s)",
                                 info.split, info.path, list(ds.keys()))
                return

        if self.meta_cache is not None and self.meta_cache.enabled:
            info_like = SimpleNamespace(path=info.path, name=info.name,
                                        split=info.split, data_dir=info.data_dir)
            ex = getattr(ds, "_ex_iterable", None)
            files = extract_data_sources(ex) if ex is not None else []
            loader = _loader_for_files(files) if files else None
            if files and loader:
                rec = self.meta_cache.build_record(
                    info_like, files, None, self.preprocess_sig,
                    self.token_sig, loader)
                if self.meta_cache.save(rec, info_like):
                    logger.info("Metadata cached for %s/%s (%d shards)",
                                info.path, info.name or "default", len(files))

        for sample in ds:
            if detected is None:
                detected = detect_text_fields(sample, text_fields)
            text = extract_text(sample, detected)
            if text:
                yield {**sample, "text": text}
                count += 1
                if limit and count >= limit:
                    return

        logger.info("Streamed %d samples from %s/%s",
                    count, info.path, info.name or "default")


class BuilderCache:
    """Persistent cache of script-dataset builder identity + iterator state.

    Script datasets have no shard files to cache, so the record stores the
    builder fingerprint/class, script revision, HF revision, split/subset and
    the preprocessing/tokenizer context — plus the raw-row offset the last
    stream reached so a warm restart resumes the iterator instead of touching
    the Hub. Runtime resume state is NOT part of the fingerprint: only the
    identity + processing context invalidate.

    Layout (sibling of the metadata cache, under its root):

        <dir>/<safe_name>/builder_record.json
    """

    def __init__(
        self,
        root: str | Path,
        enabled: bool = True,
        fingerprint_version: int = BUILDER_SCHEMA_VERSION,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.fingerprint_version = fingerprint_version

    @staticmethod
    def safe_dir_name(repo: str, name: Optional[str], split: str = "train") -> str:
        return DatasetMetadataCache.safe_dir_name(repo, name, split)

    def record_dir(self, info) -> Path:
        return self.root / self.safe_dir_name(info.path, info.name, info.split)

    def fingerprint_for(
        self,
        info,
        revision: Optional[str],
        script_revision: Optional[str],
        builder_class: Optional[str],
        preprocess_sig: str,
        token_sig: str,
    ) -> str:
        return compute_builder_fingerprint(
            repo=info.path, name=info.name, split=info.split,
            data_dir=info.data_dir, revision=revision,
            script_revision=script_revision, builder_class=builder_class,
            preprocess_sig=preprocess_sig, token_sig=token_sig,
            fingerprint_version=self.fingerprint_version,
        )

    def build_record(
        self,
        info,
        revision: Optional[str],
        script_revision: Optional[str],
        builder_class: Optional[str],
        preprocess_sig: str,
        token_sig: str,
    ) -> Dict[str, Any]:
        fp = self.fingerprint_for(info, revision, script_revision,
                                  builder_class, preprocess_sig, token_sig)
        now = time.time()
        return {
            "schema_version": BUILDER_SCHEMA_VERSION,
            "fingerprint_version": self.fingerprint_version,
            "driver": DRIVER_KIND_SCRIPT,
            "repo": info.path,
            "name": info.name,
            "split": info.split,
            "data_dir": info.data_dir,
            "revision": revision,
            "script_revision": script_revision,
            "builder_class": builder_class,
            "fingerprint": fp,
            "preprocess_sig": preprocess_sig,
            "token_sig": token_sig,
            "resume": {"shard": 0, "offset": 0},
            "created_at": now,
            "updated_at": now,
        }

    def get(self, repo: str, name: Optional[str] = None, split: str = "train") -> Optional[Dict[str, Any]]:
        """Load a cached builder record without verification. None if absent."""
        rec_dir = self.root / self.safe_dir_name(repo, name, split)
        rec_path = rec_dir / "builder_record.json"
        try:
            if not rec_path.exists():
                return None
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            if rec.get("schema_version") != BUILDER_SCHEMA_VERSION:
                return None
            if rec.get("split") != split:
                return None
            return rec
        except Exception as e:
            logger.debug("Builder cache read failed for %s/%s: %s", repo, name, e)
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
        """Return the cached builder record if its fingerprint still matches
        the current repo/name/split + processing context, else None."""
        if not self.enabled:
            return None
        repo = repo or info.path
        name = name if name is not None else info.name
        split = split if split is not None else info.split
        rec = self.get(repo, name, split)
        if rec is None:
            return None
        expected = self.fingerprint_for(
            info, rec.get("revision"), rec.get("script_revision"),
            rec.get("builder_class"), preprocess_sig, token_sig)
        if rec.get("fingerprint") != expected:
            logger.debug("Builder cache fingerprint mismatch for %s/%s", repo, name)
            return None
        return rec

    def save(self, rec: Dict[str, Any], info) -> bool:
        """Persist a builder record (atomically). True on success."""
        if not self.enabled:
            return False
        try:
            rec_dir = self.record_dir(info)
            rec_dir.mkdir(parents=True, exist_ok=True)
            import uuid

            tmp = rec_dir / f".tmp-{uuid.uuid4().hex}"
            tmp.write_text(json.dumps(rec, indent=2, default=str, ensure_ascii=False),
                           encoding="utf-8")
            try:
                with open(tmp, "ab") as f:
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                pass
            os.replace(tmp, rec_dir / "builder_record.json")
            return True
        except Exception as e:
            logger.warning("Builder cache save failed for %s/%s (%s)",
                           info.path, info.name, e)
            return False

    def invalidate(self, repo: str, name: Optional[str] = None, split: str = "train") -> None:
        """Remove a stale record so the next run re-resolves the builder."""
        rec_dir = self.root / self.safe_dir_name(repo, name, split)
        try:
            if rec_dir.exists():
                import shutil
                shutil.rmtree(rec_dir)
                logger.info("Builder cache invalidated for %s/%s",
                            repo, name or "default")
        except Exception as e:
            logger.warning("Builder cache invalidation failed for %s/%s (%s)",
                           repo, name, e)

    def save_resume(self, rec: Dict[str, Any], offset: int) -> None:
        """Persist the iterator position (raw rows already consumed). The
        resume state is runtime state — it never participates in the
        fingerprint."""
        rec["resume"] = {"shard": 0, "offset": int(offset or 0)}
        rec["updated_at"] = time.time()
        if self.enabled:
            self.save(rec, SimpleNamespace(
                path=rec.get("repo"), name=rec.get("name"),
                split=rec.get("split", "train"), data_dir=rec.get("data_dir")))

    def reset_resume(self, rec: Dict[str, Any]) -> None:
        """Clear the iterator position (dataset exhausted naturally — the
        next run starts fresh, mirroring shard-progress delete-on-complete)."""
        self.save_resume(rec, 0)


def detect_driver(
    info,
    meta_cache: Optional[DatasetMetadataCache],
    builder_cache: Optional[BuilderCache],
    preprocess_sig: str,
    token_sig: str,
    revision: Optional[str] = None,
    load_builder: Optional[Callable] = None,
    token: Optional[str] = None,
) -> Tuple[DatasetDriver, Dict[str, Any]]:
    """Pick the right driver family for a dataset and make sure a usable
    record exists (never touching row data).

    Order:
      1. local filesystem root          -> LocalDatasetDriver (zero HF)
      2. verified metadata record       -> FileDatasetDriver (warm start)
      3. verified builder record        -> ScriptDatasetDriver (warm start)
      4. live builder inspection        -> file (sources found) or script
      5. resolution failure             -> StreamingDatasetDriver (cold path)

    Returns (driver, diagnostics) where diagnostics carries:
      driver_kind, metadata_hit, builder_hit, repo_resolution_skipped,
      first_resolution, resolve_error.
    """
    diag: Dict[str, Any] = {
        "driver_kind": DRIVER_KIND_STREAMING,
        "metadata_hit": False,
        "builder_hit": False,
        "repo_resolution_skipped": False,
        "first_resolution": False,
        "resolve_error": None,
    }
    info_like = SimpleNamespace(path=info.path, name=info.name,
                                split=info.split, data_dir=info.data_dir)

    # 1. Local filesystem datasets — no Hub involvement at all.
    #    Checks BOTH path and data_dir roots: doc/registry entries use
    #    path="json" + data_dir="data/docs/<name>" and must not be sent to
    #    the Hub's "json"-loader resolution.
    local_root = info.data_dir or info.path
    try:
        local_exists = Path(local_root).exists()
    except Exception:
        local_exists = False
    if local_exists:
        files, loader = local_file_list(local_root)
        if files and loader:
            cache = meta_cache if meta_cache is not None else DatasetMetadataCache(Path("."), enabled=False)
            rec = cache.verify(info_like, preprocess_sig, token_sig) if cache.enabled else None
            if rec is None:
                rec = cache.build_record(info_like, files, None, preprocess_sig, token_sig, loader)
                if cache.enabled and cache.save(rec, info_like):
                    diag["first_resolution"] = True
            else:
                diag["metadata_hit"] = True
            diag["driver_kind"] = DRIVER_KIND_LOCAL
            diag["repo_resolution_skipped"] = True
            return LocalDatasetDriver(rec, meta_cache=meta_cache), diag

    # 2. Verified metadata record (file family) — warm start, no resolution.
    if meta_cache is not None and meta_cache.enabled:
        rec = meta_cache.verify(info_like, preprocess_sig, token_sig)
        if rec is not None:
            diag["driver_kind"] = DRIVER_KIND_FILE
            diag["metadata_hit"] = True
            diag["repo_resolution_skipped"] = True
            return FileDatasetDriver(rec, meta_cache=meta_cache), diag

    # 3. Verified builder record (script family) — warm start, no resolution.
    if builder_cache is not None and builder_cache.enabled:
        brec = builder_cache.verify(info_like, preprocess_sig, token_sig)
        if brec is not None:
            diag["driver_kind"] = DRIVER_KIND_SCRIPT
            diag["builder_hit"] = True
            diag["repo_resolution_skipped"] = True
            return ScriptDatasetDriver(brec, builder_cache=builder_cache), diag

    # 4. Live resolution (metadata only): classify via the builder.
    lb = load_builder or load_dataset_builder  # module attr → injectable
    token = token if token is not None else os.environ.get("HF_TOKEN")
    try:
        builder = lb(
            info.path, name=info.name or None,
            data_dir=info.data_dir or None,
            revision=revision or None, token=token,
        )
        files, loader = inspect_builder(builder, info.split)
        if files and loader:
            cache = meta_cache if meta_cache is not None else DatasetMetadataCache(Path("."), enabled=False)
            rec = cache.build_record(
                info_like, files, revision, preprocess_sig, token_sig, loader)
            if cache.enabled and meta_cache is not None and meta_cache.save(rec, info_like):
                logger.info("Metadata resolved+cached: %s/%s (%d shards)",
                            info.path, info.name or "default", len(files))
                diag["first_resolution"] = True
            diag["driver_kind"] = DRIVER_KIND_FILE
            return FileDatasetDriver(rec, meta_cache=meta_cache), diag

        # Script family — never a file list; cache builder identity only.
        bclass, srev = script_identity(builder)
        cache = builder_cache if builder_cache is not None else BuilderCache(Path("."), enabled=False)
        brec = cache.build_record(
            info_like, revision, srev, bclass, preprocess_sig, token_sig)
        if cache.enabled and builder_cache is not None and builder_cache.save(brec, info_like):
            logger.info("Builder resolved+cached: %s/%s (script=%s, rev=%s)",
                        info.path, info.name or "default",
                        bclass, srev or "?")
            diag["first_resolution"] = True
        diag["driver_kind"] = DRIVER_KIND_SCRIPT
        return ScriptDatasetDriver(brec, builder_cache=builder_cache), diag
    except Exception as e:
        logger.warning("Driver resolution failed for %s/%s (%s)",
                       info.path, info.name or "default", e)
        diag["resolve_error"] = str(e)
        diag["driver_kind"] = DRIVER_KIND_STREAMING
        cold_info = info if isinstance(info, DatasetInfo) else DatasetInfo(
            path=info.path,
            category=getattr(info, "category", "general"),
            weight=float(getattr(info, "weight", 0.0)) or 1.0,
            quality_score=float(getattr(info, "quality_score", 0.5)),
            name=info.name,
            split=info.split,
            data_dir=info.data_dir,
            max_samples=getattr(info, "max_samples", None),
        )
        logger.warning(
            "Falling back to cold streaming for %s/%s (no metadata/builder record).",
            cold_info.path, cold_info.name or "default")
        return StreamingDatasetDriver(
            info=cold_info,
            meta_cache=meta_cache,
            preprocess_sig=preprocess_sig, token_sig=token_sig,
        ), diag


def load_dataset_builder(*args, **kwargs):  # pragma: no cover - thin re-export
    from datasets import load_dataset_builder as _lb
    return _lb(*args, **kwargs)
