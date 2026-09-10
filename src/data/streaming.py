from __future__ import annotations

import itertools
import logging
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from datasets import (
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
    load_dataset_builder,
)

from src.data.metadata_cache import (
    DatasetMetadataCache,
    GATED_DATASET_HELP,  # noqa: F401 (re-exported for back-compat)
    _handle_load_error,
    _is_gated_error,  # noqa: F401 (re-exported for back-compat)
    _loader_for_files,
    extract_data_sources,
    rewrite_hf_url,
    stream_from_record,
)
from src.data.registry import DatasetRegistry, DatasetInfo, detect_text_fields, extract_text

logger = logging.getLogger(__name__)


class ShardCoordinator:
    """Ordered, shard-parallel streaming of a resolved dataset record.

    A record's shard files are downloaded/decoded concurrently by up to
    `workers` threads (each thread owns one shard at a time), while the
    coordinator re-emits samples in exact shard order — all samples of shard k
    before any of shard k+1, each in file order. Downstream processing
    therefore observes precisely the original sequential stream order.

    Worker responsibilities per shard:
      1. open the shard file (Arrow iterable) — timed
      2. skip `resume_offset` raw rows (islice) — mid-shard resume
      3. apply the streaming-layer extraction gate (dataset-specific field
         hints first, generic detection fallback) and count raw rows
      4. enqueue gated samples tagged (shard_idx, seq) — bounded queue gives
         backpressure so downloads never run ahead of the consumer

    Closing contract: the consumer MUST call ``close()`` (e.g. in a finally
    block) when it stops pulling early — workers are daemon threads and would
    otherwise keep reading. After close(), ``progress_state()`` returns the
    (last_shard, last_offset) position for persisted resume.

    Samples carry ``_shard`` so the consumer can tally per-shard acceptance
    statistics for intelligent shard selection.
    """

    def __init__(
        self,
        record: Dict[str, Any],
        plan: List[Tuple[int, int]],
        workers: int = 4,
        text_fields: Optional[List[str]] = None,
        token: Optional[str] = None,
        limit: Optional[int] = None,
        on_progress: Optional[callable] = None,
    ) -> None:
        self.record = record
        self.files: List[str] = list(record.get("files") or [])
        self.loader = record.get("loader") or _loader_for_files(self.files)
        self.plan = plan
        self.workers = max(1, int(workers))
        self.text_fields = text_fields
        self.token = token
        self.limit = limit
        self.on_progress = on_progress
        self._stop = threading.Event()
        self._runs: Dict[int, "_ShardRun"] = {}
        self._plan_pos = 0
        self._emit_pos = 0
        self._current: Optional["_ShardRun"] = None
        self._gated = 0
        self._consumed_raw: Dict[int, int] = {}
        self._raw_done: Dict[int, int] = {}
        self._timings: Dict[str, float] = {
            "arrow_open_sec": 0.0, "arrow_open_max_sec": 0.0,
            "extraction_sec": 0.0, "network_wait_sec": 0.0,
        }
        self._failed_shards: List[int] = []
        self._first_row_seen = False
        self._first_row_timeout = float(
            os.environ.get("DATA_FILE_FIRST_ROW_TIMEOUT", "600"))
        self._created = time.monotonic()
        # Last sample-consumed / shard-finished instant — guards against both
        # a dead first row AND a mid-stream stall (network/pyarrow hang).
        self._last_progress = self._created
        logger.info("Streaming begins — %d shards, %d parallel workers%s%s",
                    len(self.files), self.workers,
                    f", resume at shard {plan[0][0]} offset {plan[0][1]}" if plan else "",
                    f", gated cap {limit}" if limit else "")

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        self._top_up()
        if not self._runs:
            raise StopIteration
        while True:
            if (time.monotonic() - self._last_progress) > self._first_row_timeout:
                raise TimeoutError(
                    f"Streaming made no progress for {self._first_row_timeout:.0f}s "
                    f"({len(self._runs)} shards in flight). Causes: (1) gated dataset "
                    "— ensure HF_TOKEN is set and the repo terms are accepted; (2) Xet "
                    "backend stall — HF_HUB_DISABLE_XET=1 is set by default; (3) network "
                    "outage; (4) a mid-stream download hang. Raise with "
                    "DATA_FILE_FIRST_ROW_TIMEOUT=<seconds> if a stall legitimately "
                    "lasts longer.")
            if self._stop.is_set():
                raise StopIteration
            if self._emit_pos >= len(self.plan):
                raise StopIteration
            idx = self.plan[self._emit_pos][0]
            run = self._runs.get(idx)
            if run is None:
                # not started yet (worker top-up in progress) — poll
                self._top_up()
                if not self._runs:
                    raise StopIteration
                self._timings["network_wait_sec"] += 0.05
                continue
            self._current = run
            try:
                sample = run.queue.get(timeout=0.25)
            except queue.Empty:
                if run.done.is_set():
                    self._finish_shard(idx, run)
                    self._top_up()
                    continue
                self._timings["network_wait_sec"] += 0.25
                continue
            sample["_shard"] = idx
            self._first_row_seen = True
            self._last_progress = time.monotonic()
            self._consumed_raw[idx] = max(
                self._consumed_raw.get(idx, 0), int(sample.get("_raw_seq", 0)))
            self._gated += 1
            if self.limit and self._gated >= self.limit:
                self._stop.set()
            return sample

    def _top_up(self) -> None:
        while self._plan_pos < len(self.plan) and len(self._runs) < self.workers:
            self._start_run(self.plan[self._plan_pos])
            self._plan_pos += 1

    def _start_run(self, shard: Tuple[int, int]) -> None:
        run = _ShardRun(shard)
        run.thread = threading.Thread(
            target=self._worker, args=(run,), daemon=True,
            name=f"shard-{shard[0]}")
        run.thread.start()
        self._runs[shard[0]] = run

    def _worker(self, run: "_ShardRun") -> None:
        try:
            t0 = time.perf_counter()
            for sample in self._stream_shard(run):
                if self._stop.is_set():
                    break
                put = False
                while not put:
                    try:
                        run.queue.put(sample, timeout=0.5)
                        put = True
                    except queue.Full:
                        # Consumer stopped reading (target reached / closed);
                        # abandon the sample instead of blocking forever, so
                        # the pyarrow reader is released on close.
                        if self._stop.is_set():
                            break
            run.open_sec = time.perf_counter() - t0
        except Exception as e:
            logger.warning("Shard %d stream failed: %s", run.shard[0], e)
            run.failed = True
        finally:
            run.done.set()

    def _stream_shard(self, run: "_ShardRun"):
        """Open a shard file (Arrow iterable), skip resume rows, apply the
        streaming-layer extraction gate (identical to stream_dataset), and
        yield gated samples tagged (_shard, _raw_seq) for exact resume."""
        idx, offset = run.shard
        url = rewrite_hf_url(self.files[idx])
        try:
            it = load_dataset(
                self.loader, data_files=[url], split="train",
                streaming=True, token=self.token,
            )
        except Exception as e:
            logger.warning("Shard %d open failed (%s) — retrying once", idx, e)
            it = load_dataset(
                self.loader, data_files=[url], split="train",
                streaming=True, token=self.token,
            )
        if offset > 0:
            it = itertools.islice(it, offset, None)
        detected = None
        t_extract = 0.0
        for sample in it:
            if self._stop.is_set():
                break
            run.raw_count += 1
            if detected is None:
                detected = detect_text_fields(sample, self.text_fields)
            t0 = time.perf_counter()
            text = extract_text(sample, detected)
            t_extract += time.perf_counter() - t0
            if not text:
                continue
            yield {**sample, "text": text, "_raw_seq": run.raw_count}
        self._timings["extraction_sec"] += t_extract

    def _finish_shard(self, idx: int, run: "_ShardRun") -> None:
        open_sec = run.open_sec or 0.0
        self._timings["arrow_open_sec"] += open_sec
        self._timings["arrow_open_max_sec"] = max(
            self._timings["arrow_open_max_sec"], open_sec)
        if run.failed:
            self._failed_shards.append(idx)
        else:
            # Shard completion is progress even when it gated zero samples
            # (its queue was drained and the worker finished cleanly).
            self._last_progress = time.monotonic()
        self._raw_done[idx] = run.raw_count
        del self._runs[idx]
        if self._current is run:
            self._current = None
        self._emit_pos += 1
        if self.on_progress is not None:
            try:
                self.on_progress(idx, run.raw_count, failed=run.failed)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    def progress_state(self) -> Tuple[int, int]:
        """(next shard to read, raw offset within it) — the exact resume point.

        The offset counts raw rows CONSUMED by the consumer (not merely pulled
        into a worker buffer), so resuming skips exactly what was already
        handed to downstream processing — no duplicates, no gaps. Returns
        (-1, 0) when the whole plan was exhausted."""
        if self._emit_pos < len(self.plan):
            idx = self.plan[self._emit_pos][0]
            consumed = self._consumed_raw.get(idx, 0)
            return idx, consumed
        return -1, 0

    def raw_rows(self) -> int:
        """Raw rows consumed across all shards (finished + current)."""
        current = self._current.raw_count if self._current is not None else 0
        return sum(self._raw_done.values()) + current

    def gated_count(self) -> int:
        return self._gated

    def timings(self) -> Dict[str, float]:
        return dict(self._timings)

    def failed_shards(self) -> List[int]:
        return list(self._failed_shards)

    def close(self) -> None:
        self._stop.set()
        for run in self._runs.values():
            run.thread.join(timeout=5)
        self._runs.clear()


class _ShardRun:
    __slots__ = ("shard", "queue", "done", "thread", "failed", "open_sec", "raw_count")

    def __init__(self, shard: Tuple[int, int]) -> None:
        self.shard = shard
        self.queue: queue.Queue = queue.Queue(maxsize=512)
        self.done = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.failed = False
        self.open_sec: Optional[float] = None
        self.raw_count: int = 0


def resolve_and_cache(
    path: str,
    split: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    revision: Optional[str] = None,
    meta_cache: Optional[DatasetMetadataCache] = None,
    preprocess_sig: str = "",
    token_sig: str = "",
    builder_cache=None,
) -> bool:
    """Resolve a dataset's driver record WITHOUT downloading any data and store
    it in the metadata cache (file family) or builder cache (script family),
    so later stream_dataset calls take the fast path.

    Uses load_dataset_builder + as_streaming_dataset (metadata only), the same
    shard-source capture used by the cold path. Returns True when a usable
    record is available afterwards (either it already was, or resolution
    succeeded). Never raises — resolution failures are logged and return False.
    """
    info_like = SimpleNamespace(path=path, name=name, split=split, data_dir=data_dir)
    if builder_cache is not None:
        from src.data.drivers import DRIVER_KIND_STREAMING, detect_driver

        drv, diag = detect_driver(
            info_like, meta_cache, builder_cache, preprocess_sig, token_sig,
            revision=revision)
        return (diag.get("driver_kind") != DRIVER_KIND_STREAMING
                or drv.record is not None)
    if meta_cache is not None and meta_cache.enabled:
        rec = meta_cache.verify(info_like, preprocess_sig, token_sig)
        if rec is not None:
            return True
        local = Path(path)
        try:
            if local.exists():
                files = [str(local)] if local.is_file() else sorted(
                    str(p) for p in local.rglob("*") if p.is_file())
                loader = _loader_for_files(files) if files else None
                if files and loader:
                    rec = meta_cache.build_record(info_like, files, None,
                                                  preprocess_sig, token_sig, loader)
                    if meta_cache.save(rec, info_like):
                        logger.info("Metadata resolved+cached (local): %s (%d files)",
                                    path, len(files))
                        return True
                return False
        except Exception as e:
            logger.warning("Local metadata resolution failed for %s: %s", path, e)
            return False
        try:
            builder = load_dataset_builder(
                path,
                name=name if name else None,
                data_dir=data_dir if data_dir else None,
                revision=revision if revision else None,
                token=os.environ.get("HF_TOKEN"),
            )
            it = builder.as_streaming_dataset(split)
            ex = getattr(it, "_ex_iterable", None)
            files = extract_data_sources(ex) if ex is not None else []
            loader = _loader_for_files(files) if files else None
            if files and loader:
                rec = meta_cache.build_record(info_like, files, revision, preprocess_sig, token_sig, loader)
                if meta_cache.save(rec, info_like):
                    logger.info("Metadata resolved+cached: %s/%s (%d shards)",
                                path, name or "default", len(files))
                    return True
            else:
                logger.info("No file-list metadata captured for %s/%s (script/streaming dataset)",
                            path, name or "default")
                return False
        except Exception as e:
            _handle_load_error(path, e)
            return False
    return False


def stream_dataset(
    path: str,
    split: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    streaming: bool = True,
    limit: Optional[int] = None,
    text_fields: Optional[List[str]] = None,
    meta_cache: Optional[DatasetMetadataCache] = None,
    preprocess_sig: str = "",
    token_sig: str = "",
) -> Generator[Dict[str, Any], None, None]:
    kwargs: dict = {"path": path, "split": split, "streaming": streaming}
    if name:
        kwargs["name"] = name
    if data_dir:
        kwargs["data_dir"] = data_dir

    detected_fields = None
    count = 0

    info_like = SimpleNamespace(path=path, name=name, split=split, data_dir=data_dir)

    # Fast path: reuse previously resolved metadata (no repo/shard resolution).
    if meta_cache is not None and meta_cache.enabled:
        rec = meta_cache.verify(info_like, preprocess_sig, token_sig)
        if rec is not None:
            logger.info("Dataset cache found: %s/%s", path, name or "default")
            logger.info("  Repository unchanged | metadata reused (%d shards)",
                        rec.get("num_shards", 0))
            logger.info("  Arrow reused — direct iterable, no HF resolution")
            logger.info("  Streaming begins (no HF resolution)")
            try:
                for sample in stream_from_record(rec, limit=limit, token=os.environ.get("HF_TOKEN")):
                    if detected_fields is None:
                        detected_fields = detect_text_fields(sample, text_fields)
                    text = extract_text(sample, detected_fields)
                    if text:
                        yield {**sample, "text": text}
                        count += 1
                        if limit and count >= limit:
                            return
                return
            except Exception as e:
                logger.warning("Fast-path streaming failed for %s (%s) — "
                               "re-resolving dataset metadata", path, e)
                meta_cache.invalidate(path, name, split)
        else:
            logger.info("Dataset cache not found (or fingerprint changed) for %s/%s",
                        path, name or "default")

    try:
        ds = load_dataset(**kwargs)
    except Exception as e:
        _handle_load_error(path, e)
        return

    if ds is None:
        logger.error("load_dataset returned None for %s", path)
        return

    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        try:
            ds = ds[split]
        except KeyError:
            logger.exception("Split '%s' not found in %s (available: %s)",
                             split, path, list(ds.keys()))
            return

    if meta_cache is not None and meta_cache.enabled:
        ex = getattr(ds, "_ex_iterable", None)
        files = extract_data_sources(ex) if ex is not None else []
        from src.data.metadata_cache import _loader_for_files
        loader = _loader_for_files(files) if files else None
        if files and loader:
            rec = meta_cache.build_record(info_like, files, None, preprocess_sig, token_sig, loader)
            if meta_cache.save(rec, info_like):
                logger.info("Metadata cached for %s/%s (%d shards)",
                            path, name or "default", len(files))
        else:
            logger.info("No file-list metadata captured for %s/%s (script/streaming dataset)",
                        path, name or "default")

    for sample in ds:
        if detected_fields is None:
            detected_fields = detect_text_fields(sample, text_fields)
            logger.debug("Detected fields for %s: %s", path, detected_fields)
        text = extract_text(sample, detected_fields)
        if text:
            yield {**sample, "text": text}
            count += 1
            if limit and count >= limit:
                return

    logger.info("Streamed %d samples from %s/%s", count, path, name or "default")


def stream_dataset_with_fallbacks(
    info: DatasetInfo,
    registry: DatasetRegistry,
    limit: Optional[int] = None,
    meta_cache: Optional[DatasetMetadataCache] = None,
    preprocess_sig: str = "",
    token_sig: str = "",
    skip_counters: Optional[Dict[str, int]] = None,
) -> Generator[Dict[str, Any], None, None]:
    tried: List[str] = []
    chain = [info]
    for fb_path in info.fallbacks:
        fb_entry = registry.get_by_path_category(fb_path, info.category)
        if fb_entry:
            chain.append(fb_entry)

    for entry in chain:
        key = f"{entry.path}/{entry.name or 'default'}"
        if key in tried:
            continue
        tried.append(key)
        logger.info("Loading dataset: %s/%s (cat=%s, weight=%.3f, qs=%.2f)",
                     entry.path, entry.name or "default", entry.category,
                     entry.weight, entry.quality_score)
        count = 0
        try:
            for sample in stream_dataset(
                path=entry.path,
                split=entry.split,
                name=entry.name,
                data_dir=entry.data_dir,
                streaming=entry.streaming,
                limit=limit,
                text_fields=entry.text_fields,
                meta_cache=meta_cache,
                preprocess_sig=preprocess_sig,
                token_sig=token_sig,
            ):
                yield sample
                count += 1
        except Exception:
            logger.exception("Stream failed for %s", key)
            continue
        if count > 0:
            if key != f"{info.path}/{info.name or 'default'}":
                registry.log_fallback(f"{info.path}/{info.name or 'default'}", key)
            return
        if skip_counters is not None:
            skip_counters[key] = skip_counters.get(key, 0) + 1
        logger.warning("Dataset %s returned 0 samples, trying fallback %s", key,
                       entry.fallbacks if entry is info else "none")

    logger.error("All fallbacks exhausted for %s/%s", info.path, info.name or "default")


class StreamingManager:
    def __init__(self, registry: DatasetRegistry) -> None:
        self.registry = registry
        self._skip_counters: Dict[str, int] = {}

    def stream_all(self, limit_per_dataset: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        for info in self.registry.all_entries():
            key = f"{info.path}/{info.name or 'default'}"
            self._skip_counters[key] = 0
            yield from stream_dataset_with_fallbacks(
                info, self.registry, limit=limit_per_dataset,
                skip_counters=self._skip_counters)

    def stream_category(self, category: str, limit_per_dataset: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        for info in self.registry.all_entries():
            if info.category == category:
                yield from stream_dataset_with_fallbacks(info, self.registry, limit=limit_per_dataset)

    def skip_rate(self) -> Dict[str, float]:
        total = sum(self._skip_counters.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self._skip_counters.items()}


# Backward-compatible exposure of the richer MassiveDataCollector implementation.
# Prefer the full-featured collector from src.massive_data_collector when available.
try:
    from src.massive_data_collector import MassiveDataCollector as _RichMassiveDataCollector  # type: ignore
except Exception:
    _RichMassiveDataCollector = None

if _RichMassiveDataCollector is not None:
    MassiveDataCollector = _RichMassiveDataCollector
else:
    class MassiveDataCollector:
        """Legacy wrapper for backward compatibility."""
        def __init__(self, datasets_cfg: Optional[List] = None) -> None:
            self.datasets = datasets_cfg or []

        def get_dataset_list(self):
            return self.datasets

        def stream_single_dataset(self, ds_info, limit=None, theme="all", skip_samples=0, raw_text=True):
            try:
                info = DatasetInfo(
                    path=ds_info.path,
                    category=ds_info.category,
                    weight=getattr(ds_info, 'weight', 1.0),
                    quality_score=getattr(ds_info, 'quality_score', 0.5),
                    name=getattr(ds_info, 'name', None),
                    split=getattr(ds_info, 'split', 'train'),
                    text_fields=getattr(ds_info, 'text_fields', None),
                )
                registry = DatasetRegistry()
                registry.register(info)
                yield from stream_dataset_with_fallbacks(info, registry, limit=limit)
            except Exception:
                logger.exception("stream_single_dataset(%s) failed",
                                 ds_info.path if hasattr(ds_info, 'path') else str(ds_info))
                return
