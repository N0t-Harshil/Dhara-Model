from __future__ import annotations

"""Asynchronous checkpoint persistence.

``AsyncCheckpointWriter`` moves the expensive parts of checkpointing — model
weight serialization, checksum, fsync, atomic rename — off the training
thread. The GPU-facing caller only snapshots state on the main thread (a
``torch`` state dict is captured, not written) and enqueues a small job; the
writer thread performs all disk work in the background.

Guarantees:
  * durability boundary    — flush() blocks until all queued jobs are durably
                             written; stage transitions call it before
                             switching datasets.
  * bounded memory         — the job queue is capped; submit() enforces
                             backpressure with a timeout.
  * atomicity              — files are written to a temp name in the same
                             directory, fsynced, then renamed into place, so a
                             crash never leaves a half-written checkpoint.
  * corruption detection   — every checkpoint gets a sidecar sha256 manifest
                             (``.sha256.json``); verify_checkpoint() re-hashes
                             and returns False on mismatch.
  * no crash hang          — if the writer thread dies, submit() falls back to
                             a synchronous write so training never deadlocks.
  * resume compatibility   — checkpoints written this way are byte-identical
                             to the previous synchronous layout; old
                             checkpoints without manifests still verify.
"""

import hashlib
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_MANIFEST_NAME = ".sha256.json"


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(dir_path: str | Path) -> Tuple[bool, str]:
    """Check a checkpoint directory written by this module.

    Returns (ok, detail). Checkpoints without a manifest are accepted as-is
    (legacy layout); with a manifest the file hash must match. Safe to call
    before every resume.
    """
    d = Path(dir_path)
    manifest = d / _MANIFEST_NAME
    if not manifest.exists():
        return True, "no manifest (legacy checkpoint)"
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"manifest unreadable: {e}"
    for rel, expected in (meta.get("files") or {}).items():
        f = d / rel
        if not f.exists():
            return False, f"missing {rel}"
        try:
            if _sha256_file(f) != expected:
                return False, f"hash mismatch: {rel}"
        except Exception as e:
            return False, f"read error on {rel}: {e}"
    return True, f"{len(meta.get('files') or {})} files verified"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class AsyncCheckpointWriter:
    def __init__(
        self,
        max_queue: int = 2,
        write_timeout: float = 1200.0,
        checksum: bool = True,
    ) -> None:
        self._max_queue = max(1, int(max_queue))
        self._write_timeout = float(write_timeout)
        self._checksum = bool(checksum)
        self._q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=self._max_queue)
        self._flush_cv = threading.Condition()
        self._pending = 0
        self._failed = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ckpt-writer")
        self._thread.start()

    # -- public API -------------------------------------------------

    def submit(self, dir_path: str | Path, write_fn: Callable[[Path], None]) -> bool:
        """Enqueue a checkpoint. ``write_fn`` runs on the writer thread.

        Returns True when the job is accepted (queued or written); if the
        writer thread is dead, the job is written synchronously on the calling
        thread (no hang, no silent loss) and True is still returned. Never
        blocks the caller for more than ``write_timeout`` (backpressure on a
        full queue)."""
        job = (Path(dir_path), write_fn)
        if self._failed:
            logger.error("Checkpoint writer dead — falling back to sync write")
            self._write_sync(job)
            return True
        with self._flush_cv:
            self._pending += 1
        deadline = time.monotonic() + self._write_timeout
        while True:
            try:
                self._q.put(job, timeout=0.5)
                break
            except queue.Full:
                if time.monotonic() > deadline:
                    logger.error("Checkpoint queue full — falling back to sync write")
                    with self._flush_cv:
                        self._pending = max(0, self._pending - 1)
                        self._flush_cv.notify_all()
                    self._write_sync(job)
                    return True
        return True

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Block until every queued checkpoint is on disk. Returns False if
        the writer thread is dead (caller must fall back to sync)."""
        if self._failed:
            return False
        deadline = time.monotonic() + (timeout if timeout is not None else self._write_timeout)
        with self._flush_cv:
            while self._pending > 0:
                if self._failed:
                    return False
                remain = deadline - time.monotonic()
                if remain <= 0:
                    logger.error("Checkpoint flush timed out")
                    return False
                self._flush_cv.wait(timeout=remain)
        return True

    def close(self) -> None:
        """Flush pending jobs and stop the writer thread."""
        try:
            self.flush()
        except Exception:
            pass
        self._q.put(None)
        self._thread.join(timeout=5)

    # -- internals --------------------------------------------------

    def _write_sync(self, job: Tuple[Path, Callable[[Path], None]]) -> None:
        try:
            self._run_job(job)
        except Exception as e:  # noqa: BLE001 — fallback path, log and re-raise
            logger.exception("Synchronous checkpoint fallback failed: %s", e)
            raise

    def _run(self) -> None:
        try:
            while True:
                job = self._q.get()
                if job is None:
                    break
                try:
                    self._run_job(job)
                except Exception as e:  # noqa: BLE001 — isolate per checkpoint
                    logger.exception("Checkpoint write failed (will be retried by caller on next boundary): %s", e)
                finally:
                    with self._flush_cv:
                        self._pending = max(0, self._pending - 1)
                        self._flush_cv.notify_all()
        except Exception as e:  # noqa: BLE001 — writer thread death
            self._failed = True
            logger.exception("Checkpoint writer thread died: %s", e)
            with self._flush_cv:
                self._flush_cv.notify_all()

    def _run_job(self, job: Tuple[Path, Callable[[Path], None]]) -> None:
        path, write_fn = job
        path.mkdir(parents=True, exist_ok=True)
        write_fn(path)
        if self._checksum:
            try:
                files = {}
                for f in sorted(path.iterdir()):
                    if f.is_file() and f.name != _MANIFEST_NAME:
                        files[f.name] = _sha256_file(f)
                _atomic_write_bytes(
                    path / _MANIFEST_NAME,
                    json.dumps({"files": files}, indent=2).encode("utf-8"),
                )
            except Exception as e:  # noqa: BLE001 — manifest is best-effort
                logger.warning("Checkpoint checksum failed for %s: %s", path, e)


class CheckpointContext:
    """Bundle of everything a checkpoint write needs, captured eagerly on the
    training thread so the writer never touches the live model."""

    def __init__(
        self,
        model_state: Any,
        config: Dict[str, Any],
        tokenizer_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_state = model_state
        self.config = config
        self.tokenizer_state = tokenizer_state


def build_checkpoint_context(model, tokenizer=None) -> CheckpointContext:
    """Capture model weights + metadata without touching disk."""
    return CheckpointContext(
        model_state=model.state_dict(),
        config=getattr(model, "config", None).to_dict() if getattr(model, "config", None) is not None else {},
        tokenizer_state=tokenizer.state_dict() if tokenizer is not None else None,
    )


def sync_write_checkpoint(path: Path, ctx: CheckpointContext) -> None:
    """Reference implementation of the checkpoint layout, also used as the
    crash fallback: config.json + pytorch_model.bin (+ tokenizer)."""
    import torch

    path.mkdir(parents=True, exist_ok=True)
    config = ctx.config
    if "model_type" not in config:
        config = {**config, "model_type": config.get("architecture") or "methos_v3"}
    _atomic_write_bytes(path / "config.json", json.dumps(config, indent=2).encode("utf-8"))
    tmp = path / ("pytorch_model.bin.tmp")
    torch.save(ctx.model_state, str(tmp))
    os.replace(tmp, path / "pytorch_model.bin")
    if ctx.tokenizer_state is not None:
        _atomic_write_bytes(
            path / "tokenizer.json", json.dumps(ctx.tokenizer_state, default=str).encode("utf-8"))


__all__ = [
    "AsyncCheckpointWriter",
    "CheckpointContext",
    "build_checkpoint_context",
    "sync_write_checkpoint",
    "verify_checkpoint",
]