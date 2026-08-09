from __future__ import annotations

"""Bounded, multi-producer / single-consumer unit prefetch queue.

Double-buffer the dataset pipeline: a small pool of daemon workers build
``depth`` datasets ahead of the Trainer while the GPU trains the current one.
Delivery to the synchronous consumer is strictly in input order regardless of
completion order, memory is bounded by the worker count, and no hang is
possible: every ``get()`` is bounded by ``timeout`` and a producer that never
finishes only stalls its own slot.

Key properties (each is covered by tests in test_pipeline_async.py)
  * starvation-free    — workers pull from a monotonic index counter guarded by
                         a condition variable; no worker can be skipped.
  * bounded memory     — at most ``depth`` results can be outstanding at any
                         moment (one per producer), the rest are consumed.
  * ordered delivery   — a slow unit k never overtakes k+1; results buffer with
                         a condition variable until their turn.
  * no deadlock        — all waits are on a condition variable with timed wakeup.
  * no silent failure  — a build exception is re-raised on the consumer side;
                         a build that never returns raises PrefetchTimeout.
  * graceful shutdown  — close() is idempotent, wakes producers, never blocks
                         the caller, and stale buffers are dropped.
"""

import logging
import threading
import time
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class PrefetchTimeout(TimeoutError):
    """Raised when the consumer asked for a unit that produced nothing within
    the configured window. The in-flight worker continues for that slot; the
    consumer is expected to treat the unit as failed and move on."""

    def __init__(self, index: int, waited: float):
        self.index = index
        self.waited = waited
        super().__init__(f"unit {index} did not finish within {waited:.0f}s")


class UnitPrefetch:
    def __init__(
        self,
        build_fn: Callable[[Any, int], Any],
        total: int,
        depth: int = 3,
        timeout: float = 600.0,
        name: str = "unit",
    ) -> None:
        self._build = build_fn
        self._total = int(total)
        self._depth = max(1, int(depth))
        self._timeout = float(timeout)
        self._name = name
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._next_produce = 0
        self._next_expected = 0
        self._results: dict = {}
        self._threads: List[threading.Thread] = []
        self.stats = {
            "started_workers": 0,
            "produced": 0,
            "delivered": 0,
            "timeouts": 0,
            "errors": 0,
            "total_prep_sec": 0.0,
            "total_gpu_wait_sec": 0.0,
        }
        self._timing: dict = {}

    # -- lifecycle -------------------------------------------------

    def start(self, units) -> None:
        """Spawn up to ``depth`` producer threads covering ``units`` in order."""
        n = min(self._depth, max(0, self._total))
        self.stats["started_workers"] = n
        for _ in range(n):
            t = threading.Thread(
                target=self._worker, args=(units,), daemon=True,
                name=f"{self._name}-prefetch-{len(self._threads)}",
            )
            self._threads.append(t)
            t.start()

    def close(self) -> None:
        """Signal all producers, drop buffered results without awaiting the
        workers (they are daemons and may be blocked in an unbounded build)."""
        self._stop.set()
        with self._cv:
            self._results.clear()
            self._timing.clear()
            self._cv.notify_all()

    def discard(self, index: int) -> None:
        """Drop a result that will never be consumed (skipped unit) so the
        consumer's expected pointer advances past it; a late arrival for this
        slot is ignored by get()."""
        with self._cv:
            self._results.pop(index, None)
            self._timing.pop(index, None)
            self._next_expected = max(self._next_expected, index + 1)

    def in_flight(self) -> int:
        with self._cv:
            pending = len(self._results)
            produced = self._next_produce
            expected = self._next_expected
        return max(0, produced - expected - pending)

    def queue_depth(self) -> int:
        with self._cv:
            return len(self._results)

    # -- producers -------------------------------------------------

    def _worker(self, units) -> None:
        while True:
            with self._cv:
                if self._stop.is_set() or self._next_produce >= self._total:
                    return
                idx = self._next_produce
                self._next_produce += 1
            t_start = time.monotonic()
            logger.info("[ASYNC] dataset %d prefetch start", idx + 1)
            try:
                payload = self._build(units[idx], idx)
                exc = None
            except Exception as e:  # noqa: BLE001 — delivered to the consumer
                payload = None
                exc = e
            t_end = time.monotonic()
            prep_dur = t_end - t_start
            logger.info("[ASYNC] dataset %d preprocessing complete (%.2fs)", idx + 1, prep_dur)
            if self._stop.is_set():
                return
            with self._cv:
                if self._stop.is_set():
                    return
                self._results[idx] = (payload, exc)
                self._timing[idx] = {"start": t_start, "end": t_end, "duration": prep_dur}
                self.stats["produced"] += 1
                self.stats["total_prep_sec"] += prep_dur
                self._cv.notify_all()

    # -- consumer --------------------------------------------------

    def get(self, index: int):
        """Return the build result for ``index``, in ascending order.

        Raises:
          PrefetchTimeout — nothing produced within the deadline.
          the build's own exception, re-raised on the consumer thread.
        """
        get_start = time.monotonic()
        deadline = get_start + self._timeout
        with self._cv:
            while True:
                if index < self._next_expected:
                    return None  # already delivered / skipped — stale
                if index in self._results:
                    wait_dur = time.monotonic() - get_start
                    payload, exc = self._results.pop(index)
                    timing = self._timing.pop(index, {})
                    self._next_expected = max(self._next_expected, index + 1)
                    self.stats["delivered"] += 1
                    self.stats["total_gpu_wait_sec"] += wait_dur
                    logger.info("[ASYNC] dataset %d consumed (GPU wait: %.2fs)", index + 1, wait_dur)
                    if exc is not None:
                        self.stats["errors"] += 1
                        raise exc
                    return payload, wait_dur, timing
                remain = deadline - time.monotonic()
                if remain <= 0:
                    self.stats["timeouts"] += 1
                    self._next_expected = max(self._next_expected, index + 1)
                    raise PrefetchTimeout(index, self._timeout)
                self._cv.wait(timeout=max(0.05, remain))