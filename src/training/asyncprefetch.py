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
  * no duplicate build — a unit scheduled twice is built once; later
                         occurrences share the first payload (counted in
                         ``stats["duplicates_prevented"]``).
  * bounded retries    — a failing build is retried up to ``retries`` times
                         (config ``data.async_pipeline.retry_count``) before
                         its exception is delivered to the consumer; retried
                         attempts are counted in ``stats["retries"]``.
"""

import inspect
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# GPU wait below this threshold counts as a prefetch hit (dataset was already
# prepared when the trainer asked for it).
PREFETCH_HIT_THRESHOLD = 0.05

# Build-failure phase taxonomy. Every failure is tagged with the phase that
# raised it so retry decisions and the ASYNC report are explicit instead of
# text-sniffed from an exception message.
BUILD_PHASE_CONSTRUCTION = "construction"
BUILD_PHASE_CACHE_LOAD = "cache_load"
BUILD_PHASE_WRAPPER = "wrapper"
BUILD_PHASE_SANITY = "sanity"
BUILD_PHASE_QUEUE_PUT = "queue_put"
BUILD_PHASE_RESULT_HANDOFF = "result_handoff"
BUILD_PHASE_TYPE_VALIDATION = "type_validation"
BUILD_PHASE_TRAINER_HANDOFF = "trainer_handoff"
BUILD_PHASE_TELEMETRY = "telemetry"
BUILD_PHASE_LOGGING = "logging"
BUILD_PHASES = (
    BUILD_PHASE_CONSTRUCTION,
    BUILD_PHASE_CACHE_LOAD,
    BUILD_PHASE_WRAPPER,
    BUILD_PHASE_SANITY,
    BUILD_PHASE_QUEUE_PUT,
    BUILD_PHASE_RESULT_HANDOFF,
    BUILD_PHASE_TYPE_VALIDATION,
    BUILD_PHASE_TRAINER_HANDOFF,
    BUILD_PHASE_TELEMETRY,
    BUILD_PHASE_LOGGING,
)

# Phases that can never succeed by retrying: the inputs are already poisoned
# and every attempt will fail identically.
NON_RETRYABLE_PHASES = frozenset({
    BUILD_PHASE_TYPE_VALIDATION,
    BUILD_PHASE_RESULT_HANDOFF,
    BUILD_PHASE_WRAPPER,
    BUILD_PHASE_SANITY,
})

# Exception classes that are never retried regardless of phase (TypeError is
# the NoneType.__format__ / wrong-payload signature from the production
# failure; it is deterministic and re-running it just rebuilds the same
# poisoned state 4x). ValueError stays retryable to preserve the existing
# retry policy tests (transient "corrupt shard" style errors).
NON_RETRYABLE_EXC = (TypeError,)


def _tag_phase(exc, phase: str, **ctx) -> None:
    """Attach phase + context to an exception in place so the prefetch worker
    can classify where a build failed without parsing error text."""
    if exc is None:
        return
    try:
        if not getattr(exc, "_build_phase", None):
            exc._build_phase = phase  # type: ignore[attr-defined]
        if ctx:
            merged = dict(getattr(exc, "_build_ctx", None) or {})
            merged.update(ctx)
            exc._build_ctx = merged  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — tagging must never mask the failure
        pass


def _phase_of(exc) -> tuple:
    return (getattr(exc, "_build_phase", None) or BUILD_PHASE_CONSTRUCTION,
            getattr(exc, "_build_ctx", None) or {})


def _retryable(exc) -> bool:
    """Retry decision: TypeError / phase-tagged poison is never retried;
    everything else is retried up to ``retries``."""
    if isinstance(exc, NON_RETRYABLE_EXC):
        return False
    if _phase_of(exc)[0] in NON_RETRYABLE_PHASES:
        return False
    return True


def _unit_identity(u: Any) -> Any:
    """Identity used for duplicate-build prevention: ``(path, name)`` for real
    dataset units, the raw (hashable) value otherwise — e.g. the plain ints used
    in synthetic tests. Returns None when no identity can be derived."""
    if u is not None and (hasattr(u, "path") or hasattr(u, "name")):
        return (getattr(u, "path", None) or ""), (getattr(u, "name", None) or "")
    try:
        hash(u)
        return u
    except TypeError:
        return None


class PrefetchTimeout(TimeoutError):
    """Raised when the consumer asked for a unit that produced nothing within
    the configured window. The in-flight worker continues for that slot; the
    consumer is expected to treat the unit as failed and move on."""

    def __init__(self, index: int, waited: float):
        self.index = index
        self.waited = waited
        super().__init__(f"unit {index} did not finish within {waited:.0f}s")


# Sentinel stored in ``_shared[first]`` when a duplicated unit's FIRST slot
# fails (timeout / error / discard). Waiting duplicate slots observe it and
# fail fast instead of blocking until their full timeout for a payload that
# can never arrive.
_FAILED = object()


class UnitBuildCancelled(Exception):
    """Raised by a cooperatively-cancellable build when its slot's cancel
    event (or the pipeline's global cancellation) fired mid-build.

    This is NOT a failure: the consumer already stopped waiting for the slot
    (timeout, shutdown, duplicate discard). The worker treats it as a quiet
    out-of-band stop — no failure journaling, no error counter, no retry —
    and moves on to the next unit.
    """


class UnitPrefetch:
    def __init__(
        self,
        build_fn: Callable[[Any, int], Any],
        total: int,
        depth: int = 3,
        timeout: float = 600.0,
        name: str = "unit",
        retries: int = 0,
        max_workers: Optional[int] = None,
        cache_status: Optional[Callable[[Any, int], bool]] = None,
        retry_backoff_base: float = 1.0,
        retry_backoff_max: float = 30.0,
    ) -> None:
        self._build = build_fn
        self._total = int(total)
        self._depth = max(1, int(depth))
        self._timeout = float(timeout)
        self._name = name
        self._retries = max(0, int(retries))
        self._retry_backoff_base = max(0.0, float(retry_backoff_base))
        self._retry_backoff_max = max(self._retry_backoff_base,
                                      float(retry_backoff_max))
        self._max_workers = max(1, int(max_workers)) if max_workers else None
        self._cache_status = cache_status
        # Cooperative cancellation: a per-index Event lets the consumer cancel
        # a slot whose build is no longer awaited (timeout / shutdown). The
        # build only observes it if it accepts ``cancel_event`` (see
        # ``_invoke_build``); DataPipeline builds do.
        self._build_accepts_cancel = self._supports_cancel(build_fn)
        self._cancel_events: Dict[int, threading.Event] = {}
        self._active_builds: Dict[int, float] = {}
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
            "duplicates_prevented": 0,
            "retries": 0,
            "nonretryable": 0,
            "cancelled": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "total_prep_sec": 0.0,
            "total_gpu_wait_sec": 0.0,
        }
        self._timing: dict = {}
        self._duplicates: dict = {}
        self._dup_remaining: dict = {}
        self._shared: dict = {}
        # Per-dataset state machine (NOT_SCHEDULED/SCHEDULED/BUILDING/READY/
        # CONSUMED/TRAINING/TRAINED/FAILED_RETRYABLE/FAILED_PERMANENT/SKIPPED)
        # plus per-failure diagnostics for the ASYNC report.
        self._state: dict = {}
        self._failures: dict = {}

    # -- per-dataset state machine ---------------------------------

    def _set_state(self, index: int, state: str) -> None:
        with self._cv:
            old = self._state.get(index, "NOT_SCHEDULED")
            self._state[index] = state
            counts = self.stats.setdefault("state_counts", {})
            counts[old] = counts.get(old, 0) - 1 if counts.get(old, 0) else 0
            counts[state] = counts.get(state, 0) + 1
            if counts.get(old, 0) <= 0:
                counts.pop(old, None)

    def states(self) -> dict:
        with self._cv:
            return dict(self._state)

    def state_counts(self) -> dict:
        with self._cv:
            return dict(self.stats.get("state_counts", {}))

    def note_training_state(self, index: int, state: str) -> None:
        if state not in ("TRAINING", "TRAINED"):
            raise ValueError(f"note_training_state only accepts TRAINING/TRAINED, got {state!r}")
        self._set_state(index, state)

    def failure_snapshot(self) -> dict:
        with self._cv:
            return {k: dict(v) for k, v in self._failures.items()}

    def _record_failure(self, index: int, exc, phase: str, ctx: dict,
                        attempt: int, cache_status, retryable: bool,
                        identity) -> None:
        """Store phase-classified diagnostics for the ASYNC report. The full
        sanitized traceback is kept on the first attempt only (later retries
        update the message without re-rendering the stack)."""
        try:
            detail = {
                "index": index,
                "identity": str(_unit_identity(identity) if identity is not None else None),
                "phase": phase,
                "context": dict(ctx or {}),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "attempt": attempt,
                "cache_status": cache_status,
                "retryable": bool(retryable),
                "type_validation": {
                    "intended_type": ctx.get("intended_type"),
                    "actual_type": ctx.get("actual_type"),
                },
                "full_traceback": None,
            }
            with self._cv:
                existing = self._failures.get(index, {})
                if existing.get("full_traceback") is None and attempt == 1:
                    from src.utils.hf_auth import format_sanitized_traceback
                    detail["full_traceback"] = format_sanitized_traceback(exc)
                else:
                    detail["full_traceback"] = existing.get("full_traceback")
                self._failures[index] = detail
        except Exception:  # noqa: BLE001 — failure telemetry never breaks the queue
            pass

    # -- lifecycle -------------------------------------------------

    def start(self, units) -> None:
        """Spawn up to ``depth`` producer threads covering ``units`` in order.

        Identical units scheduled more than once are built exactly once: the
        first occurrence is built normally and later occurrences share that
        payload via ``get()`` (duplicate builds prevented, tracked in
        ``stats["duplicates_prevented"]``).
        """
        units = list(units)
        with self._cv:
            for idx in range(self._total):
                self._state[idx] = "NOT_SCHEDULED"
        seen: dict = {}
        for idx, u in enumerate(units):
            ident = _unit_identity(u)
            if ident is None:
                continue
            if ident in seen:
                first = seen[ident]
                self._duplicates[idx] = first
                self._dup_remaining[first] = self._dup_remaining.get(first, 0) + 1
                self.stats["duplicates_prevented"] += 1
                self._set_state(idx, "SKIPPED")
                logger.info("[ASYNC] unit %d duplicate of unit %d — "
                            "duplicate build prevented", idx + 1, first + 1)
            else:
                seen[ident] = idx
                self._set_state(idx, "SCHEDULED")
        n = min(self._depth, max(0, self._total))
        if self._max_workers is not None:
            n = min(n, self._max_workers)
        self.stats["started_workers"] = n
        for _ in range(n):
            t = threading.Thread(
                target=self._worker, args=(units,), daemon=True,
                name=f"{self._name}-prefetch-{len(self._threads)}",
            )
            self._threads.append(t)
            t.start()

    def close(self, join_timeout: float = 2.0) -> None:
        """Signal all producers, drop buffered results, cancel in-flight
        cancellable builds, and join the workers for a bounded window.

        A build still inside a non-abortable section (e.g. blocked in a
        network open) is logged as a leftover daemon instead of being awaited
        forever — close() never blocks the caller beyond ``join_timeout``.
        """
        self._stop.set()
        self.cancel_all()
        with self._cv:
            self._results.clear()
            self._timing.clear()
            self._shared.clear()
            self._duplicates.clear()
            self._dup_remaining.clear()
            self._cv.notify_all()
        self.wait_idle(timeout=join_timeout)

    def discard(self, index: int) -> None:
        """Drop a result that will never be consumed (skipped unit) so the
        consumer's expected pointer advances past it; a late arrival for this
        slot is ignored by get()."""
        with self._cv:
            if index in self._duplicates:
                first = self._duplicates.pop(index)
                remaining = self._dup_remaining.get(first, 1) - 1
                if remaining > 0:
                    self._dup_remaining[first] = remaining
                else:
                    self._shared.pop(first, None)
                    self._dup_remaining.pop(first, None)
            elif index in self._dup_remaining:
                # This slot is a FIRST occurrence with duplicates waiting on
                # its payload — mark it failed so they wake immediately.
                if self._shared.get(index) is None:
                    self._shared[index] = _FAILED
                self._cv.notify_all()
            self._results.pop(index, None)
            self._timing.pop(index, None)
            self._next_expected = max(self._next_expected, index + 1)
            self._set_state(index, "SKIPPED")

    def in_flight(self) -> int:
        with self._cv:
            pending = len(self._results)
            produced = self._next_produce
            expected = self._next_expected
        return max(0, produced - expected - pending)

    def queue_depth(self) -> int:
        with self._cv:
            return len(self._results)

    # -- ownership / cancellation -----------------------------------

    @staticmethod
    def _supports_cancel(build_fn: Callable) -> bool:
        """Whether the build callable accepts a ``cancel_event`` third kwarg
        (DataPipeline builds do). Signature inspection is done once so a
        runtime TypeError inside a build can never be confused with an arity
        mismatch."""
        try:
            sig = inspect.signature(build_fn)
        except (TypeError, ValueError):
            return False
        params = list(sig.parameters.values())
        if any(p.name == "cancel_event" for p in params):
            return True
        if any(p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD) for p in params):
            return True
        req = [p for p in params if p.default is inspect.Parameter.empty
               and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.POSITIONAL_ONLY)]
        return len(req) >= 3

    def _cancel_event(self, index: int) -> threading.Event:
        with self._cv:
            ev = self._cancel_events.get(index)
            if ev is None:
                ev = threading.Event()
                self._cancel_events[index] = ev
            return ev

    def cancel(self, index: int) -> None:
        """Cancel the in-flight build for one slot. The consumer already
        advanced past the slot (timeout / failure); the build observes the
        event at its next checkpoint and exits without producing. No-op for
        slots that already delivered or never started."""
        with self._cv:
            if index < self._next_expected or index in self._results:
                return
        self._cancel_event(index).set()

    def cancel_all(self) -> None:
        for idx, ev in list(self._cancel_events.items()):
            ev.set()  # noqa: B909 — iterating a snapshot is fine

    def active_builds(self) -> Dict[int, float]:
        """index -> monotonic seconds since the build started, for every slot
        a worker is currently inside (the 'ghost work' window)."""
        with self._cv:
            return dict(self._active_builds)

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Join every producer thread (bounded). Returns True when all exited,
        False if some are still inside a build (logged as leftover daemons)."""
        deadline = time.monotonic() + max(0.0, timeout)
        threads = list(self._threads)
        for t in threads:
            remain = deadline - time.monotonic()
            if remain <= 0:
                break
            t.join(timeout=remain)
        alive = [t.name for t in threads if t.is_alive()]
        if alive:
            logger.warning(
                "[ASYNC] prefetch '%s' left %d daemon worker(s) still building: %s — "
                "builds are not abortable at this point; they end when the "
                "process exits", self._name, len(alive), ", ".join(alive))
        return not alive

    # -- producers -------------------------------------------------

    def _worker(self, units) -> None:
        while True:
            with self._cv:
                # Backpressure: wait if outstanding results reach buffer depth.
                # The claim count bounds *inflight* units (builds in progress +
                # buffered), so a worker can never slip a race-standard put past
                # the limit: outstanding = claimed (next_produce) − consumed
                # (next_expected). Without it, two workers that both passed the
                # `len(results)` gate while the buffer was one below depth could
                # put depth+1 results.
                while not self._stop.is_set() and (
                    len(self._results) >= self._depth
                    or (self._next_produce - self._next_expected) >= self._depth
                ):
                    self._cv.wait()
                if self._stop.is_set() or self._next_produce >= self._total:
                    return
                idx = self._next_produce
                self._next_produce += 1
                if idx in self._duplicates:
                    continue  # built via its first occurrence — duplicate prevented
            self._set_state(idx, "BUILDING")
            t_start = time.monotonic()
            logger.info("[ASYNC] dataset %d prefetch start", idx + 1)
            cancel_ev = self._cancel_event(idx)
            if cancel_ev.is_set():
                logger.info("[ASYNC] dataset %d already cancelled before build", idx + 1)
                self.stats["cancelled"] += 1
                self._set_state(idx, "CANCELLED")
                continue
            payload, exc = None, None
            cancelled = False
            attempt = 0
            with self._cv:
                self._active_builds[idx] = time.monotonic()
            try:
                while True:
                    attempt += 1
                    try:
                        payload = self._invoke_build(units[idx], idx,
                                                     cancel_ev=cancel_ev)
                        exc = None
                        break
                    except UnitBuildCancelled:
                        # Slot no longer awaited (timeout / shutdown): this is
                        # not a build failure — exit quietly and keep covering
                        # the remaining units.
                        cancelled = True
                        break
                    except Exception as e:  # noqa: BLE001 — retried, then delivered
                        cstat = None
                        if self._cache_status is not None:
                            try:
                                cstat = bool(self._cache_status(units[idx], idx))
                            except Exception:  # noqa: BLE001
                                cstat = None
                        phase, pctx = _phase_of(e)
                        retryable = _retryable(e)
                        if not retryable:
                            from src.utils.hf_auth import format_sanitized_traceback
                            logger.error(
                                "[ASYNC] dataset %d build failed (non-retryable, "
                                "phase=%s, cache=%s) — full traceback:\n%s",
                                idx + 1, phase, cstat,
                                format_sanitized_traceback(e))
                            self.stats["nonretryable"] += 1
                            self._record_failure(
                                idx, e, phase, pctx, attempt,
                                cache_status=cstat, retryable=False,
                                identity=units[idx])
                            self._set_state(idx, "FAILED_PERMANENT")
                            payload = None
                            exc = e
                            break
                        if attempt <= self._retries:
                            self.stats["retries"] += 1
                            self._record_failure(
                                idx, e, phase, pctx, attempt,
                                cache_status=cstat, retryable=True,
                                identity=units[idx])
                            if attempt == 1:
                                from src.utils.hf_auth import format_sanitized_traceback
                                logger.warning(
                                    "[ASYNC] dataset %d build failed (attempt 1/%d, "
                                    "phase=%s, cache=%s) — full traceback:\n%s",
                                    idx + 1, self._retries + 1, phase, cstat,
                                    format_sanitized_traceback(e))
                            else:
                                logger.warning("[ASYNC] dataset %d build failed "
                                               "(attempt %d/%d, phase=%s) — retrying: "
                                               "%s: %s",
                                               idx + 1, attempt, self._retries + 1,
                                               phase, type(e).__name__, e)
                            delay = min(
                                self._retry_backoff_max,
                                self._retry_backoff_base * (2 ** (attempt - 1)))
                            wake = delay + random.uniform(0.0, delay * 0.25)
                            logger.info(
                                "[ASYNC] dataset %d retry backoff — next attempt "
                                "in %.1fs (attempt %d/%d)",
                                idx + 1, wake, attempt + 1, self._retries + 1)
                            with self._cv:
                                # Wait is woken early by close()/stop (notify_all)
                                # so a teardown never waits out the full backoff.
                                self._cv.wait(timeout=wake)
                            continue
                        self._record_failure(
                            idx, e, phase, pctx, attempt,
                            cache_status=cstat, retryable=True,
                            identity=units[idx])
                        payload = None
                        exc = e
                        break
            finally:
                with self._cv:
                    self._active_builds.pop(idx, None)
            if cancelled:
                self.stats["cancelled"] += 1
                self._set_state(idx, "CANCELLED")
                logger.info("[ASYNC] dataset %d build cancelled (slot no longer "
                            "awaited — timed out or shutdown)", idx + 1)
                continue
            if self._stop.is_set():
                return
            t_end = time.monotonic()
            prep_dur = t_end - t_start
            if exc is None:
                logger.info("[ASYNC] dataset %d preprocessing complete (%.2fs)",
                            idx + 1, prep_dur)
            else:
                phase, _pctx = _phase_of(exc)
                logger.error("[ASYNC] dataset %d build FAILED after %.2fs "
                             "(phase=%s, %s: %s) — not a preprocessing "
                             "completion; consumer will re-raise it",
                             idx + 1, prep_dur, phase, type(exc).__name__, exc)
            if self._stop.is_set():
                return
            with self._cv:
                if self._stop.is_set():
                    return
                if cancel_ev is not None and cancel_ev.is_set():
                    # Cancelled while finishing the build — drop the payload
                    # and keep covering the remaining units (a lone cancelled
                    # slot must never retire a producer permanently).
                    self.stats["cancelled"] += 1
                    self._set_state(idx, "CANCELLED")
                    continue
                if idx < self._next_expected:
                    # Stale arrival: the slot was discarded (skipped unit),
                    # timed out, or delivered early while the build was in
                    # flight. Drop the result so this worker's buffer slot is
                    # not leaked for the rest of the run (a leaked slot would
                    # permanently shrink the depth budget and could stall every
                    # producer at the backpressure wait), then keep covering.
                    self.stats["produced"] += 1
                    if idx in self._dup_remaining:
                        # First slot died before producing — fail its duplicates.
                        if self._shared.get(idx) is None:
                            self._shared[idx] = _FAILED
                        self._cv.notify_all()
                    continue
                self._results[idx] = (payload, exc)
                self._timing[idx] = {"start": t_start, "end": t_end, "duration": prep_dur}
                if exc is None and idx in self._dup_remaining:
                    self._shared[idx] = payload  # consumed by duplicate slots
                self.stats["produced"] += 1
                self.stats["total_prep_sec"] += prep_dur
                if exc is None:
                    self._set_state(idx, "READY")
                elif self._state.get(idx) not in ("FAILED_PERMANENT",):
                    self._set_state(idx, "FAILED_RETRYABLE")
                self._cv.notify_all()

    def _invoke_build(self, unit: Any, index: int,
                      cancel_ev: Optional[threading.Event]) -> Any:
        """Call the build, forwarding the cooperative cancel event when the
        build is cancellable. The three-argument form is used by the training
        pipeline's ``build_pretrain_dataset_unit(..., cancel_event=...)``."""
        if self._build_accepts_cancel:
            return self._build(unit, index, cancel_ev)
        return self._build(unit, index)

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
                if index in self._duplicates:
                    first = self._duplicates[index]
                    while first not in self._shared:
                        remain = deadline - time.monotonic()
                        if remain <= 0:
                            self.cancel(index)
                            self.stats["timeouts"] += 1
                            self._next_expected = max(self._next_expected, index + 1)
                            logger.error(
                                "[UNIT TIMEOUT] dataset %d exceeded "
                                "prefetch_timeout (%.0fs) while waiting for "
                                "duplicate of %d — build cancelled, continuing",
                                index + 1, self._timeout, first + 1)
                            raise PrefetchTimeout(index, self._timeout)
                        self._cv.wait(timeout=max(0.05, remain))
                    if self._shared[first] is _FAILED:
                        # First occurrence timed out / errored / was discarded:
                        # fail this duplicate immediately instead of hanging.
                        self.stats["timeouts"] += 1
                        self._next_expected = max(self._next_expected, index + 1)
                        remaining = self._dup_remaining.get(first, 1) - 1
                        if remaining > 0:
                            self._dup_remaining[first] = remaining
                        else:
                            self._shared.pop(first, None)
                            self._dup_remaining.pop(first, None)
                        self._cv.notify_all()
                        logger.error(
                            "[UNIT TIMEOUT] dataset %d (duplicate of %d) — "
                            "first occurrence failed; failing duplicate, continuing",
                            index + 1, first + 1)
                        raise PrefetchTimeout(index, self._timeout)
                    wait_dur = time.monotonic() - get_start
                    payload = self._shared[first]
                    self._next_expected = max(self._next_expected, index + 1)
                    self.stats["delivered"] += 1
                    self.stats["total_gpu_wait_sec"] += wait_dur
                    self._set_state(index, "CONSUMED")
                    if wait_dur < PREFETCH_HIT_THRESHOLD:
                        self.stats["prefetch_hits"] += 1
                    else:
                        self.stats["prefetch_misses"] += 1
                    remaining = self._dup_remaining.get(first, 1) - 1
                    if remaining > 0:
                        self._dup_remaining[first] = remaining
                    else:
                        self._shared.pop(first, None)
                        self._dup_remaining.pop(first, None)
                    logger.info("[ASYNC] dataset %d consumed (duplicate of %d, "
                                "shared build — GPU wait: %.2fs)",
                                index + 1, first + 1, wait_dur)
                    self._cv.notify_all()
                    return payload, wait_dur, {}
                if index in self._results:
                    wait_dur = time.monotonic() - get_start
                    payload, exc = self._results.pop(index)
                    timing = self._timing.pop(index, {})
                    self._next_expected = max(self._next_expected, index + 1)
                    self.stats["delivered"] += 1
                    self.stats["total_gpu_wait_sec"] += wait_dur
                    if wait_dur < PREFETCH_HIT_THRESHOLD:
                        self.stats["prefetch_hits"] += 1
                    else:
                        self.stats["prefetch_misses"] += 1
                    if exc is None:
                        logger.info("[ASYNC] dataset %d consumed (GPU wait: %.2fs)",
                                    index + 1, wait_dur)
                    else:
                        logger.error("[ASYNC] dataset %d build failure delivered "
                                     "after %.2fs — re-raising: %s: %s",
                                     index + 1, wait_dur, type(exc).__name__, exc)
                    # Wake producers that may be blocked on buffer depth
                    self._cv.notify_all()
                    if exc is not None:
                        self.stats["errors"] += 1
                        if index in self._dup_remaining and self._shared.get(index) is None:
                            # First slot errored — fail its duplicates now.
                            self._shared[index] = _FAILED
                            self._cv.notify_all()
                        raise exc
                    # Consumed only on success — a delivered failure keeps its
                    # FAILED_PERMANENT/FAILED_RETRYABLE terminal state so the
                    # ASYNC report reflects the actual outcome.
                    self._set_state(index, "CONSUMED")
                    return payload, wait_dur, timing
                remain = deadline - time.monotonic()
                if remain <= 0:
                    self.cancel(index)
                    self.stats["timeouts"] += 1
                    if index in self._dup_remaining and self._shared.get(index) is None:
                        # First slot timed out — fail its duplicates now.
                        self._shared[index] = _FAILED
                        self._cv.notify_all()
                    self._next_expected = max(self._next_expected, index + 1)
                    active = self._active_builds.get(index)
                    phase_hint = ""
                    st = self._state.get(index)
                    if st:
                        phase_hint = f", in-flight state={st}"
                    if active is not None:
                        phase_hint += (f", build running for "
                                       f"{time.monotonic() - active:.0f}s")
                    logger.error(
                        "[UNIT TIMEOUT] dataset %d exceeded prefetch_timeout "
                        "(%.0fs) — build cancelled and will be aborted at its "
                        "next checkpoint%s; marking failed, continuing to the "
                        "next dataset", index + 1, self._timeout, phase_hint)
                    raise PrefetchTimeout(index, self._timeout)
                self._cv.wait(timeout=max(0.05, remain))