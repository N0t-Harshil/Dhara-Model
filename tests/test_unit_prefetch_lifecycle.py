from __future__ import annotations

"""UnitPrefetch lifecycle / cancellation tests (Phases 3 & 4).

Covers cooperative cancellation (a build that accepts ``cancel_event`` is
aborted at its checkpoint instead of running forever on a daemon thread),
bounded close/join, and the invariant that a cancelled build is never counted
as a failure. Hermetic: no network, no GPU, no subprocesses.
"""

import threading
import time

import pytest

from src.training.asyncprefetch import (
    PrefetchTimeout,
    UnitBuildCancelled,
    UnitPrefetch,
)


def _pf(build_fn, total, depth=2, timeout=5.0, retries=0, max_workers=None,
        retry_backoff_base=0.05, retry_backoff_max=0.2):
    return UnitPrefetch(
        build_fn=build_fn, total=total, depth=depth, timeout=timeout,
        name="test-lifecycle", retries=retries, max_workers=max_workers,
        retry_backoff_base=retry_backoff_base,
        retry_backoff_max=retry_backoff_max,
    )


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _mk_blocking_build(only_index=0):
    """Build fn where only ``only_index`` blocks (with cancel-awareness);
    every other slot returns instantly. Mirrors one slow unit among fast ones."""
    release = threading.Event()
    started = threading.Event()
    cancel_seen = threading.Event()

    def build_fn(item, idx, cancel_event=None):
        if idx != only_index:
            return f"u{idx}"
        started.set()
        while not release.is_set():
            if cancel_event is not None and cancel_event.is_set():
                break
            release.wait(timeout=0.05)
        if cancel_event is not None and cancel_event.is_set():
            cancel_seen.set()
        return f"u{idx}"

    return build_fn, release, started, cancel_seen


class TestCooperativeCancellation:
    def test_timeout_cancels_in_flight_build_and_next_unit_delivered(self):
        build_fn, _release, started, cancel_seen = _mk_blocking_build(only_index=0)
        pf = _pf(build_fn, total=2, depth=2, timeout=0.4)
        pf.start([0, 1])
        assert started.wait(2.0), "slot 0 build must start"
        with pytest.raises(PrefetchTimeout):
            pf.get(0)
        assert pf.stats["timeouts"] == 1
        assert cancel_seen.wait(3.0), "in-flight build must observe cancel"
        assert _wait_until(lambda: pf.stats["cancelled"] == 1)
        payload, wait_s, timing = pf.get(1)
        assert payload == "u1"
        pf.close()

    def test_build_cancelled_is_never_counted_as_failure(self):
        def build_fn(item, idx, cancel_event=None):
            raise UnitBuildCancelled()

        pf = _pf(build_fn, total=1, depth=1, timeout=0.3, retries=3)
        pf.start([0])
        with pytest.raises(PrefetchTimeout):
            pf.get(0)
        assert pf.stats["errors"] == 0, "cancel is not an error"
        assert pf.stats["nonretryable"] == 0, "cancel is not non-retryable"
        assert pf.stats["retries"] == 0, "cancel must never trigger retries"
        assert _wait_until(lambda: pf.stats["cancelled"] == 1)
        assert pf.state_counts().get("CANCELLED", 0) == 1
        pf.close()

    def test_explicit_cancel_stops_build_and_worker_moves_on(self):
        build_fn, _release, started, cancel_seen = _mk_blocking_build(only_index=0)
        pf = _pf(build_fn, total=2, depth=2, timeout=5.0)
        pf.start([0, 1])
        assert started.wait(2.0)
        pf.cancel(0)
        assert cancel_seen.wait(3.0)
        assert _wait_until(lambda: pf.stats["cancelled"] == 1)
        assert pf.is_alive()
        _release.set()
        pf.close()
        assert not pf.is_alive()

    def test_cancel_is_idempotent_and_noop_for_delivered_slots(self):
        def build_fn(item, idx, cancel_event=None):
            return f"u{idx}"

        pf = _pf(build_fn, total=2, depth=2, timeout=5.0)
        pf.start([0, 1])
        payload, _, _ = pf.get(0)
        assert payload == "u0"
        pf.cancel(0)  # delivered — must be a no-op
        pf.cancel(0)
        pf.cancel(999)  # never scheduled — no-op
        payload1, _, _ = pf.get(1)
        assert payload1 == "u1"
        assert pf.stats["cancelled"] == 0
        pf.close()


class TestOwnerfulClose:
    def test_close_joins_idle_workers(self):
        def build_fn(item, idx, cancel_event=None):
            return f"u{idx}"

        pf = _pf(build_fn, total=2, depth=2, timeout=5.0)
        pf.start([0, 1])
        pf.get(0)
        pf.get(1)
        pf.close()
        assert not pf.is_alive(), "close() must drain idle workers"

    def test_close_bounded_join_reports_leftover_daemons(self):
        build_fn, release, started, _seen = _mk_blocking_build(only_index=0)
        pf = _pf(build_fn, total=1, depth=1, timeout=5.0)
        pf.start([0])
        assert started.wait(2.0)
        # join_timeout=0: close() must return immediately even though the
        # worker is still inside a (non-abortable) build.
        pf.close(join_timeout=0.0)
        assert pf.is_alive(), "non-abortable build may still be running"
        assert not pf.wait_idle(timeout=0.0)
        release.set()
        pf.close(join_timeout=1.0)
        assert not pf.is_alive()

    def test_active_builds_reports_in_flight_slots(self):
        build_fn, release, started, _seen = _mk_blocking_build(only_index=0)
        pf = _pf(build_fn, total=1, depth=1, timeout=5.0)
        pf.start([0])
        assert started.wait(2.0)
        assert 0 in pf.active_builds(), "slot 0 must be reported in-flight"
        release.set()
        payload, _, _ = pf.get(0)
        assert payload == "u0"
        assert pf.active_builds() == {}
        pf.close()


class TestRetryBackoff:
    def test_retry_backoff_delays_between_attempts(self):
        calls = {"n": 0}

        def build_fn(item, idx, cancel_event=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient shard failure")
            return f"u{idx}"

        pf = _pf(build_fn, total=1, depth=1, timeout=5.0, retries=3,
                 retry_backoff_base=0.1, retry_backoff_max=0.4)
        pf.start([0])
        payload, _, _ = pf.get(0)
        pf.close()
        assert payload == "u0"
        assert calls["n"] == 3, "1 initial + 2 retries"
        assert pf.stats["retries"] == 2
        assert pf.stats["errors"] == 0

    def test_retry_backoff_never_blocks_shutdown(self):
        calls = {"n": 0}

        def build_fn(item, idx, cancel_event=None):
            calls["n"] += 1
            raise ValueError("always fails")

        pf = _pf(build_fn, total=1, depth=1, timeout=5.0, retries=2,
                 retry_backoff_base=60.0, retry_backoff_max=120.0)
        pf.start([0])
        t0 = time.time()
        pf.close(join_timeout=0.5)
        assert time.time() - t0 < 5.0, "close() must not wait out backoff"
        pf.close(join_timeout=1.0)


class TestQueueSafety:
    def test_depth_bound_holds_under_mixed_timeouts_and_successes(self):
        """Every slot is eventually consumed-or-timed-out in order, claimed
        inflight work never exceeds the buffer depth, and no worker or slot is
        leaked after close()."""
        def build_fn(item, idx, cancel_event=None):
            if idx % 3 == 0:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        return f"u{idx}"
                    time.sleep(0.02)
            time.sleep(0.002)
            return f"u{idx}"

        total = 12
        depth = 2
        pf = _pf(build_fn, total=total, depth=depth, timeout=0.15)
        pf.start(list(range(total)))
        consumed = 0
        delivered = []
        max_inflight = 0
        deadline = time.time() + 30
        while consumed < total and time.time() < deadline:
            max_inflight = max(max_inflight, pf.in_flight())
            try:
                payload, _, _ = pf.get(consumed)
                delivered.append(payload)
            except PrefetchTimeout:
                pass
            finally:
                consumed += 1
        assert len(delivered) + pf.stats["timeouts"] == total
        assert delivered == [f"u{i}" for i in range(total) if i % 3 != 0]
        assert max_inflight <= depth, f"inflight {max_inflight} exceeded depth {depth}"
        pf.close()
        assert not pf.is_alive()