"""Phase 2/3 (mandate §2-3): bounded, observable cleanup-worker topology.

Acceptance: the default pool size is capped at a small documented bound (not
min(32, cpu_count)); an explicit cleanup_pool_size still overrides; a fork that
is refused at creation (the "os.fork is unsafe while filelock is changing
descriptor ownership" guard) falls back to *spawn* — never a silent
"sequential" followed by a second 32-worker fork attempt; only when every
backend fails does work degrade to sequential, with an explicit warning.
"""
import logging
import multiprocessing
import os

import pytest

from src.data.pipeline import DEFAULT_CLEANUP_POOL_WORKERS, DataPipeline


def _pipe(cleanup_pool_size):
    class D:
        hf_token = ""
        quality = type("Q", (), {
            "deduplication": type("U", (), {"method": "exact", "threshold": 0.85})(),
            "contamination": type("C", (), {"benchmarks": []})(),
        })()
        metadata_cache = type("M", (), {"dir": ".", "enabled": False,
                                        "fingerprint_version": 1})()
    D.cleanup_pool_size = cleanup_pool_size

    class Cfg:
        data = D()

    pipe = DataPipeline.__new__(DataPipeline)
    pipe.cfg = Cfg()
    pipe._cleanup_pool = None
    return pipe


def _teardown(pipe):
    try:
        pipe.close()
    finally:
        pool = pipe._cleanup_pool
        pipe._cleanup_pool = None
        if pool is not None:
            try:
                pool.terminate()
            except Exception:  # noqa: BLE001
                pass


def test_cleanup_pool_bounded_default():
    """No configuration → capped at DEFAULT_CLEANUP_POOL_WORKERS, never all cores."""
    expected = min(DEFAULT_CLEANUP_POOL_WORKERS, os.cpu_count() or 2)
    pipe = _pipe(None)
    try:
        p = pipe._get_cleanup_pool()
        assert p is not None
        assert p._processes == expected, (
            f"default pool must be bounded at {DEFAULT_CLEANUP_POOL_WORKERS}, "
            f"got {p._processes} (nproc={os.cpu_count()})")
    finally:
        _teardown(pipe)


def test_cleanup_pool_config_override():
    """cleanup_pool_size overrides the default bound (both above and below)."""
    for size in (3, 16):
        pipe = _pipe(size)
        try:
            p = pipe._get_cleanup_pool()
            assert p is not None
            assert p._processes == size
        finally:
            _teardown(pipe)


def test_cleanup_pool_fork_guard_falls_back_to_spawn(monkeypatch, caplog):
    """A fork refused at creation (filelock descriptor-ownership guard) must
    fall back to spawn and still produce a bounded pool + an explicit warning."""
    real = multiprocessing.get_context
    guard_msg = "os.fork is unsafe while filelock is changing descriptor ownership"

    def fake(name):
        if name == "fork":
            raise RuntimeError(guard_msg)
        return real(name)

    monkeypatch.setattr(multiprocessing, "get_context", fake)

    pipe = _pipe(2)
    try:
        with caplog.at_level(logging.INFO, logger="src.data.pipeline"):
            p = pipe._get_cleanup_pool()
        assert p is not None, "fork refusal must NOT degrade to sequential when spawn works"
        assert p._processes == 2
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "fork" in joined and "spawn" in joined
    finally:
        _teardown(pipe)


def test_cleanup_pool_sequential_only_when_all_backends_fail(monkeypatch, caplog):
    """Only when both fork AND spawn fail does cleanup degrade to sequential,
    with an explicit warning — and the next request still retries."""
    def fake(name):
        raise RuntimeError(f"{name} backend exploded")

    monkeypatch.setattr(multiprocessing, "get_context", fake)

    pipe = _pipe(2)
    try:
        with caplog.at_level(logging.INFO, logger="src.data.pipeline"):
            p1 = pipe._get_cleanup_pool()
            p2 = pipe._get_cleanup_pool()
        assert p1 is None and p2 is None
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "cleanup will run sequentially" in joined
    finally:
        _teardown(pipe)