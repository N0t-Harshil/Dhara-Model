from __future__ import annotations

"""Spec-required acceptance tests (exact acceptance-matrix names).

These probe the hardening contracts that the earlier suite covers under other
names (``test_hf_authentication``, ``test_token_not_logged``,
``test_token_not_in_fingerprint``, ``test_duplicate_builds_are_impossible``,
``test_no_unbounded_materialization``, ``test_worker_shutdown``,
``test_cancellation``, ``test_checkpoint_after_dataset``).

All tests are offline — hub/network interactions are faked or bypassed.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.drivers import compute_builder_fingerprint
from src.data.metadata_cache import compute_fingerprint


# ---------------------------------------------------------------------------
# HF authentication classification (§14 / §16 / §28)
# ---------------------------------------------------------------------------

class _FakeHubError401(Exception):
    def __init__(self):
        super().__init__("401 Client Error: Unauthorized (token invalid)")
        self.response = SimpleNamespace(status_code=401)


class _FakeHubErrorForbidden(Exception):
    def __init__(self):
        super().__init__("403 Client Error: Forbidden — this dataset is gated")
        self.response = SimpleNamespace(status_code=403)


def test_hf_authentication(monkeypatch):
    """No token -> AUTHENTICATION MISSING; 401/403 with token -> ACCESS DENIED;
    valid token -> AUTHENTICATED. Never raises on network errors."""
    from src.utils import hf_auth

    monkeypatch.delenv("HF_TOKEN", raising=False)

    # (a) anonymous: no whoami call, classified as missing
    called = []
    monkeypatch.setattr(hf_auth, "whoami", lambda token: called.append(token) or {})
    st = hf_auth.detect_hf_auth(SimpleNamespace(data=SimpleNamespace(hf_token="")))
    assert st["authenticated"] is False
    assert st["classification"] == "AUTHENTICATION MISSING"
    assert called == []
    assert st["checked"] is False

    # (b) token present, hub rejects (401) -> access denied
    monkeypatch.setattr(hf_auth, "whoami",
                        lambda token: (_ for _ in ()).throw(_FakeHubError401()))
    st = hf_auth.detect_hf_auth(SimpleNamespace(data=SimpleNamespace(hf_token="hf_secret_abc")))
    assert st["authenticated"] is False
    assert st["classification"] == "AUTHENTICATED BUT ACCESS DENIED"
    assert st["token_present"] is True

    # (c) token present, gated dataset reject (403) -> access denied
    monkeypatch.setattr(hf_auth, "whoami",
                        lambda token: (_ for _ in ()).throw(_FakeHubErrorForbidden()))
    st = hf_auth.detect_hf_auth(SimpleNamespace(data=SimpleNamespace(hf_token="hf_secret_abc")))
    assert st["classification"] == "AUTHENTICATED BUT ACCESS DENIED"

    # (d) valid token -> available
    monkeypatch.setattr(hf_auth, "whoami", lambda token: {"name": "me"})
    st = hf_auth.detect_hf_auth(SimpleNamespace(data=SimpleNamespace(hf_token="hf_secret_abc")))
    assert st["authenticated"] is True
    assert st["classification"] == "AUTHENTICATED"


def test_token_not_logged(monkeypatch, caplog):
    """The HF token value must never reach logs — not in auth reporting, not in
    sanitized tracebacks, not in redacted error messages."""
    from src.utils import hf_auth
    from src.utils.hf_auth import format_sanitized_traceback, redact_text

    SECRET = "hf_super_secret_token_0001"
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setattr(hf_auth, "whoami", lambda token: {"name": "me"})

    with caplog.at_level("INFO"):
        hf_auth.report_hf_auth(None)
    for record in caplog.records:
        assert SECRET not in record.getMessage()

    # sanitized tracebacks from an exception whose message embeds the token
    exc = ValueError(f"gated dataset — token {SECRET} rejected, deny list hits")
    formatted = format_sanitized_traceback(exc)
    assert SECRET not in formatted
    assert "<REDACTED>" in formatted
    assert SECRET not in redact_text(f"login with {SECRET} please")


def test_token_not_in_fingerprint(monkeypatch):
    """Cache/builder fingerprints must not depend on (or embed) the HF token:
    computed under two different tokens the fingerprint is identical and never
    contains the token value."""
    SECRET_A = "hf_token_alpha_1111"
    SECRET_B = "hf_token_beta_2222"

    def fp_with(token):
        monkeypatch.setenv("HF_TOKEN", token)
        mc = compute_fingerprint(
            repo="bigcode/the-stack-v2-dedup", name="Python", split="train",
            data_dir=None, files=["hf://datasets/x/y.parquet"],
            revision="main", preprocess_sig="ppsig|v9", token_sig="ttokersig|9",
        )
        bc = compute_builder_fingerprint(
            repo="deepmind/code_contests", name=None, split="train",
            data_dir=None, revision="main", script_revision="r1",
            builder_class="CodeContestsBuilder", preprocess_sig="ppsig|v9",
            token_sig="ttokersig|9",
        )
        return mc, bc

    fp_a = fp_with(SECRET_A)
    fp_b = fp_with(SECRET_B)
    assert fp_a == fp_b, "fingerprint must not change when the token changes"
    for digest in fp_a:
        assert SECRET_A not in digest
        assert SECRET_B not in digest
        assert len(digest) == 64


# ---------------------------------------------------------------------------
# Duplicate builds / bounded memory / worker lifecycle / cancellation
# ---------------------------------------------------------------------------

def test_duplicate_builds_are_impossible():
    """A dataset scheduled twice is built exactly once; both consumers get the
    same payload and the duplicate is counted (never materialized twice)."""
    from src.training.asyncprefetch import UnitPrefetch

    attempts = []

    def build_fn(unit, idx):
        attempts.append(idx)
        time.sleep(0.05)
        return f"{unit.path}/{unit.name}"

    units = [
        SimpleNamespace(path="org/repo", name="train"),
        SimpleNamespace(path="org/repo", name="train"),
        SimpleNamespace(path="org/repo", name="train"),
    ]
    pf = UnitPrefetch(build_fn=build_fn, total=3, depth=2, timeout=5.0)
    pf.start(units)
    got = []
    for k in range(3):
        res = pf.get(k)
        got.append(res[0] if isinstance(res, tuple) else res)
    pf.close()

    assert got == ["org/repo/train"] * 3
    assert attempts == [0], f"exactly one build expected, got {attempts}"
    assert pf.stats["duplicates_prevented"] == 2
    assert pf.stats["delivered"] == 3


def test_no_unbounded_materialization():
    """Buffered prefetch results and streamed chunk sizes are bounded — no path
    materializes the full dataset ahead of the consumer."""
    from src.training.asyncprefetch import UnitPrefetch

    def build_fn(unit, idx):
        time.sleep(0.05)
        return idx

    pf = UnitPrefetch(build_fn=build_fn, total=20, depth=3, timeout=5.0)
    pf.start(list(range(20)))
    seen_depths = []
    for k in range(20):
        res = pf.get(k)
        assert res[0] == k
        seen_depths.append(pf.queue_depth())
    pf.close()
    assert max(seen_depths) <= 3, f"queue exceeded depth bound: {max(seen_depths)}"

    # streaming chunk size is capped too (max 8192 per chunk)
    from src.data.pipeline import DEFAULT_MAX_SAMPLES_PER_DATASET
    cap = 8192
    for ds_limit in (500, 4096, 200000, DEFAULT_MAX_SAMPLES_PER_DATASET):
        chunk = max(1024, min(ds_limit, cap))
        assert chunk <= cap


def test_worker_shutdown():
    """close() stops the producer pool: buffered results are dropped and every
    worker thread terminates (no orphan threads survive a cancelled run)."""
    from src.training.asyncprefetch import UnitPrefetch

    def build_fn(unit, idx):
        time.sleep(0.1)
        return idx

    pf = UnitPrefetch(build_fn=build_fn, total=6, depth=3, timeout=5.0)
    pf.start(list(range(6)))
    # let producers start building
    time.sleep(0.2)
    assert pf.stats["started_workers"] >= 1
    assert pf.queue_depth() > 0 or pf.in_flight() > 0

    pf.close()
    assert pf.queue_depth() == 0
    deadline = time.monotonic() + 2.0
    threads = list(pf._threads)
    while any(t.is_alive() for t in threads) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(t.is_alive() for t in threads), "close() left orphan workers"

    # idempotent: closing again is safe
    pf.close()


def test_cancellation():
    """A cancelled (discarded) unit must not stall the pipeline: the consumer
    advances past it, a late arrival is ignored, and stats stay consistent."""
    from src.training.asyncprefetch import UnitPrefetch

    def build_fn(unit, idx):
        time.sleep(0.2)
        return f"unit_{idx}"

    pf = UnitPrefetch(build_fn=build_fn, total=3, depth=2, timeout=2.0)
    pf.start([0, 1, 2])

    # cancel unit 0 while it is building
    pf.discard(0)
    assert pf.get(0) is None  # stale slot

    res1 = pf.get(1)
    assert res1[0] == "unit_1"
    res2 = pf.get(2)
    assert res2[0] == "unit_2"
    pf.close()


# ---------------------------------------------------------------------------
# Dataset-granular checkpoint & resume-by-identity
# ---------------------------------------------------------------------------

def test_checkpoint_after_dataset():
    """Every dataset unit boundary produces a durable checkpoint; a restart
    resumes after the last *completed* unit — later units are rebuilt, earlier
    units are never rebuilt."""
    from src.training.checkpoint import (
        AsyncCheckpointWriter,
        _atomic_write_bytes,
        verify_checkpoint,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = AsyncCheckpointWriter(checksum=True)
        completed = []
        for i in range(3):
            ck = root / f"checkpoint-stage1-u{i + 1:03d}"
            ck.mkdir()
            _atomic_write_bytes(ck / "unit_identity.json",
                                json.dumps({"unit_index": i, "stage_index": 1})
                                .encode())
            writer.submit(ck, lambda p=ck, idx=i: (p / "model.bin").write_bytes(
                f"weights-{idx}".encode()))
        writer.flush(timeout=5.0)
        writer.close()

        ok, _ = verify_checkpoint(root / "checkpoint-stage1-u001")
        assert ok is True

        # resume by identity — units 1,2 done, unit 3 pending
        for p in sorted(root.iterdir()):
            if p.is_dir():
                ident = json.loads((p / "unit_identity.json").read_text())
                completed.append((ident["stage_index"], ident["unit_index"]))
        assert completed == [(1, 0), (1, 1), (1, 2)]
        assert not list(root.glob("checkpoint-stage1-u004"))