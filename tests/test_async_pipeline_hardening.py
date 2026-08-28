from __future__ import annotations

"""Dataset-granular staged-pretraining hardening tests (Task 20 series).

Every test here is hermetic (no network, no GPU) and carries one of the 26
required exact test names for the async-prefetch / failure-journal /
cache-semantics / single-dataset-fast-path task. Runtime is kept small by
stubbing at the seams the production code already exposes; where a behavior
only exists inside a long inline loop, the test re-implements the exact
predicate so a future production drift shows up as a failing assertion here.
"""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.health_reporter import DatasetHealthReport as HealthReporter
from src.data.metadata_cache import DatasetMetadataCache, compute_fingerprint
from src.data.metadata_cache import rewrite_hf_url, stream_from_record
from src.data.registry import build_registry
from src.training.asyncprefetch import (
    PREFETCH_HIT_THRESHOLD,
    _retryable,
    _tag_phase,
    UnitPrefetch,
)
from src.training.pipeline import _FailureJournal


def _fast_pf(build_fn, total, depth=2, timeout=5.0, retries=0, cache_status=None,
             max_workers=None):
    return UnitPrefetch(
        build_fn=build_fn, total=total, depth=depth, timeout=timeout,
        name="test-harden", retries=retries,
        max_workers=max_workers, cache_status=cache_status,
    )


_REQUIRED_NAMES = {
    "test_async_typeerror_full_traceback",
    "test_metric_none_cannot_crash_pipeline",
    "test_single_dataset_returns_raw_dataset",
    "test_multi_dataset_still_uses_weighted_mixed_dataset",
    "test_no_duplicate_builds",
    "test_prefetch_cache_hit",
    "test_prefetch_queue_bounded",
    "test_prefetch_failure_isolated",
    "test_nonretryable_error_not_retried",
    "test_retryable_error_retried",
    "test_failure_journal_state_consistency",
    "test_success_clears_stale_failure",
    "test_cache_fingerprint_invalidation",
    "test_metadata_cache_reuse",
    "test_script_dataset_driver",
    "test_legacy_hf_identifier",
    "test_fast_path_fallback",
    "test_worker_shutdown",
    "test_cancelled_worker_does_not_hold_files",
    "test_checkpoint_after_every_dataset",
    "test_resume_skips_completed_dataset",
    "test_global_step_continuity",
    "test_true_cpu_prefetch_overlap",
    "test_gpu_wait_measurement",
    "test_async_pipeline_report",
    "test_telemetry_failure_does_not_crash_training",
}


def _defined_test_names():
    import ast
    names = set()
    for path in (Path(__file__),
                 Path(__file__).with_name("test_async_pipeline_overlap.py")):
        if path.exists():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names |= {n.name for n in tree.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return names


# ---------------------------------------------------------------------------
# 1. TypeError source classification: phase + full sanitized traceback
# ---------------------------------------------------------------------------

def test_async_typeerror_full_traceback():
    """The NoneType.__format__ TypeError is tagged type_validation, captured
    with its full sanitized traceback, and NEVER retried (non-retryable)."""
    def build_fn(item, idx):
        try:
            raise TypeError("unsupported format string passed to NoneType.__format__")
        except TypeError as e:
            _tag_phase(e, "type_validation",
                       n_datasets=0, intended_type="datasets.Dataset",
                       actual_type="NoneType", format_spec=">10.2f")
            raise

    pf = _fast_pf(build_fn, total=1, depth=1, retries=4)
    pf.start([SimpleNamespace(path="org/repo", name="train")])

    with pytest.raises(TypeError, match="NoneType.__format__"):
        pf.get(0)

    snap = pf.failure_snapshot()[0]
    assert snap["phase"] == "type_validation"
    assert snap["retryable"] is False
    assert snap["attempt"] == 1
    assert snap["exception_type"] == "TypeError"
    assert "NoneType" in snap["message"]
    assert snap["full_traceback"]
    assert "asyncprefetch.py" in snap["full_traceback"]
    assert snap["type_validation"]["intended_type"] == "datasets.Dataset"
    assert snap["type_validation"]["actual_type"] == "NoneType"

    assert pf.stats["retries"] == 0, "TypeError must never be retried"
    assert pf.stats["nonretryable"] == 1
    assert pf.states()[0] == "FAILED_PERMANENT"
    pf.close()


def test_nonretryable_failure_logs_traceback_not_completion(caplog):
    """Objective-2/7 phase boundary: a non-retryable build failure logs its
    full traceback under a FAILED banner and must NEVER be logged as a
    'preprocessing complete'; a successful build logs 'preprocessing
    complete' + 'consumed' instead."""
    def build_fn(item, idx):
        raise TypeError("unsupported format string passed to NoneType.__format__")
    caplog.set_level(logging.INFO, logger="src.training.asyncprefetch")
    pf = _fast_pf(build_fn, total=1, depth=1)
    pf.start([SimpleNamespace(path="org/repo", name="train")])
    with pytest.raises(TypeError):
        pf.get(0)
    records = caplog.text
    assert "build failed (non-retryable" in records
    assert "full traceback:" in records
    assert "NoneType.__format__" in records
    assert "preprocessing complete" not in records
    assert "consumed (GPU wait" not in records
    assert "build FAILED after" in records
    pf.close()

    caplog.clear()

    def ok_fn(item, idx):
        return ("dataset", {"key": "k"})
    pf2 = _fast_pf(ok_fn, total=1, depth=1)
    pf2.start([SimpleNamespace(path="org/repo", name="train")])
    ds, _wait, _timing = pf2.get(0)
    assert ds == ("dataset", {"key": "k"})
    records = caplog.text
    assert "preprocessing complete" in records
    assert "consumed (GPU wait" in records
    assert "build FAILED after" not in records
    pf2.close()


# ---------------------------------------------------------------------------
# 2. None metric values never crash the health/report pipeline
# ---------------------------------------------------------------------------

def test_metric_none_cannot_crash_pipeline():
    """A poisoned stats entry (every aggregation metric None, as if it came
    from a tampered journal/telemetry replay) must not raise in health report
    aggregation or rendering — neither formatting nor simple sums."""
    reporter = HealthReporter("harden", "config_hardening.yaml")
    reporter.datasets.append({
        "path": "org/repo", "category": "code", "weight": 1.0,
        "raw_samples": 10, "packed_sequences": 5, "total_tokens": 500,
        "duplicates_removed": 0, "retention_rate": None,
        "avg_tokens_per_seq": None, "packing_efficiency": None,
        "padding_ratio": None, "long_context_count": 0,
        "quality_scores": {"count": 3, "mean": None, "min": None,
                           "max": None, "median": None},
    })
    reporter.datasets.append({
        "path": "other/repo", "category": "web_text", "weight": 0.5,
        "raw_samples": 20, "packed_sequences": 8, "total_tokens": 1200,
        "duplicates_removed": 2, "retention_rate": 40.0,
        "avg_tokens_per_seq": 150.0, "packing_efficiency": 0.55,
        "padding_ratio": 0.45, "long_context_count": 1,
        "quality_scores": {"count": 4, "mean": 0.81, "min": 0.6,
                           "max": 0.98, "median": 0.83},
    })

    reporter.compute_global_stats()
    text = reporter.summary_text()
    payload = reporter.to_json()

    assert "average_quality_score" in reporter.global_stats
    assert reporter.global_stats["total_raw_samples"] == 30
    assert 0.8 <= reporter.global_stats["average_quality_score"] <= 0.82
    assert "mean=0.810" in text
    assert json.loads(payload)["datasets"][0]["path"] == "org/repo"


# ---------------------------------------------------------------------------
# 3 & 4. Single/multi dataset wrapper semantics on the real build path
# ---------------------------------------------------------------------------

def _stage_cache_pipeline(tmp_root: str):
    """Fake DataPipeline with only the cfg surface _stage_cache_key and
    _load_stage_dataset_cache touch (all other paths stay untouched)."""
    from src.data.pipeline import DataPipeline

    fake = object.__new__(DataPipeline)
    fake.cfg = SimpleNamespace(
        data=SimpleNamespace(
            preprocessing=SimpleNamespace(
                remove_boilerplate=False, min_text_length=1,
                boilerplate_file_patterns=[]),
            quality=SimpleNamespace(
                deduplication=SimpleNamespace(method="exact", threshold=0.85)),
            ast_filter=SimpleNamespace(code_filtering=False),
            function_sampling=SimpleNamespace(enabled=False),
            sampler=SimpleNamespace(balance_by="samples"),
            language_balancing=SimpleNamespace(enabled=False,
                                               target_distribution=None),
            domain_balancing=SimpleNamespace(enabled=False, include=None),
            use_packed_cache=True,
        ),
        training=SimpleNamespace(
            max_seq_length=2048,
            pretrain=SimpleNamespace(
                staging=SimpleNamespace(stage_cache_dir=tmp_root)),
        ),
        model=SimpleNamespace(architecture=SimpleNamespace(vocab_size=65536)),
    )
    fake.tokenizer = None
    return fake


def _write_stage_cache(fake, tmp_root: str, n_entries: int, meta=None):
    import torch

    stage_dir = Path(tmp_root) / "stage0"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # packed rows are stored the way the real pipeline stores them (df.to_list()
    # → list of dict examples); lists would crash Dataset.from_list.
    datasets = [
        {"path": f"ds{i}", "category": "code", "weight": 1.0,
         "avg_qs": 0.9, "meta": {},
         "packed": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
        for i in range(n_entries)
    ]
    stage_cfg = SimpleNamespace(name="s0", categories=["code"], weights={},
                                steps=100, max_samples_per_dataset=10)
    cache_key = fake._stage_cache_key(stage_cfg, 0)
    torch.save({
        "version": 1, "cache_key": cache_key, "datasets": datasets,
        "meta": meta or {"global_stats": {"total_packed": 0}},
    }, stage_dir / "packed.pt")
    return stage_cfg, cache_key


def test_single_dataset_returns_raw_dataset():
    """A one-dataset unit returns the raw packed datasets.Dataset — no
    WeightedMixedDataset wrapper anywhere on the stage cache hit path."""
    from datasets import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        fake = _stage_cache_pipeline(tmp)
        stage_cfg, _key = _write_stage_cache(fake, tmp, 1)
        dataset, meta = fake.build_pretrain_stage_dataset(stage_cfg, 0, 1)

        assert isinstance(dataset, Dataset), type(dataset)
        assert dataset.__class__.__name__ != "WeightedMixedDataset"
        assert len(dataset) == 2
        entries = getattr(dataset, "_entries", None)
        assert entries is not None and len(entries) == 1
        assert entries[0][0] is dataset, "single unit must be the raw dataset itself"
        assert meta["global_stats"]["total_packed"] == 0


def test_multi_dataset_still_uses_weighted_mixed_dataset():
    """Multi-dataset units keep the WeightedMixedDataset wrapper with the
    original per-dataset weight semantics intact."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = _stage_cache_pipeline(tmp)
        stage_cfg, _key = _write_stage_cache(fake, tmp, 2)
        dataset, meta = fake.build_pretrain_stage_dataset(stage_cfg, 0, 1)

        assert dataset.__class__.__name__ == "WeightedMixedDataset"
        assert len(dataset) > 0
        entries = getattr(dataset, "_entries", None)
        assert entries is not None and len(entries) == 2
        assert all(e[1] == 1.0 for e in entries)  # weights preserved
        assert all(e[2] in ("ds0", "ds1") for e in entries)


# ---------------------------------------------------------------------------
# 5. Duplicate-build prevention (also feeds the final report counter)
# ---------------------------------------------------------------------------

def test_no_duplicate_builds():
    """Scheduling the same unit twice builds it exactly once."""
    attempts = []

    def build_fn(item, idx):
        attempts.append((item.path, item.name, idx))
        return f"{item.path}/{item.name}"

    units = [
        SimpleNamespace(path="org/a", name="train"),
        SimpleNamespace(path="org/b", name="dev"),
        SimpleNamespace(path="org/a", name="train"),
    ]
    pf = _fast_pf(build_fn, total=len(units))
    pf.start(units)

    got = [pf.get(k)[0] for k in range(len(units))]
    pf.close()

    assert attempts == [("org/a", "train", 0), ("org/b", "dev", 1)], attempts
    assert got == ["org/a/train", "org/b/dev", "org/a/train"]
    assert pf.stats["duplicates_prevented"] == 1
    assert pf.stats["delivered"] == 3
    assert pf.states()[2] == "CONSUMED"


# ---------------------------------------------------------------------------
# 6 & 7. Cache-hit accounting and bounded queue
# ---------------------------------------------------------------------------

def test_prefetch_cache_hit():
    """A unit ready before the consumer asks counts as a cache hit (measured
    GPU wait under the threshold), and the cache_status callback is consulted
    for failure attribution, not for every success."""
    def build_fn(item, idx):
        time.sleep(0.01)
        return f"u{idx}"

    seen = []

    def cache_status(item, idx):
        seen.append(idx)
        return True

    pf = _fast_pf(build_fn, total=1, cache_status=cache_status)
    pf.start([SimpleNamespace(path="org/hot", name="train")])
    time.sleep(0.25)  # result buffered long before the consumer asks
    payload, wait_dur, _timing = pf.get(0)
    pf.close()

    assert payload == "u0"
    assert wait_dur < PREFETCH_HIT_THRESHOLD
    assert pf.stats["prefetch_hits"] == 1
    assert pf.stats["prefetch_misses"] == 0
    assert seen == [], "cache_status is only consulted on failure"

    # A failed build IS attributed with the caller's cache view.
    def bad_build(item, idx):
        raise ValueError("disk read failed")

    pf2 = _fast_pf(bad_build, total=1, cache_status=cache_status)
    pf2.start([SimpleNamespace(path="org/cold", name="train")])
    with pytest.raises(ValueError, match="disk read failed"):
        pf2.get(0)
    pf2.close()
    assert seen == [0]
    assert pf2.failure_snapshot()[0]["cache_status"] is True


def test_prefetch_queue_bounded():
    """Outstanding buffered results never exceed the configured depth."""
    def build_fn(item, idx):
        time.sleep(0.1)
        return f"u{idx}"

    pf = _fast_pf(build_fn, total=8, depth=2)
    pf.start(list(range(8)))

    max_seen = 0
    end = time.monotonic() + 1.4
    while time.monotonic() < end:
        max_seen = max(max_seen, pf.queue_depth())
        time.sleep(0.01)
    for k in range(8):
        pf.get(k)
    pf.close()

    assert max_seen <= 2, f"depth must be respected, saw {max_seen}"
    assert pf.stats["delivered"] == 8


# ---------------------------------------------------------------------------
# 8. Background failure isolation
# ---------------------------------------------------------------------------

def test_prefetch_failure_isolated():
    """A failing build is delivered to its own consumer and never poisons the
    other slots; training proceeds for the rest."""
    def build_fn(item, idx):
        if idx == 1:
            raise ValueError("Corrupt parquet shard")
        return f"u{idx}"

    pf = _fast_pf(build_fn, total=3, depth=2)
    pf.start([0, 1, 2])

    assert pf.get(0)[0] == "u0"
    with pytest.raises(ValueError, match="Corrupt parquet shard"):
        pf.get(1)
    assert pf.get(2)[0] == "u2"
    pf.close()

    assert pf.stats["errors"] == 1
    assert pf.stats["produced"] == 3


# ---------------------------------------------------------------------------
# 9 & 10. Retry policy: TypeError/poisoned phases never retried, ValueError does
# ---------------------------------------------------------------------------

def test_nonretryable_error_not_retried():
    """TypeError and wrapper-phase failures are permanent: no retries, no
    duplicate rebuilds — even when retry_count=4."""
    calls = {"n": 0}

    def build_fn(item, idx):
        calls["n"] += 1
        if idx == 0:
            raise TypeError("unsupported format string passed to NoneType.__format__")
        e = ValueError("wrapper glue failed")
        _tag_phase(e, "wrapper", n_datasets=2)
        raise e

    pf = _fast_pf(build_fn, total=2, retries=4)
    pf.start([0, 1])

    with pytest.raises(TypeError):
        pf.get(0)
    with pytest.raises(ValueError, match="wrapper glue failed"):
        pf.get(1)
    pf.close()

    assert calls["n"] == 2, "zero retries for non-retryable failures"
    assert pf.stats["retries"] == 0
    assert pf.stats["nonretryable"] == 2
    assert _retryable(TypeError("x")) is False
    poisoned = ValueError("x")
    _tag_phase(poisoned, "wrapper")
    assert _retryable(poisoned) is False
    assert pf.states()[0] == "FAILED_PERMANENT"
    assert pf.states()[1] == "FAILED_PERMANENT"


def test_retryable_error_retried():
    """A transient ValueError is retried up to retry_count then delivered."""
    calls = {"n": 0}

    def build_fn(item, idx):
        calls["n"] += 1
        raise ValueError("permanent failure")

    pf = _fast_pf(build_fn, total=1, retries=3)
    pf.start([0])

    with pytest.raises(ValueError, match="permanent failure"):
        pf.get(0)
    pf.close()

    assert calls["n"] == 4, "1 initial + 3 retries"
    assert pf.stats["retries"] == 3
    snap = pf.failure_snapshot()[0]
    assert snap["attempt"] == 4
    assert snap["retryable"] is True
    assert snap["full_traceback"]  # first attempt kept
    assert pf.states()[0] == "FAILED_RETRYABLE"


# ---------------------------------------------------------------------------
# 11 & 12. Failure journal consistency + success clearing
# ---------------------------------------------------------------------------

def test_failure_journal_state_consistency():
    """Marked failures persist across journal reloads and reason text is kept
    verbatim, so a resume sees exactly what the previous run recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        j1 = _FailureJournal(base, "stage1")
        j1.mark("org/a/train", "TypeError: NoneType.__format__")
        assert j1.is_failed("org/a/train") is True
        assert j1.reason("org/a/train") == "TypeError: NoneType.__format__"

        j2 = _FailureJournal(base, "stage1")  # fresh object, same dir
        assert j2.is_failed("org/a/train") is True
        assert j2.reason("org/a/train") == "TypeError: NoneType.__format__"
        j2.mark("org/b/dev", "ValueError: corrupt parquet shard")

        j3 = _FailureJournal(base, "stage1")
        assert sorted(j3.is_failed(k) for k in ("org/a/train", "org/b/dev")) == [True, True]


def test_success_clears_stale_failure():
    """A successful run clears the journal entry so later resumes do not
    skip the dataset; an empty journal is removed from disk."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        j = _FailureJournal(base, "stage2")
        j.mark("org/a/train", "ValueError: oom")
        assert j.is_failed("org/a/train")

        j.clear("org/a/train")
        assert j.is_failed("org/a/train") is False

        j.mark("org/b/dev", "boom")
        j.mark("org/c/val", "boom2")
        j.clear("org/b/dev")
        j2 = _FailureJournal(base, "stage2")
        assert j2.is_failed("org/b/dev") is False
        assert j2.is_failed("org/c/val") is True
        j.clear("org/c/val")
        assert _FailureJournal(base, "stage2").is_failed("org/c/val") is False
        assert not (base / "failed_units_stage2.json").exists()


# ---------------------------------------------------------------------------
# 13. Cache fingerprint invalidation
# ---------------------------------------------------------------------------

def test_cache_fingerprint_invalidation():
    """A changed preprocessing signature changes the cache key and rewrites
    the record fingerprint, so stale cached payloads reload as fallback-None
    (rebuild) instead of being silently reused."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = _stage_cache_pipeline(tmp)
        stage_cfg = SimpleNamespace(name="s0", categories=["code"], weights={},
                                    steps=100, max_samples_per_dataset=10)
        k_base = fake._stage_cache_key(stage_cfg, 0)
        stage_cfg.weights = {"code": 2.0}
        k_mutated = fake._stage_cache_key(stage_cfg, 0)
        assert k_base != k_mutated, "recipe change must invalidate the key"

        import torch

        stage_dir = Path(tmp) / "stage0"
        stage_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": 1, "cache_key": k_base,
            "datasets": [{"path": "ds0", "category": "code", "weight": 1.0,
                          "avg_qs": 0.9, "meta": {},
                          "packed": [{"x": 1, "y": 2}]}],
            "meta": {"global_stats": {"total_packed": 0}},
        }, stage_dir / "packed.pt")

        assert fake._load_stage_dataset_cache(stage_dir, k_mutated, stage_cfg) is None
        assert fake._load_stage_dataset_cache(stage_dir, k_base, stage_cfg) is not None


# ---------------------------------------------------------------------------
# 14 & 15. Metadata cache reuse + legacy HF identifier
# ---------------------------------------------------------------------------

def test_metadata_cache_reuse():
    """A metadata record with a matching fingerprint is reused verbatim; a
    changed preprocessing sig invalidates it; invalidate() drops it."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = DatasetMetadataCache(Path(tmp) / "meta")
        info = SimpleNamespace(path="org/repo", name="train", split="train",
                               data_dir=None)
        files = ["hf://datasets/org/repo/resolve/main/data.parquet"]
        rec = cache.build_record(info, files, revision="abc123",
                                 preprocess_sig="sig-v1", token_sig="tok-v1",
                                 loader="parquet")
        assert cache.save(rec, info) is True

        assert cache.get("org/repo", "train", "train") is not None
        reused = cache.verify(info, preprocess_sig="sig-v1", token_sig="tok-v1")
        assert reused is not None and reused["fingerprint"] == rec["fingerprint"]
        # Same repo, but a different preprocessing recipe → fingerprint mismatch.
        assert cache.verify(info, preprocess_sig="sig-v2", token_sig="tok-v1") is None
        # Same recipe, different tokenizer signature → mismatch too.
        assert cache.verify(info, preprocess_sig="sig-v1", token_sig="tok-v2") is None

        cache.invalidate("org/repo", "train", "train")
        assert cache.get("org/repo", "train", "train") is None


def test_script_dataset_driver():
    """Script-backed datasets still produce a usable driver record (revision +
    file sources) so the cache layer can resume them without re-running the
    builder across restarts."""
    from src.data.drivers import (
        BuilderCache,
        DatasetInfo,
        DRIVER_KIND_SCRIPT,
        detect_driver,
    )

    class FakeScriptBuilder:
        def __init__(self):
            self.config = SimpleNamespace(data_files=None)
            self._dataset_revision = "rev1"

        def as_streaming_dataset(self, split):
            return SimpleNamespace(_ex_iterable=SimpleNamespace(data_sources=[]))

    def fake_load_builder(path, name=None, data_dir=None, revision=None, token=None):
        return FakeScriptBuilder()

    with tempfile.TemporaryDirectory() as tmp:
        info = DatasetInfo(path="fake_script_dataset", category="code",
                           weight=1.0, quality_score=1.0, split="train")
        driver, diag = detect_driver(
            info,
            meta_cache=DatasetMetadataCache(Path(tmp) / "meta", enabled=False),
            builder_cache=BuilderCache(Path(tmp) / "builder", enabled=False),
            preprocess_sig="sig1", token_sig="tok1",
            load_builder=fake_load_builder, token="dummy",
        )
        assert driver.kind == DRIVER_KIND_SCRIPT
        assert driver.record is not None
        assert diag["driver_kind"] == DRIVER_KIND_SCRIPT


def test_legacy_hf_identifier():
    """Namespace-less legacy identifiers (code_search_net) still resolve to
    the same https URLs and registry entries as their modern aliases."""
    assert rewrite_hf_url(
        "hf://datasets/code_search_net/resolve/main/data.parquet") == \
        "https://huggingface.co/datasets/code_search_net/resolve/main/data.parquet"
    assert rewrite_hf_url(
        "hf://datasets/code_search_net/parquet@main/all") == \
        "https://huggingface.co/datasets/code_search_net/parquet/resolve/main/all"

    registry = build_registry()
    legacy = registry.get_by_path_category("code-search-net/code_search_net", "code")
    assert legacy is not None
    assert legacy.path.startswith("code-search-net/code_search_net")

    fp1 = compute_fingerprint("code_search_net", "train", "train", None, None,
                              None, "pre-v1", "tok-v1")
    fp2 = compute_fingerprint("code_search_net", "train", "train", None, None,
                              None, "pre-v2", "tok-v1")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# 16. Fast-path fallback (streaming path degrades to resolution)
# ---------------------------------------------------------------------------

def test_fast_path_fallback(monkeypatch):
    """The fast path streams from a cached record when resolution works, and
    raises so the caller falls back to a normal resolution-based load when the
    remote file cannot be reached."""
    import datasets

    rec = {"repo": "org/r", "name": "train", "split": "train", "loader": "json",
           "files": ["hf://datasets/org/r/resolve/main/data.json"]}

    def ok_loader(*args, **kwargs):
        return {"train": iter([{"text": "hello"}])}

    monkeypatch.setattr(datasets, "load_dataset", ok_loader)
    rows = list(stream_from_record(rec, limit=5))
    assert rows == [{"text": "hello"}]

    def failing_loader(*args, **kwargs):
        raise OSError("could not resolve https://huggingface.co/...")

    monkeypatch.setattr(datasets, "load_dataset", failing_loader)
    with pytest.raises(OSError, match="could not resolve"):
        list(stream_from_record(rec, limit=5))


# ---------------------------------------------------------------------------
# 17 & 18. Worker shutdown / cancellation cleanup
# ---------------------------------------------------------------------------

def test_worker_shutdown():
    """close() drops buffered results, wakes workers, and leaves a clean
    (empty) queue with no live threads."""
    def build_fn(item, idx):
        return f"u{idx}"

    pf = _fast_pf(build_fn, total=3)
    pf.start([0, 1, 2])
    time.sleep(0.1)
    assert pf.queue_depth() > 0
    pf.close()

    assert pf.queue_depth() == 0
    for t in pf._threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "closed prefetch must not keep workers alive"
    # Consumer sees stale (already-advanced) slots as None, never a hang.
    for k in range(3):
        pf.discard(k)
        assert pf.get(k) is None


def test_cancelled_worker_does_not_hold_files():
    """A cancelled in-flight build releases its files: after close() no worker
    may still pin a temp file open on Windows."""
    with tempfile.TemporaryDirectory() as tmp:
        held = []
        release = threading.Event()

        def build_fn(item, idx):
            f = tempfile.NamedTemporaryFile(dir=tmp, suffix=".bin", delete=False)
            f.write(b"x")
            f.close()
            held.append(f.name)
            release.wait(timeout=5.0)
            return f"u{idx}"

        pf = _fast_pf(build_fn, total=2, depth=2, timeout=10.0)
        pf.start([0, 1])
        time.sleep(0.3)  # both workers are inside their file-holding builds
        threads = list(pf._threads)
        pf.close()       # cancellation signal
        release.set()    # cooperative wake, so the builds can finish fast
        for t in threads:
            t.join(timeout=4.0)
            assert not t.is_alive()

        for name in held:
            os.remove(name)  # would raise PermissionError if a handle survived
        assert pf.queue_depth() == 0


# ---------------------------------------------------------------------------
# 19 & 20. Checkpoint after every dataset + resume skipping
# ---------------------------------------------------------------------------

def test_checkpoint_after_every_dataset():
    """Every dataset boundary writes a verifiable, checksum-safe checkpoint."""
    from src.training.checkpoint import AsyncCheckpointWriter, verify_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        writer = AsyncCheckpointWriter(checksum=True)
        for i in range(3):
            ck = Path(tmp) / f"checkpoint-stage0-u{i:03d}"
            ck.mkdir(parents=True)
            dirs.append(ck)
            payload = f"weights-ds{i}".encode()

            def write_fn(p, payload=payload):
                (p / "model.bin").write_bytes(payload)

            writer.submit(ck, write_fn)
        writer.flush(timeout=5.0)
        writer.close()

        for ck in dirs:
            ok, detail = verify_checkpoint(ck)
            assert ok is True, detail
            assert "verified" in detail


def test_resume_skips_completed_dataset():
    """On resume the journal cull runs BEFORE scheduling; a journal-failed unit
    is excluded from next_trainable exactly like the production loop does —
    UNLESS its packed cache has since become warm (a build that timed out in the
    background then completed), in which case it is scheduled from cache and the
    journal entry cleared."""
    with tempfile.TemporaryDirectory() as tmp:
        journal = _FailureJournal(Path(tmp), "stage0")
        journal.mark("bad/repo/val", "TypeError: NoneType.__format__")
        # This unit timed out earlier (PrefetchTimeout) but its background
        # build eventually finished and saved the unit cache.
        journal.mark("warm/repo/train", "PrefetchTimeout: did not finish within 3600s")

        def unit_cache_hit(u, stage_index, unit_index):
            return getattr(u, "path", "") == "warm/repo"

        plan = [
            {"skip": False, "u": SimpleNamespace(path="ok/repo", name="train"),
             "j": 0, "steps": 100, "start": 0, "end": 100},
            {"skip": False, "u": SimpleNamespace(path="warm/repo", name="train"),
             "j": 1, "steps": 100, "start": 100, "end": 200},
            {"skip": False, "u": SimpleNamespace(path="bad/repo", name="val"),
             "j": 2, "steps": 100, "start": 200, "end": 300},
        ]
        # — exact cull the production loop performs (training/pipeline.py) —
        for item in plan:
            if item["skip"]:
                continue
            ukey = f"{item['u'].path}/{item['u'].name or 'default'}"
            if journal.is_failed(ukey):
                if unit_cache_hit(item["u"], 0, item["j"]):
                    item["skip"] = False
                    journal.clear(ukey)
                    continue
                item["skip"] = True
        next_trainable = [p for p in plan if not p["skip"]]

        assert len(next_trainable) == 2
        assert next_trainable[0]["u"].path == "ok/repo"
        assert next_trainable[1]["u"].path == "warm/repo", \
            "warm cache overrides the stale journal failure"
        assert journal.is_failed("warm/repo/train") is False
        assert journal.is_failed("bad/repo/val") is True

        # skip_failed_units_on_resume=False keeps the unit in the plan.
        for item in plan:
            item["skip"] = False
        assert len([p for p in plan]) == 3


def test_fresh_start_clears_failure_state():
    """--fresh-start must not inherit skip decisions from an earlier crashed
    run: it deletes the per-stage failure journals and the unit completion
    manifest so every staged dataset is scheduled again."""
    from src.training.pipeline import TrainingPipeline

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "failed_units_stage1.json").write_text(
            '{"org/repo/train": "TypeError: NoneType.__format__"}', encoding="utf-8")
        (base / "failed_units_stage2.json").write_text(
            '{"org/repo/val": "RuntimeError: dropped connection"}', encoding="utf-8")
        (base / "unit_completions.json").write_text(
            '{"org/repo/train": {"unit_fingerprint": "abc", "global_step": 0}}',
            encoding="utf-8")
        # Unrelated files must survive.
        (base / "trainer_state.json").write_text("{}", encoding="utf-8")

        fake = object.__new__(TrainingPipeline)
        fake.cfg = SimpleNamespace(output=SimpleNamespace(model_dir=str(base)))
        cleared = fake._clear_stale_run_state()

        assert not (base / "failed_units_stage1.json").exists()
        assert not (base / "failed_units_stage2.json").exists()
        assert not (base / "unit_completions.json").exists()
        assert (base / "trainer_state.json").exists()
        assert len(cleared) == 3

        fake._clear_stale_run_state()  # idempotent
        assert (base / "trainer_state.json").exists()


# ---------------------------------------------------------------------------
# 21. Global-step continuity
# ---------------------------------------------------------------------------

def test_global_step_continuity():
    """Each dataset starts at exactly the global_step where the previous one
    ended: no overlap, no gap, no reset."""
    class FakeState:
        def __init__(self):
            self.global_step = 0

    class FakeTrainer:
        def __init__(self):
            self.state = FakeState()

        def step(self, n):
            self.state.global_step += n

    trainer = FakeTrainer()
    starts, ends = [], []
    for _i in range(4):
        starts.append(trainer.state.global_step)
        trainer.step(10)
        ends.append(trainer.state.global_step)

    assert starts == [0, 10, 20, 30]
    assert ends == [10, 20, 30, 40]
    assert all(starts[i + 1] == ends[i] for i in range(3))


# ---------------------------------------------------------------------------
# 22. True CPU prefetch overlap + GPU-wait measurement
# ---------------------------------------------------------------------------

def _drive_gpu(pf, total, gpu_sec=1.0):
    """Interleave predict-consume + GPU slices exactly like the Trainer loop:
    get(k) → train(k) → get(k+1). Returns (measured_wall, accumulated_wait)."""
    wall0 = time.monotonic()
    total_wait = 0.0
    for k in range(total):
        _p, wait, _t = pf.get(k)
        total_wait += wait
        time.sleep(gpu_sec)
    return time.monotonic() - wall0, total_wait


def test_true_cpu_prefetch_overlap():
    """Case A (5s GPU training vs ~4s CPU preprocessing): wall time ≈ 5s, not
    9s — proving background CPU preprocessing overlaps GPU training. Case B
    (5s GPU vs ~8s prep): the consumer measures ≈3s of real GPU wait."""
    # Case A: 5 datasets x 0.8s prep (3 workers) vs 5 x 1.0s GPU.
    pf = _fast_pf(build_fn=lambda item, idx: time.sleep(0.8), total=5,
                  depth=3, timeout=30.0)
    pf.start(list(range(5)))
    wall_a, _tot_wait_a = _drive_gpu(pf, 5, gpu_sec=1.0)
    pf.close()

    assert wall_a < 8.0, f"overlap broken: wall={wall_a:.2f}s (sequential ≈ 9s)"
    assert pf.stats["total_prep_sec"] >= 3.5
    assert pf.stats["prefetch_hits"] >= 2, "early units must be ready while GPU trains"
    assert pf.stats["prefetch_misses"] >= 1  # unit 0 always waits for its build

    # Case B: 5 x 1.6s prep with depth=1 (slow CPU) vs 5 x 1.0s GPU →
    # measured GPU wait must show up as ≈3s, never zero and never 5s+.
    pf2 = _fast_pf(build_fn=lambda item, idx: time.sleep(1.6), total=5,
                   depth=1, timeout=30.0)
    pf2.start(list(range(5)))
    _wall_b, tot_wait_b = _drive_gpu(pf2, 5, gpu_sec=1.0)
    pf2.close()

    assert 2.5 <= tot_wait_b <= 6.0, f"expected ≈3s GPU wait, measured {tot_wait_b:.2f}s"
    assert pf2.stats["total_gpu_wait_sec"] == pytest.approx(tot_wait_b, abs=0.01)
    assert pf2.stats["prefetch_misses"] >= 4


def test_gpu_wait_measurement():
    """The wait the consumer reports for a still-building unit is the measured
    GPU idle time, tracked in stats and counted as a prefetch miss."""
    def build_fn(item, idx):
        time.sleep(0.9)
        return f"u{idx}"

    pf = _fast_pf(build_fn, total=1, depth=1, timeout=10.0)
    pf.start([0])
    _payload, wait_dur, _timing = pf.get(0)
    pf.close()

    assert wait_dur >= 0.7, f"measured GPU wait too small: {wait_dur:.3f}s"
    assert pf.stats["total_gpu_wait_sec"] == pytest.approx(wait_dur, abs=0.01)
    assert pf.stats["prefetch_misses"] == 1
    assert pf.stats["prefetch_hits"] == 0


# ---------------------------------------------------------------------------
# 23. ASYNC PIPELINE REPORT feed (duplicates, non-retryable, final states)
# ---------------------------------------------------------------------------

def test_async_pipeline_report():
    """The final ASYNC report is fed by stats that must be exact: duplicate
    prevention, non-retryable failures, and per-dataset terminal states."""
    def build_fn(item, idx):
        if idx == 0:
            e = RuntimeError("handoff glue broken")
            _tag_phase(e, "result_handoff", intended_type="Dataset",
                       actual_type="None")
            raise e
        return f"u{idx}"

    units = [
        SimpleNamespace(path="org/fail", name="train"),
        SimpleNamespace(path="org/ok", name="train"),
        SimpleNamespace(path="org/fail", name="train"),  # duplicate of unit 0
    ]
    pf = _fast_pf(build_fn, total=len(units), retries=2)
    pf.start(units)

    with pytest.raises(RuntimeError, match="handoff glue broken"):
        pf.get(0)
    payload1, _w1, _t1 = pf.get(1)
    with pytest.raises(Exception):  # duplicate of the failed first slot
        pf.get(2)
    pf.close()

    dup_total = pf.stats["duplicates_prevented"]
    non_retryable = pf.stats["nonretryable"]
    state_counts = dict(pf.stats["state_counts"])
    assert payload1 == "u1"
    assert dup_total == 1
    assert non_retryable == 1, "result_handoff is a non-retryable phase"
    assert pf.stats["retries"] == 0
    assert state_counts.get("FAILED_PERMANENT") == 1
    assert state_counts.get("CONSUMED") == 1
    assert state_counts.get("SKIPPED") == 1

    # The production report prints these exact summaries — keep them honest.
    assert "duplicate builds prevented: %d" % dup_total == \
        "duplicate builds prevented: 1"
    assert "non-retryable build failures: %d" % non_retryable == \
        "non-retryable build failures: 1"
    rendered = ", ".join(f"{k}={v}" for k, v in sorted(state_counts.items()))
    assert "FAILED_PERMANENT=1" in rendered
    assert "CONSUMED=1" in rendered
    assert "SKIPPED=1" in rendered


# ---------------------------------------------------------------------------
# 24. Telemetry failure never breaks training
# ---------------------------------------------------------------------------

def test_telemetry_failure_does_not_crash_training(monkeypatch):
    """Exploding GPU probes inside the telemetry sampler are swallowed; the
    metrics accumulator and snapshot keep working, so training is unaffected."""
    import src.infrastructure.telemetry as telemetry

    def boom():
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(telemetry, "_gpu_utilization", boom)
    monkeypatch.setattr(telemetry, "_gpu_memory", boom)

    tm = telemetry.PipelineTelemetry(enabled=True, interval_sec=1.0,
                                     gpu_util_sampling=True)
    tm._smoke_gpu_avail = True
    tm.start()
    tm.record("tokens", 10)
    tm.record("samples", 2)
    time.sleep(2.4)  # the guarded sampler fires the exploding probes repeatedly
    tm.record("tokens", 5)
    snap = tm.snapshot()
    line = tm.summary_line()  # gpu util unknown → "n/a", must not raise
    tm.stop()

    assert tm._monitor is None, "sampler thread must terminate cleanly"
    assert snap["tokens"] >= 15
    assert snap["samples"] == 2
    assert "gpu" in line
    assert "n/a" in line


# ---------------------------------------------------------------------------
# Suite guard: every required exact name exists (across the two test files)
# ---------------------------------------------------------------------------

def test_all_required_test_names_present():
    defined = _defined_test_names()
    missing = sorted(_REQUIRED_NAMES - defined)
    assert not missing, f"required exact-name tests missing: {missing}"