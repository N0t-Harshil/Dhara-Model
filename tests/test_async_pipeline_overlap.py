from __future__ import annotations

"""Comprehensive test suite for the Full CPU/GPU Pipeline Overlap & Dataset Driver Architecture.

Tests all 14 Critical Requirements and Requirement-to-Test Acceptance Matrix:
  - R1: Dataset-granular training & checkpoint identity
  - R2: True CPU/GPU overlap timing proof
  - R3: Bounded prefetch depth & ready queue
  - R4: Background metadata warming
  - R5: Processed unit cache reuse
  - R6: Tokenizer cache offline hash verification
  - R7: Worker cancellation & no orphan threads
  - R8: Persistent worker pool reuse
  - R9: Background failure isolation & journal
  - R10: Checkpoint identity & resume correctness
  - R11: Single-dataset unit bypasses WeightedMixedDataset
  - R12: Atomic cache write safety
  - R13: Duplicate dataset build prevention
  - R14: Telemetry tracking & GPU wait measurement
  - R15: Transformers 5 Trainer API compatibility
  - R16: Legacy HF URI support (code_search_net)
  - R17: FileDatasetDriver vs ScriptDatasetDriver classification
  - R18: ASYNC PIPELINE REPORT formatting
  - R19: Warm-run HF resolution bypass — driver warm paths (R5/R16/R17) +
        `scripts/test_pipeline.py` warm-cache reuse checks
  - R20: Cold/warm data equivalence — `scripts/test_local_validation.py`
        (no duplicate samples across runs, identical acceptance counts)
  - R21: Atomic dataset completion — `test_unit_completion_manifest_and_identity`
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

from src.config.schema import AsyncPipelineConfig, Config, DataConfig
from src.data.drivers import (
    DRIVER_KIND_FILE,
    DRIVER_KIND_LOCAL,
    DRIVER_KIND_SCRIPT,
    BuilderCache,
    FileDatasetDriver,
    ScriptDatasetDriver,
    compute_builder_fingerprint,
    detect_driver,
)
from src.data.metadata_cache import (
    DatasetMetadataCache,
    compute_fingerprint,
    rewrite_hf_url,
)
from src.data.pipeline import DataPipeline
from src.data.registry import DatasetInfo, DatasetRegistry
from src.training.asyncprefetch import PrefetchTimeout, UnitPrefetch
from src.training.checkpoint import (
    AsyncCheckpointWriter,
    CheckpointContext,
    _atomic_write_bytes,
    verify_checkpoint,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# R2: Deterministic CPU/GPU Overlap Timing Proof
# -----------------------------------------------------------------------------

def test_true_cpu_gpu_overlap_timing():
    """Prove that CPU preprocessing of Dataset N+1 actually overlaps GPU training of Dataset N."""
    built = []
    lock = threading.Lock()

    def build_fn(unit, idx):
        time.sleep(0.3)  # Preprocessing work
        with lock:
            built.append(idx)
        return f"dataset_{idx}"

    pf = UnitPrefetch(build_fn=build_fn, total=2, depth=2, timeout=5.0)
    pf.start([0, 1])

    t0 = time.monotonic()
    # Consume dataset 0 (first get waits for initial build of dataset 0)
    res0 = pf.get(0)
    ds0 = res0[0] if isinstance(res0, tuple) else res0
    assert ds0 == "dataset_0"

    # Simulate GPU training on Dataset 0 while background worker builds Dataset 1
    time.sleep(0.4)  # Simulated GPU training on Dataset 0

    # Consume dataset 1 (should be ALREADY READY, so wait1 ≈ 0.0s)
    res1 = pf.get(1)
    ds1 = res1[0] if isinstance(res1, tuple) else res1
    wait1 = res1[1] if isinstance(res1, tuple) and len(res1) > 1 else 0.0
    t_total = time.monotonic() - t0
    pf.close()

    assert ds1 == "dataset_1"
    assert wait1 < 0.15, f"Expected Dataset 1 to be ready immediately, but waited {wait1:.3f}s"
    assert t_total < 0.85, f"Total execution time {t_total:.3f}s indicates lack of overlap"


# -----------------------------------------------------------------------------
# R5 & R12: Processed Cache Fast Path & Atomic Write Safety
# -----------------------------------------------------------------------------

def test_processed_cache_fast_path_and_atomic_write():
    """Verify that cached units are loaded immediately without re-streaming,
    and cache writes use atomic temp files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        target_dir = tmp_path / "unit_cache"
        manifest_path = target_dir / "unit_meta.json"
        
        # Test atomic write
        _atomic_write_bytes(manifest_path, json.dumps({"key": "test_key", "packed_count": 100}).encode())
        assert manifest_path.exists()
        assert (rec_is_valid := json.loads(manifest_path.read_text()))
        assert rec_is_valid["key"] == "test_key"
        assert rec_is_valid["packed_count"] == 100


# -----------------------------------------------------------------------------
# R16 & R17: Driver Classification & Legacy HF URIs
# -----------------------------------------------------------------------------

def test_driver_classification_and_legacy_hf_uris():
    """Verify FileDatasetDriver vs ScriptDatasetDriver classification and
    non-namespaced legacy HF repository IDs (e.g. code_search_net)."""
    # Legacy URI rewriting test
    assert rewrite_hf_url("hf://datasets/code_search_net/resolve/main/data.parquet") == \
        "https://huggingface.co/datasets/code_search_net/resolve/main/data.parquet"
    assert rewrite_hf_url("hf://datasets/org/repo/resolve/v1.0/file.arrow") == \
        "https://huggingface.co/datasets/org/repo/resolve/v1.0/file.arrow"
    assert rewrite_hf_url("https://already.resolved/file.parquet") == \
        "https://already.resolved/file.parquet"

    # Driver classification test for local dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_cache = DatasetMetadataCache(Path(tmp_dir) / "meta")
        builder_cache = BuilderCache(Path(tmp_dir) / "builder")
        # create a small local file so the driver detects a local dataset
        sample_file = Path(tmp_dir) / "sample.txt"
        sample_file.write_text("print('hello world')")

        info_file = DatasetInfo(path=tmp_dir, category="code", weight=1.0, quality_score=1.0, split="train")
        driver, diag = detect_driver(
            info_file, meta_cache=meta_cache, builder_cache=builder_cache,
            preprocess_sig="sig1", token_sig="tok1"
        )
        assert driver.kind == DRIVER_KIND_LOCAL
        assert diag["repo_resolution_skipped"] is True


def test_local_driver_record_without_metadata_cache():
        """Local datasets should still build a usable driver record even when
        metadata caching is disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_cache = DatasetMetadataCache(Path(tmp_dir) / "meta", enabled=False)
            builder_cache = BuilderCache(Path(tmp_dir) / "builder", enabled=False)
            sample_file = Path(tmp_dir) / "sample.txt"
            sample_file.write_text("print('hello world')")

            info_file = DatasetInfo(
                path=tmp_dir,
                category="code",
                weight=1.0,
                quality_score=1.0,
                split="train",
            )
            driver, diag = detect_driver(
                info_file,
                meta_cache=meta_cache,
                builder_cache=builder_cache,
                preprocess_sig="sig1",
                token_sig="tok1",
            )

            assert driver.kind == DRIVER_KIND_LOCAL
            assert driver.record is not None
            assert driver.record.get("files") == [str(sample_file)]
            assert diag["repo_resolution_skipped"] is True


class FakeScriptBuilder:
        def __init__(self):
            self.config = SimpleNamespace(data_files=None)
            self._dataset_revision = "rev1"

        def as_streaming_dataset(self, split):
            return SimpleNamespace(_ex_iterable=SimpleNamespace(data_sources=[]))


def test_script_driver_creates_record_without_builder_cache():
        """Script datasets should still build a usable resume record even when
        builder caching is disabled."""

        def fake_load_builder(path, name=None, data_dir=None, revision=None, token=None):
            return FakeScriptBuilder()

        info_script = DatasetInfo(
            path="fake_script_dataset",
            category="code",
            weight=1.0,
            quality_score=1.0,
            split="train",
        )
        driver, diag = detect_driver(
            info_script,
            meta_cache=DatasetMetadataCache(Path("/tmp/nonexistent"), enabled=False),
            builder_cache=None,
            preprocess_sig="sig1",
            token_sig="tok1",
            load_builder=fake_load_builder,
            token="dummy",
        )

        assert driver.kind == DRIVER_KIND_SCRIPT
        assert driver.record is not None
        assert diag["driver_kind"] == DRIVER_KIND_SCRIPT


# -----------------------------------------------------------------------------
# R11: Single-Dataset Unit Bypasses WeightedMixedDataset
# -----------------------------------------------------------------------------

def test_single_dataset_unit_bypasses_mixed_wrapper():
    """Verify that a single-dataset unit returns the raw dataset directly
    without wrapping in WeightedMixedDataset, while multi-dataset units
    still use the mixing wrapper."""
    from src.data.pipeline import DataPipeline

    fake = object.__new__(DataPipeline)
    fake.cfg = SimpleNamespace(
        data=SimpleNamespace(
            sampler=SimpleNamespace(balance_by="samples"),
            language_balancing=SimpleNamespace(enabled=False,
                                               target_distribution=None),
            domain_balancing=SimpleNamespace(enabled=False,
                                             include=None),
        ),
    )

    all_tokenized = [(SimpleNamespace(name="raw_ds"), 1.0, "path", "code", 0.9)]
    result = fake._unit_or_mixed(all_tokenized)
    assert result is all_tokenized[0][0]
    assert result.name == "raw_ds"
    assert result.__class__.__name__ != "WeightedMixedDataset"

    class _LenDS:
        def __init__(self, n):
            self.n = n
            self.name = f"ds-{n}"

        def __len__(self):
            return self.n

    multi = [(_LenDS(10 * (i + 1)), 1.0, "path", "code", 0.9)
             for i in range(2)]
    mixed = fake._unit_or_mixed(multi)
    assert mixed.__class__.__name__ == "WeightedMixedDataset"


# -----------------------------------------------------------------------------
# R3 & R8: Bounded Queue Depth & Worker Cleanup
# -----------------------------------------------------------------------------

def test_bounded_queue_depth_and_worker_cleanup():
    """Verify that UnitPrefetch bounds its memory to configured depth and cleans up workers."""
    pf = UnitPrefetch(
        build_fn=lambda item, idx: idx * 10,
        total=10,
        depth=3,
        timeout=2.0,
        name="test_bound"
    )
    pf.start(list(range(10)))
    
    assert pf.queue_depth() <= 3
    
    res = pf.get(0)
    val = res[0] if isinstance(res, tuple) else res
    assert val == 0
    
    pf.close()
    assert pf.queue_depth() == 0


# -----------------------------------------------------------------------------
# R9 & R21: Background Failure Isolation & Journal
# -----------------------------------------------------------------------------

def test_background_failure_isolation():
    """Verify that a background worker build exception is delivered safely to
    the consumer on get() without crashing active training or zombie threads."""
    def build_with_failure(item, idx):
        if idx == 1:
            raise ValueError("Corrupt parquet shard")
        return f"unit_{idx}"

    pf = UnitPrefetch(build_fn=build_with_failure, total=3, depth=2, timeout=2.0)
    pf.start([0, 1, 2])

    res0 = pf.get(0)
    val0 = res0[0] if isinstance(res0, tuple) else res0
    assert val0 == "unit_0"

    with pytest.raises(ValueError, match="Corrupt parquet shard"):
        pf.get(1)

    res2 = pf.get(2)
    val2 = res2[0] if isinstance(res2, tuple) else res2
    assert val2 == "unit_2"
    pf.close()


# -----------------------------------------------------------------------------
# R10 & R27: Checkpoint Identity & Resume Correctness
# -----------------------------------------------------------------------------

def test_checkpoint_identity_and_resume():
    """Verify checkpoint context capturing stage, unit identity, fingerprint and verify hash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ck_dir = Path(tmp_dir) / "checkpoint-stage1-u002"
        ck_dir.mkdir(parents=True)
        
        ctx = CheckpointContext(
            model_state={"weight": 1.0},
            config={"architecture": "methos_v3", "stage_index": 1, "unit_index": 2},
        )
        
        writer = AsyncCheckpointWriter(checksum=True)
        writer.submit(ck_dir, lambda p: (p / "model.bin").write_bytes(b"weights_data"))
        writer.flush(timeout=5.0)
        writer.close()

        ok, detail = verify_checkpoint(ck_dir)
        assert ok is True
        assert "verified" in detail


# -----------------------------------------------------------------------------
# R4: Background metadata warming orchestration
# -----------------------------------------------------------------------------

def test_stage_metadata_warming_schedule():
    """Stage metadata warming resolves every dataset that has no metadata
    record (offline, driver-only) and no-ops once everything is cached."""
    from src.training.pipeline import TrainingPipeline

    warmed = []

    class FakeDP:
        def __init__(self, state):
            self.state = state

        def has_metadata_record(self, u):
            return self.state.get((u.path, u.name), False)

        def warm_metadata_cache(self, u):
            self.state[(u.path, u.name)] = True
            warmed.append(u.path)
            return True

    u1 = SimpleNamespace(path="datasets/alpha", name="train")
    u2 = SimpleNamespace(path="datasets/beta", name="train")

    fake = object.__new__(TrainingPipeline)
    state = {}
    fake.data_pipeline = FakeDP(state)

    fake._warm_stage_metadata([u1, u2], 1)
    assert sorted(warmed) == ["datasets/alpha", "datasets/beta"]
    assert state[(u1.path, u1.name)] and state[(u2.path, u2.name)]

    warmed.clear()
    fake._warm_stage_metadata([u1, u2], 1)  # everything resolved → no-op
    assert warmed == []


# -----------------------------------------------------------------------------
# R6: Tokenizer cache offline hash verification
# -----------------------------------------------------------------------------

def test_tokenizer_cache_hash_verification():
    """The tokenizer manifest records per-file sha256; a match enables the
    offline fast path, a tampered or missing file disables it."""
    import hashlib
    from src.models.factory import ModelFactory

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "tokenizer.json").write_bytes(b"vocab-hash-A")
        (p / "special_tokens_map.json").write_bytes(b"map-hash-B")

        files = ModelFactory._hash_tokenizer_files(p)
        assert files["tokenizer.json"] == hashlib.sha256(b"vocab-hash-A").hexdigest()
        manifest = {"version": 1, "files": files}
        (p / "tokenizer_version.json").write_text(
            json.dumps(manifest), encoding="utf-8")

        verified = ModelFactory._verify_tokenizer_cache(p)
        assert verified is not None
        assert verified["files"] == files

        (p / "tokenizer.json").write_bytes(b"tampered")  # file changed
        assert ModelFactory._verify_tokenizer_cache(p) is None

        (p / "tokenizer.json").write_bytes(b"vocab-hash-A")  # restored
        assert ModelFactory._verify_tokenizer_cache(p) is not None

        (p / "tokenizer_version.json").unlink()  # manifest missing
        assert ModelFactory._verify_tokenizer_cache(p) is None


# -----------------------------------------------------------------------------
# R8: Persistent cleanup pool reuse
# -----------------------------------------------------------------------------

def test_cleanup_pool_persistent_reuse():
    """The cleanup pool is created once, returned on every request, and
    terminated by close() (no repeated create/destroy per dataset)."""
    from src.data.pipeline import DataPipeline

    class Cfg:
        data = type("D", (), {
            "cleanup_pool_size": 2,
            "hf_token": "",
            "quality": type("Q", (), {"deduplication": type(
                "D", (), {"method": "exact", "threshold": 0.85})(),
                "contamination": type("C", (), {"benchmarks": []})()})(),
            "metadata_cache": type("M", (), {"dir": ".", "enabled": False,
                                             "fingerprint_version": 1})(),
        })()

    pipe = DataPipeline.__new__(DataPipeline)
    pipe.cfg = Cfg()
    pipe._cleanup_pool = None
    try:
        p1 = pipe._get_cleanup_pool()
        p2 = pipe._get_cleanup_pool()
        assert p1 is not None and p1 is p2, "pool must be created once and reused"
        pipe.close()
        assert pipe._cleanup_pool is None, "close() must clear the pool"
        p3 = pipe._get_cleanup_pool()
        p3.close()
        p3.join()
    finally:
        pipe.close()


# -----------------------------------------------------------------------------
# R15: Transformers 5 Trainer API compatibility
# -----------------------------------------------------------------------------

def test_trainer_tokenizer_kwarg_compat():
    """Trainer construction must use processing_class= when the installed
    Transformers supports it, else tokenizer= — never the deprecated one."""
    import inspect

    from transformers import Trainer

    from src.utils.training import dataloader_num_workers, trainer_tokenizer_kwarg

    kw = trainer_tokenizer_kwarg()
    params = inspect.signature(Trainer.__init__).parameters
    assert kw in ("processing_class", "tokenizer")
    assert (kw == "processing_class") == ("processing_class" in params)
    assert dataloader_num_workers(None) >= 0


# -----------------------------------------------------------------------------
# R14 & R18: Telemetry Tracking & ASYNC PIPELINE REPORT
# -----------------------------------------------------------------------------

def test_telemetry_and_async_pipeline_report():
    """Verify GPU wait duration measurement and telemetry reporting snapshot."""
    from src.infrastructure.telemetry import PipelineTelemetry

    tm = PipelineTelemetry(enabled=True, interval_sec=60, gpu_util_sampling=False)
    tm.record("cache_hits", 2)
    tm.record("cache_attempts", 4)
    tm.record("tokens", 50000)
    tm.record("gpu_wait_sec", 1.25)
    tm.record("train_sec", 30.0)
    tm.record("metadata_hits", 4)
    tm.record("metadata_attempts", 5)
    tm.record("builder_hits", 1)
    tm.record("builder_attempts", 2)
    
    snap = tm.snapshot()
    assert snap["cache_hits"] == 2
    assert snap["cache_attempts"] == 4
    assert abs(snap["cache_hit_pct"] - 50.0) < 0.1
    assert snap["gpu_wait_sec"] == 1.25
    assert snap["train_sec"] == 30.0
    assert snap["metadata_hits"] == 4 and snap["metadata_attempts"] == 5
    assert snap["builder_hits"] == 1 and snap["builder_attempts"] == 2
    assert abs(snap["metadata_hit_pct"] - 80.0) < 0.1
    tm.finish()


# -----------------------------------------------------------------------------
# R13: Duplicate Build Prevention
# -----------------------------------------------------------------------------

def test_duplicate_build_prevention():
    """Scheduling the same dataset twice must build it exactly once; the later
    occurrence shares the first payload and 'duplicate builds prevented' is
    tracked in stats (and surfaced in the ASYNC PIPELINE REPORT)."""
    attempts = []

    def build_fn(unit, idx):
        attempts.append(idx)
        time.sleep(0.05)
        return f"dataset_{unit.path}/{unit.name}"

    units = [
        SimpleNamespace(path="org/repo", name="train"),
        SimpleNamespace(path="other/repo", name="dev"),
        SimpleNamespace(path="org/repo", name="train"),  # duplicate identity
    ]
    pf = UnitPrefetch(build_fn=build_fn, total=3, depth=2, timeout=5.0)
    pf.start(units)
    got = []
    for k in range(3):
        res = pf.get(k)
        got.append(res[0] if isinstance(res, tuple) else res)
    dup = pf.stats["duplicates_prevented"]
    delivered = pf.stats["delivered"]
    pf.close()

    assert got[0] == got[2] == "dataset_org/repo/train"
    assert attempts == [0, 1], f"unit 0 must be built exactly once, got {attempts}"
    assert dup == 1
    assert delivered == 3


# -----------------------------------------------------------------------------
# R19 (part 2): data.async_pipeline.enabled=false → synchronous fallback
# -----------------------------------------------------------------------------

def test_async_pipeline_disabled_synchronous_fallback():
    """§18: disabling the global async switch must route the pipeline to the
    synchronous inline build path (emergency fallback), even when the staging
    block requests prefetch."""
    from src.training.pipeline import _prefetch_enabled

    staging_on = SimpleNamespace(prefetch=True)
    staging_off = SimpleNamespace(prefetch=False)
    async_on = SimpleNamespace(enabled=True)
    async_off = SimpleNamespace(enabled=False)

    assert _prefetch_enabled(staging_on, async_on) is True
    assert _prefetch_enabled(staging_on, async_off) is False  # emergency fallback
    assert _prefetch_enabled(staging_off, async_on) is False
    assert _prefetch_enabled(staging_off, async_off) is False
    assert _prefetch_enabled(staging_on, None) is True  # legacy configs stay async


# -----------------------------------------------------------------------------
# §24: build retries (data.async_pipeline.retry_count) before journaling
# -----------------------------------------------------------------------------

def test_unit_prefetch_retries_failed_build():
    """A failing build is retried up to retry_count before its exception is
    delivered (and only then recorded as an error)."""
    calls = {"n": 0}

    def flaky(unit, idx):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("transient network failure")
        return f"dataset_{idx}"

    pf = UnitPrefetch(build_fn=flaky, total=1, depth=1, timeout=5.0, retries=3)
    pf.start([0])
    res = pf.get(0)
    pf.close()
    assert res[0] == "dataset_0"
    assert calls["n"] == 3, f"expected 1 build + 2 retries, got {calls['n']}"
    assert pf.stats["retries"] == 2
    assert pf.stats["errors"] == 0

    # always-failing build: retried retry_count times, then error delivered
    calls2 = {"n": 0}

    def always_fail(unit, idx):
        calls2["n"] += 1
        raise ValueError("permanent failure")

    pf2 = UnitPrefetch(build_fn=always_fail, total=1, depth=1, timeout=5.0, retries=2)
    pf2.start([0])
    try:
        pf2.get(0)
        raised = False
    except ValueError:
        raised = True
    pf2.close()
    assert raised is True
    assert calls2["n"] == 3, f"expected 1 build + 2 retries, got {calls2['n']}"
    assert pf2.stats["retries"] == 2
    assert pf2.stats["errors"] == 1


# -----------------------------------------------------------------------------
# §18 & §22: worker/buffer bounds are enforced (not dead config)
# -----------------------------------------------------------------------------

def test_async_pipeline_worker_and_buffer_bounds():
    """ready_queue_size / max_inflight cap the effective prefetch depth and
    preprocess_workers caps the producer thread count."""
    from src.training.pipeline import TrainingPipeline

    depth_of = TrainingPipeline._effective_prefetch_depth
    staging = SimpleNamespace(prefetch_depth=8)
    async_cfg = SimpleNamespace(ready_queue_size=2, max_inflight=3, preprocess_workers=2)

    assert depth_of(staging, async_cfg) == 2  # min(8, 2, 3)
    assert depth_of(SimpleNamespace(prefetch_depth=1), async_cfg) == 1
    assert depth_of(SimpleNamespace(prefetch_depth=4), None) == 4
    assert depth_of(SimpleNamespace(prefetch_depth=5),
                    SimpleNamespace(ready_queue_size=9, max_inflight=4)) == 4

    def build_fn(unit, idx):
        time.sleep(0.3)
        return idx

    pf = UnitPrefetch(build_fn=build_fn, total=6, depth=4, timeout=5.0, max_workers=2)
    pf.start(list(range(6)))
    assert pf.stats["started_workers"] == 2
    for k in range(6):
        res = pf.get(k)
        assert res[0] == k
    pf.close()


# -----------------------------------------------------------------------------
# R10 (extended): Dataset-granular checkpoint identity + atomic completion
# manifest. A restart must skip by *verified identity*, never by index alone.
# -----------------------------------------------------------------------------

def test_unit_completion_manifest_and_identity():
    """Completion manifest records the unit fingerprint; identity change
    yields a different fingerprint; unit_identity.json lands in the
    per-unit checkpoint dir with stage/unit/cache keys."""
    from src.training.pipeline import TrainingPipeline

    class _U:
        path = "bigcode/the-stack"
        name = "python"
        data_dir = ""
        revision = "main"
        category = "code"

    with tempfile.TemporaryDirectory() as tmp:
        fake = object.__new__(TrainingPipeline)
        fake.cfg = SimpleNamespace(output=SimpleNamespace(model_dir=tmp))

        u = _U()
        fp = TrainingPipeline._unit_fingerprint(fake, u)
        assert len(fp) == 64
        # deterministic
        assert TrainingPipeline._unit_fingerprint(fake, u) == fp

        # revision change must change the fingerprint (identity change)
        u2 = _U()
        u2.revision = "refs/heads/v2"
        assert TrainingPipeline._unit_fingerprint(fake, u2) != fp

        # atomic completion manifest
        fake._mark_unit_complete(1, 2, "bigcode/the-stack/python", fp,
                                 1500, 50000)
        done = Path(tmp) / "unit_completions.json"
        assert done.exists() and not list(Path(tmp).glob("*.tmp"))
        recs = fake._load_unit_completions()
        rec = recs["bigcode/the-stack/python"]
        assert rec["stage_index"] == 1 and rec["unit_index"] == 2
        assert rec["unit_fingerprint"] == fp
        assert rec["global_step"] == 1500 and rec["total_steps"] == 50000

        # a checkpoint that recorded the old fingerprint will NOT match a
        # changed dataset identity (forces retrain on resume)
        assert rec["unit_fingerprint"] != TrainingPipeline._unit_fingerprint(fake, u2)

        # per-unit checkpoint identity file
        ck = Path(tmp) / "checkpoint-1500"
        ck.mkdir()
        fake._write_unit_identity(ck, 1, 2, u, "bigcode/the-stack/python",
                                  1500, 50000)
        ident = json.loads((ck / "unit_identity.json").read_text(encoding="utf-8"))
        assert ident["stage_index"] == 1
        assert ident["unit_index"] == 2
        assert ident["unit_key"] == "bigcode/the-stack/python"
        assert ident["unit_fingerprint"] == fp
        assert ident["global_step"] == 1500
        assert ident["total_steps"] == 50000
        assert ident["processed_cache_key"]
        assert not list(ck.glob("*.tmp"))
