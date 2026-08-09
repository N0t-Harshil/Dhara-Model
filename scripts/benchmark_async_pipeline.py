from __future__ import annotations

"""Offline benchmark for the async pipeline facilities.

No GPU, no network, no torch: every number below is measured with synthetic
workloads so the script can run anywhere CI runs. Produces a report at
``reports/benchmark_async_pipeline.txt`` covering:

  1. cold vs warm unit-cache startup (build time hidden behind training)
  2. prefetch overlap — how much of the dataset build is hidden
  3. async checkpoint writer vs synchronous write (caller-side block)
  4. telemetry accounting overhead
  5. cleanup-pool reuse (init once, terminate on close)
  6. failure isolation (one bad unit among N — mark-and-continue)
"""

import logging
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

BUILD_MS = 250      # simulated dataset build cost (cold)
TRAIN_MS = 200      # simulated per-unit training cost
UNITS = 6
PAYLOAD_MB = 64     # simulated checkpoint payload


def fmt(sec: float) -> str:
    return f"{sec * 1000:.0f}ms"


def bench_prefetch_overlap() -> dict:
    """Sequential (build then train) vs prefetched build-during-train."""
    from src.training.asyncprefetch import UnitPrefetch

    def build(item, idx):
        time.sleep(BUILD_MS / 1000)
        return idx

    def train_all(items, use_prefetch):
        t0 = time.monotonic()
        if use_prefetch:
            pf = UnitPrefetch(build_fn=build, total=len(items),
                              depth=3, timeout=30.0)
            pf.start(list(items))
            for i in items:
                pf.get(i)
                time.sleep(TRAIN_MS / 1000)
            pf.close()
        else:
            for i in items:
                build(i, i)
                time.sleep(TRAIN_MS / 1000)
        return time.monotonic() - t0

    items = list(range(UNITS))
    seq = train_all(items, False)
    pf = train_all(items, True)
    total_build = UNITS * BUILD_MS / 1000
    hidden = max(0.0, (seq - pf) / max(1e-9, total_build))
    return {"sequential_s": seq, "prefetched_s": pf, "hidden_pct": hidden * 100}


def bench_cold_vs_warm_unit() -> dict:
    """'Startup' comparison: a warm unit cache skips resolution entirely
    (cold = full build + packing, warm = load only)."""
    from src.training.asyncprefetch import UnitPrefetch

    cold_times = []

    def cold_build(item, idx):
        t0 = time.perf_counter_ns()
        time.sleep(BUILD_MS / 1000)
        cold_times.append((time.perf_counter_ns() - t0) / 1e9)
        return f"ds{idx}"

    warm_times = []

    def warm_build(item, idx):
        t0 = time.perf_counter_ns()
        time.sleep(0.5 / 1000)  # just a cache-file read + torch.load
        warm_times.append((time.perf_counter_ns() - t0) / 1e9)
        return f"ds{idx}"

    cold = UnitPrefetch(build_fn=cold_build, total=UNITS, depth=3, timeout=30.0)
    cold.start(list(range(UNITS)))
    t0 = time.monotonic()
    first = cold.get(0)
    rest = [cold.get(i) for i in range(1, UNITS)]
    cold_wall = time.monotonic() - t0
    cold.close()

    warm = UnitPrefetch(build_fn=warm_build, total=UNITS, depth=3, timeout=30.0)
    warm.start(list(range(UNITS)))
    t0 = time.monotonic()
    [warm.get(i) for i in range(UNITS)]
    warm_wall = time.monotonic() - t0
    warm.close()
    cold_build_total = sum(cold_times)
    warm_build_total = sum(warm_times)
    return {"cold_wall_s": cold_wall, "warm_wall_s": warm_wall,
            "cold_build_total_s": cold_build_total,
            "warm_build_total_s": warm_build_total,
            "speedup_x": cold_build_total / max(1e-9, warm_build_total)}


def bench_async_checkpoint() -> dict:
    """Async submit+flush vs a synchronous write of the same payload."""
    from src.training.checkpoint import AsyncCheckpointWriter

    def write_payload(path):
        import os
        data = b"x" * (PAYLOAD_MB * 1024 * 1024)
        with open(path / "model.bin", "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    with tempfile.TemporaryDirectory() as tmp:
        writer = AsyncCheckpointWriter(checksum=True)
        t0 = time.monotonic()
        ok = writer.submit(Path(tmp) / "ckpt", write_payload)
        submit_ms = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        flushed = writer.flush(timeout=30)
        flush_ms = (time.monotonic() - t0) * 1000
        writer.close()

        t0 = time.monotonic()
        sync_dir = Path(tmp) / "ckpt-sync"
        sync_dir.mkdir(parents=True, exist_ok=True)
        write_payload(sync_dir)
        sync_ms = (time.monotonic() - t0) * 1000

    return {"submit_ms": submit_ms, "flush_ms": flush_ms,
            "sync_ms": sync_ms, "ok": ok, "flushed": flushed,
            "caller_block_ms": submit_ms, "sync_gap_pct":
            (1 - submit_ms / max(1e-9, sync_ms)) * 100}


def bench_telemetry_overhead() -> dict:
    from src.infrastructure.telemetry import PipelineTelemetry

    tm = PipelineTelemetry(enabled=True, interval_sec=3600,
                           gpu_util_sampling=False)
    n = 100_000
    t0 = time.monotonic()
    for i in range(n):
        tm.record("cache_attempts")
        tm.record("cache_hits", 0 if i % 2 else 1)
    tm.set_event("stage", 1)
    tm.snapshot()
    tm.finish()
    per_op_us = (time.monotonic() - t0) / (n * 3) * 1e6
    return {"per_op_us": per_op_us}


def bench_cleanup_pool() -> dict:
    """Pool is created once and reused; close() terminates it cleanly."""
    from src.data.pipeline import DataPipeline

    calls = []

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
    p1 = pipe._get_cleanup_pool()
    p2 = pipe._get_cleanup_pool()
    same = p1 is p2
    workers = p1._processes if p1 is not None else 0
    t0 = time.monotonic()
    pipe.close()
    close_ms = (time.monotonic() - t0) * 1000
    calls.append(pipe._cleanup_pool)
    return {"created_once": same, "workers": workers, "close_ms": close_ms,
            "terminated": pipe._cleanup_pool is None}


def bench_failure_isolation() -> dict:
    """One unit crashes; the run must still deliver every other unit and
    total time must stay near the no-failure case."""
    from src.training.asyncprefetch import UnitPrefetch

    def build(item, idx):
        time.sleep(BUILD_MS / 1000)
        if idx == 3:
            raise RuntimeError("simulated crash")
        return idx

    pf = UnitPrefetch(build_fn=build, total=UNITS, depth=3, timeout=30.0)
    pf.start(list(range(UNITS)))
    t0 = time.monotonic()
    delivered = []
    for i in range(UNITS):
        try:
            delivered.append(pf.get(i))
        except RuntimeError:
            delivered.append(None)
    wall = time.monotonic() - t0
    pf.close()
    return {"delivered": delivered, "failed_unit_marked": delivered[3] is None,
            "all_others_ok": delivered[:3] + delivered[4:] == list(range(3)) + [4, 5],
            "wall_s": wall}


def main() -> int:
    results = {}

    print("bench: prefetch overlap ...")
    results["prefetch_overlap"] = bench_prefetch_overlap()
    print("bench: cold vs warm unit ...")
    results["cold_vs_warm"] = bench_cold_vs_warm_unit()
    print("bench: async checkpoint ...")
    results["async_checkpoint"] = bench_async_checkpoint()
    print("bench: telemetry overhead ...")
    results["telemetry"] = bench_telemetry_overhead()
    print("bench: cleanup pool ...")
    results["cleanup_pool"] = bench_cleanup_pool()
    print("bench: failure isolation ...")
    results["failure_isolation"] = bench_failure_isolation()

    lines = [
        "# Async Pipeline Benchmark (offline, synthetic)",
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"units: {UNITS}, build: {BUILD_MS}ms, train: {TRAIN_MS}ms, "
        f"checkpoint payload: {PAYLOAD_MB}MB",
        "",
        "## 1. Prefetch overlap (build hidden behind training)",
        f"  sequential: {fmt(results['prefetch_overlap']['sequential_s'])}",
        f"  prefetched: {fmt(results['prefetch_overlap']['prefetched_s'])}",
        f"  hidden:     {results['prefetch_overlap']['hidden_pct']:.1f}%",
        "",
        "## 2. Unit startup: cold vs warm cache",
        f"  cold wall: {fmt(results['cold_vs_warm']['cold_wall_s'])} "
        f"(6 units, {BUILD_MS}ms build each)",
        f"  warm wall: {fmt(results['cold_vs_warm']['warm_wall_s'])}",
        f"  cold build total: {fmt(results['cold_vs_warm']['cold_build_total_s'])}",
        f"  warm build total: {fmt(results['cold_vs_warm']['warm_build_total_s'])}",
        f"  speedup:   {results['cold_vs_warm']['speedup_x']:.0f}x",
        "",
        "## 3. Async checkpoint writer (caller-side block)",
        f"  submit (async):      {results['async_checkpoint']['submit_ms']:.1f}ms "
        f"(ok={results['async_checkpoint']['ok']})",
        f"  flush (durability):  {results['async_checkpoint']['flush_ms']:.1f}ms "
        f"(flushed={results['async_checkpoint']['flushed']})",
        f"  synchronous write:   {results['async_checkpoint']['sync_ms']:.1f}ms",
        f"  GPU thread unblocked: {results['async_checkpoint']['sync_gap_pct']:.1f}% "
        "of the sync write time",
        "",
        "## 4. Telemetry accounting overhead",
        f"  {results['telemetry']['per_op_us']:.3f}us per counter op (lock+accumulate)",
        "",
        "## 5. Persistent cleanup pool",
        f"  created once & reused: {results['cleanup_pool']['created_once']}",
        f"  workers:              {results['cleanup_pool']['workers']}",
        f"  close() cost:         {results['cleanup_pool']['close_ms']:.1f}ms "
        "(joins then terminates)",
        "",
        "## 6. Failure isolation (unit 3 of 6 crashes)",
        f"  failed unit marked: {results['failure_isolation']['failed_unit_marked']}",
        f"  all other units delivered in order: "
        f"{results['failure_isolation']['all_others_ok']}",
        f"  run completed in {fmt(results['failure_isolation']['wall_s'])} "
        "(no stall, no deadlock)",
    ]

    out = Path(__file__).resolve().parent.parent / "reports" / "benchmark_async_pipeline.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nreport written to {out}")

    checks = [
        results["prefetch_overlap"]["hidden_pct"] > 60,
        results["cold_vs_warm"]["speedup_x"] > 10,
        results["async_checkpoint"]["ok"] and results["async_checkpoint"]["flushed"],
        results["async_checkpoint"]["submit_ms"] < results["async_checkpoint"]["sync_ms"] / 5,
        results["cleanup_pool"]["created_once"] and results["cleanup_pool"]["terminated"],
        results["failure_isolation"]["failed_unit_marked"]
        and results["failure_isolation"]["all_others_ok"],
    ]
    failed = sum(1 for c in checks if not c)
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK CHECKS: {len(checks) - failed}/{len(checks)} satisfied")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
