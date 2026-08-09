from __future__ import annotations

"""Tests for the async pipeline facilities: UnitPrefetch, AsyncCheckpointWriter,
and PipelineTelemetry.

Each facility is a black box: no pipeline mocks, no torch, no GPU. Everything
here must pass in a pure-CPU, offline environment.
"""

import logging
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  OK  {name}")
        PASS += 1
    else:
        suffix = f" -- {detail}" if detail else ""
        print(f"  FAIL {name}{suffix}")
        FAIL += 1


def test_unit_prefetch_ordered_delivery() -> None:
    print("\n--- UnitPrefetch: ordered delivery ---")
    from src.training.asyncprefetch import UnitPrefetch

    built = []
    pf = UnitPrefetch(
        build_fn=lambda item, idx: (built.append(idx) or (f"ds{idx}", {"idx": idx})),
        total=8, depth=2, timeout=5.0)
    pf.start(list(range(8)))
    got = [pf.get(i) for i in range(8)]
    pf.close()
    test("all units delivered in order",
         got == [(f"ds{i}", {"idx": i}) for i in range(8)])
    test("all units built exactly once", sorted(built) == list(range(8)),
         f"built={sorted(built)}")
    test("no result buffered after consumption", pf.queue_depth() == 0)


def test_unit_prefetch_overlap() -> None:
    print("\n--- UnitPrefetch: producer/consumer overlap ---")
    from src.training.asyncprefetch import UnitPrefetch

    build2_started = threading.Event()
    release_build2 = threading.Event()
    overlap = []

    def build(item, idx):
        if idx == 2:
            build2_started.set()
            release_build2.wait(5)
        time.sleep(0.05)
        return idx

    pf = UnitPrefetch(build_fn=build, total=4, depth=3, timeout=5.0)
    pf.start(list(range(4)))
    first = pf.get(0)
    test("unit 2 built while unit 0/1 waited", build2_started.wait(2))
    release_build2.set()
    rest = [pf.get(i) for i in range(1, 4)]
    pf.close()
    test("all delivered in order", [first] + rest == [0, 1, 2, 3])


def test_unit_prefetch_starvation_free() -> None:
    print("\n--- UnitPrefetch: no worker starvation ---")
    from src.training.asyncprefetch import UnitPrefetch

    slow = {1, 4}

    def build(item, idx):
        if idx in slow:
            time.sleep(0.15)
        return idx

    pf = UnitPrefetch(build_fn=build, total=6, depth=2, timeout=5.0)
    pf.start(list(range(6)))
    got = [pf.get(i) for i in range(6)]
    pf.close()
    test("slow units never overtake fast ones", got == list(range(6)),
         f"got={got}")


def test_unit_prefetch_failure_re_raised() -> None:
    print("\n--- UnitPrefetch: build failure reaches consumer ---")
    from src.training.asyncprefetch import UnitPrefetch

    def build(item, idx):
        if idx == 1:
            raise ValueError("boom")
        return idx

    pf = UnitPrefetch(build_fn=build, total=3, depth=3, timeout=5.0)
    pf.start(list(range(3)))
    test("unit 0 ok", pf.get(0) == 0)
    try:
        pf.get(1)
        raised = False
    except ValueError as e:
        raised = str(e) == "boom"
    test("unit 1 exception re-raised on consumer", raised)
    test("consumer can continue after failure", pf.get(2) == 2)
    pf.close()


def test_unit_prefetch_timeout() -> None:
    print("\n--- UnitPrefetch: timeout ---")
    from src.training.asyncprefetch import PrefetchTimeout, UnitPrefetch

    def build(item, idx):
        if idx == 0:
            time.sleep(10)  # never finishes within the window
        return idx

    pf = UnitPrefetch(build_fn=build, total=3, depth=2, timeout=0.3)
    pf.start(list(range(3)))
    t0 = time.monotonic()
    try:
        pf.get(0)
        raised = False
    except PrefetchTimeout as e:
        raised = e.index == 0
    test("timeout raised for stuck unit", raised)
    test("waited ~timeout, not more", 0.2 < time.monotonic() - t0 < 3.0)
    # The late result must be ignored once the consumer moved on.
    test("later unit still deliverable", pf.get(2) == 2)
    pf.close()


def test_unit_prefetch_discard_and_stale() -> None:
    print("\n--- UnitPrefetch: discard + stale late results ---")
    from src.training.asyncprefetch import UnitPrefetch

    built = []

    def build(item, idx):
        built.append(idx)
        return idx

    pf = UnitPrefetch(build_fn=build, total=4, depth=3, timeout=5.0)
    pf.start(list(range(4)))
    pf.discard(0)
    test("discarded result ignored (stale get returns None)", pf.get(0) is None)
    test("later indexes unaffected", [pf.get(i) for i in (1, 2, 3)] == [1, 2, 3])
    pf.close()


def test_unit_prefetch_close_graceful() -> None:
    print("\n--- UnitPrefetch: graceful close ---")
    from src.training.asyncprefetch import UnitPrefetch

    pf = UnitPrefetch(build_fn=lambda item, idx: idx, total=4, depth=2, timeout=1.0)
    pf.start(list(range(4)))
    pf.close()
    pf.close()  # idempotent
    test("close idempotent", True)
    test("no results left behind", pf.queue_depth() == 0)


def test_async_ckpt_writer_durability() -> None:
    print("\n--- AsyncCheckpointWriter: durability ---")
    from src.training.checkpoint import AsyncCheckpointWriter, verify_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        writer = AsyncCheckpointWriter(checksum=True)

        def write_fn(path):
            (path / "model.bin").write_bytes(b"w" * 1024)
            (path / "meta.json").write_text('{"step": 7}', encoding="utf-8")

        ok = writer.submit(Path(tmp) / "ckpt-1", write_fn)
        test("submit accepted", ok)
        test("flush durable", writer.flush(timeout=10))
        ck = Path(tmp) / "ckpt-1"
        test("files on disk", (ck / "model.bin").exists() and (ck / "meta.json").exists())
        verified, detail = verify_checkpoint(ck)
        test("checksum manifest verified", verified, detail)
        writer.close()


def test_async_ckpt_writer_checksum_detection() -> None:
    print("\n--- AsyncCheckpointWriter: corruption detection ---")
    from src.training.checkpoint import AsyncCheckpointWriter, verify_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        writer = AsyncCheckpointWriter(checksum=True)

        def write_fn(path):
            (path / "model.bin").write_bytes(b"data" * 64)

        writer.submit(Path(tmp) / "ckpt-1", write_fn)
        writer.flush(timeout=10)
        ck = Path(tmp) / "ckpt-1"
        (ck / "model.bin").write_bytes(b"corrupted" * 16)
        ok, detail = verify_checkpoint(ck)
        test("tampered file detected", not ok, detail)
        writer.close()


def test_async_ckpt_writer_sync_fallback() -> None:
    print("\n--- AsyncCheckpointWriter: dead-writer fallback ---")
    from src.training.checkpoint import AsyncCheckpointWriter

    with tempfile.TemporaryDirectory() as tmp:
        writer = AsyncCheckpointWriter(checksum=False)
        writer._failed = True
        written = []

        def write_fn(path):
            written.append(path)

        ok = writer.submit(Path(tmp) / "ckpt-1", write_fn)
        test("submit falls back to sync when writer dead", ok is True)
        test("job still written", len(written) == 1)


def test_async_ckpt_writer_legacy_verify() -> None:
    print("\n--- AsyncCheckpointWriter: legacy checkpoint accepted ---")
    from src.training.checkpoint import verify_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "legacy"
        ck.mkdir()
        (ck / "pytorch_model.bin").write_bytes(b"x" * 8)
        ok, detail = verify_checkpoint(ck)
        test("no manifest -> accepted", ok and "legacy" in detail, detail)


def test_async_ckpt_atomic_write() -> None:
    print("\n--- AsyncCheckpointWriter: atomic temp rename ---")
    from src.training.checkpoint import _atomic_write_bytes

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "f.bin"
        _atomic_write_bytes(target, b"hello")
        test("atomic write landed", target.read_bytes() == b"hello")
        leftovers = list(Path(tmp).glob("f.bin.*"))
        test("no temp files left", len(leftovers) == 0, f"left: {leftovers}")


def test_telemetry_counters() -> None:
    print("\n--- PipelineTelemetry: counters ---")
    from src.infrastructure.telemetry import PipelineTelemetry

    tm = PipelineTelemetry(enabled=True, interval_sec=60, gpu_util_sampling=False)
    tm.record("cache_hits")
    tm.record("cache_attempts", 3)
    tm.record("tokens", 1000)
    tm.set_event("stage", 2)
    s = tm.snapshot()
    test("counters accumulate", s["cache_hits"] == 1 and s["cache_attempts"] == 3)
    test("event value kept", s["stage"] == 2)
    test("rates computed", s["tokens_per_sec"] >= 0)
    test("cache hit pct", abs(s["cache_hit_pct"] - 33.33) < 0.1, str(s["cache_hit_pct"]))
    test("summary line renders", "tok/s=" in tm.summary_line())
    tm.finish()


def test_telemetry_disabled() -> None:
    print("\n--- PipelineTelemetry: disabled no-op ---")
    from src.infrastructure.telemetry import PipelineTelemetry

    tm = PipelineTelemetry(enabled=False, interval_sec=60)
    tm.record("anything", 42)
    s = tm.snapshot()
    test("disabled records nothing", "anything" not in s)
    tm.finish()


def test_telemetry_monitor_lifecycle() -> None:
    print("\n--- PipelineTelemetry: monitor lifecycle ---")
    from src.infrastructure.telemetry import PipelineTelemetry

    tm = PipelineTelemetry(enabled=True, interval_sec=1, gpu_util_sampling=False)
    tm.start()
    time.sleep(2.2)  # >= 2 sample ticks
    tm.stop()
    test("monitor started and stopped without raising", True)


def test_unit_prefetch_stress_volume() -> None:
    print("\n--- UnitPrefetch: stress (400 units, mixed speed, depth 8) ---")
    import random
    from src.training.asyncprefetch import UnitPrefetch

    random.seed(42)
    N = 400
    built = []
    lock = threading.Lock()

    def build(item, idx):
        delay = random.uniform(0, 0.004) if idx % 7 else random.uniform(0, 0.02)
        time.sleep(delay)
        with lock:
            built.append(idx)
        return f"u{idx}"

    pf = UnitPrefetch(build_fn=build, total=N, depth=8, timeout=10.0)
    pf.start(list(range(N)))
    t0 = time.monotonic()
    got = []
    for i in range(N):
        r = pf.get(i)
        if r is None:
            break
        got.append(r)
    elapsed = time.monotonic() - t0
    pf.close()
    test("stress: all 400 units delivered", len(got) == N, f"got {len(got)}")
    test("stress: delivered in order", got == [f"u{i}" for i in range(N)])
    test("stress: every unit built exactly once",
         sorted(built) == list(range(N)),
         f"built {len(built)}/400")
    test("stress: no deadlock/stall (whole run < 20s)", elapsed < 20.0,
         f"{elapsed:.1f}s")


def test_ckpt_submit_flush_no_hang() -> None:
    print("\n--- AsyncCheckpointWriter: submit/flush race ---")
    from src.training.checkpoint import AsyncCheckpointWriter

    N = 200
    with tempfile.TemporaryDirectory() as tmp:
        writer = AsyncCheckpointWriter(checksum=False)

        def write_fn(path):
            (path / "model.bin").write_bytes(b"x")

        t0 = time.monotonic()
        for i in range(N):
            writer.submit(Path(tmp) / f"c{i}", write_fn)
            if not writer.flush(timeout=10):
                test("flush never returns False on a live writer", False)
                break
        elapsed = time.monotonic() - t0
        test("200 submit+flush cycles never hang", elapsed < 60.0,
             f"{elapsed:.1f}s")
        test("all cycles durable", all((Path(tmp) / f"c{i}" / "model.bin").exists()
                                       for i in range(N)))
        writer.close()


def test_ckpt_writer_stress_concurrent() -> None:
    print("\n--- AsyncCheckpointWriter: stress (50 ckpts, 4 submitter threads) ---")
    from src.training.checkpoint import AsyncCheckpointWriter, verify_checkpoint

    N = 50
    with tempfile.TemporaryDirectory() as tmp:
        writer = AsyncCheckpointWriter(checksum=True)
        results = {}
        errors = []

        def submitter(base):
            try:
                for i in range(base, min(base + (N + 3) // 4, N)):
                    def write_fn(path, i=i):
                        (path / "model.bin").write_bytes(
                            f"m{i}".encode() * 128)

                    ok = writer.submit(Path(tmp) / f"ckpt-{i}", write_fn)
                    results[i] = ok
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=submitter, args=(b,))
                   for b in (0, 13, 26, 39)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        test("stress: all submits accepted", all(results.get(i) for i in range(N)),
             f"{sum(1 for i in range(N) if results.get(i))}/{N} ok")
        test("stress: no submitter exceptions", len(errors) == 0,
             f"{errors[:2]}")
        test("stress: flush durable", writer.flush(timeout=30))
        verified = all(verify_checkpoint(Path(tmp) / f"ckpt-{i}")[0]
                       for i in range(N))
        test("stress: every checkpoint verifies", verified)
        writer.close()


def main() -> None:
    tests = [
        test_unit_prefetch_ordered_delivery,
        test_unit_prefetch_overlap,
        test_unit_prefetch_starvation_free,
        test_unit_prefetch_failure_re_raised,
        test_unit_prefetch_timeout,
        test_unit_prefetch_discard_and_stale,
        test_unit_prefetch_close_graceful,
        test_async_ckpt_writer_durability,
        test_async_ckpt_writer_checksum_detection,
        test_async_ckpt_writer_sync_fallback,
        test_async_ckpt_writer_legacy_verify,
        test_async_ckpt_atomic_write,
        test_telemetry_counters,
        test_telemetry_disabled,
        test_telemetry_monitor_lifecycle,
        test_unit_prefetch_stress_volume,
        test_ckpt_submit_flush_no_hang,
        test_ckpt_writer_stress_concurrent,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            global FAIL
            FAIL += 1
            import traceback
            traceback.print_exc()
            print(f"  EXCEPTION {t.__name__}: {e}")

    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
