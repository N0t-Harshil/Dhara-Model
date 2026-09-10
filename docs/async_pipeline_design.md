# Architecture Specification: Async Data Pipeline & CPU/GPU Overlap Engine

## Overview

The `specialized-coding-model` dataset pipeline is designed for maximum throughput, dataset-granular training, zero-idle GPU transition, and robust failure isolation.

```
                      DATASET SCHEDULER & PIPELINE
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
Stage Metadata Warm         Driver Resolution            Background Unit Prefetch
 (Stage N+3 / N+2)         (File/Script/Local)           (Stage N+1 Prep/Cache)
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                         BOUNDED READY QUEUE (Depth)
                                    │
                                    ▼
                       GPU TRAINER (Stage N Train)
```

---

## 1. Dataset Driver Architecture

The pipeline abstracts dataset loading into four specialized driver families:

1. **FileDatasetDriver** (`DRIVER_KIND_FILE`):
   - For file-backed HuggingFace datasets (Parquet, Arrow, CSV, JSON).
   - Resolves the remote source file list once and caches it in `DatasetMetadataCache`.
   - On warm runs, streams directly from cached file lists with zero HuggingFace repository enumeration (`Resolving data files...` calls avoided).

2. **ScriptDatasetDriver** (`DRIVER_KIND_SCRIPT`):
   - For iterable / generator / script datasets (e.g. The Stack V2, OpenCoder, CodeParrot, CodeSearchNet, Code Contests).
   - Does NOT invent fake file lists. Caches builder class identity, script revision, and raw iterator row offsets (`resume_offset`) in `BuilderCache`.
   - On warm runs, initializes the builder directly without Hub inspection.

3. **LocalDatasetDriver** (`DRIVER_KIND_LOCAL`):
   - For datasets rooted on the local filesystem.
   - Enumerates files directly from local disk with zero HuggingFace network involvement.

4. **StreamingDatasetDriver** (`DRIVER_KIND_STREAMING`):
   - Fallback driver for cold resolution when metadata caches are disabled or unavailable.

---

## 2. Asynchronous Prefetch & CPU/GPU Overlap

### Multi-Level Pipeline Concurrency
While the **GPU** trains **Dataset N**:
- **CPU Worker Pool**: Preprocesses, cleans, deduplicates, quality-filters, tokenizes, packs, and writes unit cache for **Dataset N+1**.
- **Network Layer**: Resolves driver metadata and prepares stream for **Dataset N+2**.
- **Metadata Layer**: Warm-caches source file lists for future stage groups (**Dataset N+3**).

### Prefetch Queue (`UnitPrefetch`)
- Bounded to `prefetch_depth` datasets in flight.
- Strict ordered delivery to the GPU Trainer.
- **Duplicate-build prevention**: scheduling the same dataset twice is
  impossible — `start()` derives a unit identity (`path`/`name`), builds the
  first occurrence once, and later occurrences share that payload via `get()`
  (`[ASYNC] unit N duplicate of unit M — duplicate build prevented`, counted in
  `stats["duplicates_prevented"]`).
- **Synchronous fallback**: the whole engine is bypassed when
  `data.async_pipeline.enabled: false` — each unit is built inline on the
  training thread (`_prefetch_enabled` requires both the staging `prefetch`
  opt-in and the global switch). Emergency off-switch with identical dataset
  semantics.
- Emits explicit log markers:
  - `[ASYNC] dataset N training start`
  - `[ASYNC] dataset N prefetch start`
  - `[ASYNC] dataset N preprocessing complete`
  - `[ASYNC] dataset N build cancelled (slot no longer awaited)`
  - `[ASYNC] dataset N retry backoff — next attempt in X.Xs (attempt a/b)`
  - `[UNIT TIMEOUT] dataset N exceeded prefetch_timeout (Xs) — build cancelled ...`
  - `[ASYNC] unit N cache hit`
  - `[ASYNC] dataset N training end`
  - `[ASYNC] dataset N consumed (GPU wait: X.XXs)`
  - `[ASYNC] GPU wait before dataset N = X.XX sec`
  - `[POOL] closing shared cleanup pool (N workers)`

### Cooperative Cancellation
Every slot owns a `cancel` event. The build callable may accept a
`cancel_event` kwarg (detected once via `inspect.signature` — a runtime
`TypeError` inside a build can never be mistaken for an arity mismatch, so
arity detection is never re-done on an exception); `DataPipeline` builds do,
checking it at cheap boundaries (dataset loop, chunk read, tokenize, cache
save) plus a global pipeline `cancel()` flag that aborts all in-flight
builds at shutdown. When the consumer stops waiting for a slot (timeout,
skip, shutdown) it calls `cancel(index)`; the build then either raises
`UnitBuildCancelled` (quietly counted in `stats["cancelled"]` — never an
error, never retried, never journaled) or, if it finishes anyway, the worker
drops the payload and **continues covering the remaining units** — a
cancelled or stale slot can never retire a producer permanently. On timeout
`get()` cancels the slot *before* advancing `next_expected` so the event is
always delivered to the owner.

---

## 3. Dataset-Granular Checkpoint & Resume Correctness

- Pretraining executes unit-by-unit inside each stage group.
- Checkpoints are written after every dataset unit completes.
- Checkpoints carry full identity context:
  ```json
  {
    "stage_index": 1,
    "unit_index": 2,
    "unit_key": "opc-fineweb-code-corpus",
    "unit_fingerprint": "abc123def456",
    "global_step": 1500,
    "total_steps": 50000
  }
  ```
- The fingerprint is a sha256 over the dataset identity
  (path/name/data_dir/revision/category). A revision bump changes it, so a
  resumed run retrains the unit instead of trusting its index.
- **Atomic dataset completion** — a unit is *complete* only after, in order:
  1. training finishes,
  2. the model checkpoint is saved and flushed (`_flush_checkpoints`),
  3. `unit_identity.json` is written into the latest checkpoint directory,
  4. the completion record is atomically committed to `unit_completions.json`
     (in `<output.model_dir>`, temp file + `os.replace`).
  A crash before step 4 leaves the unit incomplete; on restart the plan skips
  a unit only when the manifest fingerprint matches, otherwise it retrains.
- On restart, the scheduler inspects `global_step` and dataset checkpoints,
  skipping already completed units and resuming seamlessly.

---

## 4. Single-Dataset Unit Optimization

When a training unit consists of a single dataset, `build_pretrain_dataset_unit` returns the raw packed `Dataset` object directly, eliminating the overhead of the `WeightedMixedDataset` sampler layer.

---

## 5. Telemetry & Reporting

At the conclusion of training, the pipeline outputs the `ASYNC PIPELINE REPORT`:

```
============================================================
ASYNC PIPELINE REPORT
============================================================
datasets scheduled: 22
datasets trained: 22
datasets skipped from processed cache: 18
failed datasets: 0 | retried datasets: 0
timed-out datasets: 0 | built-then-cancelled (timeout/shutdown): 0
cache hits: 18 | cache misses: 4
metadata cache hits: 20 | misses: 2
builder cache hits: 1 | misses: 1
prefetch hits: 20 | prefetch misses: 2 | prefetch hit rate: 90.9%
GPU training time: 15120.00 sec
GPU wait time: 134.00 sec
pipeline idle: 0.9%
average dataset preparation: 12.4 sec
average hidden preparation: 12.2 sec
duplicate builds prevented: 0
============================================================
```

Per-unit `[TELEMETRY]` banners report up-time, samp/s, tok/s, GPU%, cache hit
%, net wait, build time and cache-hit flag; per-stage summaries give cache hit
rate and `GPU busy % = train_sec / wall`. GPU wait is measured inside
`UnitPrefetch.get()` and accumulated to `gpu_wait_sec` / `train_sec`
counters feeding the report. Prefetch hits = datasets already prepared when
the trainer asked for them (GPU wait < 50 ms).

---

## 6. Configuration reference

`data.async_pipeline` (all keys configurable; defaults in `src/config/schema.py`):

| key                 | default | meaning                                              |
|---------------------|---------|------------------------------------------------------|
| `enabled`           | `true`  | master switch; `false` = synchronous fallback        |
| `prefetch_depth`    | 2       | staging depth capped by the two bounds below (ge 1)  |
| `ready_queue_size`  | 2       | cap on buffered ready results — bounds effective depth (ge 1) |
| `preprocess_workers`| 4       | cap on background build worker threads (ge 1)        |
| `metadata_workers`  | 2       | fan-out of background metadata-warming threads (ge 1)|
| `max_inflight`      | 4       | cap on outstanding background builds — bounds effective depth (ge 1) |
| `retry_count`       | 3       | build retries before a failure reaches the journal (ge 0) |
| `retry_backoff_base`| 1.0 s   | exponential backoff base between retry attempts (ge 0)   |
| `retry_backoff_max` | 30.0 s  | cap on the per-attempt backoff delay (ge 0)              |

All knobs are enforced at runtime: `_effective_prefetch_depth` =
min(staging depth, ready_queue_size, max_inflight); worker threads are capped
by `preprocess_workers`; metadata warming splits the next stage's datasets
across up to `metadata_workers` threads; `retry_count` bounds background
build retries, delayed by exponential backoff with 25% jitter
(`retry_backoff_base * 2^(attempt-1)`, capped at `retry_backoff_max`); the
backoff wait is woken early by teardown so `close()` never drains it.

Staging-level knobs (`training.pretrain.staging`): `prefetch`, 
`prefetch_depth`, `prefetch_timeout` (ge 5s), `skip_failed_units_on_resume`,
`abort_on_unit_error`. Prefetch requires the staging opt-in **and** the
global `enabled` switch.

---

## 7. Verification

The requirement matrix R1–R21 is covered across suites:
`tests/test_async_pipeline_overlap.py` houses R1–R18 (19 tests: overlap timing,
drivers, legacy URIs, bounded queue, metadata warming, tokenizer hash cache,
cleanup-pool reuse, cancellation, failure isolation, checkpoint identity,
singleton bypass, atomic writes, duplicate-build prevention, Telemetry,
Trainer v5 API, report format); R19 (warm-run HF resolution bypass) and R20
(cold/warm data equivalence) are covered by `scripts/test_pipeline.py` (124
checks) and `scripts/test_local_validation.py` (258 checks); R21 (atomic
dataset completion) by `test_unit_completion_manifest_and_identity`.
`scripts/test_pipeline_async.py` adds 43 checks including 400-unit stress,
50-checkpoint 4-thread stress, and the submit/flush race regression.
`scripts/benchmark_async_pipeline.py` (6/6 checks → `reports/benchmark_async_pipeline.txt`)
measures the overlap, cold→warm speedup, checkpoint durability, telemetry
overhead and failure isolation offline with synthetic workloads.

Lifecycle / shutdown suites:
`tests/test_shutdown_coordinator.py` (9 tests — first-signal `SystemExit`
with `128+signum`, second-signal force-exit, watchdog grace + disarm on
`reset()`, idempotence)
and `tests/test_unit_prefetch_lifecycle.py` (10 tests — cooperative
cancellation semantics, cancel≠failure, idempotent cancel, bounded `close()`
join with leftover-daemon reporting, `active_builds`/`wait_idle`, retry
backoff delay + teardown cut-through, and a mixed timeout/success stress
test pinning `in_flight() <= depth`). Phase 5 stress caught a latent
producer leak (a cancelled/stale slot retired its worker thread; fixed by
`continue` instead of `return`).

`tests/test_health_reporter.py` (12 tests) pins the total-function safety of
`DatasetHealthReport.summary_text()`: a cached dataset with no single detected
language yields a `None` distribution label that previously crashed the
report formatter (`TypeError: unsupported format string passed to
NoneType.__format__`, `health_reporter.py:223`); `None` now renders as
`unknown`/`N/A` and the exact packed-cache + async-worker call sequence is
regression-tested.
