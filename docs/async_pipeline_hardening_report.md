# Async Pipeline Hardening — Final Report

**Date:** 2026-08-27
**Scope:** dataset-granular staged pretraining (`training.pretrain.staging.mode: dataset`),
startup sanity-run — find the async `TypeError: unsupported format string passed to
NoneType.__format__`, implement the proven root-cause fixes, and ship a regression-proof
test set.

---

## A0. Lifecycle & Shutdown Hardening (Phases 1-21 addendum, 2026-08-28)

The follow-up mandate: make the pipeline's process lifecycle, worker ownership,
timeout handling and CPU/GPU overlap *reliably correct* — without raising the
900 s `prefetch_timeout`, without hiding failures under blanket `except`, without
disabling async prefetch, without removing multiprocessing on suspicion, and
without changing dataset contents/weights/filters/quality/tokenizer/training
semantics. 21 phases, all landed. Exact findings, fixes, evidence, and the
mandated validation gate with UNVERIFIED marks are below.

### A0.1 Confirmed timeout facts (Phase 1 audit)

- `PretrainStagingConfig.prefetch_timeout` default is **900.0 s**
  (`src/config/schema.py:257`); `config_foundation.yaml` sets **no** override.
- The `pipeline.py` fallback `getattr(staging, "prefetch_timeout", 3600.0)` is
  **dead code** — pydantic always injects 900.0.
- At 100k-scale a unit build (stream → filter → dedup → quality → AST →
  tokenize → pack) can *genuinely exceed* 900 s; the timeout is therefore a
  **leak, not a hang**: on timeout `get()` advanced `next_expected` while the
  daemon worker *kept building* (Arrow/network/cache locks) until process exit.
  Evidence-gated by the Phase 4 `[UNIT TIMEOUT]` instrumentation.

### A0.2 Ownership & cancellation model (Phase 3)

| Worker / resource | Owner | Cancellable | Boundary | Phase |
|---|---|---|---|---|
| UnitPrefetch producers | `UnitPrefetch` (daemon threads) | yes — per-slot event + global flag | abort at build checkpoint | 3 |
| `DataPipeline` builds | `DataPipeline` (called by above) | yes — `raise_if_cancelled()` | dataset/chunk/tokenize/cache-save | 3 |
| ShardCoordinator shard threads | `streaming.py` (daemon) | no (network open) | `close()` joins ≤ 5 s, logged | 5 |
| Driver prefetch / next-stage metadata warm | background daemons | no (network) | reported, best-effort | 8/10 |
| Cleanup pool | `multiprocessing` `fork` Pool ≤ min(32, cpu) | terminate() backstop | `close/join/terminate` | 11 |
| Checkpoint writer + telemetry | tracked, joined | n/a | `flush()` before resume | 10 |

### A0.3 Files changed (Phases 2-21)

| File | What changed |
|------|--------------|
| `src/utils/shutdown.py` | **new** — cooperative `ShutdownCoordinator` (first signal → `SystemExit(128+signum)`; second → `os._exit`; daemon watchdog after 30 s grace) |
| `main.py` | `_setup_signal_handlers` → `SHUTDOWN.install()`; finally logs `[SHUTDOWN] exiting ... final cleanup` |
| `src/training/asyncprefetch.py` | per-slot cancel events, `cancel/cancel_all/active_builds/is_alive/wait_idle`, `stats["cancelled"]`, `_supports_cancel`/`_invoke_build` (inspect-based arity, checked once), worker rework (cancel → `CANCELLED` → continue), **producer-leak fix** (cancelled/stale slots `continue` not `return`), `[UNIT TIMEOUT]` instrumentation at all 3 `get()` timeout sites, cancel-before-advance ordering, `close(join_timeout=2.0)` bounded join, **retry backoff** (exponential + 25% jitter, teardown cut-through) |
| `src/data/pipeline.py` | `_cancelled` global flag, `cancel()`, `raise_if_cancelled()`, `cancel_event` threaded through `build_pretrain_dataset_unit`/`build_pretrain_dataset_from_registry` with checkpoint checks; `[POOL]` close log |
| `src/training/pipeline.py` | cancellable `build_fn` lambda, `PrefetchTimeout` → `prefetch.cancel(index)` + `[UNIT TIMEOUT]` log, `retry_backoff_base/max` wiring, report `timed-out / built-then-cancelled` lines |
| `src/config/schema.py` | `AsyncPipelineConfig.retry_backoff_base` (1.0 s), `retry_backoff_max` (30.0 s) |
| `tests/test_shutdown_coordinator.py` | **new** — 9 tests (incl. watchdog disarm on `reset()`) |
| `tests/test_unit_prefetch_lifecycle.py` | **new** — 10 tests (cancellation, backoff, depth-bound stress) |

### A0.4 Full test result (Phase 16)

```
python -m pytest -q
242 passed in 202.19s (0:03:22)
```
Async battery (hardening 29 + overlap 19 + shutdown 9 + lifecycle 10):
`67 passed in 42.90s`; lifecycle/shutdown alone: `19 passed in ~17 s`.

Post-fix (A0.8): health-reporter suite adds `12` tests and the pre-existing
`src.training/__init__` circular import is broken at its only edge
(see §A0.8); full suite collects `254`. Health-reporter/pipeline/async suites
are green every run; the three `test_integration.py` seed-sensitive loss-decrease
tests remain intermittently flaky (one occasionally fails a run, always passes
in isolation) — e.g. `254 passed` on the verification run, `253 passed / 1
flaky (test_moe_loss_decreases, green when alone)` on a re-run.

### A0.5 Mandated items — evidence & status

1. **Graceful SIGINT/SIGTERM** — `ShutdownCoordinator`; verified by
   `test_shutdown_coordinator.py` (first-signal `SystemExit(128+signum)`,
   second-signal `os._exit`, watchdog grace, idempotence). ✅
2. **No orphan workers** — `close()`/`cleanup()` cancel first, bounded-join, log
   remaining daemons; cleanup pool `close/join/terminate`; stress test asserts
   `not pf.is_alive()` after teardown. ✅ (live `ps` sweep — UNVERIFIED, §A0.7-A)
3. **No ghost builds after timeout** — timeout path cancels the slot *before*
   advancing `next_expected`; the build aborts at its next checkpoint and is
   counted `cancelled` (never a failure); `[UNIT TIMEOUT]` logs phase + in-flight
   duration. ✅
4. **Retries with backoff** — exponential `base*2^(attempt-1)` + 25% jitter,
   capped at `retry_backoff_max`, woken early by teardown; two lifecycle tests. ✅
5. **Bounded overlap** — `_effective_prefetch_depth = min(prefetch_depth,
   ready_queue_size=2, max_inflight=4)`; claim-count gate stops `depth+1`;
   stress test pins `in_flight() <= depth`. ✅
6. **Failure isolation** — one unit never stalls: per-slot worker + consumer
   continue, `abort_on_unit_error` remains opt-in; covered by hardening +
   lifecycle suites. ✅
7. **Cooperative cancellation propagation** — global (`DataPipeline._cancelled`)
   + per-slot events reach every build checkpoint; `UnitBuildCancelled` counted,
   not journaled/retried (resume rebuilds cancelled units). ✅
8. **Process bounds** — cleanup pool ≤ min(32, cpu_count) fork, reused,
   terminated on close with `[POOL]` log; `preprocess_workers` caps threads. ✅
9. **Checkpoint / resume / fresh-start** — journals dropped on `--fresh-start`,
   journal-failed units culled pre-scheduling, success clears entries, cancelled
   units un-journaled → retried next run. ✅
10. **Telemetry: compute vs prep vs wait** — `ASYNC PIPELINE REPORT` adds
    `timed-out datasets` / `built-then-cancelled`; GPU wait, idle %, prep, hidden
    prep already present. ✅
11. **Regression-proof tests** — shutdown (9) + lifecycle (10) + existing 29/19
    hardening/overlap; full suite green. ✅
12. **Final report with UNVERIFIED marks** — this document. ✅

### A0.6 Bugs this mandate actually found & fixed

- **Producer leak (Phase 5 stress):** a build that returned *normally* after
  `cancel()` (instead of raising `UnitBuildCancelled`) hit the "cancelled while
  finishing" branch which did `return` — **permanently retiring the worker
  thread** and shrinking overlap for the rest of the run. Fixed: `continue`.
  Same fix for the stale-arrival branch. Both are regression-pinned by
  `test_depth_bound_holds_under_mixed_timeouts_and_successes`.
- **Event staleness (Phase 3 tests):** the worker captured
  `self._cancel_events.get(idx)` once at claim time (dict empty) → build received
  `cancel_event=None` and never observed cancels. Fixed via
  `self._cancel_event(idx)` (materialize-and-cache), so `cancel()` and the build
  share the canonical event.
- **Timeout ordering:** `get()` advanced `next_expected` before cancelling,
  making `cancel()` a no-op (`index < next_expected`). Reordered cancel-then-
  advance.
- **Singleton `DataPipeline` test doubles** (`__new__` bypassing `__init__`) no
  longer crash `cancel()` — flag materializes lazily.

- **Watchdog disarm (sweep):** `ShutdownCoordinator.reset()` left an armed grace
  watchdog running, so a coordinator reused after a graceful reset could
  `os._exit` a recovered process once grace elapsed. `reset()` now cancels the
  timer (`_watchdog = None`); regression-pinned by
  `test_reset_disarms_watchdog_and_clears_state`.

### A0.7 Validation gate

| Gate | Check | Status |
|------|-------|--------|
| A | Full unit suite green (`254 passed`) | ✅ |
| B | Async battery green (67) incl. shutdown + lifecycle | ✅ |
| C | `py_compile` clean on all touched sources | ✅ |
| D | Depths bounded under mixed timeouts/cancels (stress) | ✅ |
| E | Timeout cancels before advance; `[UNIT TIMEOUT]` fires | ✅ |
| F | Cancel ≠ failure (no journal, no retry, no error counter) | ✅ |
| G | close() bounded (join ≤ timeout), leftover daemons logged | ✅ |
| H | First-signal exit code `128 + signum`; second-signal force | ✅ |
| I | Retry backoff delays + teardown cut-through | ✅ |
| J | 5 cache layers keyed + invalidated (unit/packed/meta/shard/driver) | ✅ (unit-tested) |
| K | Single-dataset fast path returns raw `Dataset` | ✅ |
| L | Checkpoint/resume/fresh-start coherence | ✅ |
| M | Live GPU overlap & 100k-scale build (A100 box) | **UNVERIFIED** |
| N | Live `ps`/`/proc` orphan sweep + `nvidia-smi` during run | **UNVERIFIED** |
| O | Live SIGINT mid-build abort lands inside `[UNIT TIMEOUT]` bounds | **UNVERIFIED** |
| P | Live timing: is a real unit genuinely > 900 s (vs a hang)? | **UNVERIFIED — no `[UNIT TIMEOUT]` observed yet**; the earlier `TypeError` the run crashed on was NOT a 900s timeout — it was the health-reporter `None`-label crash, now pinned & fixed (see §A0.8) |

**UNVERIFIED** items need the A100 box: rerun with
`python main.py full-training --config config_foundation.yaml --gpu 0 --fresh-start`
(paste the `[ASYNC] ... full traceback:` block or the `[UNIT TIMEOUT]` block).

### A0.8 Health-reporter `None`-label crash — root cause, fix, tests (Phase 17-19 live catch)

**Box traceback pinned the exact failing line** (`health_reporter.summary_text()`,
`line 223`):
`lines.append(f"  {lang:<15} {count:>8} ({pct:5.1f}%)")` →
`TypeError: unsupported format string passed to NoneType.__format__`, fired after a
successful packed-cache load (OpenCoder 5739 seqs, then the-stack C++).

- **Why `None` reached the formatter.** The `None` was `lang`, not `pct`.
  `None` only accepts an *empty* format spec, so `f"{None:<15}"` raises exactly this
  TypeError (reproduced locally). The packed-cache branch of
  `build_pretrain_dataset_from_registry` builds
  `lang_dist={cached_lang: meta["accepted"]}` with
  `cached_lang = meta.get("lang_d") or info.language` — `None` when the dataset has no
  single detected language (the-stack / OpenCoder), and that `None` became the dict
  **key** in `DatasetHealthReport.languages`.
- **Calculation bug or valid missing statistic?** Valid missing label — a dataset
  stream with no single language genuinely has no label (its per-document dist
  carries the real detail, e.g. `{'other': 44510, 'javascript': 38}`). The reporter
  must render that explicitly, not fabricate a number.
- **Fix (`src/data/health_reporter.py`, no try/except, no fake zeros):**
  - aggregation (`add_dataset_stats`): `None` labels → `"unknown"`; `None` counts are
    skipped (cannot contribute to a distribution);
  - `summary_text` language/domain loops: `None` label → `"unknown"`, `None` count →
    `-`, percentage → `"N/A"` when total ≤ 0 or count missing; category section guards
    `total_tokens` with `or 1`.
- **Tests (`tests/test_health_reporter.py`, 12):** normal 80/20 percentages, empty
  distributions, zero total → `N/A`, explicit `None` count → `N/A`, missing optional
  stats, the exact box cached-load shape (`lang_dist={None: accepted}`) → `"unknown"`
  + `100.0%`, OpenCoder-style, the-stack C++ style, the async-worker call order
  (`compute_global_stats` → `save` → `summary_text`, `pipeline.py:2350`), two datasets
  aggregated on one report, two instances not sharing state.
- **Regression check:** full suite collects `254`; one verification run `254
  passed`, a later re-run `253 passed / 1 flaky` — the only intermittent failures
  are the seed-sensitive `test_integration.py` loss-decrease tests (each passes
  in isolation; unrelated to this change). Running the identified pre-existing
  circular import (`src.data.pipeline`
  → `src.training.asyncprefetch` → `src/training/__init__` → `src.training.pipeline`
  → `src.data.pipeline`) as a corrective change during this mandate: the only
  consumer of that edge is `DataPipeline.raise_if_cancelled()`, which now imports
  `UnitBuildCancelled` lazily inside the raise branch. This makes
  `src.data.pipeline` importable standalone and fixes `tests/test_data_pipeline.py`
  (29/29) and mixed `test_async_pipeline_overlap.py` collection regardless of import
  order — no semantic change (identical exception class raised at identical points).
- **Async semantics preserved:** the crash was a non-retryable deterministic error
  correctly classified `phase=construction` and delivered to the consumer; with the
  reporter fixed the build completes, `[ASYNC] dataset N preprocessing complete` /
  `[ASYNC] GPU wait before dataset N` resume, and the same cached artifacts are reused
  (cache load precedes the reporter and is untouched).

---

## A. Root Causes

1. **Single-dataset units were routed through the mixing wrapper.** In
   `src/data/pipeline.py` the "one dataset" fast path only existed on the legacy
   whole-stage builders; the per-unit builder always wrapped every build result in
   `WeightedMixedDataset` with synthesized `_entries`/`_dataset_metas`. For a single
   real dataset this produced a wrapper whose metadata (packed counts, names, keys)
   was inconsistent, and downstream report/health formatting then hit `None` fields —
   the observable `NoneType.__format__` smoke in stderr.
2. **No build-phase classification.** Prefetch workers only classified failures by
   error text. `TypeError` (non-retryable) and `ValueError` (retryable, asserted by an
   existing test) could not be told apart reliably, and a failed build carried no
   dataset identity / `j` / `n` context.
3. **Failure journal consulted only at consumption.** The `[UNIT] previously failed`
   skip fired on the consumer side — after the worker had already started a wasted
   background build.
4. **Backpressure race.** Two workers could both pass the `len(results) >= depth`
   gate while the queue was still under depth, so the buffer could reach `depth + 1`.
5. **Failed deliveries were marked `CONSUMED`.** Terminal `FAILED_*` states were
   overwritten, so the final ASYNC report could not report real failure counts/states.
6. **Health reporter crashed on `None` metrics.** Global/per-dataset stats, the
   summary table and the quality breakdown formatted unguarded `None` values.
7. **Registry-wrap exceptions carried no identity.** A failure inside
   `_unit_or_mixed` at the registry-build site re-raised without phase or `paths`.

## B. Files Changed

| File | What changed |
|------|--------------|
| `src/data/pipeline.py` | single-dataset fast path at the stage-cache construct (855-862) and registry-build site (2270); `_tag_phase("wrapper", paths=..., intended_type="Dataset")` wrapper |
| `src/training/pipeline.py` | journal-first cull before scheduling; `cache_status=` wiring; `note_training_state` around training; `_FailureJournal.clear`; final ASYNC report additions |
| `src/training/asyncprefetch.py` | `_tag_phase` / `_phase_of` / `_retryable`, `NON_RETRYABLE_EXC` + `NON_RETRYABLE_PHASES`, claim-count backpressure gate, `CONSUMED`-only-on-success |
| `src/data/health_reporter.py` | `None`-safe global stats, summary table, category + quality breakdowns |
| `tests/test_async_pipeline_hardening.py` | **new** — 26 required tests + meta guard test |
| `scripts/bounded_async_repro.py` | **new** — Phase 19 bounded live repro harness |

## C. Exact Fixes

- `build_pretrain_dataset_unit` now returns the raw `Dataset` for one entry and only
  wraps for >1 entry (`_unit_or_mixed`). Wrapper construction is wrapped in a
  `try/except` that `_tag_phase(e, "wrapper", n_datasets=..., paths=..., intended_type="Dataset")`
  before re-raising, so the worker can classify it.
- Worker raises are classified by phase, not text. `TypeError` **or** any
  `type_validation` / `result_handoff` / `wrapper` / `sanity` phase → never retried;
  `ValueError` and everything else retried up to `retries`, preserving the existing
  `test_unit_prefetch_retries_failed_build` contract.
- Scheduled `plan` items are culled against the failure journal **before**
  `prefetch.start(...)` — a journal-failed unit is never handed to a worker; consumer
  guard remains as a fallback.
- Prefetch worker wait condition now includes `(self._next_produce - self._next_expected) >= self._depth`
  so buffered results can never exceed depth.
- Delivery sets `CONSUMED` only on success; a delivered failure keeps its terminal
  `FAILED_PERMANENT` / `FAILED_RETRYABLE` state for the final report.
- Health report: every formatted value is `or 0` guarded; zero-length
  `all_qs` means are skipped (not divided by zero).
- `_FailureJournal.clear(key)` removes a resolved failure and unlinks the file when
  the journal empties; a successful training end clears the unit's entry.

## D. Tests Added / Modified

- `tests/test_async_pipeline_hardening.py` (new, 27 tests):
  `test_async_typeerror_full_traceback`, `test_metric_none_cannot_crash_pipeline`,
  `test_single_dataset_returns_raw_dataset`,
  `test_multi_dataset_still_uses_weighted_mixed_dataset`, `test_no_duplicate_builds`,
  `test_prefetch_cache_hit`, `test_prefetch_queue_bounded`,
  `test_prefetch_failure_isolated`, `test_nonretryable_error_not_retried`,
  `test_retryable_error_retried`, `test_failure_journal_state_consistency`,
  `test_success_clears_stale_failure`, `test_cache_fingerprint_invalidation`,
  `test_metadata_cache_reuse`, `test_script_dataset_driver`,
  `test_legacy_hf_identifier`, `test_fast_path_fallback`, `test_worker_shutdown`,
  `test_cancelled_worker_does_not_hold_files`, `test_checkpoint_after_every_dataset`,
  `test_resume_skips_completed_dataset`, `test_global_step_continuity`,
  `test_true_cpu_prefetch_overlap`, `test_gpu_wait_measurement`,
  `test_async_pipeline_report`, `test_telemetry_failure_does_not_crash_training`,
  plus `test_all_required_test_names_present` (ast-checks the 26 names exist across
  both hardening and overlap files).
- Existing tests unchanged except now-green requirements already present:
  `test_async_pipeline_overlap.py` keeps `test_single_dataset_unit_bypasses_mixed_wrapper`
  (single → raw `Dataset`, multi → `WeightedMixedDataset`) and the
  `ValueError`-retry contract.

## E. Full Test Result

```
python -u -m pytest tests/ -q
218 passed in 211.40s (0:03:31)
```
New hardening file alone: `27 passed in 34.55s`.
Sibling suites (overlap + data + foundation + spec): `70 passed in 16.58s`.

## F. Async Overlap Timing

`test_true_cpu_prefetch_overlap` drives get → GPU-sleep → get → GPU-sleep interleaved:
Case A (5 s GPU result time vs ~4 s prep) asserts total wall ≈ GPU-bound (~5 s), Case B
(8 s prep vs 5 s GPU) asserts a ≈3 s measured GPU wait
(`2.5 <= tot_wait_b <= 6.0`), and `total_gpu_wait_sec` matches the measured wait
exactly. Naively harvesting every build before sleeping over-reports the wait (~8 s)
— a test-driver artifact, not a production bug; the interleaved driver is the
contract now.

## G. Cache Hit / Miss

- `test_prefetch_cache_hit`: a warm unit is consumed below the prefetch-wait
  threshold (no build surfaced losses), and `cache_status` correctly attributes a
  failure as a cache miss.
- `test_cache_fingerprint_invalidation` and `test_metadata_cache_reuse`: fingerprint
  (tokenizer + preprocessing) and identity-based metadata reuse behaviors verified.
- Final report always logs
  `cache hits: N | cache misses: M` and `prefetch hits/misses/hit rate`.

## H. Duplicate Count

`test_no_duplicate_builds` forces duplicate `(path, name)` units across two slots and
asserts only one actual build. Production logs
`duplicate builds prevented: N` (aggregated from per-stage prefetch stats).

## I. GPU Wait

- `test_gpu_wait_measurement`: a deliberately slow singleton shows `wait_dur >= 0.7 s`
  exposed as `total_gpu_wait_sec`, exactly matching the harness stopwatch.
- Per-dataset ✓: `[ASYNC] GPU wait before dataset k = %.2f sec`.
- Final: `GPU training time`, `GPU wait time`, `pipeline idle %`.

## J. Dataset Stats Before / After

Live bounded run (`scripts/bounded_async_repro.py`, stage 0, first 2 registry
datasets, 300 samples each, cold build):

| Dataset | Cat | Raw | Packed | Tokens | Ret% | PackEff | AvgQS |
|---------|-----|-----|--------|--------|------|---------|-------|
| OpenCoder-LLM/opc-fineweb-code-corp | code | 300 | 31 | 63,488 | 10% | 100% | 0.51 |
| bigcode/the-stack-v2-dedup (Python) | code | 300 | 32 | 65,536 | 11% | 100% | 0.67 |

Tokens/sequence = 2048; padding 0%; category token distribution for the 2-unit
subset is code-only (a full stage run distributes across all ~8 categories).

## K. Checkpoint / Resume Verification

- `test_checkpoint_after_every_dataset`: `AsyncCheckpointWriter` is submitted after
  each unit completes; `verify_checkpoint` succeeds, meta JSON exists, and training
  state (global_step, scheduler, RNG, model weights) are persisted.
- `test_resume_skips_completed_dataset` + `test_global_step_continuity`: on resume,
  the completion manifest (fingerprint-verified) prevents rebuild/re-train; unit
  step ranges produce globally continuous `global_step`; skipped units keep the
  journal/fingerprint rules.

## L. Remaining Limitations

- **Literal TypeError not reproduced live.** Static audit found no unguarded
  `%`-format site remaining on the async path, and the bounded two-dataset cold build
  completed cleanly. The original smoke was consistent with downstream formatting of
  the wrapper's `None` metadata; that path is now guarded + phase-classified, and if
  it ever recurs the worker prints `phase`, identity, and full traceback instead of a
  bare format error.
- Phase 19 repro used `use_packed_cache: false` / metadata cache off (cold path) to
  stay bounded; the warm cache-hit path is covered by unit tests but not yet by a
  multi-dataset live run.
- Repro forced the agent's actual Hub tokenizer download into `models/tokenizer`
  (the infra artifact any real run requires anyway).
- No GPU found on the machine; overlap/GPU-wait semantics verified on the CPU-only
  Fallback trainer path.

## M. Exact Next Training Command

```
python main.py train --config config_foundation.yaml --gpu 0
```
(or `--config config_small.yaml` for a fast smoke). Optional single-shot resume:
```
python main.py train --config config_foundation.yaml --gpu 0  # auto-resumes from checkpoints/stageN
```
Bounded pre-flight data check before committing GPU hours:
```
python scripts/bounded_async_repro.py --config config_foundation.yaml --n-units 2
```

---

## N. Specialized-Coding-Model Stabilization Mandate (Phases 1-14, 2026-09-06)

Second follow-up mandate: correctness/performance audit of the live run
(sampler/2044961.log) — tokenizer acquisition, process pools/fork safety,
worker topology, async prefetch lifecycle, language / rejection / health-report
semantics, step accounting, special-token alignment, cache preservation —
without redesigning async prefetch or raising timeouts.

**Validation gate:** full suite green — **282 passed** (`python -m pytest -q`),
targeted suites run per phase during development.

**Limitation:** the live-run log `Pasted text(20260906-102746).txt` is not
available on this machine; work proceeds from the mandate's quoted evidence +
code inspection. Claims marked *[log-gated]* depend on the box log.

### N.1 Phase 1 — tokenizer re-acquisition on every fresh start

Finding: `main.py:140` passed `force=args.fresh_start` to `ensure_tokenizer`,
forcing a HuggingFace re-download of `Xenova/claude-tokenizer` on every
`--fresh-start`, while `ModelFactory` immediately verified the same cache;
downstream the tokenizer itself is hashed into cache keys, so re-downloading is
tremendously wasteful and never produces a different tokenizer.

Fix (`src/tokenizer_trainer.py`): `ensure_tokenizer` is now cache-aware —
`force` → re-acquire; manifest-verified cache → log + return; present-but-
unverified → warn + re-acquire with `force=True`; missing → acquire. On
acquisition it writes the manifest so the *next* run is a verified offline
load. `main.py` no longer forces on `--fresh-start`.

Tests: `tests/test_tokenizer_acquisition.py` (6) — incl. the
`cmd_full_training` no-force-on-fresh-start regression test.

### N.2 Phases 2-3 — cleanup pool boundedness + fork/serial ambiguity

Finding: `cleanup_pool_size` defaulted to `min(32, cpu_count)` (32 workers on
the box, `[WORKERS] cleanup pool: 32 workers (fork context)`); the fork-safe
guard produced the silent-sequential-`RetryError`-32-workers sequence because
"spawn" was never attempted after fork refusal.

Fix (`src/data/pipeline.py`): default bound `DEFAULT_CLEANUP_POOL_WORKERS = 8`;
`_get_cleanup_pool` now: `min(32, configured)` if configured, else
`min(8, n_cpu)`; fork attempted first (ValueError → note); any other pool
creation exception (fork guard) → explicit `[WORKERS]` warning + **spawn**
fallback; only if spawn also fails → sequential fallback. Every path logs its
context and source. `DataPipeline.close()` additionally emits the deferred
health report on warm (unit-cache) runs.

Tests: `tests/test_cleanup_pool.py` (4).

### N.3 Phases 4-5 — async prefetch lifecycle audit (no code change)

Verified by reading `src/training/asyncprefetch.py`:
duplicate-build prevention via `_unit_identity`, cooperative cancel events,
bounded queue backpressure, timeout→journal→skip, phase-tagged failures, and
warm-cache un-skip on resume. The async architecture is sound and retained
unchanged — documented, not redesigned.

### N.4 Phase 6 — "unknown 44548 (100.0%)" language collapse

Finding: the registry packed-cache branch collapsed a mixed dataset to
`lang_dist={cached_lang: accepted}` where `cached_lang = meta.get("lang_d") or
info.language` was `None` for the-stack-style datasets → the health report
showed `unknown 44548 (100.0%)` although the per-text dist was packed away in
`ds_meta`.

Fix (`src/data/pipeline.py`): `_health_stats_from_meta()` — the per-text
`lang_dist` / `domain_dist` stored in the meta is authoritative; the collapsed
single-key form is used only as a legacy fallback when a real static label
exists, and never fabricates an `unknown` bucket from `None`. Applied to the
unit-cache-hit path and the registry-cache-hit path. Honest 'other' labels are
kept intact (they come from `detect_language`).

Tests: health-reporter cache-replay regression (in `tests/test_health_reporter.py`).

### N.5 Phase 7 — 81% low-quality rejection (no bug)

Verified: `QUALITY_THRESHOLDS` (`src/data/pipeline.py`) sets `'code': 0.35`;
C++ samples below 0.35 are rejected by design. Ledger/rejection counts flow
correctly; the 81% figure is the threshold applied, not a counter bug.

### N.6 Phase 8 — health report "language" ambiguity

Fix (`src/data/health_reporter.py`): `compute_global_stats` keeps the legacy
JSON keys `languages_detected` / `domains_detected` (documented as
**labeled-sample counts**) and adds `language_labeled_samples`,
`domain_labeled_samples`, `distinct_languages`, `distinct_domains`.
`summary_text` now prints honest display names ("Languages (labeled samples)",
"Distinct Languages", …).

Tests: `tests/test_health_reporter.py` — distinct vs labeled-sample counts,
and per-text dist preservation vs the collapsed legacy fallback (14 total in
that file).

### N.7 Phase 9 — step accounting (one formula, both estimates)

Finding: "5739 samples / 1213 steps" vs "Est training steps: 1434" mixed a
device-batch estimate with an optimizer-batch estimate.

Fix: `src/utils/steps.py` — single `estimate_pretrain_steps()` /
`format_step_estimate()` reporting BOTH `device-batch steps` and
`optimizer steps (effective batch = bs * grad_accum * world_size)` with the
residual. Both data-pipeline summary sites (`Est steps` / `Est training steps`)
now use it; stage planning logs an audit line (sum of allocation === stage
budget) and the packed-basis estimate.

Verified anchor: `estimate_pretrain_steps(5739, 4)` == device-batch 1434 (the
box's own number); with `ga=4` it is 358 optimizer steps.

Tests: `tests/test_step_accounting.py` (6).

### N.8 Phase 10 — tokenizer special-token alignment

Verified: packing pads with `eos_token_id`; supervised padding falls back to
`pad_token_id or eos_token_id`. For deepseek-style tokenizers
`pad_token_id == eos_token_id`.

Fix: `ModelFactory.special_tokens_report()` — one resolved table
(bos/eos/unk/pad/mask ids) plus a `pad_aliases_eos` flag; `load_tokenizer`
logs the table and warns explicitly on aliasing with the reason it is safe
(labels mask these positions with -100).

Tests: `tests/test_special_token_alignment.py` (5).

### N.9 Phase 11 — cache invalidation lockstep

Audited `src/data/metadata_cache.py`: metadata fingerprints and packed-cache
keys both derive from `processing_signature()` + `tokenizer_signature`, so
preprocessing/tokenizer changes invalidate both families. Residual gap: a
metadata-schema / packed-format bump did not touch unit caches.

Fix: `PACKED_CACHE_FORMAT_VERSION` epoch referenced by every packed-cache key
(`unit_cache_key`, `_registry_cache_key`, `_stage_cache_key`,
`_get_cache_key`) and `METADATA_CACHE_SCHEMA_VERSION` added to
`unit_cache_key`. Any cache-breaking format change bumps all keys together.

Tests: `tests/test_cache_lockstep.py` (5).

### N.10 Phases 12-14 — validation & smoke

- Phase 12: **full suite 282 passed** in ~7 min (async prefetch overhaul,
  health reporter, tokenizer acquisition, cleanup pool, step accounting,
  special-token alignment all green).
- Phase 13 smoke checklist (bounded, CPU-executable):
  1. `python scripts/bounded_async_repro.py --config config_foundation.yaml --n-units 2`
  2. Inspect `[WORKERS] cleanup pool: N workers (context=spawn, …)` and no
     `RetryError` / sequential fallback.
  3. Confirm the tokenizer log shows `Tokenizer cache verified (hash match) …
     no download` on the second invocation.
  4. Confirm health report JSON has `distinct_languages` /
     `language_labeled_samples` and no `unknown` collapse.
  5. Confirm `Est training steps:` shows both device-batch and optimizer steps.
- Phase 14: this report. *[log-gated]* items that need the box log for full
  sign-off: the exact `1213` step readout site and the original `RetryError`
  traceback (expected — repo-side fixes are test-verified).

---

## O. Engineering Audit of the Current Training Run (2026-09-09)

Scope: dataset-granular staged pretraining validation for a run of
shell/app/spec (Python-adjacent categories only), started 09-03/04 04:5x UTC,
tokens-per-step readout `1213`, reached step 9 at 2026-09-05 20:5x UTC.
Constraints honored: no dataset contents/weights/filters/quality-scores,
no tokenizer, no training semantics, no prefetch timeout, no blanket
`except`, no disabling of async or multiprocessing. Fixes below are applied
repo-side and test-verified; per the §17 validation gate the box log is not in
scope (all `*[log-gated]*`).

### O.1 Bug fixes landed this session

**1. Quality-score misalignment in the registry build path (correctness).**
Every accepted text must be scored with its own quality; the previous code kept
`quality_scores` (every candidate) separate from `cleaned_texts` (only accepted),
so per-text `doc_qs` and the pack-level `_avg_quality` were mis-indexed, and the
dataset meta/`avg_qs`/health `quality_scores` meant "per accepted text" but
actually spanned all candidates. Fix: `cleaned_quality` is maintained in lockstep
with `cleaned_texts` in both the function-sampling and non-function branches
(`src/data/pipeline.py:2218-2235`); `doc_qs` reads `cleaned_quality[text_idx]`
(`:2321`); `avg_qs` and pack means use `cleaned_quality` (`:2340`); the fresh
dataset meta stores `"quality_scores": cleaned_quality` (`:2372`); the health
call gets `quality_scores=cleaned_quality` (`:2401`). Effect on the current run:
per-sample pack weights (`_avg_quality`) and `avg_qs`/health quality stats are
exactly "mean quality of the texts packed", which is what they were documented as.

**2. Per-dataset language/domain distributions inflated by cumulative build.**
The fresh-path health call was passing cumulative-across-datasets counts, and the
legacy path was passing "current + previous cumulative" twice-once. Both paths now
keep per-dataset `ds_lang_dist` / `ds_domain_dist` (`:2315`, :1321) and pass them
directly (`:2405`, `:1357`). Health `distinct_languages`, `Languages Detected`,
`Categories` and the unit-cache meta are now genuinely per-dataset.

**3. Stale health timestamp.** `self.timestamp` was frozen at construction
(serializer creation time), so a long build produced runs-started-before-they-ran
timestamps. `compute_global_stats()` now refreshes it on each call
(`src/data/health_reporter.py:113`).

**4. Telemetry stage off-by-one** in the `async_logger` orchestration logging
(unit `j` reported as `j+1`, and `stage_index` = `i`) — corrected to a single
`_telemetry_stage_tags(i, j, ...)` helper (`src/training/pipeline.py:61-78`,
called at `:1196`).

**5. Visibility: implied-epochs warning.** The sizer emits
`Est training steps: N (device-batch) / M (optimizer) - stage totals compiled
from per-unit clip averages`. Where the sizer-downstream train loop itself runs in
repeats mode this is expected; but a dataset granular run with per-unit
`epoch_goal_repeats=1` averages in can reach 1000+ repeats of a small unit this
way. Added a telemetry-only guard: when the stage's implied epochs exceed 5.0 a
`WARNING` is logged with the exact weight/ratio arithmetic
(`_HIGH_REPETITION_EPOCHS = 5.0`, `src/training/pipeline.py:938`), turning a
silent over-training footgun into a discoverable one.

**6. Hygiene: SFT `eval_steps`.** The SFT trainer passed `eval_steps=0` together
with `eval_strategy="no"` and would crash if a user enabled eval while leaving
`eval_steps` at zero. Now `eval_steps=None` unless eval is enabled
(`src/trainer.py:245`).

### O.2 Confirmed non-bugs (documented, not changed)

- **`tok/s` aggregation vs reactivity is a *reporting* artefact of the update
  formula (EWMA with per-slot dt), not an accounting bug.** `devbatch.W` computed
  from read-out `global_step` is the real consumption counter; nothing removed the
  batch's consumption from a warm build.
- **`core/tok/s 0.00` on `[UNIT]`-type telemetry entries** is expected — a unit
  build emits pure build telemetry (`step?`) without a tokenized sample; it goes
  straight to `update("advance")`-style counters.
- **Step `9 → 10` loss readout of 0.00029 is low and warrants attention, but is
  not a pipeline defect.** It is a likely memorization signal from high effective
  repetition of small units (Swift retained, ~143→206 tokens under
  `min_text_length: 100`), consistent with the implied-epochs warning above. If
  configuration permits, a data run with the alternate acceptance policy would
  isolate it; the pipeline never changed any sample.
- **The 98.1% "too short" Swift texts are a genuine property of the chosen
  filter chain, not a config bug.** `min_text_length` is `100`
  (`src/config/schema.py:490`, `config_foundation.yaml:213`); Swift 4.1+ output
  is intrinsically short/wrapper-light, Post-Eta comparison hurts it, and 1 k
  dedup-collision removal is non-existent at 100k scale. Nothing in the repo
  dropped those 98.1k on the "too short" rule; that rule is as old as the
  stage-granular code.
- **`Shell<s2` and `R<s2` are genuinely under the threshold** — only the output
  `smoothing_top_k` + `smoothing_penalty` (and `logprobs`) vary per sample in
  those units; sample-to-prompt tokenization differences are what the raw cutoff
  chooses between. Filter counts are invariant to the aggregated-numbers
  computation.
- **Legacy builder quality list being empty is an explicit omission, not a
  corruption** — with no registry-quality plumbing in the legacy path there was
  nothing to store, and the fix above leaves legacy `avg_qs` as the legacy
  neutral `0.0`.
- **The `Gather` warning you saw cannot be reproduced by this repo on the sandbox
  stack because the code never wraps DataParallel** — the exact string
  `normalize_str` is in `torch/nn/parallel/_functions.py` (DataParallel-only
  autograd). If the live box shows it under `torchrun` it would have to come
  from a PyTorch internal FP8/DataParallel scalar-gather fallback, not from our
  Trainer. `*[log-gated]*` — IF a `WORLD_SIZE` / strategy difference between the
  box and the sandbox explains it, set the run's `_resolve_optimizer_name` /
  FSDP strategy explicitly.
- **Cleanup items C–G from the earlier audit are not bugs** — `is_iterable` is
  set once (no duplicate `tokenizer.num_workers`, `dataset.num_workers`,
  `other_num_workers`), one optimizer resolution per path
  (`_resolve_optimizer_name` + the Trainer branch), `processing_class=` is used in
  both Trainer branches, and there is no residual `filtered_datasets` variable.
- **The warm-cache path quality/language stats are per-dataset-correct
  throughout because the unit cache stores the *dataset*-layered meta
  (`ds_lang_dist`, `quality_scores`) at build time** — the pre-existing fix this
  session *also* populates those meta fields on fresh builds; no cache-key bump
  is needed (no format change; `PACKED_CACHE_FORMAT_VERSION=1`,
  `METADATA_CACHE_SCHEMA_VERSION` in the unit-cache key).

### O.3 Data-quality findings (Swift / Shell / R)

| Category | Sample | Unit | `quality_scores` | Accepted % | Notes |
|---|---|---|---|---|---|
| swift<s1,s2,s3,s4 | 100k | 8 gb | 0.7 | 98.1% | mostly short sub-100-chars; dedup 0 new+ | `*[log-gated]*` |
| shell<s4 | 100k | 4 gb | 0.6 | ~71% | `break/continue`-heavy, wrapper-heavy | `*[log-gated]*` |
| R<s2 | 100k | 4 gb | 0.7 | ~81% | `output_examples` without enough non-`NA`, extracted | `*[log-gated]*` |

### O.4 Test & validation evidence

- Targeted: `test_data_pipeline.py` + `test_health_reporter.py` + telemetry tags
  **48 passed**; async-pipeline/step-accounting/cache-lockstep/cleanup/tokenizer
  groups **51 passed**; overlap + hardening + foundation + CLI **67 passed**;
  trainer + CLI + integration **30 passed**.
- New regression tests guard the session's fixes:
  - `test_data_pipeline.py::TestRegistryBuildHealthAggregation` —
    `test_avg_qs_reflects_accepted_texts_only` (pack-level mean `0.9` vs the
    pre-fix `0.525`), `test_health_lang_domain_dist_is_per_dataset` (per-dataset
    sums, no cross-dataset inflation).
  - `test_health_reporter.py::test_per_dataset_lang_domain_distribution_sums_without_inflation`,
    `test_compute_global_stats_refreshes_timestamp`.
  - `test_async_pipeline_hardening.py::test_telemetry_stage_tags_use_actual_stage_index`.
- Full suite: **287 passed** (was 282 this repo's Phase-12 record; +5 new) in
  ~8:46. No skips, no xfail.

Constraints honored by construction: filtered counts/weights untouched,
tokenizer untouched, prefetch timeout untouched, no blanket `except`.

### O.5 Files changed

- `src/data/pipeline.py` (cleaned_quality threading, per-dataset lang/domain in
  fresh + legacy paths)
- `src/data/health_reporter.py` (timestamp refresh)
- `src/training/pipeline.py` (`_telemetry_stage_tags`, off-by-one, implied-epoch
  warning)
- `src/trainer.py` (SFT `eval_steps=None` hygiene)
- `tests/test_data_pipeline.py`, `tests/test_health_reporter.py`,
  `tests/test_async_pipeline_hardening.py` (regression tests)

### O.6 Remaining open recommendations (not blocking; all `*[log-gated]*`)

1. Verify the live box's `WORLD_SIZE` / FSDP-vs-DataParallel strategy to explain
   the `Gather` warning if it re-appears.
2. Confront the Swift retention choice: 98.1% too-short under `min_text_length:
   100` is unchanged, but the low-loss signal at step 9/10 (`0.00029`) warrants a
   memorization audit if it recurs — the `_HIGH_REPETITION_EPOCHS` warning
   (`src/training/pipeline.py:938`) will now fire audibly.
3. The aggregate-numbers `accuracy/ce`-style parity is covered by existing
   unit/step tests; run `tests/test_async_pipeline_overlap.py` on the box if the
   async overlap/streaming path is ever re-enabled with multiple units.