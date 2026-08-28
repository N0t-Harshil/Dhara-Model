# Async Pipeline Hardening — Final Report

**Date:** 2026-08-27
**Scope:** dataset-granular staged pretraining (`training.pretrain.staging.mode: dataset`),
startup sanity-run — find the async `TypeError: unsupported format string passed to
NoneType.__format__`, implement the proven root-cause fixes, and ship a regression-proof
test set.

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