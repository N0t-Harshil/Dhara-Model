# Training Startup Hang — Root Cause & Fix Report

**Date:** 2026-08-01
**Observed:** `python main.py --config config_foundation.yaml --gpu 1 full-training` printed
`Loading dataset: OpenCoder-LLM/opc-fineweb-code-corpus/default` → `Resolving data files: 100%`,
then consumed 30+ min at ~70% CPU with no further progress (no deadlock — active work).

## 1. Exact function consuming 30+ min

```
src/training/pipeline.py:246  build_pretrain_dataset_from_registry()        # full_training_sequence, pretrain branch
  → src/data/pipeline.py:893  list(stream_dataset_with_fallbacks(info, registry, limit=...))
      → src/data/streaming.py:118  load_dataset(streaming=True)             # logs "Resolving data files: 100%"
      → for sample in ds: ...                                               # THE BURN: iterates the ENTIRE hub dataset
```

`list(...)` materialized **every sample of `OpenCoder-LLM/opc-fineweb-code-corpus`**
(multi-TB, ~150M samples) before training could start. The 70% CPU was gzip
decompression + JSON parsing + text extraction of the whole corpus.

## 2. Why it happened (root cause)

`limit` was resolved as `max_samples_per_dataset` only, and every caller passed `None`:

- `build_pretrain_dataset_from_registry(max_samples_per_dataset=None)` is called with no
  argument from `training/pipeline.py:246` and `:233` (curriculum).
- `DatasetInfo.max_samples` defaults to `None` (`registry.py:70`) and `opc-fineweb-code-corpus`
  is registered without it (`registry.py:211-217`).
- Result: `limit=None` → unbounded iteration of a ~150M-sample corpus in a `list(...)`.

The unbounded pattern existed in three places:
- `src/data/pipeline.py:893` — registry pretrain path (**the hang site**)
- `src/data/pipeline.py:674` — non-registry pretrain path (same `or`-chain, no hard fallback)
- `src/training/pipeline.py:273` and `:485` — SFT / stage-cache paths (`list(stream_single_dataset(...))`)

Secondary defect: `WeightedMixedDataset.__init__` eagerly called `len()` on every wrapped
dataset and precomputed the full assignment table at construction (in-memory `len()` is
O(1), so this was not the hang — but it violates the lazy-init contract and burns time
before the first batch).

## 3. Code diff

### `src/data/pipeline.py`

1. **New module constant** (line 93) — hard per-dataset cap, matches the existing
   convention `TokenizerConfig.max_samples = 100000` (`schema.py:530`):

   ```python
   DEFAULT_MAX_SAMPLES_PER_DATASET = 100_000
   ```

2. **Registry path cap + `[TIMER]` instrumentation** (lines 894-899):

   ```python
   ds_limit = max_samples_per_dataset or info.max_samples or DEFAULT_MAX_SAMPLES_PER_DATASET
   t0 = time.perf_counter()
   samples = list(stream_dataset_with_fallbacks(info, registry, limit=ds_limit))
   logger.info("[TIMER] stream+materialize %s (limit=%s): %.1fs, %d samples",
               ds_key, ds_limit, time.perf_counter() - t0, len(samples))
   ```

3. **Non-registry path cap** (line 674): `limit = max_samples_per_dataset or ds_info.max_samples`
   → `limit = max_samples_per_dataset or ds_info.max_samples or DEFAULT_MAX_SAMPLES_PER_DATASET`

4. **`WeightedMixedDataset` made lazy** (lines 292-346): `__init__` now stores references
   only (datasets, weights, targets, seed). All work — `len()` sizing, RNG creation, the
   assignment-table loops — moved to `_ensure_ready()`, invoked on first
   `__len__` / `__getitem__` / `stats()` (results are bit-identical: same seed, same math).
   `_ensure_ready` logs its own `[TIMER]`.

5. **Fixed latent crash in the balancing path** (line 392): `_build_from_stratification`
   passed an index *list* where `_sample_adaptive` expected the *group map* →
   `AttributeError: 'list' object has no attribute 'get'` whenever
   `language_balancing`/`domain_balancing` was enabled. Now passes the map + group name,
   and honors the per-group `count` (previously each group received `total_samples`
   regardless of target fraction). This path is disabled in all shipped configs; it was
   dead broken code discovered by the laziness unit test.

6. **Removed duplicate re-tokenization** (line 1026): the health report re-tokenized the
   first 1000 cleaned texts after they were already tokenized;
   now reuses `tokenized[:1000]` ids (identical lengths, no extra tokenizer calls).

7. **Additional `[TIMER]` logs** (STEP-1 instrumentation): cleanup/filter (line 968),
   tokenize+pack (line 1003), mixed-dataset construct (line 1112), `import time` added.

### `src/training/pipeline.py`

8. Import constant (line 28); SFT cap (line 274): `limit=ds_info.max_samples or
   DEFAULT_MAX_SAMPLES_PER_DATASET`; stage-cache cap (line 486): same with `ds_entry`.
9. `[TIMER]` around the registry build (lines 247-249).
10. First-batch timer in `_train_stage` (lines 337-354): before `trainer.train()`, force
    `get_train_dataloader()` + `next(iter(dl))` and log the time — this is the exact
    "first sample" measurement and the point where the lazy mixed dataset first builds.

## 4. Before timing (server, observed)

| Stage | Time |
|---|---|
| To "Resolving data files: 100%" | ~1 min |
| First dataset full iteration (`list(...)` of opc-fineweb-code-corpus, ~150M samples) | **30+ min, never completed** |
| Total before first training step | unbounded |

## 5. After timing (offline verification, 2026-08-01)

Monkeypatched-stream smoke test of `build_pretrain_dataset_from_registry` with the full
54-entry registry (`config_foundation.yaml`):

| Stage | Time |
|---|---|
| Entire registry build, 54 datasets (2 samples each, mocked stream) | **0.8 s** |
| `WeightedMixedDataset` construction | **~0 ms** (no dataset access in `__init__`) |
| First `__getitem__` (lazy `_ensure_ready`) | < 5 ms |
| Mixed dataset contents | 100 samples, 204,800 tokens, sanity checks OK |

Verification asserts: **every** registry entry receives `limit != None`; opc resolves to
exactly `100_000`; all 34/34 `test_local_validation.py` checks pass; integration Phase 1
6/6 OK (Phases 2-3 require HF network, skipped locally by design).

## 6. New expected startup latency (server, real data)

Per dataset now bounded at 100,000 raw samples max:
- First dataset (opc, cap 100K): stream + clean + tokenize + pack ≈ minutes (was: never).
- Per-dataset `[TIMER]` lines print stream/clean/tokenize/pack times and sample counts,
  so startup shows continuous progress instead of silent 30-min stalls.
- Total startup = sum over 54 bounded datasets (each capped); tokenization dominates.
  For a faster smoke run, lower the cap — e.g. set the fallback constant to 10_000,
  or add `max_samples` to any `DatasetInfo` registration (it takes precedence).

## Remaining notes

- The old behavior also affected SFT (`training/pipeline.py:273`) and stage-cache
  (`:485`) paths — both now capped.
- `test_integration.py` Phase 3 still hangs locally when HF Hub is unreachable
  (unguarded tokenizer download, pre-existing; unrelated to this fix).
- HF token remains in `config_foundation.yaml:127` (untracked file) — recommend
  switching to the `HF_TOKEN` env var before committing anything.
