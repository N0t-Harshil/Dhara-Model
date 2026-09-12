# Pipeline Architecture — Data Streaming, Caching & Staged Training

Design doc for the dataset streaming and training pipeline. Covers the overall
flow plus the caching/resume/overlap subsystems. Grounded in the implementation
(`src/data/*`, `src/training/pipeline.py`, `src/models/factory.py`).

---

## 1. Overall pipeline

```
main.py
 └─ TrainingPipeline (src/training/pipeline.py)
     initialize()          # load tokenizer, build/resume model (ModelFactory)
     run()                 # alignment? then pretraining
        pretrain_pipeline() or staged_pretrain()
        └─ DataPipeline (src/data/pipeline.py)
            build_pretrain_dataset_from_registry(...)
              for each registry entry (DatasetInfo, src/data/registry.py):
                1. metadata cache verify   (resolve_and_cache if missing)
                2. shard-parallel streaming (ShardCoordinator, src/data/streaming.py)
                3. filter chain per chunk  (boilerplate → length → language
                                            → quality → AST → dedup → function sampling)
                4. tokenize + random-window + pack (pack_sequences)
                5. stage/unit disk cache   (packed.pt + *_meta.json)
            → WeightedMixedDataset (weighted, language/domain rebalancing)
        └─ HF Trainer per unit/stage with StageBoundaryCallback
            → checkpoints per unit / stage, final model save
```

Per-dataset lifecycle, as logged:

```
Loading dataset: <path>/<name> (cat=…, weight=…, qs=…)
  [metadata cache hit]  Dataset cache found / Repository unchanged |
                        metadata reused (N shards)
  [or]  Metadata resolved+cached (local): N files
  [or]  Arrow reused — direct iterable, no HF resolution
  Streaming begins — N shards, W parallel workers [, resume at shard S offset O]
  Accepted: <n> (<p>% of <loaded>)
  [DIAG] stage timings / BOTTLENECK DETECTED / LOW ACCEPTANCE investigation
  Packed: <n> sequences (eff=…)
```

Key identity rule: the shard-parallel path must yield **exactly** the gated
samples the sequential path would have yielded, in the same order
(`{**sample, "text": text}` + `_shard`/`_raw_seq` tags), so downstream
filtering and packing are bit-identical. Verified by
`test_shard_coordinator_identity_and_resume` (parallel == sequential reference,
resume = gap-free continuation).

---

## 2. Metadata cache

`src/data/metadata_cache.py` — a record of **what files a dataset consists of**
and **how to open them**, so streaming never re-resolves HF metadata.

- Record fields: `schema_version`, `repo`, `name`, `split`, `revision`,
  `files[]`, `num_shards`, `loader`, `preprocess_sig`, `token_sig`
  (`build_record`, saved as `files.json` + `split.json`).
- `verify(info, preprocess_sig, token_sig)` recomputes the fingerprint
  (schema-version + both signatures + file list + revision); a mismatch or
  missing file → miss.
- `resolve_and_cache` (`src/data/streaming.py`) resolves a dataset's file list
  **without downloading data** (`load_dataset_builder(...).as_streaming_dataset`,
  local directories are scanned and a loader detected) and stores the record.
  When a `BuilderCache` is supplied it delegates to `detect_driver` (§13) so
  script-family datasets get a builder record instead of a file list. Used by
  `warm_metadata_cache` (background warming) and by the cold path.
- The cold streaming path writes the record **mid-stream**, so the next run
  takes the shard-parallel fast path.
- Invalidation: see §12. Fingerprint mismatch → re-resolve; `invalidate()`
  removes a record explicitly.

---

## 3. Accepted-target streaming

Motivation: materializing a full dataset to find that only a fraction survives
filtering is wasted work (The Stack V2: ~20k streamed → 79 accepted).

- Per-dataset policy: `dataset_policies` path-glob list
  (`DatasetPolicyConfig.accepted_target`, `text_fields`, `shard_workers`,
  `shard_order`); `_dataset_policy()` matches the first glob.
- `build_pretrain_dataset_from_registry` pulls the streamer in **chunks**
  (`islice(streamer, max(1024, min(ds_limit, 8192)))`) and runs the identical
  filter chain per chunk. Quality scoring and AST validation are the same pure
  functions in the same order (quality → AST → dedup), and dedup state carries
  across chunks, so the accepted set is identical to the pre-optimization
  single-pass pipeline — only the *amount pulled* changes.
- The loop stops pulling as soon as `accepted >= accepted_target`
  (default `0` = no early stop), then persists shard progress (§4) and emits
  the chunk as packed sequences.
- `ledger`/`global_stats` keep total raw/accepted/rejected counts for the
  diagnostics report (§7) and health report (§11).

---

## 4. Resume

Interrupted builds resume at the exact row, with no duplicates and no gaps.

- `ShardProgressStore` (`src/data/shards.py`) persists one record per
  dataset+fingerprint under `data.shard_progress_dir` (`cache/shards`).
  Record: `schema_version` (2), `resume {shard, offset}`, `last_shard`,
  `last_offset`, per-shard `stats` (`streamed/accepted/raw/…` + `complete`
  flags), `done`. Key is a **hashed digest** of `repo|name|split|fingerprint`
  (`shard_progress_key`) — short enough for Windows path limits.
- `ShardCoordinator` streams the plan `[(shard, offset), …]` in order; every
  emitted raw row carries `_raw_seq`, and the consumer records the max
  consumed offset per shard (`_consumed_raw`). `progress_state()` returns
  `(next_shard, raw_offset)` — the exact position, counting only rows already
  handed downstream.
- On close, `_persist_shard_progress` (`src/data/pipeline.py`) marks finished
  shards `complete` (yield-order safe), writes `resume`, per-shard stats, and
  `last_shard`; saves via temp file + `os.replace`. When **all** shards are
  complete the record is deleted — the packed cache (§9) covers full reuse.
- `build_shard_plan(n_shards, rec, order)`:
  - completed shards excluded (`complete` flags + legacy `last_shard` range),
  - `resume` entry becomes the first plan item with its raw offset,
  - `order="yield"` prioritizes remaining shards by measured
    `accepted/streamed` (untouched shards keep natural order) — `sequential`
    preserves dataset stream order exactly.
- `ShardProgressStore.load` returns a fresh record when missing; a
  `schema_version` mismatch also starts fresh.
- **Script-family datasets** (§13) resume at the driver level: the
  `ScriptDatasetDriver` persists its raw-row `resume_offset` in the builder
  record (`save_resume`), so an interrupted script stream resumes the iterator
  at the exact raw row on the next run; natural exhaustion resets it
  (`reset_resume`), mirroring the shard delete-on-complete behavior.

---

## 5. Shard cache

The metadata record (§2) *is* the shard cache: each file in `files[]` is one
shard, opened independently by a worker:

- `ShardCoordinator` starts `min(len(plan), shard_workers)` threads (default
  8, per-policy override), each running `load_dataset(loader,
  data_files=[shard_url], streaming=True, …)` and streaming its rows through
  the extraction gate (detect text fields once, extract, drop no-text), then
  into a bounded queue (512) — the consumer drains queues in plan order.
- The shard file handle (Arrow/parquet reader) lives in the worker generator;
  workers poll `stop` while enqueueing with a timeout, so closing the
  coordinator always releases readers (no leaked handles/threads on Windows).
- Timings tracked per run: `arrow_open_sec` / `arrow_open_max_sec`
  (per-shard open retried once on failure), `network_wait_sec` (queue-empty
  polling), `extraction_sec`. Open/shards reuse is logged as
  "metadata reused (N shards)" / "Arrow reused — direct iterable, no HF
  resolution".
- The `limit` (ds_limit) is a **gated-sample cap** applied in the coordinator,
  identical to `stream_dataset` semantics (limit-th gated sample included).

---

## 6. Tokenizer cache

Two mechanisms:

- **On-disk tokenizer manifest** (`src/models/factory.py`): the tokenizer
  directory carries `tokenizer_version.json` (`version: 1`) containing a
  sha256 per file. `ModelFactory.load_tokenizer` verifies the hashes; a match
  ⇒ load **offline** (no network); a mismatch ⇒ rewrites the manifest and
  loads online. Missing/any-changed file invalidates the offline fast path.
- **`tokenizer_signature`** (pipeline property): a fingerprint of the
  tokenizer identity (name/vocab size) folded into every cache key —
  metadata records (§2), shard progress (§4), registry/stage/unit dataset
  caches (§9). Changing the tokenizer automatically invalidates all derived
  caches and forces retokenization.

---

## 7. Preprocessing workers

Parallelism without changing results:

- **Shard workers** (I/O + extraction): one thread per in-flight shard,
  bounded by `shard_workers`; the consumer pulls in plan order, so upstream
  shards can run ahead by ≤ 512 rows per queue.
- **Filter-chain pool** (CPU-bound, pure functions): per-sample quality
  scoring and AST validation via `pool.starmap(..., chunksize=1024)` and
  simhash fingerprints via `pool.map(..., chunksize=2048)`; results align
  one-to-one with tasks, so chunk boundaries don't affect the outcome.
  The exact-dedup set and simhash state are applied sequentially in stream
  order ("sequential-equivalent") across chunks.
- **`[DIAG]` report**: stage timings per dataset (metadata resolve, shard
  selection, arrow open, network wait, extraction, stream, per cleanup stage,
  tokenize+pack), `BOTTLENECK DETECTED` when any stage ≥
  `bottleneck_threshold_sec` (30 s), and a `LOW ACCEPTANCE` investigation
  (per-rejection breakdown + highest-raw shards) when
  `accepted/raw < acceptance_investigation_threshold`.
- **Background threads** (see §10): unit-dataset prefetch and next-stage
  metadata warming.

---

## 8. Staged training

`src/training/pipeline.py` — pretraining runs as a sequence of stages, each a
contiguous step range of one Trainer (single optimizer/scheduler, `max_steps` =
total across stages):

- **Unit staging** (`pretrain.staging`): every registry dataset is a "training
  unit"; a stage's step budget is allocated across its units by measured
  packed-sample counts when caches are warm (`sizing_basis: packed`,
  deterministic), else by registry weights (`weight`). `StageBoundaryCallback`
  fires on the boundary step, requests a full checkpoint save, and stops
  training; the next unit **resumes from that checkpoint** with `global_step`
  unchanged (optimizer/scheduler/RNG restored).
- **Curriculum stages** (`pretrain.curriculum`): each stage filters the
  registry by category and builds its dataset via
  `build_pretrain_stage_dataset`; the same boundary/resume mechanism applies.
- Per-stage summary logs: units, cache hit rate, GPU busy %
  (`train_sec / wall`), tok/s, tokens, wall time.

---

## 9. Dataset checkpoints

Three layered disk caches, all behind `data.use_packed_cache` and all keyed so
a hit skips streaming, tokenization, and HF resolution:

| Cache | Location | Key |
|---|---|---|
| Registry dataset | `<data.cache_dir>/registry/<safe_name>_<key>.pt` | `_registry_cache_key`: path, name, weight, quality_score, ds_limit, accepted_target, max_seq_length, vocab_size, tokenizer, dedup method/threshold, AST on/off, function sampling, boilerplate/min-length, per-category quality threshold, `v1` |
| Stage | `<stage_cache_dir>/stage<N>/packed.pt` + `stage_meta.json` | `_stage_cache_key`: stage categories/limits + the registry-key inputs |
| Unit | `<stage_cache_dir>/stage<N>/u<NNN>/packed.pt` + `unit_meta.json` | `unit_cache_key`: `unit-v1`, path/name/split, ds_limit, processing signature, tokenizer signature |

- Loads are guarded (`cache_key`, `version`, non-empty); mismatches/corrupt
  files → warning + rebuild. Saves are best-effort (warning on failure).
- `unit_cache_packed_count` reads the packed count for deterministic step
  allocation (§8). Stage cache saves `global_stats`, language/domain
  distributions, rejection reasons for reporting.
- **Dataset-granular completion** (`unit_completions.json`, written atomically
  in `<model_dir>`): a unit counts as *complete* only after train → checkpoint
  flush → `unit_identity.json` written into the latest checkpoint (stage/unit
  index, `unit_key`, `unit_fingerprint`, `processed_cache_key`, global_step /
  total_steps) → completion record committed. On restart the plan skips a unit
  only when the manifest record's fingerprint **matches the current dataset
  identity** (sha256 over path/name/data_dir/revision/category); a changed
  identity — e.g. a revision bump — forces retrain even if the index was
  previously reached. A crash before the commit leaves the unit incomplete and
  it resumes/retrains.

---

## 10. CPU/GPU overlap

- **Unit prefetch** (`UnitPrefetch`, `src/training/asyncprefetch.py`): while the
  GPU trains unit *k*, up to `prefetch_depth` (default 3) later datasets are
  built on bounded daemon workers (streaming → filtering → tokenizing →
  packing). Delivery to the trainer is strictly in unit order via a condition-
  variable buffered queue; a slow unit never overtakes a fast one and memory
  is bounded by the worker count. `get()` only blocks if the next build is
  still running — never worse than sequential, usually fully hidden. Every
  `get()` is bounded by `prefetch_timeout` (default 900s); a build crash is
  re-raised on the consumer (marked failed, run continues) and a build that
  never returns raises `PrefetchTimeout` (unit skipped, worker slot released).
  `close()` is idempotent and never blocks the caller.
- **Duplicate-build prevention**: `UnitPrefetch.start()` derives an identity
  per unit (`path`/`name`, or the raw value for synthetic units); a unit
  scheduled more than once is built exactly once — later occurrences skip the
  build and `get()` returns the shared payload of the first occurrence
  (`[ASYNC] unit N duplicate of unit M — duplicate build prevented`). The count
  is tracked in `stats["duplicates_prevented"]` and reported as
  `duplicate builds prevented` in the ASYNC PIPELINE REPORT.
- **Global async switch & emergency fallback**: prefetch (and stage metadata
  warming) run only when *both* the staging block opts in (`staging.prefetch`)
  and the global `data.async_pipeline.enabled` flag is `true`
  (`_prefetch_enabled`, §10c). Setting `enabled: false` (or it being absent)
  routes the pipeline to the original synchronous inline build — an emergency
  fallback with identical dataset semantics, just no overlap.
- **Enforced worker/buffer bounds**: all `data.async_pipeline` knobs are wired
  into the engine, not just schema fields — `ready_queue_size` and
  `max_inflight` cap the effective prefetch depth, `preprocess_workers` caps
  producer threads (`_effective_prefetch_depth`, §10c), `metadata_workers`
  fans the stage metadata warming out over up to that many threads, and
  `retry_count` bounds background build retries (see below).
- **Build retries**: a failing background build is retried up to
  `data.async_pipeline.retry_count` times
  (`[ASYNC] dataset N build failed (attempt a/b) — retrying`) before its
  exception is delivered to the consumer; only then is the unit failed and
  journaled. Transient network/resolution errors are absorbed without touching
  the failure journal. Retried attempts are counted in the report.
- **Stage metadata warming**: after the last unit of stage *i*, a daemon thread
  resolves+warm-caches the metadata for stage *i+1*'s datasets
  (`[META] stage N metadata warming in background`).
- **Cleanup pool** (`DataPipeline._get_cleanup_pool`): one `multiprocessing`
  pool (fork/spawn, `min(32, cores)` or `data.cleanup_pool_size`) is created
  once and reused across all datasets/stages for the pure quality/AST/simhash
  stages; `DataPipeline.close()` (called from `TrainingPipeline.cleanup()`)
  joins then terminates it so no worker process leaks on exit.
- **Within a dataset**: bounded queues let shard workers stream ahead while
  the consumer filters; `network_wait_sec` vs `stream_sec` vs
  `extraction_sec` are measured separately so the DIAG report can tell a
  network-bound build from a CPU-bound one.
- GPU utilization is reported as `GPU busy %` per stage; `[TIMER]` lines give
  dataset-build cost per unit (cache hit vs miss).

---

## 10a. Asynchronous checkpoint writer

`src/training/checkpoint.py` — `AsyncCheckpointWriter` moves the expensive
parts of checkpointing (state-dict serialization, fsync, atomic rename,
checksum) off the training thread:

- The GPU-facing caller captures `model.state_dict()` on the main thread and
  enqueues a small job (bounded queue, backpressure with timeout).
- Every file is written to a temp name, fsynced, then renamed into place —
  a crash never leaves a half-written checkpoint.
- A sidecar sha256 manifest (`.sha256.json`) is written per checkpoint;
  `verify_checkpoint()` re-hashes on every resume and rejects corruption
  (legacy checkpoints without a manifest are accepted as-is).
- Durability boundary: `_flush_checkpoints()` blocks until queued jobs are on
  disk and is called before every resume lookup / stage switch and at the
  final pretrain save.
- **Submit/flush race (fixed)**: `submit()` increments the pending-job count
  under the flush condition variable *before* enqueueing, so a fast job that
  completes between submit and the count update can never leave `flush()`
  waiting on a live writer. The queue-full timeout path decrements the count
  again and writes synchronously. Regression: 200-cycle submit+flush stress
  test.
- If the writer thread dies, `submit()` falls back to a synchronous write —
  training never deadlocks and no checkpoint is silently lost.
- Wired through `TrainingPipeline._save_checkpoint` and gated by
  `training.async_checkpoint` / `training.checkpoint_checksum`.

---

## 10c. Async pipeline configuration

Gated by `data.async_pipeline` (all values configurable; defaults below):

| key                 | default | meaning                                              |
|---------------------|---------|------------------------------------------------------|
| `enabled`           | `true`  | master switch; `false` = synchronous fallback (§10)  |
| `prefetch_depth`    | 2       | staging depth capped by `ready_queue_size`+`max_inflight` (ge 1) |
| `ready_queue_size`  | 2       | cap on buffered ready results — bounds effective depth (ge 1) |
| `preprocess_workers`| 4       | cap on background build worker threads (ge 1)        |
| `metadata_workers`  | 2       | fan-out of background metadata-warming threads (ge 1)|
| `max_inflight`      | 4       | cap on outstanding background builds — bounds effective depth (ge 1) |
| `retry_count`       | 3       | build retries before a failure reaches the journal (ge 0) |

All values are enforced at runtime (§10); the two caps that bound depth are
applied via `_effective_prefetch_depth` (min of staging depth, ready_queue,
max_inflight).

Staging-level knobs (`training.pretrain.staging`): `prefetch` (bool), 
`prefetch_depth`, `prefetch_timeout` (ge 5s), `skip_failed_units_on_resume`,
`abort_on_unit_error`. Prefetch needs both the staging opt-in and the global
switch.

---

## 10b. Runtime telemetry

`src/infrastructure/telemetry.py` — `PipelineTelemetry`: a lock-guarded
counter accumulator plus an optional monitor thread sampling GPU util/memory
(nvidia-smi), CPU and RAM (psutil). The pipeline records cache hits/attempts,
unit failures, tokens/samples/train-seconds per unit and emits a
`[TELEMETRY]` status banner per unit (up-time, samp/s, tok/s, GPU%, cache hit
%, net wait). Nothing in telemetry can raise into training (all sampling is
best-effort). Gated by `training.telemetry.{enabled,interval_sec,gpu_util_sampling}`.

---

## 11. Failure recovery

- **Shard level**: a shard's open is retried once; a streaming failure marks
  the shard failed (`failed_shards()`), other shards continue, and the shard is
  not marked `complete` — the next run resumes it (§4). Blocked producers
  release file handles on close.
- **Dataset level**: if a record's stream yields 0 samples or raises, the
  fallback chain (`DatasetInfo.fallbacks`) is walked; only when all fallbacks
  fail is the entry skipped ("All fallbacks exhausted").
- **Cold path**: without a record, the original resolution-based streaming
  still works and caches the record mid-stream (§2).
- **Persistence**: every cache/progress save is wrapped; a failure logs a
  warning and never aborts the run. Progress reads tolerate corruption.
- **Training**: a crash mid-stage resumes from the latest `checkpoint-*`
  (global_step unchanged); `StageBoundaryCallback` saves a full checkpoint
  (optimizer/scheduler/RNG) before each boundary; an empty unit/stage dataset
  aborts staged pretraining gracefully; `_NaNSafeCallback` raises on
  non-finite parameters. Final model + tokenizer + checkpoint are saved at the
  end.
- **Unit level (staged pretrain)**: every background task is isolated — a
  unit build failure (worker crash, network error, timeout, corrupted cache)
  is recorded in a per-run failure journal (`failed_units_stage<N>.json`,
  keyed by dataset path/name) and the run continues by default; on the next
  launch the same fingerprint causes those units to be skipped
  deterministically (`skip_failed_units_on_resume`). `abort_on_unit_error`
  opts into fail-fast instead. Empty datasets are marked and skipped the same
  way.
- **Observability**: `HealthReport` aggregates per-dataset stats and errors
  (raw/accepted/packed counts, rejection reasons, token lengths, quality
  scores) and `sanity` checks run when enabled.

---

## 12. Cache invalidation rules

- **Processing signature** (`processing_signature()`): versioned hash of the
  preprocessing config — boilerplate flag + keywords, min/max text length,
  dedup method + threshold, AST code-filtering flag, function-sampling flag,
  per-category quality thresholds, sampler balance. Any change invalidates
  every downstream cache (metadata, shard progress, registry/stage/unit keys).
- **Tokenizer signature**: tokenizer name/vocab hash (see §6); change forces
  re-resolution + retokenization (all records carry it).
- **Metadata cache**: fingerprint = f(fingerprint-version, preprocess_sig,
  token_sig, files, revision); `verify()` mismatch or missing → miss;
  `schema_version` bump invalidates all records; `invalidate()` removes
  explicitly.
- **Shard progress**: key embeds the dataset fingerprint; `schema_version` 2
  mismatch → fresh record; `complete`/`resume` state drives the plan (§4);
  record deleted on full completion.
- **Dataset caches (registry/stage/unit)**: key/version embedded in
  `packed.pt` (`version: 1`); key mismatch, version mismatch, empty or
  unreadable payload → rebuild.
- **Tokenizer manifest**: per-file sha256; any change or missing manifest ⇒
  offline fast path disabled, manifest rewritten on next load.
- **Toggles**: `use_packed_cache=false` disables all packed caches (metadata
  cache stays active so streaming still runs shard-parallel).

---

## 13. Dataset drivers (File / Script / Local / Streaming)

The loading layer is unified behind a driver abstraction
(`src/data/drivers.py`). Each dataset entry resolves to exactly one driver
family; the driver owns fingerprinting, warm-start reuse, iterator resume and
streaming for its family. This makes warm startup require **zero** HF
repository resolution and extends warm-start reuse to script-module datasets
(The Stack V2, OpenCoder, CodeParrot, ...) that cannot be warm-started from a
file list.

### Driver interface

A driver provides `resolve()`, `fingerprint()`, `stream()`, `resume()`,
`cache()`, `invalidate()`, `warmup()` and `prefetch()`. Drivers are
identity-preserving: they yield the same gated samples, in the same order, as
the original sequential path, each tagged `_shard` and `_raw_seq`.

### Families and detection

`detect_driver()` classifies an entry in this order (each step is skipped
unless a cached record makes it possible — a fully warm run makes **zero**
network or builder-resolution calls):

1. **Local** — entry path is an existing local directory with recognized
   files → `LocalDatasetDriver` (never touches the Hub).
2. **File** — metadata record verifies → `FileDatasetDriver` (Arrow direct
   iterable, shard-parallel, §2/§3).
3. **Script** — builder record verifies → `ScriptDatasetDriver` (see below).
4. **Live resolution** — `load_dataset_builder()` once; if the builder carries
   a resolved `config.data_files` (auto-converted parquet/arrow repos, e.g.
   the `Parquet<repo>` builders for The Stack V2 / OpenCoder / CodeSearchNet)
   or exposes shard sources via its streaming iterable → **File** family and
   the file-list metadata record is cached; if it exposes **no** sources (pure
   IterableDataset generators) → **Script** family and a builder record is
   cached.
5. **Streaming fallback** — resolution failed (gated / not-found / offline) →
   `StreamingDatasetDriver` mirrors the original cold path; the failure is
   short-circuited (logged once with access guidance, e.g. gated datasets need
   `hf_token`) and the cold path is **not** retried, so a missing token costs
   one failed resolution per entry, not two.

### Builder cache (`<metadata_cache_dir>/builders`)

Stores **identity only** — never file lists. A record holds repo/name/split,
revision, script revision, builder class, preprocess + tokenizer
fingerprints. The fingerprint excludes resume state; any identity or
processing change invalidates (`invalidate()`) so stale data is never reused.
Writes are atomic (temp file + `os.replace`).

### Script driver semantics

- Stream = `load_dataset_builder()` → `as_streaming_dataset(split)` → skip
  `resume_offset` raw rows (`itertools.islice`). The builder re-instantiates
  from the datasets module cache (one module call per stream, no network).
- On natural exhaustion (`raw_consumed() == length`) the iterator state is
  reset (mirrors shard delete-on-complete); on interruption the raw-row offset
  is persisted (`save_resume`) and the next run resumes the iterator
  (`Iterator resumed at raw row N`).

### Diagnostics

Every entry logs `[DIAG] <family> driver: (metadata_hit|builder_hit,
repo_resolution_skipped, first_resolution, script_resumed)` so warm-run
behavior is observable without repeated "Resolving data files..." lines. The
next two entries' drivers are warmed in the background while the current one
streams.

### Legacy Hub URIs

`rewrite_hf_url()` also rewrites legacy single-component IDs
(`hf://datasets/code_search_net@abc123/...` →
`https://huggingface.co/datasets/code_search_net/resolve/abc123/...`).

---

## 14. Dhara v3 decoder, output head & LM loss

`src/dhara/layer11_decoder.py` / `src/dhara/model.py`. This section states
honestly what the output layer computes — it is *not* a sparse
vocabulary head that avoids the dense projection.

### 14.1 What the decoder actually computes

`HierarchicalSparseDecoder` stacks a semantic (concept) decoder, a language
decoder, and a `TokenDecoder`. `TokenDecoder` owns the vocab head:

- `hidden_to_vocab = nn.Linear(d_hidden, vocab_size)` — the forward pass
  computes the **full dense projection** `logits = hidden_to_vocab(h) /
  temperature` over **all** `V` vocab entries. This is an `O(V)` operation
  and it happens on every forward pass.
- `torch.topk(logits, k)` then **selects** the adaptive top-k candidate set
  (`top_indices`/`top_logits`, `k` between `adaptive_top_k_min` and
  `adaptive_top_k_max`, default max 2048). Top-k is a *selection on top of the
  full projection* — it does **not** replace it.
- Difficulty/sparsity gates only scale `k` (and shape auxiliary signals);
  they do not gate the projection itself.

### 14.2 Single decoder evaluation per micro-batch (training speed fix)

`DharaModel.forward` previously evaluated the decoder stack + dense vocab
projection **twice** per micro-batch — once for `_vocab_logits`, and once more
inside `decoder.hierarchical_log_prob(...)` for the LM loss. It now evaluates
the decoder **once** per micro-batch; the resulting `_vocab_logits` back both
the LM loss and the aux/entity outputs, and the causal shift (position `t`
predicts label `t+1`) is applied via a target mask instead of re-decoding a
shifted slice. Pure efficiency fix; the dense-mode loss math is identical and
is pinned by a regression test comparing against the old path.

### 14.3 `head_ce` — opt-in top-k loss target set

`model.architecture.dhara_v3.head_ce` (`src/config/schema.py`, also read by
`DharaModel`): `"dense"` (default) or `"topk"`.

- **`dense` (default):** LM cross-entropy over the full-vocab log-softmax of
  `_vocab_logits`.
- **`topk`:** LM cross-entropy over the decoder's adaptive top-k candidate set
  (its existing `top_indices`, bounded by `adaptive_top_k_max`) **union** the
  target token, via `logsumexp` over that small set — an approximation that
  shrinks the **loss/backward target set**.

Honest scope of `topk`: the decoder still computes the full dense vocab
projection in the forward pass for candidate scoring (see §14.1), so
`head_ce: topk` reduces the loss/backward target set but does **not** remove
the dense projection and does **not** change generation. Opt-in; enabled in
`config_foundation.yaml`.
