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
- Emits explicit log markers:
  - `[ASYNC] dataset N training start`
  - `[ASYNC] dataset N prefetch start`
  - `[ASYNC] dataset N preprocessing complete`
  - `[ASYNC] unit N cache hit`
  - `[ASYNC] dataset N training end`
  - `[ASYNC] dataset N consumed (GPU wait: X.XXs)`
  - `[ASYNC] GPU wait before dataset N = X.XX sec`

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
- On restart, the scheduler inspects `global_step` and dataset checkpoints, skipping already completed units and resuming seamlessly.

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
cache hits: 18 | cache misses: 4
prefetch hit rate: 91.3%
GPU training time: 15120.00 sec
GPU wait time: 134.00 sec
pipeline idle: 0.9%
============================================================
```
