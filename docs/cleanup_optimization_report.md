# Cleanup/Filter Optimization Report

**Goal:** reduce `cleanup/filter` stage from **1043s** (server, 100K samples/dataset) to **<120s** without changing filtering behavior.

**Status:** implemented + verified locally (10.1x speedup). Final validation pending server rerun.

---

## Phase 1 — Per-stage profiling

### Before (server, 100K opc samples, old order quality → AST → dedup)
| Stage | Time |
|---|---|
| stream + materialize (after cap fix) | 22s |
| **cleanup/filter total** | **1043s** |
| — rejected: duplicates | 28,023 |
| — rejected: AST | 27,193 |
| — rejected: quality | 52 |
| — rejected: too short | 184 |
| — accepted | 44,548 |

### Before (local synthetic 15.5K corpus, instrumented)
| Stage | Time | % |
|---|---|---|
| quality scoring | 82.94s | 86.0% |
| AST validation | 10.68s | 11.1% |
| boilerplate removal | 2.37s | 2.5% |
| everything else (extract, length, language, dedup) | <0.3s | <1% |
| **total** | **96.47s** | |

Extrapolated to 100K: quality ≈ 415s, AST ≈ 53s, boilerplate ≈ 12s.

### Component micro-profiling (1.3KB code sample)
| Function | ms/call |
|---|---|
| `document_quality_score` total | 5.58 |
| `detect_language` (70 regexes) | 1.908 |
| `_language_confidence` (re-runs detection) | 1.943 |
| `_code_quality` | 1.212 |
| `_toxicity_score` | 0.721 |
| `_formatting_quality` | 0.304 |
| `_estimate_perplexity` | 0.088 |
| pipeline `detect_language` (substring check) | 0.001 |

---

## Phase 2 — Algorithm audit (complexity per sample)

| Stage | Before | After |
|---|---|---|
| extract / empty / length | O(1) | O(1) — no change |
| boilerplate removal | O(n) regex | O(n) — no change |
| language detect (pipeline) | O(1) substring | O(1) — no change |
| exact dedup | O(1) hash set | O(1) — already optimal |
| **simhash dedup** | **O(N²) linear scan over all prior fingerprints** | **O(N²) vectorized** (numpy SWAR popcount, ~ns/element in C) |
| **quality scoring** | **O(70 × n) regexes, run twice/sample** | O(pruned × n), single run, single-pass toxicity |
| AST validation | O(n) python-only parse | O(n) — parallelized (pure function) |
| function sampling | disabled in config | — |

---

## Bottleneck

1. **Quality scoring (86%)** — ~70 MULTILINE regexes for language detection executed twice per sample.
2. **SimHash O(N²) linear scan** (~257s extrapolated @ 44.5K fingerprints) — each new fingerprint compared against every previous one.

---

## Optimizations applied

### 1. SimHash: vectorized table pass (quality.py) — final design
- **First attempt (block-index dicts) FAILED on real data:** fingerprints of near-dup clusters share the same 13-bit blocks, producing ~50K-entry hot buckets; every lookup scanned them (measured: **1,627.6s** on the server for the table pass).
- **Final: exact O(N²) vectorized scan.** `process_batch(fps)` compares each fingerprint (hamming ≤ 9, i.e. similarity ≥ 0.85) against every previously-inserted fingerprint — the *same comparison set and order* as the original linear code — but via numpy `uint64` XOR + SWAR popcount (~2ns/element in C vs ~300ns with `bin().count()` in Python).
- Chunked (256/chunk): chunk vs stored array (vectorized) + strict-upper-triangle within chunk, masked so only *inserted* (non-dup) fingerprints are compared against — decisions bit-identical to sequential.
- Verified: **0 mismatches vs original algorithm on 6,000 texts**; **0 mismatches process_batch vs one-by-one on 3,000 texts**; near-dup-heavy cluster (3K, 99.7% dups): 0.1s.
- Result on server-like 100K local bench: table pass **0.1s** (was 1,627.6s).
- Fingerprint rewritten (`_compute_simhash_fp`): set-bit majority counting, ~2x faster, identical output.

### 2. Pipeline order — KEPT original (quality → AST → exact dedup → simhash)
- A trial reorder (dedup first) changed the accepted set (44,548 → 42,959 on the server, −3.6%): duplicates are order-dependent, and dedup-before-quality compared against low-quality samples that the original pipeline rejected before dedup.
- **Reverted to the original stage order** so the accepted set is identical to pre-optimization behavior. The speedups make this irrelevant: dedup still only sees quality+AST survivors (~44.5K fps), and the vectorized table pass handles them in ~0.1s locally.

### 3. Language detection pruning (quality.py)
- `_detect_language_with_counts`: single pass, early-exit when a language reaches 5 hits, skip languages that can no longer win; same winner as before (tie order preserved).
- `_language_confidence` reuses the counts (second pass eliminated).
- `_toxicity_score`: single compiled alternation regex (one pass instead of one-per-term).

### 4. Parallelism for pure stages (pipeline.py + quality.py + ast_filter.py)
- Spawn (Windows) / fork (Linux) pool, min(32, cores); sequential fallback with warning if pool fails.
- Workers are pure functions in `quality.py` / `ast_filter.py` (light import chains) — spawned workers never import the full torch/transformers/datasets stack (which cost ~70s/worker locally).
- Dedup state stays in the main process → identical decision sequence.

### 5. Per-stage timers (Phase 1 deliverable)
- `[TIMER] cleanup/<stage> <dataset>: Xs` per stage per dataset + `cleanup/filter total` + `tokenize+pack` + pool context/teardown logs.

---

## After (local synthetic corpus, new pipeline, same stage order)

| Stage | 15.5K corpus | 100K server-like corpus |
|---|---|---|
| boilerplate | 1.3s | 5.2s |
| language | 0.1s | 0.4s |
| quality scoring (pool) | 0.9s | 52.8s* |
| AST validation (pool) | 1.8s | 11.0s* |
| exact_dedup | 0.2s | 0.9s |
| simhash fingerprint (pool) | 5.2s | 2.8s |
| **simhash dedup (table)** | **0.0s** | **0.1s** (server measured: 1,627.6s before) |
| **cleanup/filter total** | **9.6s** (was 96.47s → 10.1x) | **64.4s** |

\* local machine, 8 workers — server (32 workers, faster cores): 2.4s / 0.5s.

- Both runs well under the 120s target on 8 local cores; server expected ~25-40s.
- Filtering behavior unchanged: same stage order (quality → AST → exact dedup → simhash), same per-stage decisions (dedup equivalence 0 mismatches vs original algorithm).

---

## Files changed

- `src/data/pipeline.py` — per-stage `[TIMER]` accumulators; pool (fork/spawn) with light-module workers; pool teardown; simhash stage uses `process_batch`; stage order kept original.
- `src/data/quality.py` — `_detect_language_with_counts` + pruned `detect_language`, count-reusing `_language_confidence`, single-pass `_toxicity_score`, `_compute_simhash_fp` (+ `_pool_simhash_fp`, `_pool_quality_score`), vectorized `SimHashDeduplicator` (`process_batch` + SWAR popcount).
- `src/data/ast_filter.py` — `_pool_filter_code` worker.

## Validation

- SimHash old-vs-new equivalence: **0 mismatches / 6,000 texts**.
- `process_batch` vs one-by-one sequential: **0 mismatches / 3,000 texts**; near-dup cluster (99.7% dups): 0.1s.
- `scripts/test_local_validation.py`: **34/34 passed**.
- `test_registry_cap.py` (100K cap + lazy mixed dataset): passes.

## After — server (100K opc dataset, same stage order, 32 workers)

| Stage | Original | Now |
|---|---|---|
| boilerplate | ~12s (extrap.) | 4.2s |
| language | 0.7s | 0.7s |
| quality scoring | ~415s (extrap.) | 7.7s |
| AST validation | ~53s (extrap.) | 1.0s |
| exact_dedup | 1.0s | 1.0s |
| simhash fingerprint | sequential | 6.3s (32 workers) |
| **simhash dedup (table)** | ~257s (est.) | **45.3s** (was 1,627.6s with block-index) |
| **cleanup/filter total** | **1,043s** | **66.4s (15.7x)** |
| tokenize+pack | — | 27.0s (5,739 packed) |

**Behavior verified bit-identical:** accepted **44,548**; rejected: duplicate 28,023, AST 27,193, quality 52, too short 184 — every count matches the original run exactly.

## Remaining

- Optional: simhash_dedup (45.3s, single-threaded numpy) is now the largest stage; could be further parallelized via fork-shared snapshots, but 66.4s already beats the 120s target ~2x — not required.
- 54 datasets × ~66s ≈ ~60min per full pretrain run (was ~15h+/dataset).
