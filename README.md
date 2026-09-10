# Dhara Class Model

Train a custom code-focused LLM from scratch on **4× A100 80GB** using the
**Dhara Class Model** — an NSLT-derived architecture with
**O(1) memory** w.r.t. sequence length.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia">
    <img alt="Tests" src="https://img.shields.io/badge/tests-162%20passing-brightgreen">
   <img alt="Bugs fixed" src="https://img.shields.io/badge/bugs%20fixed-25%2B-2ea44f">
   <img alt="CLI" src="https://img.shields.io/badge/cli-8%20commands-blue">
   <img alt="Architecture" src="https://img.shields.io/badge/architecture-Dhara-8A2BE2">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Launch training (pretrain → SFT → instruction tuning)
bash scripts/train_4gpu.sh
```

No interaction needed. Training auto-resolves the tokenizer, streams datasets,
and saves checkpoints every 1,000 steps.

---

## Development Setup (Windows PC & JupyterLab GPU Server)

This project uses a hybrid development setup:
- **Local Workstation (Windows PC)**: Used for code editing, static analysis, and CPU configuration checks.
  ```cmd
  pip install -r requirements-dev.txt
  python main.py config-validate
  ```
- **Remote Training Environment (JupyterLab Linux GPU Server)**: Used for full GPU model training, distributed FSDP, and CUDA benchmarks.
  ```bash
  pip install -r requirements.txt
  bash scripts/train_4gpu.sh
  ```
All file paths use relative forward slashes (`/`), environment configurations auto-adapt to available CUDA devices, and scripts remain strictly bash-compatible.

---

## What is the Dhara Class Model?

A **non-Transformer** LLM built on the Dhara architecture (V4 redesign) — a **workspace-centric** hierarchical reasoning pipeline with **O(1) memory** w.r.t. sequence length:

```
Input → [Tokenizer] → [Embedding+RoPE] → [MemoryManager] → [Executive Controller]
                                                                    │
                                                               (gate enforcement)
                                                                    │
         ┌──────────────────────────────────────────────────────────┘
         ▼
    [Intent] → [Hierarchical Planner] → [ODE Reasoning] → [Workspace (central hub)]
         │                                                    │     │
         ▼                                                    ▼     ▼
    [WorldModel]                                    [SymbolicToolRouter] ← [DebateSandbox]
                                                           │                │
                                                           └─────► Workspace
         ┌────────────────────────────────────────────────────────┘
         ▼
    [QualityAssurance (Reflection + Verification + Curiosity)]
         │
         ▼
    [Workspace (final)] → [HierarchicalDecoder] → Output
```

| Transformer | Dhara V4 |
|---|---|
| O(T²) attention or O(T·d_kv) KV-cache | **O(1)** compressed state — fixed size, any length |
| Single forward pass | **Executive-controlled with gate enforcement**: modules are actually skipped when gate < threshold |
| Self-attention + FFN per layer | **14 specialized components** with workspace-as-central-hub communication |
| Neural function approximation | **Symbolic tools** (real `eval()`/`exec()`) with learned routing |
| Single next-token loss | **14 auxiliary losses** (intent, planning, verification, calibration, tools, novelty, etc.) |
| MatMul-heavy | Scan-heavy (HSSM recurrence + Neural ODE) |

---

## Features

- **8 CLI commands** — train, reserved-train, generate, test, benchmark, download tokenizer, validate config, system info
- **3-stage training** — pretrain (1M) → SFT (100K) → instruction tuning (50K)
- **HF PreTrainedModel integration** — DharaModel works with HuggingFace Trainer, checkpoint save/load
- **DharaConfig** — Pydantic-validated config with cross-field constraints (`PretrainedConfig` subclass)
- **FSDP full-shard** across 4 GPUs (ZeRO-3), enabled for single-GPU with CPU offload
- **GPU reservation** — `reserved-training` command waits for free GPU, locks it, then trains
- **9 benchmarks** — HumanEval, MBPP, MMLU, GSM8K, HellaSwag, ARC, TruthfulQA, Winogrande, BBH
- **162 pytest tests + 431 script-suite checks** — all passing
- **Claude-grade tokenizer** — `Xenova/claude-tokenizer` (BPE, ~100K vocab)
- **5 SSM scan backends** — sequential, vectorized, Triton, TorchScript JIT, CUDA
- **Dhara 14-component V4 architecture** — workspace-as-central-hub communication, symbolic tools with learned routing, merged QualityAssurance (reflection + verification + curiosity), RL-trained Executive Controller with gate enforcement, 14 auxiliary training losses
- **Causal LM training** — proper label shift (position i predicts i+1), per-position decoder context

---

## CLI Overview

| Command | What it does |
|---|---|---|
| `python main.py full-training` | Run pretrain → SFT → instruction tuning |
| `python main.py reserved-training` | Wait for free GPU, lock it, then train |
| `python main.py generate --prompt "..."` | Generate text from a checkpoint |
| `python main.py test` | Run the 162-test suite |
| `python main.py benchmark` | Run benchmarks against a checkpoint |
| `python main.py download-tokenizer` | Download a HuggingFace tokenizer |
| `python main.py config-validate` | Validate `config.yaml` |
| `python main.py info` | Print system info |

### Training

```bash
# 4-GPU (production)
bash scripts/train_4gpu.sh

# Single GPU (debug)
python main.py full-training --gpu 3

# Fresh start (ignore checkpoints)
python main.py full-training --fresh-start

# Wait until a GPU is free, reserve it, then train (shared cluster)
python main.py reserved-training --fresh-start
python main.py reserved-training --fresh-start --min-free-gb 60 --poll-interval 10
python main.py reserved-training --fresh-start --gpu 3  # skip wait, just reserve
```

Training auto-resumes from the latest checkpoint. Each stage uses its own
learning rate, batch size, and optimizer:

| Stage | Steps | LR | Batch (eff.) | Tokens (est.) | Optimizer |
|---|---|---|---|---|---|
| Pretrain | 1M | 2e-4 | 64 | ~262B | adamw_fused |
| SFT | 100K | 5e-6 | 16 | ~6.5B | adamw_fused |
| Instruction Tuning | 50K | 1e-5 | 16 | ~3.3B | adamw_fused |
| **Total** | **1.15M** | — | — | **~272B** | — |

This config targets a **~100M param** model with **intentional overtraining**
(~130× Chinchilla-optimal). Fixed-size SSM state may benefit from more steps.
For Chinchilla-optimal (2B tokens), set `pretrain.max_steps: 7600`.

### Generation

```bash
python main.py generate --prompt "Write a Python HTTP server"
echo "def fibonacci(n):" | python main.py generate

python main.py generate \
  --prompt "Write a Rust HTTP server" \
  --checkpoint models/dhara/sft \
  --max-new-tokens 2048 \
  --temperature 0.8 \
  --top-p 0.95
```

### Testing

```bash
python main.py test                          # all 162 pytest tests
python main.py test --filter ssm             # SSM scan tests only
python -m pytest tests/ -v --tb=short -x      # verbose, stop on first failure
python -m pytest tests/ --cov=src            # coverage
```

### Benchmarking

```bash
python main.py benchmark                                          # all
python main.py benchmark --benchmarks "human_eval,mbpp"           # specific
python main.py benchmark --checkpoint models/dhara/best          # custom checkpoint
```

### Tokenizer

```bash
python main.py download-tokenizer                                 # Claude tokenizer
python main.py download-tokenizer --model-id google/gemma-2-27b-it
python main.py download-tokenizer --force
```

---

## Project Layout

```
.
├── main.py                     # CLI entrypoint (8 commands)
├── config.yaml                 # Training configuration (Pydantic-validated)
├── scripts/train_4gpu.sh       # 4-GPU torchrun launcher
├── src/
│   ├── dhara/              # Dhara — advanced cognitive architecture (V4 redesign)
│   │   ├── model.py            # DharaModel (PreTrainedModel), DharaForCausalLM, DharaConfig
│   │   ├── hssm.py             # Hierarchical SSM stack (3-level, vectorized)
│   │   ├── workspace.py        # CognitiveWorkspace — central dict-based communication hub
│   │   ├── executive.py        # ExecutiveController — RL-trained gate enforcement + merged LearningController
│   │   ├── quality_assurance.py# QualityAssurance — merged reflection/verification/curiosity multi-pass QA
│   │   ├── tools.py            # SymbolicToolRouter — symbolic calculator/python/search/db with learned routing
│   │   ├── losses.py           # AuxiliaryLossComputer — 14 configurable auxiliary training losses
│   │   ├── world_model.py      # World Model — entities, relations, events, cause-effect
│   │   ├── curiosity.py        # CuriosityModule — backward-compatible wrapper (internally uses QA)
│   │   ├── learning_controller.py # Backward-compatible re-export from executive.py
│   │   ├── layer1_tokenizer.py # Intelligent tokenizer with semantic metadata
│   │   ├── layer2_embedding.py # Adaptive semantic embedding + RoPE
│   │   ├── layer3_memory.py    # MemoryManager — working/semantic/long/episodic with forget/compress/retrieve/prioritize
│   │   ├── layer4_intent.py    # Intent understanding + difficulty routing
│   │   ├── layer5_planner.py   # HierarchicalPlanner — goal tree + dependencies + execution graph + cost
│   │   ├── layer6_reasoning.py # Adaptive continuous (Neural ODE) reasoning
│   │   ├── layer7_workspace.py # CognitiveWorkspace (backward-compatible re-export from workspace.py)
│   │   ├── layer8_specialists.py # DebateSandbox — debate + cross-examination + repair + consensus
│   │   ├── layer9_reflection.py  # RecursiveReflection (backward-compatible QA wrapper)
│   │   ├── layer10_verification.py # VerificationWithRepair (backward-compatible QA wrapper)
│   │   └── layer11_decoder.py  # HierarchicalDecoder — semantic → language → token (3-level)
│   ├── nslt/                   # NSLT architecture (pure PyTorch)
│   │   ├── model.py            # NSLTModel (~759 lines)
│   │   ├── layer1_ssm.py       # SSM compression engine
│   │   ├── layer2_ltc.py       # Neural ODE routing
│   │   ├── layer3_sandbox.py   # Latent sandbox reasoning
│   │   ├── layer4_output.py    # Sparse output gating
│   │   ├── ssm_scan.py         # 5 scan backends + autograd
│   │   ├── moe_ssm.py          # MoE SSM blocks
│   │   ├── mcts_sandbox.py     # MCTS latent sandbox
│   │   ├── multiscale_ssm.py   # Multi-scale hierarchy
│   │   └── vision_encoder.py   # SigLIP vision encoder
│   ├── config/schema.py        # Pydantic config (536 lines, 8 cross-field validators)
│   ├── models/factory.py       # Model creation + loading (573 lines)
│   ├── training/pipeline.py    # Training orchestration
│   ├── tokenizer_trainer.py    # Tokenizer download + BPE training
│   ├── infrastructure/
│   │   ├── distributed.py      # FSDP distributed setup (mixed_precision fix)
│   │   └── tracking.py         # WandB/MLflow/TensorBoard
│   └── data/
│       ├── pipeline.py         # Data streaming + processing
│       ├── doc_builder.py      # 18-source documentation crawler (template-safe)
│       ├── registry.py         # Dataset registry + weights
│       └── streaming.py        # Streaming with fallback chains
├── scripts/
│   ├── corpus_audit.py         # JSONL corpus quality gate
│   ├── verify_datasets.py      # HF Hub dataset verification
│   ├── production_validation.py# 8-phase production validation
│   └── test_local_validation.py# 14 local validation tests
├── tests/                      # 162 tests
└── hf_cache/                   # Dataset cache
```

For the complete reference (every config field, every CLI flag, architecture
deep-dive, troubleshooting guide, benchmark methodology), see
[`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA 12.1+ for GPU training.

---

## Test Suite

162 pytest tests across 14 files, all passing (`python -m pytest tests/ -q`):

```
tests/test_nslt.py                   # 14 — All 4 layers + full model
tests/test_integration.py            # 20 — SSM scan, training, MoE, vision, MCTS
tests/test_alignment.py              # 8 — DPO/ORPO/SimPO/KTO loss + ref-model fallback
tests/test_config.py                 # 8 — Config validation + migration
tests/test_data_collector.py         # 5 — Dataset streaming
tests/test_data_pipeline.py          # 29 — Text cleaning, quality, dedup, gaps
tests/test_async_pipeline_overlap.py # 19 — UnitPrefetch + AsyncCheckpointWriter stress
tests/test_foundation_pipeline.py    # 14 — 0-dataset foundation config
tests/test_main_cli.py               # 4 — CLI parser flags
tests/test_evaluation.py             # 5 — Safety, benchmarks
tests/test_generation.py             # 10 — Code extraction, language aliasing
tests/test_quality.py                # 10 — Quality scoring, contamination
tests/test_trainer.py                # 6 — Prompt formatting, tokenization
tests/test_validation.py             # 10 — Python syntax + execution
```

Plus script suites (not under pytest): `scripts/test_pipeline.py` 124, `scripts/test_pipeline_async.py` 43, `scripts/test_local_validation.py` 258, `scripts/benchmark_async_pipeline.py` 6 checks.

---

## Benchmarks

| Benchmark | Type | Problems | Metric |
|---|---|---|---|
| HumanEval | Code generation | 164 | pass@1 |
| MBPP | Programming tasks | 417 | pass@1 |
| MMLU | Knowledge (57 subjects) | ~14K | accuracy |
| HellaSwag | Commonsense NLI | 10K | accuracy |
| ARC | Science QA | 2,590 | accuracy |
| GSM8K | Math word problems | 1,319 | exact match |
| TruthfulQA | Factuality | 817 | mc1/mc2 |
| Winogrande | Coreference | 1,267 | accuracy |
| BBH | Reasoning (23 tasks) | 6,511 | accuracy |

---

## Troubleshooting

| Problem | Fix |
|---|---|---|
| CUDA OOM (single GPU) | `reserved-training` command auto-enables CPU offload for optimizer states |
| CUDA OOM (multi GPU) | Reduce `batch_size`, lower `max_seq_length`, enable `expandable_segments` |
| GPU busy by another user | `python main.py reserved-training` — waits/reserves automatically |
| NCCL errors | Set `NCCL_DEBUG=WARN`, `NCCL_NVLS_ENABLE=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| "Tokenizer not found" | Run `python main.py download-tokenizer` |
| Architecture mismatch | `rm -rf models/dhara/checkpoints; bash scripts/train_4gpu.sh --fresh-start` |
| "No module named 'src'" | Run from the project root directory |
| Training hangs at init | Check GPUs with `nvidia-smi`, kill stale `torchrun` processes |
| `mixed_precision` key not found | Config uses `"mixed_precision": "fp16"`; both `fp16` and `mixed_precision` are accepted |
| HSSM state tensor shape mismatch | HSSM `A` tensor uses `d_state` dimension from `num_heads`, not `d_model` |
| Missing `config.json` when loading | Models save/load via `save_pretrained()` which writes `config.json` |
| FSDP module attribute error | Trainer sets `ddp_find_unused_parameters = False` to avoid FSDP parameter tracking issues |
| Singleton array in `can_soft_mixture` | `mo_soft_assignments` handles scalar wake masks properly |
| Memory manager returns 3 values | Model expects `mem_out, mem_state, mem_meta = self.memory(...)` |
| Decoder logit shape mismatch (inference) | 3D `(batch, seq, d_hidden)` inputs are auto-flattened before decoder forward |
| Curiosity confidence dimension mismatch | Module auto-expands scalar confidence to match batch dim |

---

## Changelog

### 2026-08 — Model-correctness audit pass (third audit follow-up)

| Area | Gap fixed |
|------|-----------|
| **Triton SSM scan wrong result** | Stage mask rounding produced different per-state sums than the reference for the audit case `[1.0, 2.9, 4.16, 3.828]`. Rewritten as a fully-masked Hillis–Steele inclusive scan (masked loads/stores at every sweep step, `seq_len>0` guard). Matches the reference for prefix-sums, non-power-of-2 lengths, and 200 random cases |
| **Triton scan gradient** | Backward only handled the up-sweep case and saved `h_seq` in the ctx. Now recomputes `h_seq` inside `no_grad` and computes correct adjoints for `A/B/C/delta` (incl. the `B_bar` derivative); gradcheck vs native autograd: max delta 2.4e-7 for all 5 grads |
| **Triton chosen in autograd mode** | Auto mode picked Triton even when inputs needed gradients → custom-backward mismatch. Non-Triton path is now selected whenever any input `requires_grad` |
| **MCTS sandbox dead gradients** | `MCTSLatentSandbox`/`Efficient` built `z_0` from learned embeddings but expanded with detached noise → `energy_fn`/`policy_value`/`out_proj`/`norm` got zero gradients. Search steps expanded with the real `energy_grad`, and `_finalize_state` replays the chosen trajectory in-graph so all learned params train |
| **Causal-loss future leak** | Loss branch broadcast `z_final` into every position (`pos_hidden[:, :-1] + z_final`) → step-t targets leaked the whole sequence. Now uses `pos_hidden[:, :-1]` only; whole-sequence features feed a separate sequence-level reasoning head (`nan_to_num(main_logits)` + 0.1× CE on the final label) |
| **Padding masked in forward** | `x` was unmasked after RoPE when the loss branch ran → padding positions trained random features. Mask applied right after RoPE |
| **Auxiliary losses unwired** | `AuxiliaryLossComputer.forward` ignored memory/planning/trajectory/entity/decoder losses (decoder_loss didn't exist). All five wired; new `decoder_loss()` masks labels with `-100` |
| **`compute_log_prob` vs forward mismatch** | `layer4_output.compute_log_prob` scored the target with the *unscaled* logits while `forward` applied a gated scale — later steps diverged. Now consistent (verified `allclose` vs manual reference) |
| **Eval crash without eval dataset** | `eval_strategy="steps"` is the schema default but no `eval_dataset` was passed → eval-time crash. `_build_trainer` now carves a deterministic 5% eval split and falls back to `"no"` when the dataset can't split |
| **FSDP gather skipped on staged pretrain** | `_run_staged_pretrain` built the Trainer locally and never set `self._last_trainer` → `_state_dict_for_save()` fell back to rank-local shards. Trainer now registered on build |
| **Resume layout drift / fresh-start leak** | `_find_resume_checkpoint` only looked at the model root + `checkpoint_dir` (never found stage/`checkpoint-N` dirs); `fresh_start` still resumed staged steps from `trainer_state.json`; non-staged stages saved `save_only_model=True` so resume always restarted a fresh optimizer. Resume now scans `model_dir` recursively (newest weights), `fresh_start`/`resume_checkpoint` are threaded through `initialize` → stage resume → staged step accounting, and stages save full optimizer/scheduler state |
| **Staged step accounting** | `stage_stats["steps"]` logged the *planned* steps per unit; now the real `trainer.state.global_step` delta |
| **NaN check on every step** | `_NaNSafeCallback` iterated every named parameter each step (O(params) per step). Now throttled to once per 100 steps |
| **Alignment hyperparams unwired** | `_run_preference_training` hardcoded `batch_size=4` / `max_steps=1000`; per-method `batch_size`/`max_steps` from `method_configs` never reached the loop; avg divided by `step` (off-by-one). All wired; average now divides by the real iteration count |
| **Safety toggles never consulted** | `refusal_training`/`honesty_training`/`constitution_training` were dead flags; `safety_training_steps`/`red_teaming_iters` defaults were hardcoded. `run_safety_training` skips when `refusal_training=False` and defaults to config values |
| **Benchmark config unwired** | `benchmark_configs[].max_samples` and `evaluation.timeout` never reached the runner. Per-benchmark `limit`/`timeout` now flow through `run_evaluation` |
| **DeepSpeed config out of sync** | Generated `ds_config.json` hardcoded `train_batch_size=1`, `gradient_accumulation_steps=1`, and `bf16: True` even for fp16. Now derived from the Trainer's actual batch/grad-accum/world-size and the model dtype |
| **Dead optimizer fallback** | `"adamw_fused"` fallback gated on `hasattr(torch.optim, "AdamW")` (always true). Now resolves against the installed transformers optimizer registry and falls back to `adamw_torch`; 8-bit without CUDA also falls back |
| **Prefetch slot leak** | A skipped/timed-out unit's late build result sat in the buffer forever, permanently shrinking prefetch depth (stall cascade). Workers now drop stale results (idx < expected) |
| **Checkpoint writer close hang** | `AsyncCheckpointWriter.close()` could block up to `write_timeout` after writer death (blocking `put` on a full queue). `close()` is now idempotent with a bounded put |
| **Embedding dedup mapped to exact** | Config `method="embedding"` silently fell through to `ExactDeduplicator`. Now routes to `SemanticDeduplicator` |
| **Contamination only on SFT** | Contamination gating ran only in `build_supervised_dataset`. Added to both pretrain filter loops (registry + legacy), with ledger counters |
| **fineweb-edu 5-category overlap** | `HuggingFaceFW/fineweb-edu` `sample-10BT` was registered as a primary dataset in 5 categories — same documents counted 5×, and staged mode's unit-identity dedup silently skipped 4 of them. Each category now uses a distinct real HF config (`sample-10BT`/`sample-100BT`/`sample-350BT`/`CC-MAIN-2024-10`/`CC-MAIN-2023-50`); same for fineweb (`default`/`sample-100BT`/`CC-MAIN-2021-10`) — configs verified against the HF datasets-server splits API |
| **Unit cache key blind to category** | `unit_cache_key` omitted `category`, so same-path/same-name units in different categories shared a cache. Category folded into the key |

Touched: `src/nslt/ssm_scan.py`, `src/nslt/mcts_sandbox.py`, `src/nslt/model.py`, `src/nslt/layer4_output.py`, `src/dhara/losses.py`, `src/alignment/pipeline.py`, `src/training/pipeline.py`, `src/training/asyncprefetch.py`, `src/training/checkpoint.py`, `src/infrastructure/distributed.py`, `src/data/pipeline.py`, `src/data/registry.py`, `src/evaluation/benchmarks.py`, `tests/test_nslt.py`, `README.md`. Verified: **pytest 166/166** (15 files); `test_pipeline_async.py` / `test_pipeline.py` / `test_local_validation.py` suites unchanged. 4 new regression tests (MCTS gradient flow, compute_log_prob gate consistency, causal-loss padding mask, aux-loss wiring).

### 2026-08 — Implementation gap-fill pass (second audit follow-up)

| Area | Gap fixed |
|------|-----------|
| **FSDP checkpoint corruption (leftover)** | `_save_checkpoint` still reassigned `state = self.model.state_dict()` after the gather — the first-pass fix was shadowed and 4-GPU checkpoints were again written rank-local shards. Shadowing line deleted; the gathered `_state_dict_for_save()` result is what gets written |
| **Triton SSM scan race** | `_ssm_scan_kernel` computed `pid` but never used it for `a/b/c/y` addressing — on GPU every program (batch×state grid) read/wrote the same batch-0, state-0 region (race corruption) and y-writes for batch>0 collided with batch 0. `pid` is now decomposed into `(b_id, s_id)` and every pointer access is offset by batch/state strides |
| **DPO/KTO trained against itself** | `run_dpo`/`run_kto` never passed a reference model → `ref_model=None` fell back to the policy itself (`ref_logps = policy logps`), zeroing the KL signal (DPO loss constant, KTO degenerate). `AlignmentPipeline` now lazily deep-copies and freezes the policy as the reference (reused across methods) and passes it to both trainers |
| **Alignment/safety/eval phases dead** | `run_alignment`, `run_safety_training`, `run_red_teaming`, `run_evaluation` had zero callers — `full_training_sequence` ended after instruction tuning. Phases now gated on `cfg.training.alignment.enabled` / `cfg.training.safety.enabled` / `cfg.evaluation.automated_report`, with a bounded instruction collector feeding `generate_preference_data` → `build_preference_dataset` |
| **LatentSandbox gradient barrier** | The sim loop detached `z_k` every step AND the final selection ran under `torch.no_grad()` → `energy_fn`/`out_proj`/`norm` got zero gradients (module was a frozen random projection). The last iteration's update and the best-trajectory selection are now in-graph (search iterations stay detached for memory), so all learned params train. Applied to `LatentSandbox` and `LatentSandboxEfficient` |

Touched: `src/training/pipeline.py`, `src/nslt/ssm_scan.py`, `src/nslt/layer3_sandbox.py`, `src/alignment/pipeline.py`, `tests/test_alignment.py`, `tests/test_nslt.py`. Verified: **pytest 162/162** (14 files); `test_pipeline_async.py` 43/43; `test_pipeline.py` 124/124; `test_local_validation.py` 258/258; benchmark 6/6. 3 new regression tests added (sandbox gradient flow, frozen ref-model).

### 2026-08 — Implementation gap-fill pass

Audit found and fixed **18 gaps** across the collector, data pipeline, and training paths:

| Area | Gap fixed |
|------|-----------|
| **SFT/instruction no-op** | `MassiveDataCollector` crashed on pydantic `DatasetEntryConfig` models (`.get()`/`[]` on attribute-based objects) → every SFT/instruction dataset was silently skipped and those stages trained on nothing. Entries are now normalized (`_to_dict`); `stream_single_dataset`/`stream_samples` accept both dicts and models |
| **Format extraction drift** | `_process_entry` (production path) missed dolly/flan/orca/tool-use/QA/sentence formats that `_extract_fields` supported — production samples silently fell back to raw-text extraction. `_process_entry` now delegates to `_extract_fields` (single source of truth); duplicate dead block deleted |
| **Skip re-streaming** | `stream_single_dataset(skip_samples=…)` re-loaded and re-iterated the whole dataset per skipped sample (infinite-loop risk). Rewritten as a single pass with inline skips |
| **`reserved-training` crash** | `--gpu` used `argparse.SUPPRESS` → `args.gpu` was absent → `AttributeError` before any GPU logic. Default is now `None` |
| **FSDP corrupt checkpoints** | `save_model`/`_save_checkpoint` wrote raw `state_dict()` (rank-local shards) over the gathered `pytorch_model.bin` — on 4-GPU FSDP every rank wrote shards to the same path (last-writer-wins). Now gathers `FULL_STATE_DICT` on the main rank and skips other ranks |
| **Failed shards data loss** | Worker-crashed shards were persisted as `complete=1` and permanently skipped on resume. Failed shards now reset their progress and are retried |
| **Dead fallback chains** | `resolve_error` short-circuited the whole fallback chain (`_stream_cold` unreachable). Now continues to the next fallback entry. `WEB_FALLBACKS` (dolma/RedPajama/refinedweb) and `sql-create-context` were never registered, so their fallback slots never fired — registered as `fallback_only` entries (resolvable, never streamed as primaries) |
| **Stale shard-progress** | Progress fingerprint omitted the resolved file list — upstream file changes re-used stale per-shard offsets. Record fingerprint (incl. files/revision) folded into the progress key |
| **Mid-stream stalls** | `ShardCoordinator`'s first-row timeout only applied before the first row — a mid-stream network/pyarrow hang spun forever. Stall timeout now applies to any no-progress period |
| **Zero-weight crash** | `WeightedMixedDataset` divided by zero when all weights were 0; exhausted datasets kept re-sampling consumed indices (~100% duplicate epochs). Uniform fallback + fresh-pass restart added |
| **Stale tokenized cache** | Legacy `_get_cache_key` ignored dataset name/split and dedup/quality/boilerplate settings — changing them silently reused stale cached datasets |
| **Cross-split cache thrash** | Metadata/builder cache dirs keyed by (repo, name) only — two splits of one dataset invalidated each other. `safe_dir_name` now includes `split` |
| **Local driver misses** | `detect_driver` only checked `path` (not `data_dir`-rooted doc datasets) and mixed-extension dirs yielded no loader; hub paths shadowed by same-named local dirs. Now checks `data_dir` and filters to the dominant loadable extension |
| **Torn JSONL after timeout** | `doc_builder.scrape_all` left a daemon thread appending to `documents.jsonl` on timeout. Scrapes go to a temp file, fsynced and atomically renamed only on completion |
| **Crash-safe cache writes** | All atomic writers (metadata, builder, shard progress) lacked `fsync` — a power loss could leave empty/partial renamed files. `flush()+fsync` before every `os.replace` |
| **Health-report overwrite** | `DatasetHealthReport` used `dict.update` → language/domain totals only reflected the last dataset. Now accumulates |
| **Dead skip counters / stale resume** | `StreamingManager.skip_rate()` was dead code — now wired to real fallback skips. `ScriptDatasetDriver.reset_resume` left `_raw_consumed` stale |
| **Language/domain stratification no-op** | Packed datasets never carried `language`/`domain` → balancing collapsed into `'other'`. Hints now attached (fresh and cache-hit) |

Touched: `src/massive_data_collector.py`, `src/data/{pipeline,streaming,drivers,metadata_cache,shards,registry,health_reporter,doc_builder}.py`, `src/training/pipeline.py`, `main.py`, `scripts/test_local_validation.py`. Verified: **pytest 159/159** (14 files); `test_pipeline_async.py` 43/43; `test_pipeline.py` 124/124; `test_local_validation.py` 258/258; benchmark 6/6; 3 configs validate clean. 19 new tests added for the fixed gaps.

### 2026-07 — v1.0-pretraining final pass (data pipeline)

**Final engineering pass before freeze — status: READY TO FREEZE REPOSITORY.**

| Area | Fix |
|------|-----|
| **Template URL injection** | Template placeholder URLs (`{{ }}`, `{% %}`, `${ }`, `<% %>`) and invalid hrefs (`javascript:`, `void(0)`, `#`) are rejected in every scraper's `discover_urls()` **before queue insertion**; logged as `Filtered template URL` and counted in `filtered=N` — verified end-to-end with a synthetic Docker crawl (0 template URLs reach the network) |
| **CUDA discovery bug** | Indentation bug dropped every href except the last per page; all links now followed |
| **PostgreSQL** | URL validation accepts any `/docs/<version>/` path |
| **Linux kernel** | `MIN_TEXT_LENGTH` 3000→500 (nav sidebar excluded by selector — no boilerplate) |
| **cuDNN** | Seeds updated to `/latest/`, sub-link following re-enabled |
| **Corpus audit** | New `scripts/corpus_audit.py` — UTF-8/JSON/dup/HTML-leakage/length/token checks; corrupt 0-byte artifact removed |
| **Validation** | `test_local_validation.py` 14/14 clean; `test_integration.py` fixed for transformers v5 (`AutoConfig.for_model`, `eval_strategy`, resume from `checkpoint-N/` dirs) — training→checkpoint→resume verified offline |

**Data pipeline summary**: 18 doc sources, template-safe crawling, per-source summary (`discovered= dups= filtered= failed=`), quality-gated JSONL output. Build datasets with:

```bash
python src/data/doc_builder.py                  # all 18 sources
python src/data/doc_builder.py --sources python docker cudnn
python scripts/corpus_audit.py                  # quality gate on all JSONL
```

### 2026-07 — V4 Bug Fix Audit

**25+ bugs fixed across 14 files** in a comprehensive codebase audit:

| Category | Bugs Fixed |
|----------|-----------|
| **Buffer registration** | Workspace `shared_repr`/`confidence`, episodic buffer, `mem_priority` — all silently became plain tensors after `reset()`/`forward()` via direct reassignment. Fixed with `.data.copy_()`. |
| **Module registration** | `SymbolicSearch`/`SymbolicDatabase` not inheriting `nn.Module` — params invisible to optimizer. `_goal_proj` created dynamically in `forward()` with double registration. Fixed with proper `nn.Module` inheritance and `__init__`-time creation. |
| **Shape mismatches** | Trajectory averaging broadcast bug (`steps_f` with extra dim produced 3D output). `ODETick dz` shape wrong for 3D inputs. `B_bar` discretization used `B*dt` instead of zero-order hold formula. State chaining dim mismatch between HSSM layers. |
| **Training correctness** | Dummy `torch.zeros` logits in training path. `generate()` didn't restore `train()` mode. Wrong `hierarchical_log_prob` tuple unpacking. BCE clamping `(1e-7, 1-e7)` used subtraction (effectively `(1e-7, -9999993)`). CPU tensors in loss functions. |
| **Factory/config** | `d_model`/`max_seq_len`/`rope_base`/`dtype` kwargs silently filtered out because names didn't match `DharaConfig.__init__`. Same in `load_model` plus uninitialized `saved_config` variable. |
| **Trainer compat** | `self.model.model` / `.is_ready` / `.save_model()` assumed `SpecializedCoderModel` wrapper. Fixed with `_unwrap_model` property and `hasattr` fallbacks. |
| **Code quality** | In-place `conf_t` mutation, dead `debate_input` code, unused `CompressionAE` tuple unpacking, `position.weight` used instead of `position(positions)`, redundant `MemoryRetriever` batch broadcasting. |

For the full fix list, see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md#appendix-bug-fix-audit).

---

## License

MIT
