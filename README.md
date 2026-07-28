# Methos Class Model

Train a custom code-focused LLM from scratch on **4× A100 80GB** using the
**Methos Class Model** — an NSLT-derived architecture with
**O(1) memory** w.r.t. sequence length.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia">
   <img alt="Tests" src="https://img.shields.io/badge/tests-109%20passing-brightgreen">
   <img alt="Bugs fixed" src="https://img.shields.io/badge/bugs%20fixed-25%2B-2ea44f">
   <img alt="CLI" src="https://img.shields.io/badge/cli-8%20commands-blue">
   <img alt="Architecture" src="https://img.shields.io/badge/architecture-MethosV3-8A2BE2">
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

## What is the Methos Class Model?

A **non-Transformer** LLM built on the MethosV3 architecture (V4 redesign) — a **workspace-centric** hierarchical reasoning pipeline with **O(1) memory** w.r.t. sequence length:

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

| Transformer | MethosV3 V4 |
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
- **HF PreTrainedModel integration** — MethosV3Model works with HuggingFace Trainer, checkpoint save/load
- **MethosV3Config** — Pydantic-validated config with cross-field constraints (`PretrainedConfig` subclass)
- **FSDP full-shard** across 4 GPUs (ZeRO-3), enabled for single-GPU with CPU offload
- **GPU reservation** — `reserved-training` command waits for free GPU, locks it, then trains
- **9 benchmarks** — HumanEval, MBPP, MMLU, GSM8K, HellaSwag, ARC, TruthfulQA, Winogrande, BBH
- **109 tests** — all passing
- **Claude-grade tokenizer** — `Xenova/claude-tokenizer` (BPE, ~100K vocab)
- **5 SSM scan backends** — sequential, vectorized, Triton, TorchScript JIT, CUDA
- **MethosV3 14-component V4 architecture** — workspace-as-central-hub communication, symbolic tools with learned routing, merged QualityAssurance (reflection + verification + curiosity), RL-trained Executive Controller with gate enforcement, 14 auxiliary training losses
- **Causal LM training** — proper label shift (position i predicts i+1), per-position decoder context

---

## CLI Overview

| Command | What it does |
|---|---|---|
| `python main.py full-training` | Run pretrain → SFT → instruction tuning |
| `python main.py reserved-training` | Wait for free GPU, lock it, then train |
| `python main.py generate --prompt "..."` | Generate text from a checkpoint |
| `python main.py test` | Run the 109-test suite |
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
  --checkpoint models/methos/sft \
  --max-new-tokens 2048 \
  --temperature 0.8 \
  --top-p 0.95
```

### Testing

```bash
python main.py test                          # all 109
python main.py test --filter ssm             # SSM scan tests only
python -m pytest tests/ -v --tb=short -x      # verbose, stop on first failure
python -m pytest tests/ --cov=src            # coverage
```

### Benchmarking

```bash
python main.py benchmark                                          # all
python main.py benchmark --benchmarks "human_eval,mbpp"           # specific
python main.py benchmark --checkpoint models/methos/best          # custom checkpoint
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
│   ├── methos_v3/              # MethosV3 — advanced cognitive architecture (V4 redesign)
│   │   ├── model.py            # MethosV3Model (PreTrainedModel), MethosV3ForCausalLM, MethosV3Config
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
│   ├── nslt/                   # Methos Class Model architecture (pure PyTorch)
│   │   ├── model.py            # NSLTModel (~694 lines)
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
│   └── data/pipeline.py        # Data streaming + processing
├── tests/                      # 109 tests
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

109 tests across 11 files:

```
tests/test_nslt.py               # 17 — All 4 layers + full model
tests/test_integration.py         # 12 — SSM scan, training, MoE, vision, MCTS
tests/test_alignment.py           # 4 — DPO/ORPO/SimPO/KTO loss
tests/test_config.py              # 7 — Config validation + migration
tests/test_data_collector.py      # 3 — Dataset streaming
tests/test_data_pipeline.py       # 20 — Text cleaning, quality, dedup
tests/test_evaluation.py          # 4 — Safety, benchmarks
tests/test_generation.py          # 8 — Code extraction, language aliasing
tests/test_quality.py             # 6 — Quality scoring, contamination
tests/test_trainer.py             # 3 — Prompt formatting, tokenization
tests/test_validation.py          # 10 — Python syntax + execution
```

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
| Architecture mismatch | `rm -rf models/methos/checkpoints; bash scripts/train_4gpu.sh --fresh-start` |
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

### 2026-07 — V4 Bug Fix Audit

**25+ bugs fixed across 14 files** in a comprehensive codebase audit:

| Category | Bugs Fixed |
|----------|-----------|
| **Buffer registration** | Workspace `shared_repr`/`confidence`, episodic buffer, `mem_priority` — all silently became plain tensors after `reset()`/`forward()` via direct reassignment. Fixed with `.data.copy_()`. |
| **Module registration** | `SymbolicSearch`/`SymbolicDatabase` not inheriting `nn.Module` — params invisible to optimizer. `_goal_proj` created dynamically in `forward()` with double registration. Fixed with proper `nn.Module` inheritance and `__init__`-time creation. |
| **Shape mismatches** | Trajectory averaging broadcast bug (`steps_f` with extra dim produced 3D output). `ODETick dz` shape wrong for 3D inputs. `B_bar` discretization used `B*dt` instead of zero-order hold formula. State chaining dim mismatch between HSSM layers. |
| **Training correctness** | Dummy `torch.zeros` logits in training path. `generate()` didn't restore `train()` mode. Wrong `hierarchical_log_prob` tuple unpacking. BCE clamping `(1e-7, 1-e7)` used subtraction (effectively `(1e-7, -9999993)`). CPU tensors in loss functions. |
| **Factory/config** | `d_model`/`max_seq_len`/`rope_base`/`dtype` kwargs silently filtered out because names didn't match `MethosV3Config.__init__`. Same in `load_model` plus uninitialized `saved_config` variable. |
| **Trainer compat** | `self.model.model` / `.is_ready` / `.save_model()` assumed `SpecializedCoderModel` wrapper. Fixed with `_unwrap_model` property and `hasattr` fallbacks. |
| **Code quality** | In-place `conf_t` mutation, dead `debate_input` code, unused `CompressionAE` tuple unpacking, `position.weight` used instead of `position(positions)`, redundant `MemoryRetriever` batch broadcasting. |

For the full fix list, see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md#appendix-bug-fix-audit).

---

## License

MIT
