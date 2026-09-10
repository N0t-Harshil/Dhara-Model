# Dhara Class Model — Agent Reference

> **Purpose:** This document is the single source of truth for AI agents working on this codebase.
> Every config field, CLI flag, architecture detail, data flow, error state, and edge case is
> documented here. If an agent needs to modify the code, read this first.

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. Entry Points](#2-entry-points)
- [3. Configuration System](#3-configuration-system)
  - [3.1 Schema Definition](#31-schema-definition)
  - [3.2 Config File Location](#32-config-file-location)
  - [3.3 All Fields Reference](#33-all-fields-reference)
- [4. CLI Commands](#4-cli-commands)
  - [4.1 full-training](#41-full-training)
  - [4.2 generate](#42-generate)
  - [4.3 test](#43-test)
  - [4.4 benchmark](#44-benchmark)
  - [4.5 download-tokenizer](#45-download-tokenizer)
  - [4.6 config-validate](#46-config-validate)
  - [4.7 info](#47-info)
- [5. Training Pipeline](#5-training-pipeline)
  - [5.1 Startup Sequence](#51-startup-sequence)
  - [5.2 Stage Execution](#52-stage-execution)
  - [5.3 FSDP Details](#53-fsdp-details)
  - [5.4 Checkpointing](#54-checkpointing)
  - [5.5 Error Handling](#55-error-handling)
  - [5.6 Synthetic Data Generation for Auxiliary Losses](#56-synthetic-data-generation-for-auxiliary-losses)
- [6. Model Architecture](#6-model-architecture)
  - [6.0 DharaModel (V4 redesign)](#60-dharamodel)
  - [6.1 ExecutiveController (RL-trained, gate-enforcing)](#61-executivecontroller)
  - [6.2 MemoryManager](#62-memorymanager)
  - [6.3 HierarchicalPlanner](#63-hierarchicalplanner)
  - [6.4 DebateSandbox](#64-debatesandbox)
  - [6.5 RecursiveReflection (merged into QA)](#65-recursivereflection-merged-into-qualityassurance)
  - [6.6 VerificationWithRepair (merged into QA)](#66-verificationwithrepair-merged-into-qualityassurance)
  - [6.7 HierarchicalDecoder (3-level)](#67-hierarchicaldecoder-3-level)
  - [6.8 WorldModel](#68-worldmodel)
  - [6.9 SymbolicToolRouter (replaces InternalToolInterface)](#69-symbolictoolrouter-replaces-internaltoolinterface)
  - [6.10 LearningController (merged into Executive)](#610-learningcontroller-merged-into-executivecontroller)
  - [6.11 CuriosityModule (merged into QA)](#611-curiositymodule-merged-into-qualityassurance)
  - [6.12 QualityAssurance](#612-qualityassurance)
  - [6.13 AuxiliaryLossComputer](#613-auxiliarylosscomputer)
  - [6.14 NSLTModel](#614-nsltmodel)
  - [6.15 Layer 1: SSMCompressionEngine](#615-layer-1-ssmcompressionengine)
  - [6.16 Layer 2: LTCRoutingLayer](#616-layer-2-ltcroutinglayer)
  - [6.17 Layer 3: LatentSandbox](#617-layer-3-latentsandbox)
  - [6.18 Layer 4: SparseOutputSynthesizer](#618-layer-4-sparseoutputsynthesizer)
  - [6.19 SSM Scan Backends](#619-ssm-scan-backends)
  - [6.20 MoE SSM Block](#620-moe-ssm-block)
- [7. Tokenizer](#7-tokenizer)
  - [7.1 Sources](#71-sources)
  - [7.2 Loader Fallback Chain](#72-loader-fallback-chain)
  - [7.3 Special Tokens](#73-special-tokens)
- [8. Data Pipeline](#8-data-pipeline)
- [9. Distributed Setup](#9-distributed-setup)
- [10. Source Files](#10-source-files)
  - [10.1 src/ directory tree](#101-src-directory-tree)
  - [10.2 Key file summaries](#102-key-file-summaries)
- [11. Test Suite](#11-test-suite)
- [12. Benchmarks](#12-benchmarks)
- [13. Known Flaky Tests & Edge Cases](#13-known-flaky-tests--edge-cases)
- [14. Appendix: Bug Fix Audit](#14-appendix-bug-fix-audit)
  - [14.1 Summary](#141-summary)
  - [14.2 Complete Fix Table](#142-complete-fix-table)
  - [14.3 Common Pitfalls & Prevention Patterns](#143-common-pitfalls--prevention-patterns)
  - [14.4 Verification](#144-verification)

---

## 1. System Overview

- **Language:** Python 3.10+
- **Framework:** PyTorch 2.6, HuggingFace Transformers 4.48+, Datasets 3.3+
- **Distributed:** FSDP full-shard (ZeRO-3), NCCL backend
- **Hardware target:** 4× A100 80GB (320 GB pooled VRAM)
- **Config:** Pydantic v2 `BaseModel` in `src/config/schema.py`
- **CLI:** `argparse` in `main.py`, 8 subcommands
- **Tests:** pytest, 109 tests, all pass

---

## 2. Entry Points

| File | Purpose |
|---|---|
| `main.py` | CLI dispatcher. Defines `ArgumentParser` with 8 subcommands. Calls into services. |
| `src/training/pipeline.py` | `TrainingPipeline` — orchestrates 3-stage training. |
| `src/models/factory.py` | `ModelFactory` — create, load, save models. |
| `src/tokenizer_trainer.py` | Tokenizer download (`download_tokenizer`) and BPE training (`train_custom_tokenizer`). |
| `src/infrastructure/distributed.py` | `DistributedSetup` — process group init, device management, FSDP config. |
| `src/config/schema.py` | All pydantic models: `Config`, `ModelArchitectureConfig`, `TrainingConfig`, etc. |
| `src/dhara/model.py` | `DharaModel`, `DharaConfig`, `DharaForCausalLM` — the core architecture (V4 redesign). |

---

## 3. Configuration System

### 3.1 Schema Definition

Defined in `src/config/schema.py` (~536 lines). The root model is `Config`. All validation
is automatic via pydantic. The schema includes **8 cross-field validators** and a `@model_validator(mode="before")`
that migrates v1 config keys to v2.

```python
# v1 → v2 auto-migration:
# data_collection.*        → data.*
# fsdp.*                   → distributed.fsdp.*
# mixed_precision           → distributed.fsdp.fp16 (or kept as mixed_precision)
# rope_scaling_factor      → rope_scaling.factor
# rope_scaling_type        → rope_scaling.type
# languages key            → removed

**Config validators (8 total):**
- `check_mode_or_validation` — at least one of pretrain/sft/instruction_tuning enabled
- `validate_nslt_config` — d_state ≤ d_model, d_hidden ≥ d_model
- `validate_moe` — num_experts ≥ top_k
- `validate_tokenizer_non_hf` — vocab_size required for custom tokenizer
- `_validate_hf_fields` — hidden_size divisible by num_attention_heads
- `_validate_mixed_precision` — accepts `mixed_precision` or `fp16` key in fsdp config
- `_validate_fsdp_config` — cpu_offload allowed only with full_shard
- `_validate_no_empty_datasets` — at least one dataset configured
```

### 3.2 Config File Location

- Default: `config.yaml` in project root.
- Read by: `main.py` → `load_config("config.yaml")`.
- The path is hardcoded; agents should not change it without updating all callers.

### 3.3 All Fields Reference

```yaml
# ── Top-Level ──
project:
  name: Dhara Class Model           # str — Project name for logging
  seed: 42                           # int — Random seed (set_seed() called in pipeline init)

model:
  name: Dhara Class Model           # str — Display name
  dtype: bfloat16                    # "bfloat16" | "float16" | "float32"
  device: auto                       # "auto" → resolved to "cuda" or "cpu" at runtime
  train_from_scratch: true           # bool — If false, would try pretrained (not implemented for NSLT)
  architecture:                      # ModelArchitectureConfig (nested)
  load_in_8bit: false                # bool — Not implemented for NSLT
  load_in_4bit: false                # bool — Not implemented for NSLT

# ── Architecture ──
model.architecture:
  model_type: dhara_v3              # "dhara_v3" | "nslt" | "llama" | "mixtral" | "qwen2_moe" | "deepseek_v2"
  hidden_size: 10240                 # int — d_model (scaled for 4× A100 80GB)
  vocab_size: 128000                 # int — Must match tokenizer.vocab_size or len(tokenizer)
  max_position_embeddings: 16384     # int — Absolute max sequence length
  rope_theta: 10000000.0             # float — RoPE base frequency
  rope_scaling:
    type: yarn                       # "yarn" | "linear" | "dynamic"
    factor: 16.0                     # float
    target_max_length: 262144        # int
    original_max_position_embeddings: 16384  # int
  attention_implementation: flash_attention_2  # "sdpa" | "flash_attention_2" | "eager"
  tie_word_embeddings: false         # bool
  attention_bias: false              # bool
  attention_dropout: 0.0             # float
  hidden_act: silu                   # "silu" | "gelu" | "relu"
  rms_norm_eps: 1e-6                # float
  initializer_range: 0.02            # float
  pretraining_tp: 1                  # int — Reserved
  mlp_bias: false                    # bool
  gradient_checkpointing: true       # bool
  use_compile: false                 # bool — torch.compile, experimental
  vocab_size: 128000                 # int — Must match tokenizer.vocab_size or len(tokenizer)

  # MoE sub-config (ignored for nslt model_type)
  moe:
    num_experts: 8
    top_k: 2
    expert_capacity: null
    shared_expert_count: 1
    shared_expert_gate: true
    norm_topk_prob: true
    output_router_logits: false
    aux_loss_coef: 0.01
    jitter_noise: 0.0
    router_aux_loss_coef: 0.001

  # Vision sub-config
  multimodal:
    vision:
      enabled: false
      vision_encoder: google/siglip-so400m-patch14-384
      image_size: 384
      patch_size: 14
      vision_hidden_size: 1152
      num_vision_layers: 27
      num_attention_heads: 16
      intermediate_size: 4304
      projection_dim: 5120
      freeze_vision_encoder: true
      tie_vision_embeddings: false
      image_token_id: 128000
      max_images_per_sample: 5

  # Dhara sub-config (ignored for non-dhara_v3 model_type).
  # Full schema: see src/dhara/model.py DharaConfig.__init__
  dhara_v3:
    d_state: 4096                    # int — SSM compressed state dimension
    d_hidden: 10240                  # int — Workspace/hidden dimension
    n_ssm_layers: 6                  # int — Number of SSM compression layers
    n_hssm_levels: 3                 # int — Hierarchical SSM levels
    n_ode_steps: 8                   # int — ODE integration steps
    n_trajectories: 7                # int — Specialist agents
    n_sim_steps: 16                  # int — Simulation steps
    sparsity_pct: 1.0                # float — Vocabulary activation %
    n_experts: 7                     # int — Specialist experts
    n_debate_rounds: 3               # int — Internal debate rounds
    enable_executive: true           # bool — Enable executive controller
    enable_world_model: true         # bool — Enable world model
    enable_tools: true               # bool — Enable symbolic tools
    enable_aux_losses: true          # bool — Enable 14 auxiliary losses
    executive_gate_threshold: 0.3    # float — Gate skip threshold
    qa_max_passes: 5                 # int — QA refinement passes
    qa_converge_threshold: 0.05      # float — QA early-stop confidence delta
    # + many more fields (see DharaConfig for complete list)

  # NSLT sub-config (ignored for non-nslt model_type)
  nslt:
    d_state: 4096                    # int — SSM compressed state dimension
    d_hidden: 10240                  # int — LTC + sandbox hidden dimension
    n_ssm_layers: 12                 # int — Number of stacked SSM blocks
    n_ode_steps: 8                   # int — RK4 integration steps per token
    solver: rk4                      # "euler" | "rk4" | "adjoint"
    n_trajectories: 8                # int — Parallel sandbox trajectories
    n_sim_steps: 16                  # int — Energy descent steps
    use_efficient_sandbox: true      # bool — Memory-efficient variant
    sparsity_pct: 1.0                # float — % of vocabulary activated

# ── Training ──
training:
  max_seq_length: 4096               # int — Sequences longer than this are truncated
  response_only_loss: true           # bool — Mask prompt tokens with -100 in labels
  ignore_data_skip: true             # bool — Resume without re-skipping data
  save_steps: 1000                   # int — Checkpoint interval
  save_total_limit: 5                # int — Keep last N checkpoints
  eval_strategy: "no"                # "steps" | "epoch" | "no"
  logging_steps: 10                  # int — Logging interval

  pretrain:
    enabled: true
    learning_rate: 2e-4
    lr_scheduler_type: cosine
    warmup_steps: 10000              # 1% of max_steps
    weight_decay: 0.1
    batch_size: 2                    # Per GPU (4× A100 80GB → eff. batch = 2 × 4 × 8 = 64)
    gradient_accumulation_steps: 8
    max_steps: 1000000               # 262B tokens at 4K seq_len (see overtrain note below)
    optimizer: adamw_fused           # "adamw" | "adamw_8bit" | "adamw_fused" | "sgd"
    data_mix: {code: 0.3, web_text: 0.3, books: 0.1, math: 0.1, science: 0.1, other: 0.1}

  sft:
    enabled: true
    learning_rate: 5e-6
    lr_scheduler_type: cosine
    warmup_steps: 1000               # 1% of max_steps
    weight_decay: 0.05
    batch_size: 1
    gradient_accumulation_steps: 4
    max_steps: 100000
    optimizer: adamw_fused
    data_mix: {sft_instructions: 0.5, conversations: 0.3, code_tasks: 0.2}

  instruction_tuning:
    enabled: true
    learning_rate: 1e-5
    lr_scheduler_type: cosine
    warmup_steps: 500                # 1% of max_steps
    weight_decay: 0.05
    batch_size: 1
    gradient_accumulation_steps: 4
    max_steps: 50000
    optimizer: adamw_fused

  alignment:
    enabled: false
    methods: [dpo, kto]
    method_configs:
      dpo:
        learning_rate: 3e-7
        beta: 0.1
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
        label_smoothing: 0.0
        loss_type: sigmoid
      orpo:
        learning_rate: 3e-7
        beta: 0.05
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
      simpo:
        learning_rate: 3e-7
        gamma: 0.5
        beta: 2.0
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
      kto:
        learning_rate: 3e-7
        beta: 0.1
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
        desirable_weight: 1.0
        undesirable_weight: 1.0

  rlhf:
    enabled: false
    learning_rate: 1e-6
    batch_size: 4
    max_steps: 10000
    ppo_epochs: 4
    kl_coef: 0.05
    cliprange: 0.2
    vf_coef: 0.1

  safety:
    enabled: false
    safety_training_steps: 5000
    harmlessness_loss_coef: 0.1
    refusal_training: true
    honesty_training: true
    constitution_training: true
    red_teaming_iters: 1000
    red_team_model: null

# ── Distributed ──
distributed:
  strategy: fsdp                   # "fsdp" | "deepspeed" | "ddp" | "none"
  fsdp:
    enabled: true
    sharding_strategy: full_shard  # "full_shard" | "hybrid_shard" | "no_shard"
    transformer_layer_cls: HierarchicalSSMStack  # FSDP wrapping target
    backward_prefetch: backward_pre   # "backward_pre" | "backward_post" | "no_prefetch"
    forward_prefetch: true
    activation_checkpointing: true
    use_orig_params: true
    sync_module_states: true
    limit_all_gathers: true
    mixed_precision: bf16          # "bf16" | "fp16" | "fp32" (also accepts legacy "fp16" key)
    cpu_offload: true              # bool — Offload optimizer states to CPU (required for single-GPU 10.55B)
  deepspeed:
    zero_stage: 3
    offload_optimizer: cpu          # null | "cpu" | "nvme"
    offload_params: null            # null | "cpu" | "nvme"

# ── Data ──
data:
  cache_dir: ./hf_cache
  streaming: true
  max_cache_gb: 200
  num_download_workers: 16
  datasets:
    - path: codeparrot/codeparrot-clean
      max_samples: 100000
      split: train
      name: null
      data_dir: null
      language: null
      category: null
      weight: 1.0
  quality:
    min_length: 30
    max_length: 500000
    deduplication:
      enabled: true
      method: minhash               # "exact" | "minhash" | "embedding"
      threshold: 0.85
    contamination:
      enabled: true
      benchmarks: [human_eval, mbpp, mmlu, gsm8k, arc, hellaswag, truthfulqa]
    quality_scoring:
      enabled: true
      method: heuristic             # "heuristic" | "perplexity" | "classifier"
    language_detection:
      enabled: true
    toxicity_filtering:
      enabled: true
  curriculum:
    enabled: false
    stages: []

# ── Tokenizer ──
tokenizer:
  source: huggingface               # "huggingface" | "custom"
  huggingface_model: Xenova/claude-tokenizer
  vocab_size: 128000                # Used only when source: custom
  max_samples: 100000               # Used only when source: custom
  type: bpe                         # "bpe" | "unigram" | "wordpiece"
  force: false                      # Force re-download / re-train
  add_prefix_space: false
  add_bos_token: true
  add_eos_token: true

# ── Output ──
output:
  model_dir: ./models/dhara
  data_dir: ./data
  checkpoint_dir: ./models/dhara/checkpoints
  log_dir: ./logs
  experiment_tracking:
    enabled: false
    provider: none                  # "wandb" | "mlflow" | "tensorboard" | "none"
    project: dhara-class-model

# ── Generation defaults ──
generation:
  max_new_tokens: 32768
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repetition_penalty: 1.05
  do_sample: true
  num_beams: 1

# ── Evaluation ──
evaluation:
  benchmarks: [mmlu, hellaswag, arc, human_eval, mbpp, gsm8k, truthfulqa, winogrande, bbh]
  benchmark_configs: {}
  automated_report: true
  report_dir: ./eval_reports
  timeout: 60
  evaluate_during_training: true
  eval_frequency: 5000
```

---

## 4. CLI Commands

Defined in `main.py`. Uses `argparse` with subparsers. All commands support `-h`.

### 4.1 full-training

```python
# main.py dispatch:
# if args.command == "full-training" → cmd_full_training(args)
#   → TrainingPipeline(cfg).initialize(fresh_start=args.fresh_start, resume_checkpoint=...)
#   → pipeline.full_training_sequence()
#   → pipeline.cleanup()  # always runs (try/finally)
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fresh-start` | `store_true` | `False` | Skip checkpoint resume; start from scratch |

**Startup sequence** (in order):
1. `load_config("config.yaml")` — validates and migrates
2. `DistributedSetup(cfg)` — reads env vars `WORLD_SIZE`, `LOCAL_RANK`, `RANK`;
   calls `torch.cuda.set_device(local_rank)` then `dist.init_process_group(backend="nccl")`
3. `ensure_tokenizer("models/tokenizer", "config.yaml")` — rank 0 only, then `dist.barrier()`
4. `ModelFactory.load_tokenizer("models/tokenizer")` — loads saved tokenizer
5. `ModelFactory.create_model(cfg, tokenizer)` or `load_model()` for resume
6. `DataPipeline(cfg, tokenizer)` — initializes data streaming
7. Run each enabled stage

**Stage execution** (`_load_stage_data`):
- Calls `self.data_pipeline.collector.stream_samples(limit=None)`
- Builds dataset from samples via `build_stage_dataset`
- Creates `Trainer` via `_build_trainer` with per-stage config
- Calls `trainer.train()`, saves checkpoint, saves tokenizer

**Cleanup** (always):
- `self.tracker.finish()`
- `gc.collect()`
- `torch.cuda.empty_cache()`
- `dist.destroy_process_group()` — prevents "already initialized" on re-run

### 4.2 reserved-training

Waits for a GPU with enough free memory, locks it via a reservation tensor,
then runs full training. Designed for shared clusters where GPUs are
occupied by other users.

```
python main.py reserved-training --fresh-start
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fresh-start` | `store_true` | `False` | Skip checkpoint resume |
| `--gpu` | `str` | `None` | Skip wait, use specific GPU index |
| `--min-free-gb` | `float` | `50.0` | Minimum free GPU memory required (GB) |
| `--reserve-gb` | `float` | `1.0` | GPU memory to hold as reservation (GB) |
| `--poll-interval` | `int` | `30` | Seconds between GPU availability checks |
| `--timeout` | `int` | `None` | Max seconds to wait (default: forever) |

**Behavior:**
1. Polls `nvidia-smi --query-gpu=index,memory.free` every `--poll-interval` seconds
2. Once a GPU has `>= --min-free-gb` GB free, sets `CUDA_VISIBLE_DEVICES` to it
3. Allocates a `torch.empty(N, device="cuda:0")` tensor of `--reserve-gb` GB to
   lock the memory — prevents other processes from grabbing it between detection
   and model initialization
4. Runs `full-training` (model allocates its own params, claiming the space)
5. On completion or crash, `finally` block frees the reservation tensor

**Reservation tensor detail:** Uses fp32, so `num_elements = reserve_gb * 1024³ / 4`.
The CUDA driver prevents other processes from reusing memory owned by the
current process's allocations. When training starts, PyTorch allocates model
weight buffers which then "own" the memory instead of the reservation tensor.

### 4.3 generate

```python
# main.py dispatch:
# if args.command == "generate" → cmd_generate(args)
#   → ModelFactory.load_model(checkpoint_path, cfg, tokenizer)
#   → model.generate(**gen_kwargs)
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--prompt` | `str` | `None` | Input prompt. Reads from stdin if omitted. |
| `--checkpoint` | `str` | `models/dhara` | Path to model directory |
| `--tokenizer` | `str` | `models/tokenizer` | Path to tokenizer directory |
| `--max-new-tokens` | `int` | `1024` | Max tokens to generate |
| `--temperature` | `float` | `0.7` | Sampling temperature; 0 = greedy |
| `--top-p` | `float` | `0.9` | Nucleus sampling threshold |
| `--top-k` | `int` | `40` | Top-k sampling |

**Model loading logic** (`ModelFactory.load_model`):
1. Check `path/config.json` for `model_type`
2. If `"nslt"` → construct `NSLTModel`, load `pytorch_model.bin` via `state_dict`
3. If other → `AutoModelForCausalLM.from_pretrained(path, ...)`
4. If no tokenizer provided → `ModelFactory.load_tokenizer()`

**Tokenizer loading** (`_try_load_tokenizer`, 3-tier fallback):
1. `AutoTokenizer.from_pretrained(path, trust_remote_code=False)`
2. `PreTrainedTokenizerFast.from_pretrained(path)`
3. Manual BPE from `vocab.json` + `merges.txt`

**pad_token fix:** If `tokenizer.pad_token is None`, calls
`tokenizer.add_special_tokens({"pad_token": "<pad>"})`.

### 4.3 test

| Flag | Type | Default | Description |
|---|---|---|---|
| `--filter` | `str` | `None` | `-k` filter passed to pytest |

```python
# Implementation:
# cmd = ["pytest", "tests/", "-v"]
# if args.filter: cmd.extend(["-k", args.filter])
# subprocess.run(cmd)
```

### 4.4 benchmark

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `str` | `models/dhara` | Model path |
| `--tokenizer` | `str` | `models/tokenizer` | Tokenizer path |
| `--benchmarks` | `str` | `None` | Comma-separated names |

If `--benchmarks` is omitted, runs the default benchmark list `human_eval,mbpp`. Runs via
`BenchmarkRunner(model, tokenizer).run_benchmarks(benchmark_list)`.

### 4.5 download-tokenizer

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-id` | `str` | `Xenova/claude-tokenizer` | HF model ID |
| `--output` | `str` | `models/tokenizer` | Output directory |
| `--force` | `store_true` | `False` | Overwrite existing |

Uses `AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)`.
Always ensures `bos_token`, `eos_token`, `unk_token`, `pad_token`, `mask_token`
are set. Saves with `tokenizer.save_pretrained(output_path)`.

### 4.6 config-validate

No flags. Reads `config.yaml` via `load_config()`. Prints a validation summary on success, error message on failure. Exits with code 0 on success, 1 on failure.

**Exception handling:** Only catches `(yaml.YAMLError, ValidationError, FileNotFoundError)`.
`KeyboardInterrupt` and `SystemExit` propagate.

### 4.7 info

No flags. Prints:
- Python version (`sys.version`)
- PyTorch version (`torch.__version__`)
- CUDA available + version
- GPU count, names, free memory per device
- World size from env
- Distributed backend (nccl/gloo)

---

## 5. Training Pipeline

### 5.1 Startup Sequence

```
main.py: cmd_full_training()
  ├─ load_config("config.yaml")           → Config object
  ├─ TrainingPipeline(cfg)
  │   ├─ DistributedSetup(cfg)
  │   │   ├─ torch.cuda.set_device(local_rank)
  │   │   └─ dist.init_process_group(backend="nccl")
  │   ├─ ExperimentTracker(cfg)
  │   └─ set_seed(42)
  ├─ pipeline.initialize(fresh_start, resume_checkpoint)
  │   ├─ ensure_tokenizer("models/tokenizer", "config.yaml")
  │   │   ├─ if source=="huggingface": download_tokenizer()
  │   │   └─ if source=="custom": train_custom_tokenizer()
  │   ├─ ModelFactory.load_tokenizer("models/tokenizer")
  │   ├─ if fresh_start or no checkpoint:
  │   │   └─ ModelFactory.create_model(cfg, tokenizer) → model
  │   ├─ else:
  │   │   ├─ ModelFactory.is_compatible(cfg, tokenizer, checkpoint)
  │   │   └─ ModelFactory.load_model(checkpoint, cfg, tokenizer)
  │   ├─ DataPipeline(cfg, tokenizer)
  │   └─ AlignmentPipeline(model, tokenizer, cfg)
  └─ pipeline.full_training_sequence()
      ├─ _load_stage_data("pretrain")     → Dataset
      ├─ run_pretrain(dataset)
      ├─ _load_stage_data("sft")          → Dataset
      ├─ run_sft(dataset)
      ├─ _load_stage_data("instruction_tuning") → Dataset
      └─ run_instruction_tuning(dataset)
```

### 5.2 Stage Execution

Each `_train_stage(dataset, stage_name, stage_cfg)`:
1. Resolves optimizer via `optim_map`:
   - `"adamw"` → `"adamw_torch"`
   - `"adamw_8bit"` → `"paged_adamw_8bit"` (or `"adamw_torch"` fallback)
   - `"adamw_fused"` → `"adamw_torch_fused"` (or `"adamw_torch"` fallback)
   - `"sgd"` → `"sgd"`
2. Creates `TrainingArguments` with per-stage hyperparams
3. Creates `Trainer(model, args, train_dataset, data_collator, tokenizer, callbacks)`
4. Calls `trainer.train()`
5. Saves model to `output_dir / stage_name`
6. Saves tokenizer to same directory
7. Returns metrics dict

**Key `TrainingArguments` settings:**
- `save_only_model=True` — only saves model weights, not optimizer state
- `ddp_find_unused_parameters=False` — avoids expensive param scan (critical for FSDP)
- `remove_unused_columns=False` — keeps all dataset columns
- `report_to` → from `experiment_tracking.provider`

**Instruction tuning stage** uses `DharaForCausalLM` wrapper which adds
causal LM loss calculation: `cross_entropy(logits[:-1], labels[1:])`.
The decoder sees the processed representation of position i and predicts token i.
Per-position decoder context is initialized from a learned `decoder_context` parameter
(one per position, up to `max_position_embeddings`).

### 5.3 FSDP Details

**In `distributed.py` `get_training_args()`:**
```python
# Always applied when distributed.strategy == "fsdp" (even single-GPU)
fsdp_args = ["full_shard", "auto_wrap"]
if fsdp_cfg.cpu_offload:
    fsdp_args.append("offload")
if not self.is_distributed:
    fsdp_args.append("no_shard")
args["fsdp"] = " ".join(fsdp_args)
# Resolve mixed_precision: accept either "mixed_precision" or legacy "fp16" key
mp_value = getattr(fsdp_cfg, "mixed_precision", None) or getattr(fsdp_cfg, "fp16", "fp32")
args["fsdp_config"] = {
    "transformer_layer_cls_to_wrap": [fsdp_cfg.transformer_layer_cls],
    "backward_prefetch": fsdp_cfg.backward_prefetch,
    "forward_prefetch": fsdp_cfg.forward_prefetch,
    "activation_checkpointing": fsdp_cfg.activation_checkpointing,
    "use_orig_params": fsdp_cfg.use_orig_params,
    "sync_module_states": fsdp_cfg.sync_module_states,
    "limit_all_gathers": fsdp_cfg.limit_all_gathers,
    "fsdp_mixed_precision": mp_value,  # accepts "mixed_precision" or legacy "fp16" key
    "cpu_offload": fsdp_cfg.cpu_offload,  # bool, moves optimizer states to CPU
}
```

**Key changes from original:**
- FSDP config is returned **even for single-GPU** (removed `self.is_distributed` gate)
- Single-GPU mode adds `no_shard` to the fsdp flags (no sharding needed with 1 GPU)
- `cpu_offload: true` moves optimizer states to CPU — the only way to fit a 10.55B
  model with AdamW on a single 80 GB GPU (params + gradients ≈ 42 GB, optimizer
  states ≈ 84 GB → offloaded to CPU RAM)

**Override in `pipeline.py` `_build_trainer()`:**
```python
fsdp_config["transformer_layer_cls_to_wrap"] = [ARCH_FSDP_LAYER_MAP.get(model_type, "LlamaDecoderLayer")]
```
This ensures the correct layer class is used based on `model_type`.

**Layer targeting:** For NSLT, `transformer_layer_cls_to_wrap = ["SSMCompressionEngine"]`
wraps each SSM block as an independent FSDP unit, enabling fine-grained memory
sharding. Previously wrapped the entire `NSLTModel` as one unit, which defeated
FSDP's memory savings.

### 5.4 Checkpointing

- **Interval:** `training.save_steps` (default 1000)
- **Retention:** `training.save_total_limit` (default 5)
- **Per-stage directories:** `models/dhara/<stage_name>/`
- **Auto-resume:** `_train_stage()` scans `models/dhara/<stage_name>/` for
  `checkpoint-*` subdirectories and passes the latest to `trainer.train(resume_from_checkpoint=...)`.
  Without this, a crash mid-stage loses all progress because `trainer.train()` starts from step 0.
- **Cross-stage resume:** Pipeline checks `models/dhara/checkpoints` for latest.
  Compares architecture spec between saved `config.json` and current config.
  On mismatch → log warning, create fresh model.
- **Save format:** `model.save_pretrained(dir, safe_serialization=True)`
  + `tokenizer.save_pretrained(dir)`
  - Saves `model.safetensors`, `config.json` (DharaConfig → PretrainedConfig),
    and `training_args.bin` (TrainingArguments state)
- **Load format:** `DharaModel.from_pretrained(dir)` reads `config.json` to
  reconstruct `DharaConfig`, then loads weights from `model.safetensors`.
- **HuggingFace integration:** Because `DharaModel` extends `PreTrainedModel`
  and `DharaConfig` extends `PretrainedConfig`, checkpoint directories are fully
  compatible with `AutoModel` and the HuggingFace Hub.

### 5.5 Error Handling

- `try/finally` ensures `cleanup()` always runs (destroy process group, empty cache)
- `_load_stage_data` catches `Exception` and returns `None` (skips stage)
- `_build_trainer` uses `getattr(stage_cfg, "warmup_steps", 500)` fallbacks for
  optional fields
- `destroy_process_group()` guards with `if dist.is_initialized()`
- If training crashes mid-stage, the next launch resumes from the latest
  checkpoint (optimizer state may be lost, but model weights are preserved)

### 5.6 Synthetic Data Generation for Auxiliary Losses

Several auxiliary losses require targets that aren't present in raw text data.
Without a generation plan, these losses will silently produce zero gradients
(because `F.cross_entropy(..., reduction="mean")` with random targets still
backpropagates — but the signal is noise). The table below shows how each
auxiliary target should be produced:

| Loss | Target Needed | Generation Strategy | Implementation |
|------|--------------|---------------------|----------------|
| Intent CE | task_type, difficulty, reasoning_type labels | **Heuristic classifier** — rule-based: count code keywords → task_type, measure token entropy → difficulty, detect logical operators → reasoning_type | `src/data/aux_labels.py` (to be created) |
| Memory MSE | original hidden state | **Autoencoder** — target is the input itself (self-supervised); `compression_loss` is built into `CompressionAE.forward()` | Already implemented in `layer3_memory.py:CompressionAE` |
| Planning CE | subgoal token predictions | **Self-supervised** — predict the next subgoal embedding via masked subgoal prediction (mask 15% of subgoals, reconstruct) | Add `_masked_subgoal_loss` to `AuxiliaryLossComputer` |
| Gate BCE | binary gate targets (1=useful, 0=skip) | **Proxy task** — gate target = 1 when `loss < running_mean(loss)` for that module, else 0. Tracks whether the module is helping. | `executive.py:ModulePerformanceTracker` already computes utilities |
| Verification BCE | binary correctness labels | **Heuristic** — for code: `exec()` in sandbox, catch exception → correctness=0. For math: compare numeric answer to extracted target. For logic: check consistency with known facts. | `validator.py` already has `exec()`-based validation; wire into `aux_labels.py` |
| Calibration ECE | accuracy per confidence bin | **Self-supervised** — compare model confidence to actual correctness on held-out batches. No external labels needed. | `AuxiliaryLossComputer.calibration_loss` uses ECE formula directly |
| Trajectory L2 | smooth step-to-step deltas | **Self-supervised** — target is the zero vector (penalize large jumps). No external labels. | Already implemented in `layer6_reasoning.py` via `dz` |
| Tool Selection CE | which tool should be used | **Heuristic** — text contains `=` or numbers → calculator; contains `def ` or `import` → python; contains `find` or `search` → search; contains `lookup` or `record` → database | Router predictions compared to rule-based tool classifier |
| Entity Prediction CE | entity/relation targets | **Self-supervised** — mask 15% of entities in the world model input, predict the masked entity. | `world_model.py:EntityExtractor` with masked reconstruction head |
| Novelty Bonus | none (negative entropy) | **Self-supervised** — maximize entropy of self-evaluation scores. No labels needed. | Already implemented: `-entropy(novelty_scores)` |
| Language Group CE | language group per token | **Heuristic** — classify programming language from file extension / docstring language. | `data/pipeline.py` already tags language; map to `n_language_groups` |
| Reflection BCE | reflection correctness | **Heuristic** — check if reflection output matches ground-truth answer availability | Rule-based: answer exists ↔ reflection should say "answered" |
| Improvement CE | did correction help | **Heuristic** — if `error_score(new) < error_score(old)` → improvement=1 | Computed on-the-fly in `QualityAssurance` passes |

**Curriculum activation:** These synthetic targets are **not all active from step 1**.
The 8-phase curriculum (see Section 6.0, Curriculum Training table) gradually
activates loss groups. For example, intent and memory losses activate in Phase 2
(after the core LM loss has stabilized), while tool loss activates in Phase 5
(after the model can reliably generate code tokens).

**Fallback behavior:** If `AuxiliaryLossComputer` receives `None` for a target,
that loss is silently skipped (returns 0.0). This means incomplete synthetic
label coverage doesn't crash training — it just means that aux loss contributes
nothing. The curriculum implementation should log a warning on first skip so
the operator knows a target generator isn't wired up yet.

---

## 6. Model Architecture

### 6.0 DharaModel

**File:** `src/dhara/model.py` (~445 lines)

```python
class DharaConfig(PretrainedConfig):
    model_type = "dhara_v3"
    def __init__(self, vocab_size=128000, d_model=10240, d_state=4096, d_hidden=10240, ...):
    # + @classmethod from_pretrained, to_dict, to_json_string, save_pretrained

class DharaModel(PreTrainedModel):
    config_class = DharaConfig
    def __init__(self, config: DharaConfig):
    # supports save_pretrained(), from_pretrained(), get_input_embeddings(), set_input_embeddings()

class DharaForCausalLM(DharaModel):
    # Wrapper with LM head for causal language modeling loss
    # forward() returns CausalLMOutputWithPast
```

**HF PreTrainedModel integration benefits:**
- Checkpoints save/load seamlessly via `save_pretrained()`/`from_pretrained()`
- HuggingFace Trainer works natively (no custom training loop needed)
- Compatible with `AutoModel` registry, Hub push, and `pipeline()`
- `DharaConfig` includes Pydantic-style validation but inherits from `PretrainedConfig`

A **non-Transformer** LLM with **workspace-centric architecture** (V4 redesign). All modules communicate through a central `CognitiveWorkspace` dict-based hub rather than calling each other directly. The **Executive Controller** enforces module gates (actually skips modules below threshold) and is trained via REINFORCE with reward = accuracy − λ·compute. Neural tool approximations are replaced with **symbolic tools** (real `eval()`/`exec()`) with learned routing. Reflection, verification, and curiosity are merged into a single **QualityAssurance** multi-pass module. The model supports **14 auxiliary training losses** beyond next-token prediction.

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

**Architecture components (V4 redesign, 14 unique modules with 3 merged):**

| # | Component | File | Purpose | V4 Change |
|---|---|---|---|---|
| 1 | Tokenizer | `layer1_tokenizer.py` | Token embedding with semantic metadata | Unchanged |
| 2 | Embedding | `layer2_embedding.py` | RoPE + task-context embedding + context adapter | Unchanged |
| 3 | MemoryManager | `layer3_memory.py` | Memory with forget/compress/retrieve/prioritize | Unchanged |
| 4 | ExecutiveController | `executive.py` | Meta-controller: gate enforcement + merged LearningController + RL reward | **Gate enforcement**, **merged LC**, **RL** |
| 5 | WorldModel | `world_model.py` | Entity extraction, relation networks, event modeling, cause-effect | Unchanged |
| 6 | Intent | `layer4_intent.py` | Task type, difficulty, reasoning type, confidence prediction | Unchanged |
| 7 | HierarchicalPlanner | `layer5_planner.py` | Goal tree + dependency graph + execution graph + cost estimation | Unchanged |
| 8 | ODE Reasoning | `layer6_reasoning.py` | Neural ODE over multi-domain dynamics | Unchanged |
| 9 | CognitiveWorkspace | `workspace.py` | **Central dict-based communication hub** (read/write/read_all/clear API) | **Redesigned as central bus** |
| 10 | SymbolicToolRouter | `tools.py` | **Symbolic** calculator/python/search/db with learned routing | **Replaced neural tools** |
| 11 | DebateSandbox | `layer8_specialists.py` | N-round debate + cross-examination + repair + consensus | Unchanged |
| 12 | QualityAssurance | `quality_assurance.py` | **Merged** reflect → verify → self-evaluate → correct → converge | **Merged from 3 modules** |
| 13 | AuxiliaryLossComputer | `losses.py` | 14 configurable auxiliary training losses | **New** |
| 14 | HierarchicalDecoder | `layer11_decoder.py` | 3-level: semantic → language → token | Unchanged |

Old separate ReflectionModule, VerificationWithRepair, CuriosityModule, and LearningController are now **backward-compatible wrappers** that re-export from the merged modules.

**Config key** (in `config.yaml`): `model.architecture.model_type: dhara_v3`

**DharaConfig additional fields (V4):**

| Field | Default | Description |
|---|---|---|
| `n_experts` | 7 | Number of specialist experts in debate sandbox |
| `n_debate_rounds` | 3 | Rounds of debate/cross-examination |
| `max_refinement_passes` | 5 | Max QA refinement passes |
| `max_repair_iters` | 3 | Max QA repair iterations |
| `max_entities` | 64 | Max entities for world model |
| `n_relation_types` | 16 | Relation types in world model |
| `max_events` | 32 | Max events tracked |
| `n_tool_types` | 4 | Number of symbolic tool types (calc, python, search, db) |
| `enable_executive` | true | Enable Executive Controller |
| `enable_world_model` | true | Enable World Model |
| `enable_tools` | true | Enable Tool Interface |
| `enable_aux_losses` | true | Enable auxiliary training losses |
| `executive_gate_threshold` | 0.3 | Gate value below which modules are skipped |
| `qa_max_passes` | 5 | Max QA multi-pass iterations |
| `qa_converge_threshold` | 0.05 | QA early-stop confidence change threshold |
| `loss_weights` | null | Dict of per-loss weights (overrides defaults in AuxiliaryLossComputer) |

**Forward pass (data flow with workspace-as-hub):**

```
input_ids (B,T)                         — token IDs
  ↓
Token Embedding + Semantic Metadata     — x: (B, T, d_model) → workspace.write("memory", ...)
  ↓
AdaptiveSemanticEmbedding + RoPE        — x: (B, T, d_model)
  ↓
MemoryManager                           — mem_out: (B, T, d_model), mem_state, mem_meta
  ↓
ExecutiveController                     — gates → apply_gates() actually skips modules < 0.3
  ↓
IntentUnderstanding(mem_out)            — workspace.write("intent", ...)
  ↓
HierarchicalPlanner(mem_out, intent)    — workspace.write("plan", ...)
  ↓
ODE Reasoning(h_ctx, z)                 — workspace.write("reasoning", ...)
  ↓
CognitiveWorkspace(plan, mem, reason)   — workspace.write("workspace", fused_repr)
  ↓
WorldModel(mem_out)                     — workspace.write("world_model", ...)
  ↓
DebateSandbox(workspace_repr)           — consensus → workspace.write("specialists", ...)
  ↓
SymbolicToolRouter(consensus)           — symbolic exec → workspace.write("tools", fused)
  ↓
QualityAssurance(workspace, consensus)  — multi-pass: reflect → verify → eval → correct
  ↓
HierarchicalDecoder(corrected_h)        — logits: (B, T, V) or loss
  ↓
AuxiliaryLossComputer(module_outputs)   — 14 auxiliary losses summed with L_nll
```

**Key design decisions (V4 changes):**
- **Workspace as central hub**: All modules write outputs to workspace, no module calls another directly. `workspace.read(key)`, `workspace.write(key, value)`, `workspace.read_all()`, `workspace.clear()`.
- **Executive enforces gates**: `apply_gates()` returns which modules to skip; `forward()` uses `torch.where` to bypass skipped modules.
- **RL-trained executive**: REINFORCE with reward = −task_loss − λ·compute_penalty. Running reward buffer for baseline subtraction.
- **Symbolic tools**: Calculator uses `ast.parse` + `eval` in restricted namespace. Python uses `exec()` in sandboxed locals. Search uses embedding retrieval. Database uses structured access. No neural approximation.
- **Merged QualityAssurance**: Unified multi-pass: reflect (answered/constraints/contradiction) → verify (syntax/compilation/runtime/math/logic) → self-evaluate (usefulness/novelty/uncertainty) → correct → re-verify until convergence.
- **14 auxiliary losses**: Intent CE, memory reconstruction MSE, gate BCE, verification BCE, calibration ECE, trajectory smoothness L2, tool selection CE, novelty bonus, entity prediction CE. Each configurable per weight.
- **LearningController merged into Executive**: `ModulePerformanceTracker` maintains per-module LR, performance history, and predicted performance. All part of `ExecutiveController`.

### 6.1 ExecutiveController

**File:** `src/dhara/executive.py`

```python
class ExecutiveController(nn.Module):
    def __init__(self, d_model, d_hidden, n_modules=8, gate_threshold=0.3):
```

The meta-controller that decides which modules to activate and enforces those decisions:

- **Module gating**: 8 independent gates (memory, planner, intent, reasoning, sandbox, reflection, verification, decoder) — each produces a `[0, 1]` activation score per sample. Unlike V3, these gates are **actually enforced** during the forward pass.
- **apply_gates()**: Returns `Dict[str, bool]` — `True` means skip this module. Used by `model.forward()` to bypass modules when their gate < threshold.
- **Gate enforcement**: Model uses `torch.where(skip_mask, identity_output, module_output)` to conditionally activate modules. When a module is skipped, a gated residual connection preserves the input.
- **RL training**: REINFORCE formulation with reward = `−task_loss − 0.01·n_active`. Running reward buffer (capacity 100) tracks recent rewards for baseline calculation.
- **Value head**: Separate `nn.Linear(d_hidden, 1)` predicts expected return from state — can be used for actor-critic extensions.
- **Compute budget allocator**: Neural network head produces 8 normalized budget weights summing to 1.0 per sample.
- **Depth predictor**: Predicts reasoning depth multiplier `[0, 1]` per sample.
- **Merged LearningController**: `ModulePerformanceTracker` maintains performance history (100-step rolling buffer) per module. Adjusts module-specific LR: increases when performance < 0.5, decreases when > 0.8. `meta_learner` predicts meta-weights from pooled hidden + learned module importance.
- **Previous (V3)**: `should_skip_module()` existed but was never called. Now `apply_gates()` is called in `model.forward()`.

### 6.2 MemoryManager

**File:** `src/dhara/layer3_memory.py`

```python
class MemoryManager(nn.Module):
    def __init__(self, d_model, d_state, n_hssm_layers, n_hssm_levels,
                 working_mem_capacity, n_semantic_concepts, max_episodes):
```

Replaces the passive `HierarchicalMemoryEngine` with active memory management:

- **ForgetGate**: Learned decay — `memory * (1 - sigmoid(gate(concat(memory, age))))`. Older memories decay more.
- **CompressionAE**: Autoencoder that compresses/deduplicates memories; returns `compression_loss` as training signal.
- **PriorityScorer**: Scores each memory slot on importance `[0, 1]`.
- **MemoryRetriever**: Multi-head attention-based retrieval weighted by priority.
- **Returns 3-tuple**: `(fused_memory, new_state, meta_dict)` where `meta_dict` contains importance, compression_loss, priority.

**Backward compatible:** `HierarchicalMemoryEngine` inherits from `MemoryManager`.

### 6.3 HierarchicalPlanner

**File:** `src/dhara/layer5_planner.py`

```python
class HierarchicalPlanner(nn.Module):
    def __init__(self, d_model, max_subgoals=64, max_depth=4):
```

Extends `GlobalPlanner` with hierarchical goal decomposition:

- **Goal tree**: Depth-weighted hierarchical goals — each depth level uses a different subgoal mask.
- **Dependency graph**: `max_subgoals × 2` matrix — predecessor and successor dependencies.
- **Execution graph**: Learned execution plan embeddings.
- **Cost estimation**: `F.softplus(cost_head(h))` per subgoal; `total_cost` summed across subgoals.
- **Per-subgoal details**: SubgoalNode produces cost, depth, type logits for each subgoal.
- **Depth weights**: Softmax over `max_depth` — allows variable-depth planning per sample.

**Backward compatible:** `GlobalPlanner` inherits from `HierarchicalPlanner`.

### 6.4 DebateSandbox

**File:** `src/dhara/layer8_specialists.py`

```python
class DebateSandbox(nn.Module):
    def __init__(self, d_hidden, n_experts=7, n_debate_rounds=3):
```

Replaces independent expert proposals with interactive debate:

- **Phase 1 — Propose**: Each expert produces an initial proposal + confidence.
- **Phase 2 — Debate (N rounds)**: Each expert sees others' average proposal, produces a repair, applies a learned repair gate: `new_prop = (1 - gate) * own_prop + gate * repaired`.
- **Phase 3 — Cross-examination**: Cross-attention across all proposals generates a critique signal.
- **Phase 4 — Repair**: Experts repair based on cross-examination signal.
- **Phase 5 — Consensus**: Weighted by `softmax(confidence * critic_score)`, concatenation + linear projection.

**Backward compatible:** `SpecialistSandbox` inherits from `DebateSandbox`.

### 6.5 RecursiveReflection (merged into QualityAssurance)

**V4 change:** This module is now merged into `QualityAssurance`. The old `layer9_reflection.py` is a **backward-compatible wrapper** that inherits from `QualityAssurance`:

```python
class RecursiveReflection(QualityAssurance):
    def __init__(self, d_hidden, max_refinement_passes=5, converge_threshold=0.05):
        super().__init__(d_hidden, max_passes=max_refinement_passes, converge_threshold=converge_threshold)
```

See **6.12 QualityAssurance** for the new implementation.

### 6.6 VerificationWithRepair (merged into QualityAssurance)

**V4 change:** This module is now merged into `QualityAssurance`. The old `layer10_verification.py` is a **backward-compatible wrapper** that inherits from `QualityAssurance`. The new Verifier inside QA checks syntax, compilation, runtime correctness, math consistency, and logic consistency in each pass.

See **6.12 QualityAssurance** for the new implementation.

### 6.7 HierarchicalDecoder (3-level)

**File:** `src/dhara/layer11_decoder.py`

```python
class HierarchicalSparseDecoder(nn.Module):
    def __init__(self, d_hidden, vocab_size, d_model, n_language_groups=8, ...):
```

Replaces single-level sparse decoder with 3-level hierarchy:

- **Level 1 — SemanticDecoder**: Concept routing — projects hidden state to learned concept embeddings (`n_semantic_concepts=4096`). Weighted sum produces concept-aware output.
- **Level 2 — LanguageDecoder**: Language group routing — classifies output into language groups, adds language context bias.
- **Level 3 — TokenDecoder**: Final token prediction with adaptive top-k sparsity and learned temperature.
- **Fusion gate**: `Linear(d_hidden*3, d_hidden)` fuses original h + semantic_out + language_context.
- **hidden_to_vocab**: Handles both 2D (batch, d_hidden) and 3D (batch, seq, d_hidden) inputs.

### 6.8 WorldModel

**File:** `src/dhara/world_model.py`

```python
class WorldModel(nn.Module):
    def __init__(self, d_model, max_entities=64, n_relation_types=16, max_events=32):
```

Constructs an internal world representation before reasoning:

- **EntityExtractor**: Learns `max_entities` entity embeddings, classifies input into entity slots.
- **RelationNetwork**: Scores pairwise entity relations across `n_relation_types` — produces relation embeddings.
- **EventModel**: Classifies and tracks events with temporal encoding.
- **CauseEffectModel**: Multi-head attention over events to model cause-effect.
- **Fusion**: Concatenates and projects all world signals into `world_state`.

### 6.9 SymbolicToolRouter (replaces InternalToolInterface)

**File:** `src/dhara/tools.py`

```python
class SymbolicToolRouter(nn.Module):
    def __init__(self, d_hidden):
```

**V4 change:** Previously used neural networks to approximate tool behavior (Calculator was an MLP, Python was learned op embeddings, etc.). Now uses **true symbolic execution** with a learned router:

- **SymbolicCalculator**: Extracts numeric tokens via regex, parses expressions via `ast.parse`, evaluates in restricted namespace via `eval()` with only arithmetic operators allowed. Returns scalar result projected to hidden dim.
- **SymbolicPythonExecutor**: Executes code in sandboxed `exec()` environment with `__builtins__` restricted. Captures `result` or `output` variables from the executed namespace. Returns projected value.
- **SymbolicSearch**: Embedding-based content-addressable memory. Learns query/key/value projections on a `Parameter` memory bank. Attention-based retrieval — unchanged from V3 but now properly symbolic (the memory acts as a true retrieval store, not a neural computation).
- **SymbolicDatabase**: Slot-based structured access. Learns query projection and access gates over `Parameter` record bank. Weighted sum retrieval — returns the weighted combination of accessed records.
- **Router**: `Linear(d_hidden, 4)` produces logits → `softmax` → weighted combination of all 4 tool outputs. The router **learns** which tool to invoke per sample.
- **set_context()**: Allows setting `hidden_text` and `code_str` before forward, enabling context-aware symbolic execution.
- **Backward compatible:** `InternalToolInterface` inherits from `SymbolicToolRouter`.

### 6.10 LearningController (merged into ExecutiveController)

**V4 change:** `LearningController` and `ModulePerformanceTracker` are now merged directly into `ExecutiveController`. The old `learning_controller.py` file is a **backward-compatible re-export** that imports `ModulePerformanceTracker` and `MODULE_NAMES` from `executive.py`.

See **6.1 ExecutiveController** for the merged implementation. Key merged features:
- `ModulePerformanceTracker` maintains per-module performance history and adaptive LR
- `meta_learner` predicts meta-weights from pooled hidden + learned module importance
- `module_importance` learnable parameter tracks which modules matter most

### 6.11 CuriosityModule (merged into QualityAssurance)

**V4 change:** Curiosity and self-evaluation are now part of `QualityAssurance`. The old `curiosity.py` is a **backward-compatible standalone wrapper** that provides `SelfEvaluation` and `CuriosityModule` using the same internal components. The new SelfEvaluator inside QA checks usefulness, novelty, and uncertainty as part of the multi-pass loop.

### 6.12 QualityAssurance

**File:** `src/dhara/quality_assurance.py`

```python
class QualityAssurance(nn.Module):
    def __init__(self, d_hidden, max_passes=5, converge_threshold=0.05):
```

Merges RecursiveReflection, VerificationWithRepair, and CuriosityModule into a single unified multi-pass quality assurance pipeline:

**Sub-components:**
- **Reflector**: 4 heads — answered check, constraint check, contradiction detection, rethink gate
- **Verifier**: 5 heads — syntax correctness, compilation probability, runtime correctness, math consistency, logic consistency
- **SelfEvaluator**: 3 heads — usefulness, novelty, uncertainty
- **Corrector**: Error projection → correction MLP → correction gate
- **Confidence head**: Tracks per-pass confidence for convergence detection

**Multi-pass loop** (up to `max_passes`):
1. Add pass-specific embedding (learned per-pass positional encoding)
2. Reflect: check answered/constraints/contradiction → needs_correction mask
3. Verify: compute error scores from all 5 verifiers → max_error signal
4. Self-evaluate: usefulness, novelty, uncertainty scores
5. Correct: corrector(h, error_signal) → gated correction
6. Refine: `refine_proj(concat(h, workspace, goal_context)) → refined + correction`
7. Replace current with refined where needs_reconsider is True
8. Track confidence → early stop if `conf_change < threshold && confidence > 0.7`

**Returns:**
- `corrected_h`: Final corrected hidden state `(B, d_hidden)`
- `final_confidence`: Scalar confidence after last pass
- `confidence_trace`: Tensor of all per-pass confidences
- `n_passes`: Number of passes actually executed
- `verify_scores`: Full trace of all verification scores `(B, n_passes, 5)`
- `eval_scores`: Full trace of all evaluation scores `(B, n_passes, 3)`
- Per-module dicts for the last pass: `verify_out`, `eval_out`, `reflect_out`

### 6.13 AuxiliaryLossComputer

**File:** `src/dhara/losses.py`

```python
class AuxiliaryLossComputer(nn.Module):
    def __init__(self, weights=None):
```

Computes 14 configurable auxiliary losses to supplement the primary next-token prediction loss:

**Loss functions:**
| Loss | Function | Default Weight | Input Shape |
|---|---|---|---|
| intent | Cross-entropy on task_type, difficulty, reasoning_type | 0.05 | logits + target indices |
| memory | Reconstruction MSE | 0.01 | compressed ≈ original |
| planning | Subgoal prediction CE | 0.05 | subgoal_logits + targets |
| gate | Gate BCE | 0.01 | gate values + binary targets |
| verification | Verification correctness BCE | 0.02 | verify scores + binary targets |
| calibration | ECE | 0.005 | confidence + accuracy |
| trajectory | Trajectory smoothness L2 | 0.001 | step-to-step delta |
| tools | Tool selection CE | 0.05 | route_weights + tool labels |
| novelty | Novelty bonus (negative entropy) | 0.001 | novelty scores |
| entity | Entity prediction CE | 0.01 | entity_logits + targets |

**Usage in model:**
```python
module_outputs = {"intent": intent, "executive": exec_decision, "tools": tool_out, ...}
aux_losses = self.loss_computer(module_outputs, aux_targets)
total_aux = self.loss_computer.total_loss(aux_losses)
loss = lm_loss + total_aux
```

Each loss is only computed when the relevant module output is available in `module_outputs`, making the system robust to module enable/disable toggling.

**Config key:** All weights configurable via `DharaConfig.loss_weights` dict. Default weights used when a key is missing.

### 6.14 NSLTModel

**File:** `src/nslt/model.py` (~694 lines)

```python
class NSLTModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_hidden, n_ssm_layers,
                 max_seq_len, rope_base, sparsity_pct, n_ode_steps=8,
                 n_trajectories=8, n_sim_steps=16, use_efficient_sandbox=False,
                 dtype=torch.bfloat16):
```

**Components:**
- `TokenEmbedding(vocab_size, d_model)` — learned embeddings, scaled by `sqrt(d_model)`
- `RotaryPositionEncoding(d_model, max_seq_len, rope_base)` — RoPE
- `SSMCompressionEngine` × `n_ssm_layers` — stacked SSM blocks
- `LTCRoutingLayer(d_model, d_hidden, d_state, ...)` — neural ODE routing
- `LatentSandbox(d_hidden, d_state, n_trajectories, n_sim_steps)` — parallel reasoning
- `SparseOutputSynthesizer(d_hidden, vocab_size, sparsity_pct)` — ultra-sparse output

**Forward pass:**
1. `TokenEmbedding(input_ids)` → `[B, T, d_model]`
2. `RotaryPositionEncoding(x)` — applies RoPE in place
3. For each SSM layer: `x, h = ssm(x)` — compresses to `h [B, d_state]`
4. `LTCRoutingLayer(x, h)` → `[B, T, d_hidden]`
5. `LatentSandbox(z)` → `[B, d_hidden]` — select best trajectory
6. `SparseOutputSynthesizer(z)` → `[B, T, V]` logits (sparse)

**MoE variant:** `MoENSLTModel` in the same file wraps `NSLTModel` and adds
MoE routing on the SSM layers.

**meta device guard:** `self.to(device, dtype)` is guarded with
`if device.type != "meta"` to prevent RuntimeError when model is on meta device.

### 6.15 Layer 1: SSMCompressionEngine

**File:** `src/nslt/layer1_ssm.py` (~151 lines)

**Math:** `h_t = exp(∆_t·A)·h_{t-1} + (exp(∆_t·A)-I)/A·B_t·x_t`
**Output:** `y_t = C_t·h_t`

**Components:**
- `in_proj`: `Linear(d_model, d_inner*2)` — split into `x_inner` and `x_gate`
- `conv1d`: depthwise `Conv1d(d_inner, d_inner, kernel=4, padding=3)` — local mixing
- `SiLU` activation
- `dt_proj`: `Linear(d_inner, dt_rank)` → `F.softplus(...)` for positivity
- `A_log`: `Parameter [d_state]` — `A = -exp(A_log)` (negative diagonal, stable)
- `B_proj`, `C_proj`: `Linear(d_inner, dt_rank)` — input-dependent
- `out_proj`: `Linear(d_inner, d_model)` — project back
- `LayerNorm(d_model)` — pre-normalization

**Forward:**
1. `norm(x)` → `in_proj(x)` → chunk → `x_inner`, `x_gate`
2. `conv1d(x_inner.permute(...))` → slice to seq_len → permute back → SiLU
3. Compute `delta` from `dt_proj(x_conv)` + softplus + `dt_bias`
4. Compute `A = -exp(A_log)`, `B = B_proj(x_conv)`, `C = C_proj(x_conv)`
5. `selective_scan(x_conv, delta, A, B, C, dt_rank)` → `y, h_final`
6. `y * SiLU(x_gate)` → `out_proj(y)` → output

### 6.16 Layer 2: LTCRoutingLayer

**File:** `src/nslt/layer2_ltc.py` (~200 lines)

**Math:** `dz/dt = -[w_tau·σ(w_tau·z + b_tau)]·z + f(z, I(t))`

- **Liquid time-constant:** `τ(z) = 1 / (w_tau·σ(w_tau·z + b_tau))`
- **Solvers:** Euler (1st order), RK4 (4th order, default), adjoint (memory-efficient)
- **Input dynamics:** MLP that takes concatenated `[z, I(t)]` and produces `dz/dt`
- Runs `n_ode_steps` integration steps per input token

**Solver choice:**
| Solver | Order | Gradient Memory | Speed |
|---|---|---|---|
| `euler` | O(h) | Full (stores all intermediates) | Fastest |
| `rk4` | O(h⁴) | Full (stores 4 intermediates per step) | Fast |
| `adjoint` | O(h⁴) | O(1) (recomputes forward) | Slower |

### 6.17 Layer 3: LatentSandbox

**File:** `src/nslt/layer3_sandbox.py` (~300 lines)

- Creates K copies of state `z` → `z_1 ... z_K`
- Each evolves via gradient descent on `E(z)` for `n_sim_steps` iterations:
  ```python
  z_i = z_i - lr * torch.autograd.grad(E(z_i).sum(), z_i)[0]
  ```
- Selects trajectory with lowest final energy
- Energy = `||z - decoder(encoder(z))||² + λ₂·R(z)`

**Energy function network:**
- Encoder: `Linear(d_hidden, d_latent*2)` → SiLU → `Linear(d_latent*2, d_latent)`
- Decoder: `Linear(d_latent, d_latent*2)` → SiLU → `Linear(d_latent*2, d_hidden)`
- R-net: `Linear(d_hidden, d_hidden//4)` → SiLU → `Linear(d_hidden//4, 1)`

**Efficient variant:** `LatentSandboxEfficient` — batches all K trajectories
in a single forward pass (lower memory overhead).

**MCTS variant** (`src/nslt/mcts_sandbox.py`): Uses Monte Carlo Tree Search
instead of gradient descent. Each node is a latent state. UCB-based selection,
expansion, simulation, back-propagation. More exploration at higher compute cost.

### 6.18 Layer 4: SparseOutputSynthesizer

**File:** `src/nslt/layer4_output.py` (~236 lines)

**Adaptive top-k gating:**
```python
k = min_k + (max_k - min_k) * entropy / log(vocab_size)
```
Where `entropy = -Σ p_i·log(p_i)` of the gate distribution.

**Components:**
- `hidden_proj`: `Linear(d_hidden, d_model)` — project hidden to model dim
- `output_embedding`: `Parameter [vocab_size, d_model]` — tied embedding, not tied by default
- `logit_temperature`: `Parameter [1]` — learned temperature, abs() for positivity
- `gate`: MLP that produces logits over vocabulary
- `SparseGatingUnit`: adaptive top-k selector

**Forward (training):**
1. `h = hidden_proj(x)` — `[B, d_model]`
2. `gate_values, top_indices, _ = gate(h)` — select top-k vocab entries
3. `target_logit = target_emb · h / temp` — logit for target token
4. `selected_logits = selected_embs · h / temp` — logits for top-k
5. `all_logits = concat([selected_logits, target_logit.unsqueeze(1)])`
6. `log_probs = log_softmax(all_logits)` — normalization over [k+1]
7. Return `log_probs[:, -1]` — log prob of target token

**Forward (inference):**
1. Same gate selection (no target token available)
2. Score only the top-k selected vocabulary entries
3. Return logits over [k] — never materialize [V]

### 6.19 SSM Scan Backends

**File:** `src/nslt/ssm_scan.py` (~445 lines)

```python
def selective_scan(x, delta, A, B, C, dt_rank, mode=None, use_autograd=False):
```

| Mode | Function | Hardware | Description |
|---|---|---|---|
| `sequential` | `selective_scan_sequential` | CPU/CUDA | Python for-loop, correct gradients |
| `vectorized` | `selective_scan_vectorized` | CUDA | Parallel prefix scan via cumsum |
| `triton` | `selective_scan_triton` | CUDA+Triton | Custom Triton kernel (fastest) |
| `jit` | `selective_scan_jit` | CPU | TorchScript JIT-compiled loop |
| `auto` | dispatches | Any | Triton→vectorized→sequential fallback |

**Dispatch logic (`auto` mode):**
```python
if x.is_cuda and _HAS_TRITON:
    return selective_scan_triton(...)
if x.is_cuda:
    return selective_scan_vectorized(...)
return selective_scan_sequential(...)
```

**Custom autograd Function** (`SSMScanFunction`):
- Forward saves `h_seq` for non-Triton modes (recomputed for Triton)
- Backward recomputes `h_t` per timestep for correct B/C/delta gradients
- Gradient shapes: `grad_x [B,T,d_inner]`, `grad_delta [B,T,dt_rank]`,
  `grad_A [d_state]`, `grad_B [B,T,dt_rank]`, `grad_C [B,T,dt_rank]`

### 6.20 MoE SSM Block

**File:** `src/nslt/moe_ssm.py` (~200 lines)

Wraps `SSMCompressionEngine` with mixture-of-experts routing:
- Router network selects top-k experts per token
- Each expert is an independent `SSMCompressionEngine`
- Output is weighted sum of selected experts' outputs
- Auxiliary load-balancing loss encourages uniform expert usage

---

## 7. Tokenizer

### 7.1 Sources

| `source` | Behavior | File created |
|---|---|---|
| `huggingface` | `AutoTokenizer.from_pretrained(model_id)` | `tokenizer.json`, `tokenizer_config.json` |
| `custom` | Train `ByteLevelBPETokenizer` on dataset corpus | Same |

### 7.2 Loader Fallback Chain

In `ModelFactory._try_load_tokenizer(path)`:

```python
try:
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)
except:
    try:
        return PreTrainedTokenizerFast.from_pretrained(path)
    except:
        # Manual BPE from raw vocab.json + merges.txt
        if (path/"vocab.json").exists() and (path/"merges.txt").exists():
            backend = ByteLevelBPETokenizer(str(vocab), str(merges))
            return PreTrainedTokenizerFast(
                tokenizer_object=backend._tokenizer,
                bos_token="<s>", eos_token="</s>",
                unk_token="<unk>", pad_token="<pad>",
            )
        raise RuntimeError(f"Unable to load tokenizer from {path}")
```

### 7.3 Special Tokens

Always added if missing after loading:
- `bos_token`: `<s>`
- `eos_token`: `</s>`
- `unk_token`: `<unk>`
- `pad_token`: `<pad>`
- `mask_token`: `<mask>`

**pad_token fix (critical):** If `tokenizer.pad_token is None` after loading,
`tokenizer.add_special_tokens({"pad_token": "<pad>"})` is called. This ensures
the data collator and loss computation have a valid padding token ID.

---

## 8. Data Pipeline

**File:** `src/data/pipeline.py` (~241 lines)

`DataPipeline` wraps `MassiveDataCollector` from `src/data/streaming.py`.

**Key flows:**
1. `DataPipeline(cfg, tokenizer)` → creates `MassiveDataCollector(cfg.data.datasets)`
2. `collector.stream_samples(limit=None)` → yields dicts with keys: `language`,
   `instruction`, `input`, `output`, etc.
3. `build_stage_dataset(samples)` → tokenizes and returns HuggingFace `Dataset`

**Quality filtering** (applied per-sample):
1. Length check: `30 ≤ len(content) ≤ 500000`
2. Low-quality marker check (`is_high_quality_content`): rejects samples
   containing "lorem ipsum", "todo: add code", "your code here", etc.
3. Contamination check: rejects samples matching benchmark patterns
4. Deduplication (minhash with configurable threshold)

---

## 9. Distributed Setup

**File:** `src/infrastructure/distributed.py` (~108 lines)

`DistributedSetup` class:

**mixed_precision key fix:** The config schema accepts both `mixed_precision` and
legacy `fp16` key names under `distributed.fsdp`. The `get_training_args()` method
resolves with `getattr(fsdp_cfg, "mixed_precision", None) or getattr(fsdp_cfg, "fp16", "fp32")`,
ensuring backward compatibility with configs using the old key name.

```python
class DistributedSetup:
    def __init__(self, cfg):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = self.world_size > 1
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
```

**Critical details:**
- `torch.cuda.set_device(local_rank)` is called BEFORE `init_process_group`
  to ensure NCCL buffers are allocated on the correct GPU
- `is_main_process()` checks `self.rank == 0` (global rank), NOT `local_rank == 0`
- `auto_device()` returns `cuda:{local_rank}` or `cpu`
- `get_training_args()` returns a dict, not a `TrainingArguments` object —
  the actual `TrainingArguments` is constructed in `pipeline.py` `_build_trainer()`

---

## 10. Source Files

### 10.1 src/ directory tree

```
src/
├── __init__.py
├── dhara/                 # Dhara — cognitive architecture (V4 redesign)
│   ├── __init__.py
│   ├── model.py               # ~490 lines — DharaModel, DharaConfig, DharaForCausalLM
│   ├── hssm.py                # ~85 lines — HierarchicalSSM, HierarchicalSSMStack (vectorized, state-dim fix)
│   ├── workspace.py           # ~100 lines — CognitiveWorkspace (dict-based central hub)
│   ├── executive.py           # ~150 lines — ExecutiveController (RL gates + merged LearningController + ModulePerformanceTracker)
│   ├── quality_assurance.py   # ~180 lines — QualityAssurance (merged reflect/verify/eval/correct)
│   ├── tools.py               # ~180 lines — SymbolicToolRouter (symbolic calc/python/search/db with learned routing)
│   ├── losses.py              # ~150 lines — AuxiliaryLossComputer (14 configurable auxiliary losses)
│   ├── world_model.py         # ~110 lines — WorldModel, EntityExtractor, RelationNetwork, CauseEffectModel
│   ├── learning_controller.py # ~5 lines — Backward-compatible re-export from executive.py
│   ├── curiosity.py           # ~75 lines — Backward-compatible CuriosityModule, SelfEvaluation wrapper
│   ├── layer1_tokenizer.py    # ~38 lines — IntelligentTokenizer, SemanticMetadataEmbedding
│   ├── layer2_embedding.py    # ~87 lines — AdaptiveSemanticEmbedding, RoPE, ContextAdapter
│   ├── layer3_memory.py       # ~230 lines — MemoryManager, WorkingMemory, SemanticMemory, etc.
│   ├── layer4_intent.py       # ~40 lines — IntentUnderstanding, AdaptiveDifficultyRouter
│   ├── layer5_planner.py      # ~120 lines — HierarchicalPlanner, SubgoalNode, GoalGraphAttention
│   ├── layer6_reasoning.py    # ~57 lines — AdaptiveContinuousReasoning, ODETick
│   ├── layer7_workspace.py    # ~2 lines — Backward-compatible re-export from workspace.py
│   ├── layer8_specialists.py  # ~110 lines — DebateSandbox, SpecialistExpert, SpecialistCritic
│   ├── layer9_reflection.py   # ~20 lines — Backward-compatible RecursiveReflection QA wrapper
│   ├── layer10_verification.py# ~30 lines — Backward-compatible VerificationWithRepair QA wrapper
│   └── layer11_decoder.py     # ~145 lines — HierarchicalDecoder (3-level: semantic → language → token)
├── model.py                   # SpecializedCoderModel (higher-level wrapper)
├── trainer.py                 # ModelTrainer (legacy trainer, kept for compatibility)
├── generator.py               # Code generation utilities
├── validator.py               # Python code validation
├── tokenizer_trainer.py       # Tokenizer download + BPE training
├── dataset.py                 # Dataset utilities
├── benchmark.py               # Legacy benchmark entry
├── massive_data_collector.py  # Legacy data collector
├── knowledge_graph.py         # Knowledge graph utilities
│
├── config/
│   ├── __init__.py
│   └── schema.py              # ~536 lines — ALL pydantic models (8 cross-field validators)
│
├── models/
│   ├── __init__.py
│   └── factory.py             # ~573 lines — ModelFactory (create, load, save model)
│
├── training/
│   ├── __init__.py
│   └── pipeline.py            # ~365 lines — TrainingPipeline
│
├── infrastructure/
│   ├── __init__.py
│   ├── distributed.py         # ~108 lines — DistributedSetup
│   └── tracking.py            # ~79 lines — ExperimentTracker
│
├── data/
│   ├── __init__.py
│   ├── pipeline.py            # ~241 lines — DataPipeline
│   ├── quality.py             # ~147 lines — QualityFilter, dedup, contamination
│   └── streaming.py           # MassiveDataCollector
│
├── alignment/
│   ├── __init__.py
│   └── pipeline.py            # DPO, ORPO, SimPO, KTO, safety training
│
├── evaluation/
│   ├── __init__.py
│   ├── benchmarks.py          # ~426 lines — BenchmarkRunner
│   ├── safety.py              # SafetyEvaluator
│   └── reporting.py           # EvaluationReport
│
├── nslt/
│   ├── __init__.py
│   ├── model.py               # ~694 lines — NSLTModel, MoENSLTModel
│   ├── layer1_ssm.py          # ~151 lines — SSMCompressionEngine
│   ├── layer2_ltc.py          # ~200 lines — LTCRoutingLayer
│   ├── layer3_sandbox.py      # ~300 lines — LatentSandbox, EnergyFunction
│   ├── layer4_output.py       # ~236 lines — SparseOutputSynthesizer
│   ├── ssm_scan.py            # ~445 lines — 5 scan backends + autograd
│   ├── moe_ssm.py             # ~200 lines — MoE SSM blocks
│   ├── mcts_sandbox.py        # ~400 lines — MCTS latent sandbox
│   ├── multiscale_ssm.py      # ~241 lines — 3-level state hierarchy
│   └── vision_encoder.py      # ~200 lines — SigLIP vision encoder
│
└── utils/
    └── reproducibility.py     # set_seed() for deterministic training
```

### 10.2 Key file summaries

| File | Lines | Key classes/functions | Dependencies |
|---|---|---|---|---|
| `main.py` | ~250 | `cmd_*` dispatchers | All `src.*` modules |
| `src/config/schema.py` | 536 | `Config`, `ModelArchitectureConfig`, `TrainingConfig`, etc. (8 validators) | pydantic, yaml |
| `src/models/factory.py` | 573 | `ModelFactory` (create, load, save including HuggingFace `from_pretrained`) | transformers, torch, `src.config.schema` |
| `src/training/pipeline.py` | 365 | `TrainingPipeline` | transformers.Trainer, `src.*` |
| `src/infrastructure/distributed.py` | 108 | `DistributedSetup` | torch.distributed |
| `src/tokenizer_trainer.py` | 222 | `download_tokenizer`, `train_custom_tokenizer`, `ensure_tokenizer` | tokenizers, transformers |
| `src/dhara/model.py` | 445 | `DharaModel` (PreTrainedModel), `DharaConfig`, `DharaForCausalLM` | torch.nn, transformers |
| `src/dhara/hssm.py` | 85 | `HierarchicalSSM`, `HierarchicalSSMStack` (state-dim fix) | torch.nn |
| `src/dhara/workspace.py` | 100 | `CognitiveWorkspace` (dict-based central hub, read/write/clear API) | torch.nn |
| `src/dhara/executive.py` | 150 | `ExecutiveController` (RL gates + enforcement + merged LearningController) | torch.nn |
| `src/dhara/quality_assurance.py` | 180 | `QualityAssurance` (merged reflect/verify/eval/correct multi-pass) | torch.nn |
| `src/dhara/tools.py` | 180 | `SymbolicToolRouter` (symbolic calc/python/search/db + learned router) | torch.nn, ast |
| `src/dhara/losses.py` | 150 | `AuxiliaryLossComputer` (14 configurable auxiliary losses) | torch.nn |
| `src/dhara/world_model.py` | 110 | `WorldModel`, `EntityExtractor`, `RelationNetwork`, `CauseEffectModel` | torch.nn |
| `src/dhara/layer1_tokenizer.py` | 38 | `IntelligentTokenizer`, `SemanticMetadataEmbedding` | torch.nn |
| `src/dhara/layer2_embedding.py` | 87 | `AdaptiveSemanticEmbedding`, `RotaryPositionEncoding`, `ContextAdapter` | torch.nn |
| `src/dhara/layer3_memory.py` | 230 | `MemoryManager`, `ForgetGate`, `CompressionAE`, `PriorityScorer`, `MemoryRetriever` | torch.nn |
| `src/dhara/layer4_intent.py` | 40 | `IntentUnderstanding`, `AdaptiveDifficultyRouter` | torch.nn |
| `src/dhara/layer5_planner.py` | 120 | `HierarchicalPlanner`, `SubgoalNode`, `GoalGraphAttention` | torch.nn |
| `src/dhara/layer6_reasoning.py` | 57 | `AdaptiveContinuousReasoning`, `ODETick`, `DomainDynamics` | torch.nn |
| `src/dhara/layer7_workspace.py` | 2 | Backward-compatible re-export from workspace.py | torch.nn |
| `src/dhara/layer8_specialists.py` | 110 | `DebateSandbox`, `SpecialistExpert`, `SpecialistCritic` | torch.nn |
| `src/dhara/layer9_reflection.py` | 20 | Backward-compatible QA wrapper (RecursiveReflection) | torch.nn |
| `src/dhara/layer10_verification.py`| 30 | Backward-compatible QA wrapper (VerificationWithRepair) | torch.nn |
| `src/dhara/layer11_decoder.py` | 145 | `HierarchicalDecoder` (3-level: semantic → language → token) | torch.nn |
| `src/nslt/model.py` | 694 | `NSLTModel`, `MoENSLTModel`, `TokenEmbedding`, `RotaryPositionEncoding` | torch.nn |
| `src/nslt/ssm_scan.py` | 445 | `selective_scan`, `SSMScanFunction`, `selective_scan_triton` | torch, triton (optional) |
| `src/data/pipeline.py` | 241 | `DataPipeline` | datasets, transformers |
| `src/data/quality.py` | 147 | `QualityFilter`, `ExactDeduplicator`, `ContaminationFilter` | re, hashlib |

---

## 11. Test Suite

109 tests across 11 files.

| File | Tests | Key fixtures | What's tested |
|---|---|---|---|
| `tests/test_nslt.py` | 17 | `nslt_model` | SSM shapes, LTC ODE, sandbox, sparse output, full model forward/train/gen |
| `tests/test_integration.py` | 12 | `tiny_nslt`, `tiny_moe` | SSM scan correctness, loss convergence, MoE forward, vision encoder, MCTS |
| `tests/test_alignment.py` | 4 | — | DPO loss prefers chosen, ORPO loss finite, SimPO correctness |
| `tests/test_config.py` | 7 | — | Loading, defaults, v1→v2 migration, invalid dtype rejection |
| `tests/test_data_collector.py` | 3 | — | Chat schema, problem-solution, max_samples |
| `tests/test_data_pipeline.py` | 20 | — | Text cleaning, language detection, field extraction, quality filtering |
| `tests/test_evaluation.py` | 4 | — | Safety keywords, benchmark result formatting |
| `tests/test_generation.py` | 8 | — | Code extraction, language aliasing, model not-ready error |
| `tests/test_quality.py` | 6 | — | Length check, low-quality markers, dedup, contamination |
| `tests/test_trainer.py` | 3 | — | Constitutional prompt, supervised tokenization, format |
| `tests/test_validation.py` | 10 | — | Python syntax, execution, timeout, batch validation |

**Run all:** `python -m pytest tests/ -q`
**With coverage:** `python -m pytest tests/ --cov=src --cov-report=term-missing`

---

## 12. Benchmarks

| Name | Type | # Problems | Metric | Parameters |
|---|---|---|---|---|
| `human_eval` | Code generation | 164 | pass@1 | 512 max tokens, temp 0.8 |
| `mbpp` | Code generation | 417 | pass@1 | Same as HumanEval |
| `mmlu` | Knowledge (57 subjects) | ~14K | accuracy | 5-shot, answer letter only |
| `hellaswag` | Commonsense NLI | 10K | accuracy | 0-shot, pick correct ending |
| `arc` | Science QA | 2,590 (challenge) | accuracy | 0-shot, multiple choice |
| `gsm8k` | Math word problems | 1,319 | exact match | 8-shot CoT |
| `truthfulqa` | Factuality | 817 | mc1/mc2 | 0-shot, multiple choice |
| `winogrande` | Coreference | 1,267 | accuracy | 0-shot, fill-in-the-blank |
| `bbh` | Reasoning (23 tasks) | 6,511 | accuracy | 3-shot CoT |

All benchmarks use `BenchmarkRunner.run_benchmarks(benchmark_list)`.

---

## 13. Known Flaky Tests & Edge Cases

### Flaky test
- `test_nslt_loss_decreases` (`tests/test_integration.py`): Trains a tiny NSLT
  for 10 steps with random init. Loss may occasionally increase (~1/5 runs).
  **Not a bug** — rerun the test. To stabilize, increase model size or fix seed
  in the test fixture.

### Edge cases
1. **Tokenizer without pad_token:** Auto-fixed by `add_special_tokens({"pad_token": "<pad>"})`
2. **Checkpoint with different vocab_size:** Detected by `is_compatible()`,
   returns `False`, starts fresh
3. **Zero samples from data pipeline:** `_load_stage_data` catches `Exception`,
   logs warning, returns `None` → stage skipped
4. **Tokenizer directory doesn't exist:** `ensure_tokenizer()` downloads/trains
   automatically. If it fails, raises `RuntimeError` with actionable message
5. **Single-GPU FSDP:** FSDP works on single GPU (`no_shard`). With
   `cpu_offload: true`, optimizer states live on CPU RAM — fits the 10.55B model
   on a single A100 80GB (~42 GB GPU for params+gradients+activations).
6. **Ctrl+C during training:** `try/finally` in `cmd_full_training` and
   `cmd_reserved_training` ensures
   `dist.destroy_process_group()` is called. Safe to re-run immediately
7. **Multiple torchrun instances:** NCCL will hang if two `torchrun` processes
   share the same `MASTER_ADDR:MASTER_PORT`. Ensure only one instance runs
 8. **Vision encoder enabled but no image data:** `VisionConfig.enabled: true`
    with text-only data will cause shape errors. Keep `enabled: false` for
    text-only training
 9. **HSSM state tensor shape mismatch (fixed):** HSSM state `A` tensor must use
     `d_state` dimension derived from `num_heads`, not `d_model`. Fix: set
     `self.d_state = config.d_state or (config.num_heads * config.d_head)` and
     ensure `hidden_states.size(-1) == self.d_state`.
10. **Missing `config.json` when loading model from path (fixed):** Models now
     always save/load via `save_pretrained()` which writes `config.json` with
     `DharaConfig` serialization. Loading via `from_pretrained()` reads the
     config and reconstructs the model.
11. **Singleton array conversion in `can_soft_mixture` (fixed):** The
     `mo_soft_assignments` method in `src/dhara/layers/mixture.py` now handles
     scalar wake masks properly — wraps singleton tensors in a list before sorting.
12. **Causal LM label shift (fixed):** Instruction tuning stage uses
     `DharaForCausalLM` which correctly shifts labels for next-token prediction:
     `loss = cross_entropy(logits[:-1], labels[1:])`. Each position i sees the
     decoder context for position i (not i-1).
13. **Memory manager returns 3-tuple (fixed):** `MemoryManager.forward()` returns
     `(fused, state, meta)` instead of `(fused, state)`. The model's forward
     handles both via `mem_out, mem_state, mem_meta = self.memory(x, mem_state)`.
14. **Executive controller softmax dim (fixed):** `F.softmax` calls in executive.py
     must include `dim=-1` to avoid deprecation warnings and ensure correct routing.
15. **Recursive reflection tensor dims (fixed):** `pass_embed` expansion adapts to
     input dimensionality (supports both 2D `(batch, d_hidden)` and 3D inputs).
16. **Decoder 3D input handling (fixed):** `HierarchicalDecoder.hidden_to_vocab()`
     flattens 3D inputs `(batch, seq, d_hidden)` to 2D before forward, then
     reshapes logits back to `(batch, seq, vocab)`.
17. **Learning controller concat dims (fixed):** Uses `h.mean(dim=1, keepdim=True)`
     to avoid 1D/2D concat mismatch with module_importance.
 18. **Curiosity scalar confidence handling (fixed):** `executive_decision["confidence"].mean()`
     produces a scalar; the curiosity module expands it to match batch dim before stacking.

---

## 14. Appendix: Bug Fix Audit

### 14.1 Summary

A comprehensive codebase audit (July 2026) identified and fixed **25+ bugs across 14 files** in the Dhara module. All fixes are verified with end-to-end model tests.

### 14.2 Complete Fix Table

| # | File | Line(s) | Severity | Bug | Fix |
|---|------|---------|----------|-----|-----|
| 1 | `workspace.py` | 41-42 | **HIGH** | `reset()` used `self.shared_repr = torch.zeros(...)` which replaces the registered buffer with a plain tensor, breaking `register_buffer` | Changed to `self.shared_repr.data = ...` |
| 2 | `workspace.py` | 85-86 | **HIGH** | `update()` used `self.shared_repr = (1 - gate) * ... + ...` and `self.confidence = ...` — direct reassignment breaks buffer registration | Changed to `.data.copy_()` |
| 3 | `executive.py` | 36-37 | **HIGH** | `ModulePerformanceTracker` used `self.counts = torch.zeros(...)` and `self.successes = ...` as plain attributes — not visible to `parameters()` or `state_dict()` | Changed to `self.register_buffer(...)` |
| 4 | `quality_assurance.py` | 78-79 | **HIGH** | Dynamic `nn.Linear` created inside `forward()` with double registration (`self._goal_proj` + `self.add_module('_goal_proj_module', ...)`) — module duplicated in hierarchy | Moved to `__init__` with `d_model` parameter, single `self.goal_proj` |
| 5 | `hssm.py` | 98 | **HIGH** | `B_bar = B * dt` used first-order Euler approximation instead of correct zero-order hold discretization | Changed to `B_bar = B * (1.0 - torch.exp(-dt * A.abs())) / A` |
| 6 | `hssm.py` | 105 | **HIGH** | State chaining between HSSM levels: `final_state = state[:, -1, :]` sliced last timestep but next layer expects full state tensor | Changed to `final_state = state` (full state) |
| 7 | `layer6_reasoning.py` | 30 | **HIGH** | `ODETick dz = torch.zeros_like(z)` produced wrong shape when `z` is 3D `(batch, 1, d_hidden)` — should be `(batch, d_hidden)` | Changed to `torch.zeros(z.shape[0], z.shape[1], ...)` |
| 8 | `layer6_reasoning.py` | 57-59 | **HIGH** | Trajectory averaging: `steps_f = n_steps.unsqueeze(-1).unsqueeze(-1)` created shape `(batch, 1, 1)`. Division `(batch, d_hidden) / (batch, 1, 1)` broadcast to `(batch, 1, d_hidden)` instead of `(batch, d_hidden)` | Reduced to `steps_f = n_steps.float().unsqueeze(-1).clamp(min=1)` → `(batch, 1)` |
| 9 | `layer8_specialists.py` | 69 | **HIGH** | In-place `conf_t[:, i:i+1] = ...` mutation on a sliced tensor — modifies the original tensor, causing gradient tracking issues | Changed to `torch.cat([..., conf_t[:, :i], ..., conf_t[:, i+1:]], dim=1).detach()` |
| 10 | `layer3_memory.py` | 69 | **HIGH** | `MemoryRetriever` batch dimension: `batch` variable renamed to `q_batch` but the final `.view(batch, -1, d_model)` still used old name | Changed to `.view(q_batch, -1, d_model)` |
| 11 | `layer3_memory.py` | 76 | **HIGH** | Same `MemoryRetriever`: `memory_bank` not expanded for batch dimension; only works when query batch = 1 | Added `memory_bank.expand(q_batch, -1, -1)` for K and V projections |
| 12 | `layer3_memory.py` | 207 | **HIGH** | `self.episodic.episode_buffer = decayed` replaces the registered buffer with a plain tensor | Changed to `self.episodic.episode_buffer.data.copy_(decayed)` |
| 13 | `layer3_memory.py` | 210 | **HIGH** | `self.mem_priority = priority.detach()` replaces registered buffer | Changed to `self.mem_priority.data.copy_(priority)` |
| 14 | `losses.py` | multiple | **MEDIUM** | CPU tensors in loss functions — target tensors created with `torch.zeros(...)` on CPU, not matching logits device | Added `.to(logits.device)` to all target tensors |
| 15 | `losses.py` | 64 | **MEDIUM** | BCE clamping `torch.clamp(pred, 1e-7, 1-e7)` used subtraction — `1-e7` evaluates to `-9999993`, not `1-1e-7` | Fixed to `torch.clamp(pred, 1e-7, 1 - 1e-7)` |
| 16 | `model.py` | 337 | **MEDIUM** | Training path returned `torch.zeros(batch, seq_len-1, vocab_size)` as dummy logits instead of actual computation | Changed to `self.decoder.hidden_to_vocab(full_h)` |
| 17 | `model.py` | 328 | **MEDIUM** | `hierarchical_log_prob` returns a single tensor, but code used `_, log_probs, _ = ...` tuple unpacking (would crash at runtime) | Changed to `log_probs = self.decoder.hierarchical_log_prob(...)` |
| 18 | `model.py` | 358-390 | **MEDIUM** | `generate()` calls `self.eval()` but never restores `self.train()` — model stuck in eval mode after generation | Added `was_training` guard and `if was_training: self.train()` |
| 19 | `factory.py` | 228-275 | **MEDIUM** | `create_model` passed kwargs like `d_model`, `max_seq_len`, `rope_base`, `dtype` — but `DharaConfig.__init__` expects `hidden_size`, `max_position_embeddings`, `rope_theta`. All kwargs silently filtered out by the `co_varnames` filter | Changed to create a proper `DharaConfig` object and pass `config=` |
| 20 | `factory.py` | 479-518 | **MEDIUM** | Same kwarg mismatch in `load_model`. Also `saved_config` not initialized when checkpoint lacks config.json | Same fix + initialized `saved_config = None` |
| 21 | `trainer.py` | 157 | **MEDIUM** | `self.model.is_ready` assumes `SpecializedCoderModel` wrapper — fails on raw `PreTrainedModel` | Changed to `callable(getattr(self.model, "is_ready", False))` |
| 22 | `trainer.py` | 259 | **MEDIUM** | `model=self.model.model` assumes `.model` child attribute — `DharaModel` is directly a `PreTrainedModel` | Added `_unwrap_model` property |
| 23 | `trainer.py` | 287 | **MEDIUM** | `self.model.save_model(save_dir)` assumes wrapper method — `PreTrainedModel` uses `save_pretrained` | Added `hasattr` fallback |
| 24 | `tools.py` | 79, 95 | **LOW** | `SymbolicSearch` and `SymbolicDatabase` not inheriting `nn.Module` — their `nn.Parameter` and `nn.Linear` members invisible to optimizer | Added `nn.Module` inheritance + `super().__init__()` |
| 25 | `layer8_specialists.py` | 54-55 | **LOW** | Dead code: `debate_input` variable assigned but never used | Removed assignment |
| 26 | `layer3_memory.py` | 95-96 | **LOW** | `self.position.weight` accessed directly instead of calling `self.position(positions)` | Changed to proper embedding lookup |
| 27 | `layer3_memory.py` | 39-40 | **LOW** | `CompressionAE` returned `decoded, decoded_detached, loss` but callers used `_, _, loss = ...` — unused second return value | Removed unused `decoded_detached` path |

### 14.3 Common Pitfalls & Prevention Patterns

This section documents recurring bug patterns found in the audit and how to prevent them.

#### Pattern A: Buffer Registration (`register_buffer`)

**Problem:** A tensor registered via `self.register_buffer("name", tensor)` is reassigned later with `self.name = new_tensor`. This replaces the registered buffer with an unregistered plain tensor, breaking:
- `model.state_dict()` — buffer won't appear
- `model.to(device)` — buffer won't move
- FSDP — buffer won't be sharded

**Bad:**
```python
# In __init__
self.register_buffer("my_buffer", torch.zeros(1, 64))

# Later in forward()
self.my_buffer = new_tensor  # ← BREAKS registration
```

**Good:**
```python
# In-place mutation (preferred for buffers)
self.my_buffer.copy_(new_tensor)

# Or via .data (for shape-changing operations)
self.my_buffer.data = new_tensor.data

# Or via .data.copy_ (safe, preserves buffer identity)
self.my_buffer.data.copy_(new_tensor)
```

**Files affected:** `workspace.py` (shared_repr, confidence), `layer3_memory.py` (episodic buffer, mem_priority)

#### Pattern B: Module Registration (non-`nn.Module` classes with `nn.Parameters`)

**Problem:** A class that uses `nn.Parameter` or `nn.Linear` but doesn't inherit from `nn.Module`. Its parameters are invisible to the owning module's `parameters()` iterator and won't receive gradients.

**Bad:**
```python
class MySubComponent:  # ← Not nn.Module!
    def __init__(self, d):
        self.weight = nn.Parameter(torch.randn(d))

class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = MySubComponent(64)  # ← weight not registered!
```

**Good:**
```python
class MySubComponent(nn.Module):  # ← Inherit nn.Module
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d))
```

**Files affected:** `tools.py` (SymbolicSearch, SymbolicDatabase)

#### Pattern C: Dynamic Module Creation in `forward()`

**Problem:** Creating `nn.Linear` (or any `nn.Module`) inside `forward()`. These modules are not in the module hierarchy and won't be found by `parameters()` or moved by `to(device)`.

**Bad:**
```python
def forward(self, x, goal_embeds):
    if goal_embeds.shape[-1] != x.shape[-1]:
        proj = nn.Linear(goal_embeds.shape[-1], x.shape[-1])  # ← Dynamic!
        x = proj(x)
```

**Good:**
```python
def __init__(self, d_hidden, d_model=None):
    super().__init__()
    self.goal_proj = None
    if d_model is not None and d_model != d_hidden:
        self.goal_proj = nn.Linear(d_model, d_hidden)  # ← Created at init

def forward(self, x, goal_embeds):
    if self.goal_proj is not None:
        x = self.goal_proj(x)
```

**Files affected:** `quality_assurance.py` (old `_goal_proj`)

#### Pattern D: Tensor Shape Broadcasting with Extra Dimensions

**Problem:** When dividing a 2D tensor by a 3D tensor, broadcasting can silently produce unexpected extra dimensions.

**Bad:**
```python
masked_sum = (z_traj * mask).sum(dim=1)  # shape (batch, d_hidden)
steps_f = n_steps.unsqueeze(-1).unsqueeze(-1)  # shape (batch, 1, 1)
result = masked_sum / steps_f  # shape (batch, 1, d_hidden) — WRONG!
```

**Good:**
```python
steps_f = n_steps.float().unsqueeze(-1)  # shape (batch, 1) — matches masked_sum
result = masked_sum / steps_f  # shape (batch, d_hidden) — CORRECT
```

**Files affected:** `layer6_reasoning.py` (trajectory averaging)

#### Pattern E: In-place Tensor Mutation on Slices

**Problem:** Indexed assignment on a tensor slice mutates the original tensor, which can corrupt gradient computation.

**Bad:**
```python
conf_t[:, i:i+1] = new_values  # Mutates the source tensor through the slice
```

**Good:**
```python
conf_t = torch.cat([
    conf_t[:, :i],
    new_values,
    conf_t[:, i+1:]
], dim=1).detach()  # Creates new tensor, no mutation
```

**Files affected:** `layer8_specialists.py` (debate confidence)

#### Pattern F: HuggingFace `PretrainedConfig` kwarg name mismatch

**Problem:** When subclassing `PretrainedConfig`, custom kwarg names in `DharaModel.__init__` must match `DharaConfig.__init__` parameter names exactly. Otherwise the co_varnames filter silently drops them.

**Always pass config as a `DharaConfig` object rather than relying on kwarg passthrough:**
```python
# Correct:
config = DharaConfig(hidden_size=arch.hidden_size, ...)
model = DharaModel(config=config)

# Wrong (kwargs silently dropped):
model = DharaModel(d_model=arch.hidden_size, ...)
```

**Files affected:** `factory.py` (create_model, load_model)

### 14.4 Verification

All fixes verified with:
```python
from src.dhara.model import DharaModel, DharaConfig

# Create tiny test model
c = DharaConfig(vocab_size=100, hidden_size=64, d_state=32, d_hidden=64, ...)
m = DharaModel(config=c)

# Forward pass
x = torch.randint(0, 100, (2, 16))
out = m(x)
assert out.logits.shape == (2, 16, 100)

# Training pass (with labels)
out = m(x, labels=x)
assert out.loss is not None

# Generation
out = m.generate(x, max_new_tokens=4)
assert out.shape == (2, 20)

# Buffer registration preserved
buffers = dict(m.named_buffers())
assert "workspace.shared_repr" in buffers or any("shared_repr" in k for k in buffers)
```

Run `python main.py test` to verify no regressions across the 109-test suite.
