# Training Pipeline

## Overview

The training pipeline (`src/training/pipeline.py`) orchestrates multi-stage training for the Dhara model. It supports three YAML configurations for different training scales, four training stages, FSDP distributed training, checkpointing with resume, and experiment tracking.

## Configuration

Three YAML config files in the project root provide different training scales:

### 1. `config_foundation.yaml` — Foundation Pretraining

Target: ~160M parameter Dhara model for single NVIDIA A100 80GB.

- **Architecture**: `dhara_v3`, hidden_size=576, d_state=288, n_ssm_layers=3, vocab_size=64000
- **Training**: 50,000 steps, learning_rate=1e-4, batch_size=4, gradient_accumulation=4 (effective batch 16)
- **Sequence length**: 2048
- **Distributed**: single GPU (strategy: none)
- **Data**: Full registry mode with 54 primary datasets (+4 fallback-only) across 8 categories
- **Staging**: dataset-granular staged pretraining with 8 stages (8000 / 6000 / 10000 / 6000 / 6000 / 6000 / 6000 / 2000 = 50,000 steps), one cosine schedule over the full run

### 2. `config_small.yaml` — Small Scale

Target: ~173M parameter Dhara for single-GPU experimentation.

- **Architecture**: `dhara_v3`, hidden_size=576, vocab_size=128000, reduced subgoal/reasoning parameters
- **Training**: 50,000 steps, learning_rate=3e-4, batch_size=4, gradient_accumulation=8 (effective batch 32)
- **Data**: Uses explicit dataset list (codeparrot-clean + c4), not registry mode
- **Distributed**: single GPU

### 3. `config.yaml` — Full Production

Target: Full-scale Dhara for 4x A100 80GB (FSDP full-shard).

- **Architecture**: `dhara_v3`, hidden_size=10240, d_state=4096, n_ssm_layers=6, vocab_size=128000
- **Training**: 1,000,000 steps, learning_rate=2e-4, batch_size=2, gradient_accumulation=8 (effective batch 64)
- **Sequence length**: 4096 (extensible to 262K via YaRN RoPE scaling)
- **Distributed**: FSDP full-shard, 4 GPUs, CPU offload for optimizer states
- **Tokens seen**: ~262B (intentional 130x over Chinchilla-optimal for SSM state testing)

### Pydantic Schema Validation

All configuration parameters are validated by the Pydantic v2 schema in `src/config/schema.py`. The root model is `Config` with nested sub-configs:

- `ModelConfig` → `ModelArchitectureConfig` → `DharaConfig` / `NSLTConfig` / `MoEConfig` / `VisionConfig`
- `TrainingConfig` → `PretrainStageConfig`, `SFTStageConfig`, `InstructionTuningConfig`, `AlignmentPhaseConfig`, `RLHFConfig`, `SafetyConfig`
- `DistributedConfig` → `FSDPConfig`, `DeepSpeedConfig`
- `DataConfig` → `QualityPipelineConfig`, `CurriculumConfig`, `PreprocessingConfig`, `LanguageBalancingConfig`, `DomainBalancingConfig`, `ASTFilterConfig`, `FunctionSamplingConfig`, `WeightedSamplerConfig`
- `TokenizerConfig`, `AlignmentConfig`, `OutputConfig`, `GenerationConfig`, `EvaluationConfig`

Cross-field validators ensure consistency:
- `validate_kv_heads` — num_key_value_heads must divide num_attention_heads
- `validate_loading` — 8-bit and 4-bit loading cannot both be enabled
- `validate_top_k_range` — adaptive_top_k_min cannot exceed adaptive_top_k_max
- `validate_image_size` — image_size must be divisible by patch_size (when vision enabled)
- v1→v2 config migration validator handles legacy keys

## Training Stages

### Stage 1: Foundation Pretraining (config_foundation.yaml)

- **Steps**: 50,000
- **Learning rate**: 1e-4 (cosine schedule)
- **Warmup**: 3,000 steps
- **Batch size**: 4 per GPU, accumulation 4 → effective 16
- **Weight decay**: 0.1
- **Max grad norm**: 1.0
- **Optimizer**: AdamW (fused)
- **Dataset**: Full registry (54 primary entries, 8 categories via weighted mixture)
- **Disabled**: SFT, instruction tuning, alignment, safety, curriculum
- **FSDP**: Disabled (single GPU)
- **Async prefetch**: per-slot staged unit prefetch (daemon producers, cancellation, retry/backoff) keeps streaming/filtering/tokenization ahead of the GPU
- **DataLoader**: pretrain forces `dataloader_num_workers=0` — batches are read on the main thread (forked workers previously inherited internal locks held by the async-prefetch/streaming threads and deadlocked on their first batch fetch)

### Stage 2: Full Pretraining (config.yaml)

- **Steps**: 1,000,000
- **Learning rate**: 2e-4 (cosine schedule)
- **Warmup**: 10,000 steps
- **Batch size**: 2 per GPU × 4 GPUs × 8 grad_acc → effective 64
- **FSDP**: Full shard, CPU offload, BF16 mixed precision
- **Curriculum**: Optional 4-stage (foundation → code_basics → reasoning → full_mix)

### Stage 3: Supervised Fine-Tuning (SFT)

- **Steps**: 100,000
- **Learning rate**: 5e-6 (cosine schedule)
- **Warmup**: 1,000 steps
- **Batch size**: 1 per GPU × 4 GPUs × 4 grad_acc → effective 16
- **Data mix**: sft_instructions 50%, conversations 30%, code_tasks 20%
- **Enabled in**: config.yaml only

### Stage 4: Instruction Tuning

- **Steps**: 50,000
- **Learning rate**: 1e-5 (cosine schedule)
- **Warmup**: 500 steps
- **Batch size**: 1 per GPU × 4 GPUs × 4 grad_acc → effective 16
- **Enabled in**: config.yaml only

## Key Training Components

### Gradient Accumulation

The pipeline uses HuggingFace `Trainer`'s built-in `gradient_accumulation_steps` to achieve large effective batch sizes on limited GPU memory:

```
effective_batch = per_device_batch * num_gpus * gradient_accumulation_steps
```

For production: 2 × 4 × 8 = 64 effective batch.

### Decoder Evaluation & LM Loss (`head_ce`)

`DharaModel.forward` (`src/dhara/model.py`) now evaluates the hierarchical decoder stack + vocab projection exactly **once** per micro-batch. Previously it ran twice — once via `decoder.hidden_to_vocab(full_h)` for `_vocab_logits`, and again inside `decoder.hierarchical_log_prob(...)` for the loss. The LM loss now reuses those logits and applies the causal shift (position *t* predicts label *t+1*) via a target mask. This roughly halves the dominant per-step cost. It is a pure efficiency fix — the dense-mode loss math is identical (verified by a regression test that byte-compares against the old `hierarchical_log_prob` path).

`model.architecture.dhara_v3.head_ce` controls how the LM cross-entropy loss is reduced:

- **`"dense"`** (default) — standard full-vocab log-softmax CE; unchanged behavior, all existing configs unaffected.
- **`"topk"`** (used by `config_foundation.yaml`) — candidate-set CE computed over the decoder's own adaptive top-k token indices (the `top_indices` `HierarchicalSparseDecoder` already produces, bounded by `adaptive_top_k_max`) **union** the target token, instead of a full-vocab log-softmax.

`head_ce: topk` is opt-in and an approximation: the decoder still performs the full dense vocab projection in the forward pass for candidate scoring; the flag reduces only the loss computation and the gradient target set. The architecture, forward pass, and generation/sampling are unchanged by this flag.

### Pretrain DataLoader Worker Safety

`_build_trainer` (`src/training/pipeline.py`) forces `dataloader_num_workers=0` for the `pretrain` stage. Forked DataLoader worker processes previously inherited internal locks held by the async-prefetch/streaming threads at fork time, deadlocking every worker on its first batch fetch and pinning training at step 0 forever (main thread futex-wait, progress bar stuck at `0/50000`). Pretrain reads batches on the main thread.

### Mixed Precision (FP16/BF16)

Configured via `FSDPConfig.mixed_precision`. Default is BF16 which provides better numerical stability than FP16 and is natively supported on A100 GPUs. The `DistributedSetup` resolves mixed precision from multiple config key locations for backward compatibility.

### FSDP Full-Shard

Fully Sharded Data Parallelism (ZeRO-3) distributes model parameters, gradients, and optimizer states across all GPUs. Key settings:

- `sharding_strategy: full_shard` — shards all parameters across devices
- `transformer_layer_cls` — FSDP wrapping target (auto-mapped from `model_type` by `ARCH_FSDP_LAYER_MAP`)
- `cpu_offload: true` — offloads optimizer states to CPU (required for 10.55B model on 80GB)
- `activation_checkpointing: true` — trades compute for memory
- `mixed_precision: bf16` — keeps master weights in BF16

### Gradient Clipping

`max_grad_norm=1.0` is enforced globally. The `_NaNSafeCallback` (`src/training/pipeline.py:59`) checks all parameters for non-finite values after each step and raises a `RuntimeError` if detected.

### Learning Rate Scheduling

All stages use cosine decay with linear warmup. The scheduler type is configurable per stage via `lr_scheduler_type`. Warmup steps vary per stage (pretrain: 10K-3K, SFT: 1K, instruction tuning: 500).

## Checkpointing

### Save Format

- **Model weights**: `model.safetensors` (via `save_pretrained()`)
- **Config**: `config.json` (DharaConfig → PretrainedConfig, HuggingFace-compatible)
- **Tokenizer**: `tokenizer.json`, `special_tokens_map.json`, etc.
- **Training state**: `training_args.bin` (TrainingArguments state, managed by Trainer)

### Resume Logic

The `_train_stage()` method (`src/training/pipeline.py:308`) performs auto-resume:
1. Scans `output_dir/stage_name/` for `checkpoint-*` subdirectories
2. Sorts by step number
3. Passes the latest to `trainer.train(resume_from_checkpoint=...)`

Cross-stage resume in the pipeline `initialize()` method:
1. Checks `output_dir/checkpoints/` for latest checkpoint
2. Verifies compatibility via `ModelFactory.is_compatible()`
3. If incompatible (architecture change), creates a fresh model with a warning
4. If no checkpoint found, creates model from scratch

### Validation Smoke Test

Phase 8 of `production_validation.py` runs a complete smoke test:
1. Trains 100 steps with checkpoint saves at step 100
2. Verifies checkpoint files exist (.pt or checkpoint-*)
3. Resumes from checkpoint for 10 more steps
4. Validates all return codes are 0

## Logging & Monitoring

### ExperimentTracker

The `ExperimentTracker` (`src/infrastructure/tracking.py`) wraps three backends:

- **Weights & Biases** (`wandb`): Full experiment tracking with config logging, metric logging, and run management
- **MLflow**: MLflow experiment tracking with params and metrics logging
- **TensorBoard**: Local SummaryWriter-based logging

The tracker is initialized in the pipeline with the full config dict and logs per-stage metrics (loss, learning rate, throughput). It is guarded by `try/except ImportError` so missing backends don't crash training.

### Console Logging

The `src/utils/logging.py` module (standard Python logging) provides structured console output. The `_LoggingCallback` (`src/training/pipeline.py:40`) logs per-step metrics:
- Step number
- Training loss (4 decimal places)
- Learning rate (scientific notation)
- Iterations/second (throughput)

### HealthReporter

The `DatasetHealthReport` (`src/data/health_reporter.py`) is populated during data pipeline execution. It tracks:
- Per-dataset statistics: raw samples, retention rate, packing efficiency, padding ratio, quality scores
- Category-level token distribution
- Language and domain distributions
- Pipeline errors and warnings
- Saved as JSON to `reports/dataset_health_report.json`
