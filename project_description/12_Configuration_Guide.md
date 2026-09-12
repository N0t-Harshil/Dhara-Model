# Configuration Guide

## Overview

The configuration system uses Pydantic v2 `BaseModel` defined in `src/config/schema.py`. The root model `Config` contains 9 sub-configs covering model architecture, training, distributed setup, data pipeline, tokenizer, alignment, output, generation, and evaluation. Configuration files are YAML and are loaded by `load_config()`.

Three config files are provided for different training scales:

| Config | Model Size | GPUs | Steps | Purpose |
|--------|-----------|------|-------|---------|
| `config_foundation.yaml` | ~160M | 1× A100 | 50K | Foundation pretraining |
| `config_small.yaml` | ~173M | 1× A100 | 50K | Small-scale experimentation |
| `config.yaml` | ~10.55B | 4× A100 | 1M | Full production training |

---

## Key Configuration Parameters

| Parameter | Config | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| `model.name` | All | `"Dhara"` | string | Model identifier for logging and checkpoint naming |
| `model.architecture.model_type` | All | `"dhara_v3"` | dhara_v3, nslt, llama, mixtral, qwen2_moe, deepseek_v2 | Architecture selection. Determines which model class is instantiated by ModelFactory |
| `model.architecture.hidden_size` | foundation | 576 | 64–10240 | Main model dimension (d_model). Larger = more capacity but more memory. Foundation: tiny 576. Full: 10240 |
| `model.architecture.vocab_size` | foundation | 64000 | 1000–256000 | Vocabulary size. Foundation uses 64K (reduced cost). Full config uses 128K. Must match tokenizer |
| `model.architecture.d_state` | foundation | 288 | 64–4096 | SSM compressed state dimension. Controls the O(1) memory bottleneck. Larger = more information retained |
| `model.architecture.n_ssm_layers` | foundation | 3 | 1–12 | Number of SSM compression layers. More layers = deeper compression hierarchy but more compute |
| `model.architecture.n_trajectories` | foundation | 2 | 1–7 | Number of parallel specialist trajectories in the sandbox. 7 for full config (7 specialists). Fewer = less compute |
| `model.architecture.max_position_embeddings` | foundation | 4096 | 512–262144 | Absolute maximum sequence length before RoPE scaling. Foundation: 4096. Full: 16384 |
| `model.architecture.rope_scaling.factor` | foundation | 16.0 | 1.0–32.0 | YaRN RoPE scaling factor. Extends context window by this factor (e.g., 4096→65536 with factor 16) |
| `model.architecture.attention_implementation` | full | `flash_attention_2` | sdpa, flash_attention_2, eager | Attention kernel. FlashAttention 2 is fastest on A100. SDPA is default PyTorch. Eager is slow but debuggable |
| `model.architecture.tie_word_embeddings` | foundation | true | bool | Share weights between embedding and LM head. Saves ~vocab_size × d_model parameters (~37M for 64K×576) |
| `model.architecture.dhara_v3.head_ce` | foundation | `"dense"` | `"dense"`, `"topk"` | LM head cross-entropy reduction. `"dense"` = standard full-vocab log-softmax CE (default). `"topk"` = opt-in candidate-set CE over the decoder's adaptive top-k token indices (bounded by `adaptive_top_k_max`) union the target token; it reduces only the loss/backward target set — the dense vocab projection still runs in the forward pass. Foundation config uses `topk` |
| `training.max_seq_length` | foundation | 2048 | 512–262144 | Training sequence length. Shorter = more samples per batch. Longer = better long-range learning. Full: 4096 |
| `training.pretrain.max_steps` | foundation | 50000 | 1000–1e6 | Number of pretraining steps. Foundation: 50K. Full: 1M (262B tokens at 4K seq_len) |
| `training.pretrain.learning_rate` | foundation | 1e-4 | 1e-6–1e-3 | Peak learning rate for cosine schedule. Foundation: 1e-4. Full: 2e-4. Small: 3e-4 |
| `training.pretrain.batch_size` | foundation | 4 | 1–64 | Per-GPU batch size. Foundation single GPU: 4. Full 4 GPUs: 2 each |
| `training.pretrain.gradient_accumulation_steps` | foundation | 4 | 1–128 | Accumulate gradients over N steps before optimizer update. Effective batch = batch_size × num_gpus × accumulation |
| `training.pretrain.staging.enabled` | foundation | true | bool | Dataset-granular staged pretraining — 8 stages (8000/6000/10000/6000×4/2000) whose step counts sum to `training.pretrain.max_steps`; one cosine schedule over the full run |
| `training.pretrain.weight_decay` | all | 0.1 | 0.0–1.0 | AdamW weight decay. Higher = stronger regularization. 0.1 is standard for LLM pretraining |
| `training.pretrain.warmup_steps` | foundation | 3000 | 0–10000 | Linear warmup steps before cosine decay. ~3–10% of max_steps is typical |
| `training.pretrain.max_grad_norm` | all | 1.0 | 0.1–10.0 | Gradient clipping threshold. Prevents gradient explosion. 1.0 is conservative |
| `training.pretrain.optimizer` | all | `adamw_fused` | adamw, adamw_8bit, adamw_fused, sgd | Optimizer. `adamw_fused` = torch's fused AdamW (fastest). `adamw_8bit` = memory-efficient (slower) |
| `training.sft.learning_rate` | full | 5e-6 | 1e-7–1e-4 | SFT learning rate. Lower than pretraining to prevent catastrophic forgetting |
| `training.sft.max_steps` | full | 100000 | 1000–1e6 | Supervised fine-tuning steps. ~6.5B tokens at 4096 seq_len |
| `training.instruction_tuning.learning_rate` | full | 1e-5 | 1e-7–1e-4 | Instruction tuning learning rate. Slightly higher than SFT for fast adaptation |
| `training.instruction_tuning.max_steps` | full | 50000 | 1000–1e6 | Instruction tuning steps |
| `training.save_steps` | all | 1000 | 100–100000 | Checkpoint save interval. Lower = more checkpoints but more I/O and disk usage |
| `training.save_total_limit` | all | 5 | 1–50 | Maximum checkpoints to keep. Older ones are deleted automatically |
| `training.logging_steps` | all | 10 | 1–1000 | Logging interval in steps. Lower = more verbose logging |
| `distributed.strategy` | all | `fsdp` | fsdp, deepspeed, ddp, none | Distributed strategy. Full config uses FSDP. Foundation/small use `none` (single GPU) |
| `distributed.fsdp.sharding_strategy` | full | `full_shard` | full_shard, hybrid_shard, no_shard | FSDP sharding. `full_shard` = ZeRO-3 (shards all states). `hybrid_shard` = ZeRO-2 (shards grad+optim) |
| `distributed.fsdp.mixed_precision` | full | `bf16` | bf16, fp16, fp32 | FSDP mixed precision. BF16 is recommended for A100 (stable, no loss scaling needed) |
| `distributed.fsdp.cpu_offload` | full | true | bool | Offload optimizer states to CPU. Required for 10.55B model with AdamW on 80GB GPU |
| `data.use_registry` | foundation | true | bool | Use the dataset registry system. If false, uses the explicit `datasets` list |
| `data.hf_token` | foundation | `""` | string | HuggingFace token for gated dataset access. Should be set via environment variable `HF_TOKEN` |
| `data.cache_dir` | all | `./hf_cache` | path | Cache directory for downloaded datasets and tokenized samples |
| `data.streaming` | all | true | bool | Stream datasets from HuggingFace (true) or download entirely (false). Streaming saves disk space |
| `data.quality.min_length` | all | 30 | 10–1000 | Minimum text length after cleaning. Shorter texts are discarded |
| `data.quality.max_length` | all | 500000 | 1000–1e7 | Maximum text length. Longer texts are truncated |
| `data.quality.deduplication.method` | all | `simhash` | exact, minhash, simhash | Deduplication method. `simhash` = fuzzy (good for near-duplicates). `exact` = exact match only. `minhash` = Jaccard similarity |
| `data.quality.deduplication.threshold` | all | 0.85 | 0.0–1.0 | Similarity threshold for dedup. 0.85 = documents with 85%+ similarity are considered duplicates |
| `data.quality.contamination.benchmarks` | all | [...] | list of strings | Benchmark names to filter from training data (prevents benchmark contamination) |
| `data.preprocessing.min_text_length` | all | 100 | 10–10000 | Minimum cleaned text length (after boilerplate removal). Different from quality.min_length |
| `data.preprocessing.remove_boilerplate` | all | true | bool | Remove license headers and boilerplate comments from code files |
| `data.sampler.balance_by` | all | `tokens` | documents, tokens | Weighted sampling strategy. `tokens` = balance by token count. `documents` = balance by document count |
| `data.curriculum.enabled` | all | false | bool | Enable curriculum learning with staged dataset filtering. Foundation config has 4 predefined stages |
| `tokenizer.source` | all | `huggingface` | huggingface, custom | Tokenizer source. `huggingface` = download pretrained. `custom` = train from scratch (not fully implemented) |
| `tokenizer.huggingface_model` | all | `Xenova/claude-tokenizer` | string | HuggingFace tokenizer model ID |
| `output.model_dir` | all | `./models/dhara` | path | Directory for saving trained models |
| `output.checkpoint_dir` | all | `./models/dhara/checkpoints` | path | Directory for training checkpoints |
| `output.experiment_tracking.enabled` | all | false | bool | Enable experiment tracking (wandb/mlflow/tensorboard) |
| `output.experiment_tracking.provider` | all | `none` | wandb, mlflow, tensorboard, none | Tracking backend |
| `generation.temperature` | all | 0.7 | 0.0–2.0 | Sampling temperature. 0.0 = greedy. Higher = more random |
| `generation.top_p` | all | 0.9 | 0.0–1.0 | Nucleus sampling threshold |
| `generation.top_k` | all | 40 | 1–vocab_size | Top-k sampling |
| `evaluation.benchmarks` | full | [...] | list of strings | Benchmark names for evaluation during training |

---

## Important Parameter Interactions

### Effective Batch Size

The effective batch size determines how many tokens are seen per optimizer step:

```
effective_batch = per_device_batch_size × num_gpus × gradient_accumulation_steps
```

| Config | Per GPU | GPUs | Accumulation | Effective Batch | Seq Len | Tokens per Step |
|--------|---------|------|-------------|-----------------|---------|-----------------|
| foundation | 4 | 1 | 4 | 16 | 2048 | 32,768 |
| small | 4 | 1 | 8 | 32 | 2048 | 65,536 |
| full | 2 | 4 | 8 | 64 | 4096 | 262,144 |

### Memory vs. Model Size

- `hidden_size` (d_model) is the primary determinant of model FLOPs and memory
- `vocab_size × hidden_size` determines embedding size (can be ~40% of total params for small models)
- `d_state` controls SSM compressed state — larger = more retained information but no direct memory scaling (O(1))
- `tie_word_embeddings: true` saves `vocab_size × hidden_size` parameters
- `gradient_checkpointing` trades ~33% compute for ~50% memory savings in backward pass

### Learning Rate Scaling

Learning rate should be scaled with batch size:
- Foundation (batch 16): 1e-4
- Small (batch 32): 3e-4
- Full (batch 64): 2e-4

The cosine schedule decays LR to near-zero by `max_steps`. Warmup steps should be ~1-10% of total steps.

### RoPE Scaling

YaRN scaling extends context window without retraining:
- `max_position_embeddings`: pretrained context window
- `rope_scaling.factor`: extension multiplier (e.g., factor=16 extends 4K→64K context)
- `rope_scaling.original_max_position_embeddings`: should match `max_position_embeddings`
- `target_max_length`: desired extended context length

### Curriculum Learning

When `data.curriculum.enabled: true`, the pretraining stage is split into phases with different dataset filters:

1. **Foundation**: web_text, wiki, books (language fundamentals)
2. **Code basics**: +code, +docs, +web_text, +wiki (introduce code)
3. **Reasoning**: +code, +math, +science, +docs (advanced reasoning)
4. **Full mix**: all categories (final blend)

Each phase trains for a configurable number of steps with its own filtered dataset mixture.

---

## Validation Errors

The Pydantic schema produces clear error messages for common misconfigurations:

- `num_key_value_heads cannot exceed num_attention_heads`
- `top_k cannot exceed num_experts` (MoE config)
- `hidden_size must be divisible by num_attention_heads`
- `image_size must be evenly divisible by patch_size` (vision config)
- `load_in_8bit and load_in_4bit cannot both be enabled`
- `deepspeed config is required when strategy is 'deepspeed'`
- `adaptive_top_k_min cannot exceed adaptive_top_k_max`
- `config file not found` (wrong path)
