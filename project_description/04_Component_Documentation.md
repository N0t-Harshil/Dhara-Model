# Component Documentation

---

## `src/data/registry.py` — Central Dataset Registry

**Purpose**: Single source of truth for all 55 training datasets across 8 categories. Manages dataset metadata, weights, fallback chains, and category organization.

### Public Classes

**`DatasetInfo`** — Dataclass with 17 fields describing a single dataset entry:

| Field | Type | Description |
|---|---|---|
| `path` | str | HuggingFace dataset path (e.g. `"bigcode/the-stack-v2-dedup"`) |
| `category` | str | One of: code, web_text, docs, wiki, math, science, books, structured_knowledge |
| `weight` | float | Base sampling weight within category |
| `quality_score` | float | Static quality estimate (0-1) |
| `name` | Optional[str] | Dataset configuration/subset name |
| `split` | str | Dataset split (default: "train") |
| `data_dir` | Optional[str] | Subdirectory within dataset |
| `language` | Optional[str] | Primary programming language |
| `domain` | str | Functional domain (backend, ml, web, etc.) |
| `fallbacks` | List[str] | Ordered fallback dataset paths |
| `text_fields` | List[str] | Candidate text field names |
| `license` | str | Dataset license |
| `max_samples` | Optional[int] | Max samples to stream |
| `function_sampling` | bool | Enable function-level extraction |
| `streaming` | bool | Use streaming mode |
| `priority` | int | Loading priority (lower = higher) |
| `estimated_tokens` | Optional[int] | Estimated token count |

`DatasetInfo.load_kwargs()` — Returns dict of kwargs for `datasets.load_dataset()`.

**`DatasetRegistry`** — Dictionary-based registry with 5 public methods:

| Method | Description |
|---|---|
| `register(info)` | Register a DatasetInfo entry |
| `get(key)` | Get entry by internal key |
| `get_by_path_category(path, category)` | Lookup by path + category |
| `all_entries()` | Return all registered entries |
| `by_category(cat)` | Filter entries by category |
| `log_fallback(primary, fallback)` | Record a fallback activation |
| `normalize_weights(category_targets)` | Normalize weights per category to match targets |

### Public Functions

- **`build_registry()`** → `DatasetRegistry`: Calls 8 category-specific `_register_*()` functions, normalizes weights, returns populated registry.
- **`detect_text_fields(sample, known_fields)`** → `list`: Scans sample dict for text-containing fields using `TEXT_FIELD_PRIORITY` list.
- **`extract_text(sample, fields)`** → `str`: Extracts text from sample using ordered field candidates.

### Internal Workflow

```
build_registry()
  ├── _register_code()        → 30% of total weight
  │     ├── OPC FineWeb Code  → 15% of code
  │     ├── Stack v2          → 30% of code (18 languages)
  │     ├── CodeSearchNet     → 20% of code
  │     ├── CodeParrot        → 20% of code
  │     └── CodeContests      → 15% of code
  ├── _register_web_text()    → 20%
  ├── _register_docs()        → 15%
  │     ├── FineWeb-Edu       → 3% of docs
  │     └── 18 doc scrapers    → 97% of docs
  ├── _register_wiki()        → 10%
  ├── _register_math()        → 10%
  │     └── 6 sub-entries with fallbacks
  ├── _register_science()     → 5%
  ├── _register_books()       → 5%
  └── _register_structured()  → 5%
```

### Configuration Constants

- **`CATEGORY_WEIGHTS`** — Dict mapping 8 categories to target weights (sum = 1.0)
- **`DOC_SOURCES`** — 18 documentation sources with language, quality_score, weight, domain
- **`CODE_LANG_TARGETS`** — 19 programming language target distributions
- **`STACK_V2_LANGUAGES`** — 18 language mappings for Stack v2

### Failure Modes

- Weight normalization may have floating-point drift; corrected by adjusting the last entry
- Empty category targets cause normalization skip
- Unknown categories fall back gracefully

### Performance

- Registry builds in ~0.04s
- Thread-safe for reads (no mutation after build)

---

## `src/data/streaming.py` — Dataset Streaming with Fallbacks

**Purpose**: Load datasets via HuggingFace `datasets.load_dataset()` with transparent error handling, automatic text field detection, and multi-level fallback chains.

### Public Classes

**`StreamingManager`** — Orchestrates streaming across all registry entries:

| Method | Description |
|---|---|
| `stream_all(limit_per_dataset)` | Stream every registered dataset |
| `stream_category(category, limit)` | Stream datasets in a category |
| `skip_rate()` | Returns per-dataset skip fractions |

**`MassiveDataCollector`** — Legacy wrapper providing backward compatibility with the old collector interface.

### Public Functions

- **`stream_dataset(path, split, name, data_dir, streaming, limit, text_fields)`** → `Generator[dict]`: Core streaming function. Handles all 4 dataset return types, auto-detects text fields, preserves original fields.
- **`stream_dataset_with_fallbacks(info, registry, limit)`** → `Generator[dict]`: Tries primary entry, then each fallback in order. Logs fallback activation on success.

### Key Design Decisions

1. **Never iterate over split names**: Directly accesses `ds[split]` after checking `isinstance(ds, (DatasetDict, IterableDatasetDict))`
2. **Transparent DatasetDict handling**: Automatically extracts the target split from both `DatasetDict` and `IterableDatasetDict`
3. **Gated dataset detection**: Checks error messages for keywords ("gated", "access", "permission", "401", "403"); shows user-friendly instructions
4. **Text field auto-detection**: Runs `detect_text_fields()` on first sample; reuses detected fields for all subsequent samples
5. **Original field preservation**: Yields `{**sample, "text": text}` — all original fields are preserved

### Fallback Chain Example

```
Primary: HuggingFaceFW/fineweb
  → Fallback 1: allenai/dolma
  → Fallback 2: togethercomputer/RedPajama-Data-1T
  → Fallback 3: tiiuae/falcon-refinedweb
```

---

## `src/data/pipeline.py` — Data Assembly Pipeline

**Purpose**: Complete pretrain/SFT/preference dataset assembly. The most complex file in the project at 1123 lines. Orchestrates quality filtering, deduplication, boilerplate removal, AST-based code filtering, function sampling, weighted mixing with adaptive resampling, sequence packing, and disk caching.

### Public Classes

**`WeightedMixedDataset`** — PyTorch `Dataset` subclass implementing adaptive weighted sampling:

| Feature | Description |
|---|---|
| Adaptive resampling | `effective_weight = base_weight × quality_score × remaining_token_ratio` |
| Stratification | Optional language/domain stratified sampling |
| RNG seeding | `numpy.random.default_rng(seed=rng_seed)` for reproducibility |
| Consumption tracking | Per-dataset consumed count for adaptive weights |
| Stats | `stats()` returns weights, consumed, remaining ratios |

**`DataPipeline`** — Main pipeline orchestrator:

| Method | Description |
|---|---|
| `build_pretrain_dataset(stage_name, filter)` | Build pretrain dataset (registry or config mode) |
| `build_sft_dataset(data, style, constitution)` | Build supervised fine-tuning dataset |
| `build_preference_dataset(data)` | Build DPO preference pairs dataset |
| `build_stage_dataset(raw_samples)` | Build generic stage dataset |

### Public Functions

- **`pack_sequences(tokenized_samples, max_seq_length, eos_token_id)`** → `(packed, packing_eff)`: EOS-separated segment concatenation with random window sampling, padding tracking, and per-sequence metadata.
- **`remove_boilerplate(text, keywords)`** → `str`: Scans first 50 lines for license headers; strips contiguous license blocks.
- **`compute_quality_score(text, category, language)`** → `float`: Delegates to `document_quality_score()`.
- **`detect_domain(text, category)`** → `str`: 22-domain classifier using keyword matching.
- **`detect_language(text, dataset_name)`** → `str`: Heuristic language detection from dataset name and code patterns.
- **`random_window_sample(tokens, max_seq_length)`** → `list`: Random window extraction for long documents.
- **`passes_quality_filter(text, category, quality_score, lang)`** → `bool`: Threshold-based quality gate.

### Quality System

The `build_pretrain_dataset_from_registry()` method applies filters in strict order:

1. Boilerplate removal → 2. File pattern skip → 3. Min length check → 4. Quality scoring → 5. AST filter (code only) → 6. Exact dedup → 7. SimHash dedup → 8. Function sampling (code only)

Each dataset is cached to disk (`hf_cache/tokenized/{stage}/{sha256_hash}`) after tokenization and packing.

### Sequence Packing Details

- EOS-separated: segments separated by `eos_token_id`
- Random window sampling for docs > `max_seq_length`
- Padding tracking: target < 5%
- Metadata per sequence: `_segments` count, `_avg_quality`
- Packing efficiency reported as percentage

---

## `src/data/quality.py` — Quality Scoring & Deduplication

**Purpose**: Multi-component document quality scoring, 4 deduplication strategies, contamination filtering, and language detection.

### Quality Scoring

**`document_quality_score(text, category, language, seen_hashes)`** → `dict` with components:

| Component | Weight (code) | Weight (web_text) | Description |
|---|---|---|---|
| `length` | 0.10 | 0.10 | Normalized document length |
| `perplexity` | 0.15 | 0.30 | Estimated perplexity from word statistics |
| `language_conf` | 0.15 | 0.20 | Language signature match confidence |
| `formatting` | 0.15 | 0.25 | Line length variance, blank ratio, indent, caps |
| `code_quality` | 0.35 | 0.00 | Compilation test, identifiers, comments, functions |
| `doc_completeness` | 0.00 | 0.00 | Section headers, code blocks, structure (for docs/wiki) |
| `toxicity` | 0.10 | 0.15 | 1.0 - toxic word hits |
| `final` | — | — | Weighted sum of components |

### Deduplication Strategies

| Class | Method | Threshold | Use Case |
|---|---|---|---|
| `ExactDeduplicator` | MD5 hash of normalized prefix | Exact match | Fast first-pass dedup |
| `MinHashDeduplicator` | 128-minhash signature, 5-gram shingles | Jaccard > 0.85 | Near-duplicate detection |
| `SimHashDeduplicator` | 64-bit fingerprint, token weights | Cosine > 0.85 | Large-scale fuzzy dedup |
| `SemanticDeduplicator` | sentence-transformers embeddings | Cosine > 0.92 | Semantic near-duplicate |

### Contamination Filtering

`ContaminationFilter` checks against benchmark patterns for: HumanEval, MBPP, MMLU, GSM8K, ARC.

### Language Detection

`detect_language(text)` → `str`: Regex-based signature matching for 13 languages (python, javascript, typescript, java, cpp, c, rust, go, csharp, php, ruby, shell, sql).

### Legacy Classes

- `QualityFilter` — static methods for length/content/language checks
- `QualityScorer` — wrapper around `document_quality_score()`

---

## `src/data/doc_builder.py` — Documentation Web Scraper

**Purpose**: Web scraping pipeline for building documentation datasets from 18 sources. Used to generate the "docs" category in the registry.

**Size**: 808 lines.

### Architecture

**`DocScraper`** (abstract base) → **`SphinxScraper`** (Sphinx-specific) → 14 concrete scrapers + 4 custom scrapers.

### Scrapers

| Scraper | Source | Language |
|---|---|---|
| `PythonDocScraper` | docs.python.org/3/ | python |
| `PyTorchDocScraper` | pytorch.org/docs/stable/ | python |
| `NumPyDocScraper` | numpy.org/doc/stable/ | python |
| `FastAPIDocScraper` | fastapi.tiangolo.com/ | python |
| `OpenCVDocScraper` | docs.opencv.org/4.x/ | cpp |
| `ONNXDocScraper` | onnx.ai/onnx/ | python |
| `DockerDocScraper` | docs.docker.com/ | shell |
| `KubernetesDocScraper` | kubernetes.io/docs/ | shell |
| `PostgreSQLDocScraper` | postgresql.org/docs/current/ | sql |
| `SQLiteDocScraper` | sqlite.org/docs.html | sql |
| `MDNDocScraper` | developer.mozilla.org/ | javascript |
| `RustBookScraper` | doc.rust-lang.org/book/ | rust |
| `GoDocScraper` | go.dev/doc/ | go |
| `CUDADocScraper` | docs.nvidia.com/cuda/ | cpp |
| `CuDNNDocScraper` | docs.nvidia.com/deeplearning/cudnn/ | cpp |
| `RFCDocScraper` | rfc-editor.org/rfc/ | text |
| `LinuxKernelDocScraper` | kernel.org/doc/html/latest/ | text |
| `LangSpecScraper` | Python + Rust language references | text |

### Key Features

- Client-side redirect following (meta refresh, JS location.replace)
- Per-scraper timeout (default 600s, configurable)
- Failed URL tracking (avoids retrying)
- URL normalization (strip fragments, trailing slashes)
- Duplicate suppression per scraper instance
- Per-scraper discovery/duplicate/failed counts

### Failure Modes

- Cloudflare challenges (blocks scraping)
- Site redesigns (URL pattern breaks)
- JS-rendered content (MDN uses JS nav)
- Rate limiting (429 responses)
- Slow servers (timeout handling)

---

## `src/config/schema.py` — Configuration Schema

**Purpose**: Pydantic v2 configuration validation. 666 lines, 30+ models.

### Key Models

| Model | Fields | Validators |
|---|---|---|
| `Config` | 10+ sub-models | Cross-field validation (FSDP ↔ strategy) |
| `ModelArchitectureConfig` | model_type, vocab_size, hidden_size, methos_v3, nslt | — |
| `MethosV3Config` | 30+ architecture params | — |
| `NSLTConfig` | 7 architecture params | — |
| `TrainingConfig` | pretrain/sft/instruction/alignment/safety stages | — |
| `DataConfig` | quality, curriculum, preprocessing, ast_filter, function_sampling | — |
| `DistributedConfig` | strategy, fsdp settings | FSDP requires strategy='fsdp' |
| `FSDPConfig` | sharding_strategy, mixed_precision, cpu_offload, etc. | — |
| `MoEConfig` | num_experts, top_k, aux_loss_coef | top_k ≤ num_experts |
| `VisionConfig` | enabled, encoder, image_size, patch_size | image_size % patch_size == 0 |
| `MultimodalConfig` | vision sub-config | — |
| `DatasetEntryConfig` | path, name, category, max_samples, etc. | — |

### Auto-migration

`load_config()` function auto-detects v1 vs v2 format by checking for top-level keys like `data_collection` or `datasets`, and migrates to v2 format.

---

## `src/training/pipeline.py` — Training Orchestration

**Purpose**: Multi-stage training pipeline. 556 lines. Runs sequential training stages with curriculum learning.

### Public Classes

**`TrainingPipeline`** — Orchestrates training:

| Method | Description |
|---|---|
| `run()` | Execute full training plan |
| `run_stage(stage_name)` | Execute single training stage |
| `load_checkpoint(path)` | Resume from checkpoint |

### Stages

```
pretrain → sft → instruction_tuning → alignment
```

Each stage uses the HuggingFace `Trainer` with stage-specific config overrides (learning rate, batch size, data mix, optimizer).

### Features

- Gradient checkpointing across ranks
- Logging callback with step timing
- Checkpoint management with configurable `save_total_limit`
- _LoggingCallback logs loss and LR at each step

---

## `src/infrastructure/distributed.py` — Distributed Setup

**Purpose**: Initialize distributed training environment. 123 lines.

### Public Classes

**`DistributedSetup`** — Handles:
- Process group initialization (nccl/gloo backend)
- Device placement (CUDA set_device)
- FSDP argument preparation
- Multi-node support via WORLD_SIZE/RANK/LOCAL_RANK env vars
- `is_main_process()` check
- `get_training_args()` returns FSDP/DeepSpeed config dict

### Key Design

- Auto-detects distributed environment from env vars
- Falls back to single-process if not distributed
- Validates FSDP config against distributed strategy

---

## `src/infrastructure/tracking.py` — Experiment Tracking

**Purpose**: Unified experiment tracking across providers. 79 lines.

### Public Classes

**`ExperimentTracker`** — Wraps wandb/mlflow/tensorboard:

| Method | Description |
|---|---|
| `init(**kwargs)` | Initialize the tracking provider |
| `log_metrics(metrics, step)` | Log metrics dict |
| `log_config(config)` | Log configuration dict |
| `finish()` | End the tracking run |
| `log_checkpoint(path, metadata)` | Log checkpoint location |

### Design

- Provider selected via config (`wandb` / `mlflow` / `tensorboard` / `none`)
- Graceful fallback if provider not installed
- Configurable project name

---

## `src/models/factory.py` — Model Factory

**Purpose**: Unified model creation and weight loading. 604 lines.

### Supported Architectures

| Architecture | Config Class | Model Class | FSDP Layer |
|---|---|---|---|
| methos_v3 | MethosV3Config → PretrainedConfig | MethosV3Model | HierarchicalSSMStack |
| nslt | Custom config | NSLTModel | SSMCompressionEngine |
| llama | LlamaConfig | LlamaForCausalLM | LlamaDecoderLayer |
| mixtral | MixtralConfig | MixtralForCausalLM | MixtralDecoderLayer |
| qwen2_moe | AutoConfig | AutoModelForCausalLM | Qwen2MoeDecoderLayer |
| deepseek_v2 | AutoConfig | AutoModelForCausalLM | DeepseekV2DecoderLayer |

### Key Methods

- `build_model_config(vocab_size, arch_config)` → `PretrainedConfig`
- `create_model(model_config, pretrained_path)` → `PreTrainedModel`
- `estimate_params(model)` → parameter count
- `load_tokenizer(tokenizer_config)` → `PreTrainedTokenizerBase` with fallback chain

---

## `src/trainer.py` — Standalone Model Trainer

**Purpose**: High-level training interface. 298 lines.

### Public Classes

**`ModelTrainer`** — Full training lifecycle:

| Method | Description |
|---|---|
| `prepare_data()` | Build datasets via DataPipeline |
| `train()` | Run training loop via HuggingFace Trainer |
| `evaluate()` | Run evaluation benchmarks |
| `save_checkpoint(path)` | Save model + tokenizer + config |

### Integration

Uses `DataPipeline` for data preparation and HuggingFace `Trainer` for the training loop.

---

## `src/methos_v3/model.py` — Core MethosV3 Model

**Purpose**: Full MethosV3 model definition. 414 lines. Integrates all 11 layers into a single forward pass with auxiliary losses.

### Public Classes

| Class | Description |
|---|---|
| `MethosV3Config(PretrainedConfig)` | 50+ configuration parameters, model_type="methos_v3" |
| `MethosV3Model(PreTrainedModel)` | Core model with all 11 layers |
| `MethosV3ForCausalLM(PreTrainedModel)` | CausalLM wrapper with LM head |
| `MoEMethosV3Model(PreTrainedModel)` | Mixture-of-Experts variant |

### Forward Pass Flow

```
Input → IntelligentTokenizer → AdaptiveSemanticEmbedding → HierarchicalMemoryEngine
  → IntentUnderstanding → GlobalPlanner → CognitiveWorkspace
  → ExecutiveController (gate enforcement)
  → AdaptiveContinuousReasoning → CognitiveWorkspace
  → SpecialistSandbox (7 specialists, debate) → CognitiveWorkspace
  → InternalToolInterface → CognitiveWorkspace
  → QualityAssurance (reflect → verify → self-eval → correct) → CognitiveWorkspace
  → HierarchicalSparseDecoder → Output Logits
```

### Auxiliary Losses (14 total)

Each module contributes a differentiable loss: memory reconstruction, intent CE, planner CE, executive BCE + RL, reasoning smoothness, debate calibration, tool selection CE, QA correctness, verification, self-evaluation, calibration ECE, world model entity prediction, decoder group assignment, curiosity novelty bonus.

---

## `src/nslt/model.py` — NSLT Model

**Purpose**: Neural State-Space Liquid Transformer — non-Transformer architecture. 719 lines.

### Public Classes

| Class | Description |
|---|---|
| `TokenEmbedding` | Learned embeddings with optional tied weights |
| `RotaryPositionEncoding` | RoPE with configurable theta |
| `NSLTModel` | Full NSLT model: SSM → LTC → Sandbox → Output |
| `MoENSLTModel` | Mixture-of-Experts NSLT variant |

### Key Design Decisions

- **O(1) memory**: SSM compression replaces KV cache
- **Continuous ODE reasoning**: LTC routing with liquid time-constants
- **Energy-based reasoning**: LatentSandbox with parallel trajectory simulation
- **Sparse output**: Top-1% vocabulary gating (O(top_k) vs O(V))
- **Multimodal**: Vision encoder integration via SigLIP + projection

### Architecture Flow

```
Input → TokenEmbedding → RotaryPositionEncoding → SSMCompressionEngine
  → LTCRoutingLayer (ODE) → LatentSandbox (energy-based)
  → SparseOutputSynthesizer (top-k gating) → Output
```
