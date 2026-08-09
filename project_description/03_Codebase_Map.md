# Codebase Map

## Root Directory

| File | Purpose |
|---|---|
| `main.py` | CLI entrypoint — GPU auto-selection, subcommand dispatch (train, test, chat, etc.), config loading |
| `config.yaml` | Primary training config — Methos V4 NSLT, 4x A100 80GB, vocab_size=128000, 1M pretrain steps |
| `config_foundation.yaml` | Foundation pretraining config — 1x A100 (~160M params), vocab_size=64000, 50K pretrain steps |
| `config_small.yaml` | Small/test config — single GPU debug, 173M params, 2 datasets only |
| `ARCHITECTURE.md` | Detailed Methos V4 architecture spec — 11-layer workspace-centric cognitive architecture |
| `PROJECT_DOCUMENTATION.md` | High-level project overview and usage guide |
| `README.md` | Project README with setup instructions |
| `requirements.txt` | Production dependencies — torch 2.6, transformers 4.43, datasets, pydantic, etc. |
| `requirements-dev.txt` | Development dependencies |
| `pytest.ini` | Pytest configuration |
| `test_backward.py` | Standalone backward-compatibility test |

---

## `src/` — Main Source Package

### `src/data/` — Production Data Pipeline

**Purpose**: Complete data pipeline for pretraining, SFT, and preference optimization — from raw dataset streaming through quality filtering, deduplication, AST-based code processing, sequence packing, and disk caching.

**Files** (13 files):

| File | Lines | Responsibility |
|---|---|---|
| `registry.py` | 464 | Central dataset registry — 55 entries across 8 categories, weight management, fallback tracking |
| `streaming.py` | 549 | Dataset loading with fallback chains — handles all 4 DatasetDict/IterableDataset types |
| `drivers.py` | 722 | Dataset driver abstraction — File/Script/Local/Streaming families, builder cache, `detect_driver()` |
| `metadata_cache.py` | 419 | Metadata record cache — file lists, loader detection, fingerprints, gated-error helpers |
| `shards.py` | 184 | Shard-parallel progress store — resume state, per-shard stats, completion tracking |
| `pipeline.py` | 1123 | Complete dataset assembly — quality filtering, dedup, boilerplate removal, AST filtering, function sampling, weighted mixing, sequence packing, disk caching |
| `quality.py` | 508 | Quality scoring (8 components), 4 deduplication strategies (Exact/MinHash/SimHash/Semantic), contamination filtering, language detection |
| `doc_builder.py` | 820 | Web scraping for 18 documentation sources — Sphinx-based, RFCs, MDN, NVIDIA docs |
| `ast_filter.py` | 200 | AST-based code filtering — identifier ratio, executable ratio, autogen detection |
| `function_sampler.py` | 57 | Function-level code sampling — extract functions, sample or fallback to full code |
| `health_reporter.py` | 239 | Dataset health reporting — per-dataset stats, global stats, error tracking |
| `sanity.py` | 144 | Sanity checks on packed samples — decode quality, content markers |
| `__init__.py` | — | Package init |

**Dependencies**: `datasets`, `transformers`, `torch`, `numpy`, `requests`, `beautifulsoup4`
**Internal Dependencies**: `src.config.schema` (Config, DatasetEntryConfig)

---

### `src/config/` — Configuration Schema

**Purpose**: Pydantic v2 configuration loading and validation.

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `schema.py` | 666 | 30+ Pydantic models — Config (root), ModelConfig, TrainingConfig, DataConfig, DistributedConfig, FSDPConfig, MethosV3Config, NSLTConfig, MoEConfig, etc. Cross-field validation, auto-migration |
| `__init__.py` | — | Package init |

**Dependencies**: `pydantic>=2.0.0`, `pyyaml`
**Key Detail**: `Config` model has `model_validator` for cross-field validation (e.g., FSDP requires distributed strategy `== 'fsdp'`)

---

### `src/methos_v3/` — Core Model Architecture

**Purpose**: Methos V3/V4 — 11-layer workspace-centric cognitive architecture with SSM compression, hierarchical planning, specialist debate, symbolic tools, and quality assurance.

**Files** (22 files):

| File | Lines | Layer / Module |
|---|---|---|
| `model.py` | 414 | Core model — MethosV3Config, MethosV3Model, MethosV3ForCausalLM, MoEMethosV3Model |
| `layer1_tokenizer.py` | — | Intelligent Tokenizer — multi-granularity tokenization |
| `layer2_embedding.py` | — | Adaptive Semantic Embedding |
| `layer3_memory.py` | — | Hierarchical Memory Engine — SSM compression + HSSM multi-scale |
| `layer4_intent.py` | — | Intent Understanding + Adaptive Difficulty Router |
| `layer5_planner.py` | — | Global Planner / Hierarchical Planner |
| `layer6_reasoning.py` | — | Adaptive Continuous Reasoning (ODE) |
| `layer7_workspace.py` | — | Workspace Integration |
| `layer8_specialists.py` | — | Specialist Sandbox — 7 specialists, multi-round debate |
| `layer9_reflection.py` | — | Reflection Module |
| `layer10_verification.py` | — | Verification Module |
| `layer11_decoder.py` | — | Hierarchical Sparse Decoder (3-level: Semantic → Language → Token) |
| `executive.py` | — | Executive Controller — RL-trained gate enforcement |
| `workspace.py` | — | CognitiveWorkspace — central dict-based hub |
| `tools.py` | — | Internal Tool Interface |
| `world_model.py` | — | World Model — entity extraction, relation network |
| `hssm.py` | — | Hierarchical SSM — multi-scale memory |
| `losses.py` | — | Auxiliary Loss Computer — 14 auxiliary losses |
| `curiosity.py` | — | Curiosity Module |
| `quality_assurance.py` | — | Merged QA module — reflect + verify + self-evaluate + correct |
| `learning_controller.py` | — | Learning Controller for curriculum |
| `__init__.py` | — | Package init |

**Dependencies**: `torch`, `transformers` (PretrainedConfig, PreTrainedModel)

---

### `src/nslt/` — Neural State-Space Liquid Transformer

**Purpose**: Alternative non-Transformer architecture — O(1) memory via SSM compression, continuous ODE reasoning via LTC, energy-based reasoning via LatentSandbox, sparse output.

**Files** (10 files):

| File | Lines | Responsibility |
|---|---|---|
| `model.py` | 719 | Full NSLT model — NSLTModel, MoENSLTModel, TokenEmbedding, RotaryPositionEncoding |
| `layer1_ssm.py` | — | SSM Compression Engine — structured state-space sequence modeling |
| `layer2_ltc.py` | — | LTC Routing Layer — liquid time-constant ODE |
| `layer3_sandbox.py` | — | Latent Sandbox — energy-based parallel vector reasoning |
| `layer4_output.py` | — | Sparse Output Synthesizer — top-1% vocabulary gating |
| `mcts_sandbox.py` | — | MCTS-based Latent Sandbox — Monte Carlo tree search in latent space |
| `moe_ssm.py` | — | Mixture-of-Experts SSM Block |
| `multiscale_ssm.py` | — | Multi-scale SSM |
| `ssm_scan.py` | — | SSM scan utilities (associative scan) |
| `vision_encoder.py` | — | SigLIP Vision Encoder integration for multimodal support |

**Dependencies**: `torch`, `transformers`

---

### `src/training/` — Training Orchestration

**Purpose**: Multi-stage training orchestration — runs pretrain, SFT, instruction tuning, and alignment stages with curriculum learning.

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `pipeline.py` | 556 | TrainingPipeline — stage management, checkpointing, logging, gradient checkpointing |
| `__init__.py` | — | Package init |

**Dependencies**: `src.config`, `src.data`, `src.models`, `src.infrastructure`, `src.alignment`, `src.evaluation`

---

### `src/infrastructure/` — Distributed & Experiment Tracking

**Purpose**: Distributed training setup (FSDP/DeepSpeed/DDP) and experiment tracking (wandb/mlflow/tensorboard).

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `distributed.py` | 123 | DistributedSetup — process group init, FSDP arg preparation, device placement, multi-node support |
| `tracking.py` | 79 | ExperimentTracker — wraps wandb/mlflow/tensorboard, log metrics/configs/checkpoints |

**Dependencies**: `torch.distributed`, `wandb`/`mlflow`/`tensorboard` (optional)

---

### `src/models/` — Model Factory

**Purpose**: Unified model creation and loading — instantiates any supported architecture, loads pretrained weights, estimates parameter counts.

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `factory.py` | 604 | ModelFactory — create MethosV3, NSLT, MoE variants, LLaMA, Mixtral, Qwen2MoE, DeepSeekV2 |

**Dependencies**: `torch`, `transformers` (AutoModelForCausalLM, LlamaForCausalLM, MixtralForCausalLM)

---

### `src/alignment/` — Alignment Training

**Purpose**: Alignment training — Constitutional AI, DPO, KTO, ORPO, SimPO.

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `constitutional.py` | 125 | ConstitutionalTrainer — constitutional AI training loop |
| `dpo_trainer.py` | 296 | DPOTrainer, KTOtrainer, ORPOTrainer, SimPOTrainer — preference optimization |
| `pipeline.py` | 286 | AlignmentPipeline — orchestrates alignment probes, refusal testing, safety evaluation |

**Dependencies**: `torch`, `transformers`

---

### `src/evaluation/` — Evaluation & Safety

**Purpose**: Model evaluation benchmarks and safety checking.

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `benchmarks.py` | 438 | BenchmarkRunner — HumanEval, MBPP, MMLU, GSM8K, BBH, custom benchmarks |
| `reporting.py` | — | EvaluationReport — result formatting and reporting |
| `safety.py` | 132 | SafetyEvaluator — safety probes, honesty probes, refusal keyword detection |

**Dependencies**: `torch`, `transformers`, `numpy`

---

### `src/utils/` — Shared Utilities

**Files**:

| File | Lines | Responsibility |
|---|---|---|
| `logging.py` | — | Shared logging configuration |
| `reproducibility.py` | — | Seeding — set_seed() for torch/numpy/random |

**Dependencies**: `torch`, `numpy`

---

### Root-level `src/` Modules

| File | Lines | Purpose |
|---|---|---|
| `model.py` | 168 | SpecializedCoderModel — high-level model wrapper, config loading, tokenizer setup |
| `trainer.py` | 298 | ModelTrainer — standalone trainer: prepare_data(), train(), evaluate(), save_checkpoint() |
| `dataset.py` | 988 | CodeExample dataclass — curated Python/JS coding examples for fine-tuning |
| `benchmark.py` | 1011 | Custom coding benchmark — inspired by HumanEval/MBPP, pass@1 scoring, 100+ problems |
| `validator.py` | 318 | CodeValidator — syntax checking and sandboxed execution for Python/JS |
| `generator.py` | 240 | CodeGenerator — high-level generation with batching, multi-language, code extraction |
| `tokenizer_trainer.py` | 226 | Tokenizer trainer — ByteLevelBPETokenizer training from dataset streams |
| `knowledge_graph.py` | 130 | GraphMemory — ChromaDB + NetworkX for agentic memory and knowledge graph |
| `massive_data_collector.py` | 484 | Legacy MassiveDataCollector — streams from Hugging Face, dataset filtering by libraries |
| `synthetic_labels.py` | 123 | Synthetic label generator — intent, tool selection, QA labels for multi-task training |

---

## `scripts/` — 15 Scripts

| Script | Purpose |
|---|---|
| `verify_datasets.py` | Phase 1 validation — verifies every registered dataset loads correctly (incl. local JSONL validation) |
| `production_validation.py` | 8-phase validation pipeline — end-to-end system validation |
| `check_datasets.py` | Dataset integrity checks |
| `check_stackv2.py` | Stack v2 dataset-specific validation |
| `check_scripts.py` | Script integrity validation |
| `check_tokens_and_decode.py` | Tokenization round-trip verification |
| `param_audit.py` | Parameter count audit and model memory estimation |
| `validate_pipeline.py` | Pipeline validation — streaming, filtering, packing |
| `validate_registry.py` | Registry entry validation |
| `verify_each_entry.py` | Per-entry dataset verification |
| `test_pipeline.py` | Pipeline integration test script |
| `test_local_validation.py` | Local dataset validation — missing/malformed/empty JSONL, UTF-8, schema |
| `test_integration.py` | End-to-end integration test — registry → stream → pack → tokenize → train → checkpoint |
| `final_verify.sh` | Shell-based final verification |
| `train_4gpu.sh` | Shell launcher for 4-GPU training |

---

## `tests/` — 12 Test Files

| Test File | Coverage |
|---|---|
| `test_data_pipeline.py` | Data pipeline — streaming, filtering, packing |
| `test_quality.py` | Quality scoring — document_quality_score, deduplication, contamination |
| `test_config.py` | Config loading and validation — Pydantic schema |
| `test_trainer.py` | ModelTrainer — training loop, checkpointing |
| `test_nslt.py` | NSLT architecture — forward pass, generation, SSM |
| `test_alignment.py` | Alignment — Constitutional AI, DPO, KTO |
| `test_evaluation.py` | Evaluation — benchmark runner, reporting |
| `test_generation.py` | Code generation — Generator + validator integration |
| `test_data_collector.py` | Legacy MassiveDataCollector |
| `test_foundation_pipeline.py` | Foundation config pipeline test |
| `test_integration.py` | End-to-end integration tests |
| `test_validation.py` | Code validation — syntax checking, execution |

---

## `notebooks/` — 2 Jupyter Notebooks

| Notebook | Purpose |
|---|---|
| `setup_check.ipynb` | Environment verification — GPU, CUDA, dependencies |
| `train.ipynb` | Training notebook — interactive training session |

---

## `data/`, `models/`, `docs/`

| Directory | Purpose |
|---|---|
| `data/` | Cached datasets, tokenized data, documentation JSONL files |
| `models/` | Model checkpoints, saved weights, config files |
| `docs/` | Documentation sources (output of doc_builder scraper) |

---

## Dependencies (from `requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.6.0+cu128 | Core tensor computation and neural network framework |
| `transformers` | 4.43.3 | HuggingFace model zoo, tokenizers, trainers |
| `datasets` | >=2.14.4 | HuggingFace dataset loading and streaming |
| `accelerate` | 0.33.0 | Distributed training utilities |
| `bitsandbytes` | >=0.43.0 | 8-bit / 4-bit quantization |
| `peft` | >=0.10.0 | Parameter-Efficient Fine-Tuning (LoRA, etc.) |
| `tokenizers` | >=0.19.1 | Fast tokenization |
| `pyyaml` | — | YAML config parsing |
| `pydantic` | >=2.0.0 | Config schema validation |
| `sentencepiece` | — | Tokenizer subword modeling |
| `scipy` | — | Scientific computation |
| `einops` | — | Tensor operations (rearrange, repeat, reduce) |
| `chromadb` | — | Vector database for agentic memory |
| `sentence-transformers` | — | Embedding models for semantic dedup |
| `networkx` | — | Graph operations for knowledge graph |
| `flask` | >=3.0.0 | Web chat UI |

Optional: `flash-attn`, `wandb`, `mlflow`, `tensorboard`, `human-eval`, `pylint`
