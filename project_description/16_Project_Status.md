# Project Status

## Completed Features

### Data Pipeline
- ✅ **Dataset registry** with 54 primary entries (58 registered incl. fallback-only) across 8 categories with weight normalization to match target token distribution (`CATEGORY_WEIGHTS`)
- ✅ **Streaming with fallback chains** — datasets automatically fail over to configured fallbacks; fallback usage is logged and tracked
- ✅ **Text field auto-detection** — priority-based field scanning (`TEXT_FIELD_PRIORITY`) with fallback to longest string values
- ✅ **Metadata preservation** — packed sequences carry `_dataset`, `_category`, `_language`, `_domain`, `_segments`, `_avg_quality` metadata
- ✅ **Documentation scraping system** — 18 scrapers (Python, PyTorch, NumPy, Rust Book, Go Docs, MDN, FastAPI, CUDA, cuDNN, Linux Kernel, OpenCV, Kubernetes, Docker, PostgreSQL, SQLite, ONNX, RFCs, Lang Specs) with timeout, failure tracking, and summary logging
- ✅ **Template URL hardening** — template placeholder URLs (`{{ }}`, `{% %}`, `${ }`, `<% %>`) and invalid hrefs (`javascript:`, `void(0)`, `#`) rejected at discovery *before* queue insertion in every scraper; logged as `Filtered template URL` and counted in `filtered=N`
- ✅ **CUDA discovery bug fixed** — indentation bug processed only the last href per page; all links are now followed (page count increase on regeneration comes from real pages)
- ✅ **Client-side redirect handling** in doc builder — follows `<meta refresh>` and `location.replace/href` redirects
- ✅ **Per-scraper timeout** — 600s default, configurable via `source_timeout`
- ✅ **PostgreSQL URL validation** — accepts any `/docs/<version>/` path (not only `current/`)
- ✅ **cuDNN seeds updated** — `/latest/` structure with sub-link following and dual content selector
- ✅ **Linux kernel threshold** — `MIN_TEXT_LENGTH` 3000→500 (nav sidebar excluded via `div.document` selector, so no boilerplate pages)
- ✅ **Corpus audit tool** (`scripts/corpus_audit.py`) — UTF-8/JSON/duplicate/HTML-leakage/length/token checks over all generated JSONL with PASS/CORRUPT/HIGH_DUP statuses
- ✅ **AST-based code quality filtering** — identifier ratio, executable ratio, auto-generated code detection, syntax corruption detection
- ✅ **Function-level code sampling** — extracts individual functions/classes/methods from code files via AST
- ✅ **8-component document quality scoring** — `document_quality_score()` computes composite quality from multiple heuristics
- ✅ **Weighted mixed dataset with adaptive resampling** — `WeightedMixedDataset` dynamically adjusts sampling weights based on remaining token ratios per sub-dataset
- ✅ **Sequence packing with metadata tracking** — `pack_sequences()` packs variable-length samples into fixed-length sequences with EOS separators, tracking segment counts and quality scores
- ✅ **Boilerplate removal** — license keyword scanning (first 50 lines), boilerplate file pattern filtering (`_pb2.py`, `node_modules/`, `.min.js`, etc.)
- ✅ **Exact, SimHash, and MinHash deduplication** — configurable via `DedupConfig`
- ✅ **Benchmark contamination filtering** — filters known benchmark samples (HumanEval, MBPP, MMLU, GSM8K, ARC, HellaSwag, TruthfulQA)
- ✅ **Health reporter** — `DatasetHealthReport` tracks per-dataset and global statistics with summary text output
- ✅ **Sanity checks** — `run_sanity_checks()` validates built datasets by decoding samples and checking for anomalies

### Configuration
- ✅ **Pydantic v2 configuration schema** — full type-validated config with 40+ nested models
- ✅ **v1→v2 migration** — `migrate_v1_config()` handles old config keys (`data_collection`, `fsdp`, `rope_scaling_factor`)
- ✅ **Multiple config files** for different scales: `config.yaml`, `config_foundation.yaml`, `config_small.yaml`

### Models
- ✅ **Model factory** supporting 6 architectures: llama, mixtral, qwen2_moe, deepseek_v2, nslt, dhara_v3
- ✅ **Dhara model** — 11-layer architecture (`IntelligentTokenizer` → `AdaptiveSemanticEmbedding` → `HierarchicalMemoryEngine` → `IntentUnderstanding` → `GlobalPlanner` → `AdaptiveContinuousReasoning` → `CognitiveWorkspace` → `SpecialistSandbox` → `QualityAssurance` → `HierarchicalSparseDecoder` + `ExecutiveController`), `PreTrainedModel` compatible
- ✅ **NSLT model** — 4-layer architecture (SSM Compression Engine → LTC Routing → Latent Sandbox → Sparse Output), O(1) memory complexity, RoPE position encoding
- ✅ **Sparse output layer** — `SparseOutputSynthesizer` uses top-k vocabulary projection for candidate selection; the forward pass still computes the full-vocab projection, and the opt-in `head_ce: topk` narrows the loss/backward target set
- ✅ **Model size estimation** — parameter count estimates for all 6 architectures
- ✅ **Checkpoint compatibility checking** — `is_compatible()` validates saved config against current config
- ✅ **MoE variants** — `DharaMoEModel`, `MoENSLTModel` with expert routing and load-balancing loss

### Validation
- ✅ **8-phase production validation** — automated checks for datasets, documentation, token distribution, decoded samples, packing quality, fallbacks, smoke test, checkpoint save/load
- ✅ **Production readiness report** — consolidated decision output (READY / READY WITH MINOR WARNINGS / NOT READY)
- ✅ **Dataset verification script** (`verify_datasets.py`) — tests all registry datasets for accessibility with categorized reporting
- ✅ **Local validation suite** (`test_local_validation.py`) — 14/14 passing, clean under pytest
- ✅ **Integration test fixes (transformers v5 compatibility)** — `AutoConfig.for_model()` instead of loading a tokenizer-only repo; `eval_strategy` (removed `evaluation_strategy` API); resume from trainer `checkpoint-N/` dirs (required `trainer_state.json`)
- ✅ **Corpus audit** — no blocking issues on regenerated docs (removed 0-byte corrupt artifact)

### Infrastructure
- ✅ **Distributed training setup** — FSDP and DeepSpeed configs, `DistributedSetup` class, layer class mapping for wrapping
- ✅ **Experiment tracking** — adapter layer for wandb/mlflow/tensorboard via `ExperimentTracker`
- ✅ **Reproducibility utilities** — `set_seed()`, deterministic mode option
- ✅ **Tokenizer loading** — multi-strategy tokenizer loading (AutoTokenizer, PreTrainedTokenizerFast, custom BPE from vocab/merges files)

---

## Partially Completed Features

- ⚠️ **Training pipeline** — Foundation pretrain segment exists with curriculum learning support; SFT, instruction tuning, and alignment stages are scaffolded but may need end-to-end testing
- ⚠️ **Alignment pipeline** — DPO, ORPO, SimPO, KTO trainers exist; Constitutional AI pipeline exists; may need integration testing with real preference data
- ⚠️ **Evaluation benchmarks** — `BenchmarkRunner` scaffolded with benchmark names (MMLU, HellaSwag, ARC, HumanEval, MBPP, GSM8K, TruthfulQA, Winogrande, BBH); actual benchmark implementations may not be complete
- ⚠️ **Distributed training** — FSDP config exists with layer class mapping; actual multi-GPU training on 4x A100 needs validation
- ⚠️ **Experiment tracking** — Wraps wandb/mlflow/tensorboard via `ExperimentTracker`; adapter layer may need refinement for production use
- ⚠️ **Curriculum learning** — `CurriculumConfig` and curriculum stage handling exist in training pipeline; actual curriculum data mixing may need testing
- ⚠️ **Safety training** — `SafetyEvaluator`, safety training phase, and red teaming scaffolded; integration with alignment pipeline exists but is untested

---

## Unfinished Features

- ❌ **Full tokenizer training from scratch** — `tokenizer_trainer.py` exists but falls back to pretrained HF tokenizers (Xenova/claude-tokenizer)
- ❌ **Vision encoder integration with NSLT** — `forward_multimodal()` is scaffolded in `NSLTModel`; `SigLIPVisionEncoder` class exists; `from_multimodal_config()` classmethod exists but vision pipeline is not end-to-end tested
- ❌ **MCTS-based latent sandbox** — `MCTSLatentSandbox` and `MCTSLatentSandboxEfficient` classes exist in `mcts_sandbox.py` but are marked as experimental
- ❌ **Multi-node training support** — Single-node distributed config exists; multi-node configuration is not implemented
- ❌ **Automated HF Hub dataset upload** — No script or pipeline for uploading processed datasets to HuggingFace Hub
- ❌ **REST API for model serving** — No inference server or API layer exists

---

## Experimental Components

- 🧪 **MCTS latent sandbox** (`src/nslt/mcts_sandbox.py`) — Monte Carlo Tree Search-based reasoning in latent space; alternative to energy-based sandbox
- 🧪 **MoE variants** (`DharaMoEModel`, `MoENSLTModel`) — Mixture-of-Experts SSM blocks; not benchmarked against base variants
- 🧪 **Triton SSM scan kernel** (`src/nslt/ssm_scan.py`) — `selective_scan_vectorized()` function scaffolded for GPU-accelerated SSM scan; not verified against native implementation
- 🧪 **Curricula learning** — Config parameters exist and are wired through the training pipeline; effectiveness not validated
- 🧪 **Auxiliary losses** — `AuxiliaryLossComputer` in `src/dhara/` supports multiple auxiliary loss functions; loss weight tuning is experimental
- 🧪 **World model** — `WorldModel` module in `src/dhara/` for entity tracking and event prediction; not validated on real tasks

---

## Deprecated Code

- `MassiveDataCollector` in `streaming.py` — Legacy wrapper class used by the old `DataPipeline.build_pretrain_dataset()` path. Replaced by `stream_dataset_with_fallbacks()` in registry mode.
- `src/massive_data_collector.py` — Standalone module; all functionality moved into `src/data/streaming.py` and `src/data/pipeline.py`
- Some `_register_*` functions in `registry.py` — Code comments in `_register_science`, `_register_books`, `_register_structured` note that previous datasets were removed due to compatibility issues (s2orc, pubmed, pg19, WIT, wikidata, etc.)

---

## Unused Code

- `scripts/check_stackv2.py` — Debugging script for Stack v2 dataset inspection
- `scripts/check_tokens_and_decode.py` — Debugging script for token inspection
- `scripts/param_audit.py` — Parameter auditing script
- `scripts/check_datasets.py` — May be superseded by `verify_datasets.py`
- `scripts/validate_registry.py` — May be superseded by `verify_datasets.py`
- `src/massive_data_collector.py` — Standalone module not integrated with the new registry-based data pipeline
- `src/synthetic_labels.py` — Not referenced in the main pipeline or training code
- `src/knowledge_graph.py` — Standalone knowledge graph module; not integrated with any model or data pipeline
- `src/generator.py` — If not used by main training or evaluation scripts

---

## Files That Can Likely Be Removed

| File | Reason |
|---|---|
| `src/massive_data_collector.py` | Replaced by `src/data/streaming.py` registry pipeline |
| `src/synthetic_labels.py` | Not referenced in main pipeline |
| `src/knowledge_graph.py` | Standalone, not integrated |
| `src/generator.py` | Check if used by main training |
| `scripts/check_stackv2.py` | Debugging script |
| `scripts/check_tokens_and_decode.py` | Debugging script |
| `scripts/param_audit.py` | Auditing script, not part of pipeline |
| `scripts/check_datasets.py` | Superseded by `verify_datasets.py` |
| `scripts/validate_registry.py` | Superseded by `verify_datasets.py` |

---

## Files Requiring Refactoring

| File | Lines | Issue |
|---|---|---|
| `src/data/pipeline.py` | 2604 lines | Very long; could be split into `quality.py`, `packing.py`, `dataset_building.py` |
| `src/dhara/model.py` | 490 lines | `forward()` method is complex with many branch paths |
| `scripts/production_validation.py` | 778 lines | 29KB; could be split by phase into separate modules |
| `src/config/schema.py` | 765 lines | Many config classes; could be split into sub-modules |
| `src/nslt/model.py` | 759 lines | Combines `NSLTModel`, `MoENSLTModel`, and support modules; could be split |
