# Future Work

Categorized by priority level. Items marked with a dagger (†) are currently blocked by external dependencies.

---

## Critical

These issues must be resolved before production training can proceed reliably.

- **Complete test coverage for training pipeline**
  - Target: tests/test_trainer.py, tests/test_foundation_pipeline.py
  - Verification: 10-step training with loss decrease, optimizer stepping, gradient accumulation correctness
  - Need: parameterized fixtures for tiny model configs, gradient flow assertions

- **Fix PyTorch doc scraping (Cloudflare bypass)**
  - Target: src/data/doc_builder.py (PyTorchDocScraper)
  - Options: (1) Switch to readthedocs.io mirror, (2) Integrate cloudscraper library, (3) Use Playwright/Selenium with stealth
  - Risk: If unfixed, pytorch-docs contributes 0 pages to the corpus

- **Track config_foundation.yaml in version control** ✅ **RESOLVED**
  - Target: git tracking for config_foundation.yaml — now tracked in git, all config changes versioned and reviewable

- **Request GAIR/MathPile access** ✅ **RESOLVED**
  - Target: https://huggingface.co/GAIR/MathPile
  - Access granted — 0.010 weight now loads; fallback: open-web-math/open-web-math (still configured)

- **Complete production validation and fix all Phase failures**
  - Target: scripts/production_validation.py
  - Run all 8 phases, fix all warnings and failures until status is "READY FOR FULL PRETRAINING"
  - Key metrics: corpus availability >= 90%, padding < 5%, token distribution within 3% of targets

- **Remove hardcoded HF token from config_foundation.yaml** ✅ **RESOLVED**
  - Target: config_foundation.yaml
  - `data.hf_token` is now empty in all configs; set the `HF_TOKEN` environment variable instead

---

## High

Significant improvements that should be implemented before or during production training.

- **Implement all test placeholder implementations**
  - Target: tests/test_alignment.py, tests/test_evaluation.py, tests/test_generation.py, tests/test_integration.py
  - Many tests currently pass trivially (scaffolded asserts)
  - Need: actual forward pass, loss computation, and metric assertions

- **Fix PostgreSQL doc scraping**
  - Target: src/data/doc_builder.py (PostgreSQLDocScraper)
  - Currently only discovers 1 page due to Bootstrap/JS navigation
  - Research: sitemap.xml, static HTML mirror, or alternative host

- **Improve Docker/Kubernetes doc scraping**
  - Target: src/data/doc_builder.py (DockerDocScraper, KubernetesDocScraper)
  - Currently yields 9-12 pages; investigate JS-rendered navigation or rate limiting
  - May require headless browser or sitemap-based discovery

- **Add validation for alignment pipeline**
  - Target: src/alignment/pipeline.py
  - Tests for DPO/ORPO/SimPO/KTO training, constitutional alignment, red teaming
  - Verify reward model training, preference pair handling, loss computation

- **Add integration tests for end-to-end pipeline**
  - Target: tests/test_integration.py
  - Full pipeline test: config → data loading → tokenization → training → evaluation
  - Use tiny model configs for fast execution

- **Implement more rigorous packing quality metrics**
  - Target: src/data/pipeline.py (pack_sequences)
  - Add: per-dataset packing efficiency, cross-contamination checks, sequence boundary quality
  - Current: only aggregate padding percentage and average segments

- **Enhance NSLT test coverage**
  - Target: tests/test_nslt.py
  - Add gradient flow tests per sub-module (SSM, LTC, Sandbox, SparseOutput)
  - Add ODE solver accuracy comparison (Euler vs RK4 vs adjoint)
  - Add numerical stability tests (NaN/Inf propagation)

---

## Medium

Valuable improvements that enhance quality, performance, or maintainability.

- **Replace MassiveDataCollector with StreamingManager everywhere**
  - Target: src/data/streaming.py, all callers in pipeline.py
  - Legacy wrapper adds unnecessary indirection; StreamingManager provides cleaner API
  - Migration: update DataPipeline to use StreamingManager directly

- **Add more programming languages to STACK_V2_LANGUAGES**
  - Target: src/data/registry.py:157
  - Currently 19 language targets. Add: Ruby, Dart, Elixir, Haskell, OCaml, Zig, Nim
  - Update CODE_LANG_TARGETS accordingly with proportional weights

- **Implement tokenizer training from scratch**
  - Target: src/tokenizer_trainer.py
  - Currently uses `Xenova/claude-tokenizer` (pretrained BPE tokenizer)
  - Custom tokenizer optimized for code + technical documentation vocabulary

- **Add multi-node training support**
  - Target: src/infrastructure/distributed.py
  - Currently single-node multi-GPU only (4× A100)
  - Add: NCCL init with multi-node env vars, gradient checkpoint distribution, data sharding

- **Performance optimization for SSM scan kernels**
  - Target: src/nslt/ssm_scan.py
  - Current: Python reference implementation
  - Options: Triton kernels, CUDA custom ops, or selective scan (Mamba-style)

- **Add curriculum stage for long-context adaptation**
  - Target: config_foundation.yaml, src/training/pipeline.py
  - After pretraining at 2048/4096, add a fine-tuning stage at 8192/16384 with RoPE scaling
  - Use the existing YaRN scaling config

- **Add AST filtering support for more languages**
  - Target: src/data/ast_filter.py
  - Currently Python, JavaScript, TypeScript, Rust, Go, Java, C/C++
  - Add: Ruby, PHP, Kotlin, Swift, Lua

---

## Low

Nice-to-have improvements that can be addressed as time permits.

- **Add more documentation sources**
  - Candidates: Rust standard library docs, Go standard library reference, Julia docs, Swift docs
  - Each requires a new scraper class (most can extend SphinxScraper)

- **Add benchmark contamination dataset expansion**
  - Target: src/data/quality.py (ContaminationFilter)
  - Expand benchmark list beyond HumanEval, MBPP, MMLU, GSM8K, ARC, HellaSwag, TruthfulQA
  - Add MATH, BBH, AGIEval, TheoremQA

- **Implement automated HF Hub dataset caching**
  - Target: src/data/pipeline.py
  - Automatically check HF Hub for pre-tokenized/packed versions of datasets
  - Skip local processing when remote cache is available

- **Add Docker support with docker-compose**
  - Target: Dockerfile, docker-compose.yml
  - Containerized training environment with pinned dependencies
  - Volume mounts for cache, models, and data directories

- **Pre-commit hooks for formatting/linting**
  - Target: .pre-commit-config.yaml
  - ruff, black, mypy, pytest
  - CI integration via GitHub Actions

- **Add piped import aliases for all merged modules**
  - Target: src/dhara/__init__.py
  - Old modules (ReflectionModule, VerificationWithRepair, CuriosityModule, LearningController) should be clearly marked as deprecated wrappers
  - Add deprecation warnings to all backward-compatible re-exports

---

## Nice to Have

Polishing and features for broader accessibility.

- **Gradio/Streamlit demo interface**
  - Interactive web UI for model inference
  - Show reasoning steps, specialist debate, QA passes

- **REST API for inference**
  - FastAPI-based serving endpoint
  - Support: text generation, batch inference, streaming

- **Model quantization (GPTQ/AWQ)**
  - Reduce 10.55B model to 4-bit for single-GPU inference
  - Evaluate quality degradation vs memory savings

- **LoRA/QLoRA fine-tuning integration**
  - PEFT-based fine-tuning for task-specific adaptation
  - Support: target modules, rank config, alpha scaling

- **Synthetic data generation pipeline**
  - Generate auxiliary loss targets (task type, difficulty, reasoning type, tool selection)
  - Rule-based classifiers + LLM-based generation

- **Automated hyperparameter search**
  - Optuna or Ray Tune integration
  - Search: learning rate, warmup steps, weight decay, gradient accumulation
  - Use the small config (173M) for fast iteration

---

## Research Ideas

- **Ablation: SSM vs Transformer backbone for Dhara**
  - Compare: SSM-based compressed state vs standard Transformer hidden state
  - Metrics: perplexity, throughput, memory usage at various sequence lengths

- **Ablation: Workspace-centric vs standard residual architecture**
  - Compare: dict-based hub communication vs standard residual stream
  - Metrics: gradient flow, module utilization, training efficiency

- **MCTS-based decoding vs standard autoregressive generation**
  - Replace greedy/beam search with Monte Carlo Tree Search
  - Use the LatentSandbox energy function as the evaluation metric

- **Multi-modal extension of NSLT**
  - Vision encoder already scaffolded in VisionConfig + MultimodalConfig
  - Integrate SigLIP encoder with projection to d_model
  - Train on image-caption data

- **Continuous-time ODE vs discrete reasoning steps**
  - Compare: Neural ODE (continuous depth) vs Transformer layers (discrete depth)
  - Metrics: parameter efficiency, reasoning accuracy, gradient stability

- **Energy-based vs autoregressive reasoning in LatentSandbox**
  - LatentSandbox currently uses energy minimization for parallel reasoning
  - Compare with autoregressive step-by-step reasoning
  - Metrics: correctness, diversity of solutions, computational cost
