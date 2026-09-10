# Dhara Class Model — Project Description

**Dhara** (formerly "Methos V3/V4") is a non-Transformer, workspace-centric cognitive
architecture for a large language model: instead of a dense Transformer stack, it uses
structured state-space (SSM) compression with **O(1) memory** in sequence length and a
hierarchical, multi-agent reasoning pipeline. The project ships a production-grade data
pipeline (54-dataset registry, web documentation scraping, quality/dedup, streaming with
fallbacks), a staged pretraining orchestration, two model implementations (**Dhara** and
the alternative **NSLT**), and a full alignment/evaluation stack — 26 test files / 287
tests green.

## Design Principles

- **Correctness-aware supervision.** Training data is tagged as *correct* versus
  *possibly incorrect*, not merely ranked by "most likely". Medical data in particular
  is curated so the model learns to treat uncertain labels as uncertain — it is trained
  to reason about the confidence of its ground truth rather than regressing
  uncritically toward a single most-likely answer.
- **Multi-agent sandbox for many domains.** Reasoning happens in a sandbox that hosts
  multiple specialist agents covering many domains. These agents are orchestrated
  around a shared workspace, debate one another, and converge on a verified answer —
  instead of a single monolithic forward pass.

## Architecture Layers

Dhara is an 11-layer workspace-centric pipeline, driven by an RL-trained
**Executive Controller** that enforces gates between stages:

1. **Tokenization** — Intelligent tokenizer that attaches semantic metadata (token
   category, language, doc role, task type) to each token, so later layers route on
   meaning rather than raw subword ids.
2. **Embedding** — Adaptive semantic embedding over the metadata-rich tokens, with
   RoPE position encoding.
3. **Memory** — MemoryManager across working, semantic, long-term, and episodic memory
   with forget/compress/retrieve/prioritize operations.
4. **Intent** — Intent understanding and difficulty routing (5 levels) selects how
   deeply the query is processed.
5. **Planning** — HierarchicalPlanner builds goal trees with dependencies, execution
   graphs, and cost estimates for multi-step tasks.
6. **Reasoning** — Adaptive continuous reasoning over a Neural ODE with liquid
   time-constants; trajectories can step forward and be revised.
7. **Workspace** — CognitiveWorkspace is the central dict-like hub where all layers and
   agents read/write shared state (the "chalkboard").
8. **Specialist debate** — DebateSandbox hosts per-domain specialist agents
   (code, math, science, medicine, web, DB, …) that debate, cross-examine, repair, and
   reach consensus — the multi-domain multi-agent core.
9. **Reflection** — Recursive reflection reviews the reasoning trace for gaps or
   contradictions.
10. **Verification** — VerificationWithRepair checks outputs against the trace and
    repair loop (now unified in QualityAssurance).
11. **Decoding** — HierarchicalDecoder renders the final answer semantic → language →
    token, with ultra-sparse output gating.

Supporting modules: **WorldModel** (entities, relations, cause-effect event prediction),
**SymbolicToolRouter** (calculator/Python/search/DB with learned routing), and
**AuxiliaryLossComputer** with 14 configurable auxiliary losses for supervision on the
intermediate cognitive stages.

## Data & Training

- **Registry**: 54 primary datasets across 8 categories (code 0.30, web text 0.20,
  docs 0.15, Wikipedia 0.10, math 0.10, science 0.05, books 0.05, structured knowledge
  0.05), plus 4 fallback-only entries; every entry has a fallback chain so a gated or
  downed source never blocks training.
- **Docs**: 18 built-in web scrapers build a self-hosted documentation corpus.
- **Quality**: AST-based code filtering, function-level sampling, simhash dedup,
  contamination filtering against public benchmarks, toxicity filtering.
- **Foundation pretraining**: 8 staged category groups (code → docs → web → wiki → math
  → science → books → structured) over 50k steps on a single cosine schedule, with
  per-stage checkpoints, disk cache, and async prefetch.
- **Configs**: `config_foundation.yaml` (~160M), `config_small.yaml` (~173M),
  `config.yaml` (full-scale ~10.55B, FSDP + CPU offload on 4× A100 80GB).

## Status

Pre-release. Data pipeline and model plumbing are production-polished; full pretraining
rollouts are the current milestone. See `project_description/` for architecture, API,
and release-readiness documentation.