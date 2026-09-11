# Dhara Class Model — Project Description

**Dhara** (formerly "Methos V3/V4") is a non-Transformer, workspace-centric cognitive
architecture for a large language model. Instead of a dense Transformer stack, it uses
structured state-space (SSM) compression with **O(1) memory** in sequence length and a
hierarchical, multi-agent reasoning pipeline. Today the model is being trained to
generate code. The longer-term goal is a model that can reason across any domain — and,
critically, be honest about uncertainty: it should say what is actually correct and also
tag what might not be correct, report not just the most likely answer but also what the
answer could be, and do this for medical data as well as everything else, using its
sandbox of multiple specialist agents to tackle each field.

The project ships a production-grade data pipeline (54-dataset registry, web
documentation scraping, quality/dedup, streaming with fallbacks), a staged pretraining
orchestration, two model implementations (**Dhara** and the alternative **NSLT**), and a
full alignment/evaluation stack — 26 test files / 287 tests green.

## Architecture Layers

Dhara is an 11-layer workspace-centric pipeline. Instead of every module calling its
neighbors directly, each layer writes its output to a central **CognitiveWorkspace** — a
dict-based hub with `read`, `write`, `read_all`, and `clear` operations — and later
layers read from it. An RL-trained **ExecutiveController** enforces gates between stages
using REINFORCE, learning when each module should activate for a given input.

1. **Tokenization** — IntelligentTokenizer attaches semantic metadata (token category,
   language, document role, task type) to every token embedding, so later layers route
   on meaning rather than raw subword ids.
2. **Embedding** — AdaptiveSemanticEmbedding applies Rotary Position Encoding (RoPE,
   base theta ~10M) with task-context conditioning and context-adapter blocks.
3. **Memory** — HierarchicalMemoryEngine fuses four memory systems: working memory
   (~512 slots), semantic memory over learned concept embeddings (~4096), a
   long-context SSM that compresses the sequence, and episodic memory (~256 episodes).
   It returns a fused representation plus a compressed state dict.
4. **Intent** — IntentUnderstanding predicts task type, difficulty level, and reasoning
   type; an AdaptiveDifficultyRouter then decides how many reasoning steps the query
   deserves (easy → shallow, hard → deep).
5. **Planning** — GlobalPlanner produces a goal tree with subgoal embeddings,
   dependency graphs, execution plans, and per-subgoal cost estimates for multi-step
   tasks.
6. **Reasoning** — AdaptiveContinuousReasoning evolves trajectories through a Neural ODE
   over multi-domain dynamics, with configurable solve steps and domain-specific
   dynamics.
7. **Executive control** — ExecutiveController gates the pipeline modules using learned
   activation scores (REINFORCE-trained), deciding which components run for a given
   input.
8. **Specialist debate** — SpecialistSandbox runs multi-round debate among specialist
   agents across domains (7 default agents) with cross-examination, repair, and
   consensus mechanisms — the multi-domain, multi-agent core.
9. **Tools and world model** — InternalToolInterface executes symbolic tools
   (calculator, Python, web search, database) with learned routing; the WorldModel
   extracts entities, builds relation networks, and models cause-effect events.
10. **Quality assurance** — a merged multi-pass module performing reflection,
    verification, self-evaluation, and correction until convergence (unifies the former
    reflection / curiosity / verification paths).
11. **Decoding** — HierarchicalSparseDecoder renders the answer in three levels —
    semantic concept routing → language-group routing → adaptive top-k token selection —
    instead of a full-vocabulary softmax.

Supporting modules: **AuxiliaryLossComputer** provides 14 auxiliary losses beyond
next-token prediction — intent cross-entropy, memory reconstruction MSE, verification
BCE, **calibration (expected calibration error)**, trajectory smoothness, tool
selection, entity prediction, and more — so the intermediate cognitive stages get direct
supervision. The **CuriosityModule** drives exploration of novel or uncertain states.

## Where It Is Today

- **Primarily code generation.** The current training mix is code-heavy (code 30% of
  the registry weight), and the pipeline is optimized for producing and reasoning about
  programs.
- **Foundation pretraining**: 8 staged category groups (code → docs → web → wiki → math
  → science → books → structured) over 50k steps on a single cosine schedule, with
  per-stage checkpoints, disk cache, and async prefetch.
- **Data**: 54 primary datasets across 8 categories plus 4 fallback-only entries, every
  entry with a fallback chain; 18 built-in web scrapers for a self-hosted docs corpus;
  AST-based code filtering, simhash dedup, contamination filtering.
- **Configs**: `config_foundation.yaml` (~160M), `config_small.yaml` (~173M),
  `config.yaml` (full-scale ~10.55B, FSDP + CPU offload on 4× A100 80GB).

## The Goal: Certainty-Aware Reasoning Beyond Code

*This direction was outlined in a discussion with the professor and is the guiding
vision the architecture is built around.*

Training the model on medical data instead of (or in addition to) code is the test case
for a much deeper capability the architecture is designed for:

- **Say what is correct, and tag what might not be.** A medical answer cannot just be
  confident — it must also flag uncertainty. The model should be trained to distinguish
  what is actually correct from what might be incorrect, marking its level of
  confidence instead of hiding it behind a single most-likely output.
- **Report what could be, not just what is most likely.** Beyond the top guess, the
  model should surface the plausible alternatives — the diagnoses, treatments, or
  answers that *could* be right — so a human (or another agent) can weigh them.
- **Not only medical.** This certainty-aware behavior should generalize: finance, law,
  engineering, research — anywhere a confidently wrong answer is worse than an honest
  uncertain one.
- **The sandbox makes this possible.** The DebateSandbox can hold multiple agents
  covering many domains at once. A medical question can be taken up by a diagnostician
  agent, a pharmacologist agent, and a skeptical reviewer agent that debate
  cross-examine each other's claims, flag the points they are unsure about, and converge
  on an answer annotated with what is verified versus what remains possible.

## Status

Pre-release. Data pipeline and model plumbing are production-polished; full pretraining
rollouts are the current milestone. See `project_description/` for architecture, API,
and release-readiness documentation.