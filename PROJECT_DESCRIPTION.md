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
full alignment/evaluation stack — 26 test files / 292 tests green.

---

## How to use this document

This document exists to be *taught and explained live*. Every part is written so it can
be read aloud as the skeleton of a lecture, and it is structured around the three
questions a class will actually have:

1. **How does this work?** — the mechanism, in code-accurate detail.
2. **Why does each component exist?** — the reasoning, including the failure mode the
   component is designed to avoid.
3. **Which parts are new, and which are adopted from prior work?** — at the level of
   *both* components and the individual mechanisms inside them.

Every layer section therefore follows the same template:

> **What it does** → **Why it exists** → **Key mechanisms** → **Novelty (component and
> mechanism)** → **Honest caveats**.

Novelty is coded in three buckets:

- **(A) adopted** — a mechanism from prior work, essentially as published.
- **(S) synthesis** — several known ideas combined here into an integrated module.
- **(N) novel** — the framing or design is this project's contribution, even when it
  rests on prior foundations.

---

## Presenter's roadmap (60–75 minutes)

A workable lecture plan using this document, with suggested timings. Each item tells
you which part(s) to read aloud and what to *draw* on the board while talking.

| Minutes | What you teach | Read | Board diagram |
|---|---|---|---|
| 0–5 | What Dhara is, in one paragraph | Part 1 | The 11-layer pipeline (Part 4.1) |
| 5–15 | The problem: attention's quadratic cost, monolithic stacks, uncalibrated confidence; why not a Transformer | Part 2, 2.5 | Quadratic vs. linear memory growth curve |
| 15–25 | The five design principles (each exists to avoid a concrete failure) | Part 3 | Five cards: statement → failure → mechanism |
| 25–35 | Prerequisite concepts so the layers make sense (SSM, scan, RoPE, CE loss, workspace, REINFORCE) | Part 1.5 | SSM recurrence `h_t = A h_{t-1} + B x_t` |
| 35–55 | Deep dive: each layer — what it does, why it exists, key mechanism, honest gap | Part 5 (+4.2) | One box per layer, arrows through the workspace |
| 55–60 | Supervision: the 14-loss family and "why calibration as a loss" | Part 6 | Aux-loss table |
| 60–65 | Engineering reality: staged pretraining, the step-0 deadlock we found, the ops tooling, telemetry reading | Parts 4.4–4.5, 9.5 | Training log annotated in the margin |
| 65–70 | Honest limitations (say these *before* anyone asks) | Part 8 | The train/serve skew diagram |
| 70–75 | Questions | Part 11 | — |

Optional extras if you have time: the certainty-aware vision (Part 10), the alignment/eval
stack (Part 9.6), and the glossary (end of document).

---

# Part 1 — Executive summary (a 30-second talk track)

> "Dhara is a non-Transformer language model. Instead of a stack of attention blocks, it
> is a pipeline of specialized cognitive modules — tokenization, memory, intent,
> planning, reasoning, executive control, specialist debate, tools, quality assurance,
> decoding — communicating through a shared workspace, supervised by auxiliary losses at
> every stage, and compressing the sequence with a state-space core so memory stays
> constant rather than quadratic in sequence length. Two ideas drive it. First, that
> several specialized processes sharing a global workspace reason better than one
> monolithic network — this is the hypothesis, not yet the proof. Second, that a learned
> executive controller should decide *which* of those processes to run for each input,
> so easy questions skip most layers. Most of the mechanisms are adopted from
> established work — state-space models, neural ODEs, RoPE, multi-agent debate,
> reflection/verification loops, REINFORCE, tool use, auxiliary losses. What is new is
> how they are composed: the workspace-centric pipeline with a dict-based shared buffer,
> a hierarchical (multi-level) SSM core, specialist debate executed directly inside the
> training graph, a three-level conditioning decoder, and calibration used as a training
> signal. The aim of the composition is a model that can be honest about what it does not
> know — which is also the end goal for our medical-data direction."

---

# Part 1.5 — The concepts you should teach before the architecture

A class cannot judge the architecture unless it first has the pieces. This part is a
compressed foundations lecture: teach these seven ideas, and every layer in Part 5
becomes obvious. None of this is new; all of it is prior art that Dhara adopts.

## 1.5.1 What a language model is, and what "training" means

A language model is a function that, given a sequence of tokens, produces a probability
distribution over the *next* token:

    P(token | preceding tokens)

- **Tokens** are the atoms of text. Dhara uses `vocab_size = 64000`: an
  intelligent/sentencepiece-style tokenizer that turns text into ids, reserving special
  tokens like `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`.
- **Next-token prediction** is the training task. We feed a sequence and ask the model
  to predict token *t+1* from tokens 0..t. This is permissive: any text is its own
  training signal, so you can train on the whole internet without labels.
- **The loss.** Standard training minimizes *cross-entropy*: given the model's predicted
  distribution and the true token, we maximize the log-probability of the true token.
  For a vocabulary of V tokens the softmax costs O(V) per position:

      loss = -log P(true_token)          (per position, averaged over the batch)

  With V = 64000 and 4096 positions, this one term is an enormous tensor: 64000 × 4096
  logits per example. *This is precisely the term that dominates training cost in Dhara
  too* — more below in Part 9.5.

**Why next-token prediction is a reasonable universal objective.** Generating good next
tokens requires an implicit model of syntax, semantics, world knowledge, and reasoning.
It is crude supervision but infinitely abundant; the art is the *architecture* that turns
abundant input into a useful internal computation.

## 1.5.2 Why Transformers are expensive on long text (the quadratic problem)

A Transformer computes self-attention: every position attends to every previous position.

    attention(a, b) ∝ softmax( (Q a) · (K b) / sqrt(d) )

- **Compute** per step is O(sequence²) because you need a similarity for *every pair*
  of positions.
- **Memory** per step is O(sequence) even with caching — the "KV cache" that stores one
  key and value vector per position so generation does not recompute everything.

For a 200-page codebase or a multi-year medical record (say 100k tokens), a quadratic
model is impractical on fixed hardware; even linear-cache models spend most of their
memory on the cache. **The whole Dhara project is, in part, an attempt to remove this
term from the cost** by replacing the attention stack with a *state-space* core that keeps
a fixed-size state (Part 1.5.3).

## 1.5.3 State-space models (SSM) — the O(1)-memory replacement for attention

A state-space model is a recurrent update with a hidden state h:

    h_t = A h_{t-1} + B x_t      (state update)
    y_t = C h_t                  (read-out of the state)

Key facts to teach:
- **Memory is fixed.** The state h has a size chosen by the architect (Dhara:
  `d_state = 288` per channel) and does *not* grow with sequence length. This is the
  O(1)-in-length claim — the entire history is compressed into the state.
- **But recurrence is sequential.** A naive loop over t is slow to train. The trick is
  the **parallel scan**: because the recurrence is *linear*, the whole sequence can be
  processed with a prefix-sum-like operation in parallel. `h_1 .. h_T` are computed in
  O(log T) sequential steps with O(T) parallel work.
- **Modern variants** (S4, H3, Mamba) make A, B, C *input-dependent* and add gating
  (SiLU), which gives the model selectivity: it can choose what to keep in state.
- **Dhara's twist (the "Hierarchical" HSSM):** instead of one state channel per layer,
  each layer keeps **three stacked state channels** with separate B/C projections
  (`n_levels = 3`), a richer compressed state for the same parameter cost — a small,
  concrete contribution over single-level S4/Mamba.

Honest caveat to mention in class: Dhara's scan is a cumulative-sum parallel form, not a
hardware-aware selective-scan kernel (no CUDA `scan`). Fine at training sizes, to be
replaced at full scale.

## 1.5.4 Rotary position encodings (RoPE) and why position matters

Sequences are ordered, so the model needs some notion of "token i is near token j".
**RoPE** (Su et al., 2021) bakes relative position into the representations by rotating
pairs of coordinates by an angle proportional to absolute position, for each attention
head. Two facts to teach:
- **Relative** structure falls out for free: the dot product of two rotated vectors
  depends on their *distance*, not their absolute addresses.
- **Context extension** by rescaling the base rotation period (YaRN, Peng et al., 2023)
  lets a model trained at 4096 context run at longer contexts with mild degradation.
  Dhara's base θ ≈ 10M (vs. Llama's 500k) is a deliberate choice: finer resolution at
  short offsets, appropriate for a model whose compression core does the long-range work.

## 1.5.5 Cross-entropy, calibration, and why probabilities lie

Two different things are both called "the probability":
- **The softmax over tokens** is a *training* output: it is trained to be correct, but
  nothing forces it to be *well-calibrated* (see 1.5.6).
- **Calibrated confidence** means: when the model says 0.8, it is right 80% of the time.

A language model trained with plain cross-entropy tends to be *overconfident* on
out-of-distribution input. This is why Dhara carries machinery *whose job is* confidence:
a confidence head, verification heads, and a calibration *loss* (Part 6.3). In class:
ask "does a high softmax mean the model is sure?" — the answer is "no, and the whole
point of the calibration loss is to change that."

## 1.5.6 The Global Workspace theory — where "workspace-centric" comes from

Cognitive science (Baars's Global Workspace Theory; Dehaene's Global Neuronal
Workspace; Minsky's Society of Mind) argues that cognition is many specialized
processors competing for access to a **shared broadcast buffer**. Whatever gets into the
workspace is what a whole set of downstream processes can act on. Dhara is a direct
adaptation: every module **writes** to `CognitiveWorkspace` and **reads** from it, instead
of being wired to each other point-to-point:

    worker_specialists ──write──▶ workspace ──broadcast──▶ consumer_modules

This is, to be honest, a *theory of mind architecture made into a neural topology* — an
engineering bet, not an established law.

## 1.5.7 REINFORCE and why an "executive" can learn to route

The ExecutiveController decides which modules run. To train that decision when
correctness is ultimately measured by task loss, the project uses **REINFORCE**
(Williams, 1992), a *policy gradient* algorithm:

- Treat "which gates the executive set" as a decision the model sampled; the reward is
  `-task_loss - λ·(number of active gates)` (do well, and be cheap).
- The gradient is `reward × log-probability-of-the-decision`, with a baseline
  subtracted to reduce variance. Over many steps, decisions that led to low loss get
  more probable.
- A second knob, **Adaptive Computation Time** (Graves, 2016), is the conceptual basis
  for spending more steps on hard inputs: difficulty→step-count routing.

**Honest note for class:** the REINFORCE loop is implemented and unit-tested, but in the
current training forward the executive is called without a task-loss arm, so the reward
buffer does not fill and the controller is at best weakly trained. This is a *design
statement*, not a current-run fact — and the distinction is stated throughout the doc.

---

# Part 2 — Problem statement and motivation

## 2.1 What problem are we attacking?

Large language models are currently almost synonymous with the Transformer decoder:
alternate self-attention and MLP blocks, learned with next-token prediction. That stack
has three properties we consider limitations for the target use cases:

1. **Quadratic memory/compute in sequence length (or linear KV-cache memory).** Long
   documents — a 200-page codebase, a multi-year medical record, a research paper plus
   its references — are exactly what a reasoning model should handle, and they are the
   regime where self-attention is most expensive.

2. **One monolithic flow.**
   A single stack is forced to be simultaneously a *scanner* (understand the input),
   a *retriever* (pull in the right memory), a *reasoner* (structure the answer), a
   *verifier* (check the answer), and a *producer* (emit text). Cognitive science has
   argued for decades that these are separable processes coordinated through a shared
   workspace rather than one undifferentiated network.

3. **Confidence is a side effect, not a target.**
   Softmax probabilities are outputs of the loss, not calibrated statements of
   certainty. For code, a confident wrong function is annoying; for medicine, it is
   dangerous. We want a model whose confidence *means* something and whose "I am not
   sure, and here is what else it could be" is an explicit capability — not an accident
   of sampling.

Dhara is our attempt at an architecture designed around those three problems from the
outset rather than patched on top of attention.

## 2.2 Why not a Transformer at all?

Because the design goals demand properties a Transformer does not give for free:

- **Fixed memory** requires a recurrent (state-based) core. We adopt the linear-time
  state-space family (S4/H3/Mamba) as the core, which keeps a fixed-size state while
  still being trainable in parallel across the sequence.
- **Separable specialization** requires either multiple networks or multiple heads glued
  together with engineered wiring. The workspace pattern is the cleanest way to let
  them share state without bespoke couplings.
- **Honest uncertainty** requires the model to have heads (and losses) whose *job is*
  confidence/verification/calibration, not just a probability layer at the end.

The bet is not that attention is useless — it is used for retrieval and graph-style
operations *inside* the subsystems — but that a Transformer is the wrong *top-level*
organizer for these goals.

## 2.3 The two ways to frame this design (both defensible)

- **Engineering framing:** a modular pipeline of specialized, cross-supervised modules
  with a learned router, sharing a memory core that scales to long context for free.
- **Cognitive-science framing:** a Global-Workspace architecture (Baars; Dehaene) where
  specialists broadcast into and read from a shared buffer, and an executive —
  like a metacognitive monitor — decides the depth (and modules) of processing.

Both framings are true at different abstraction levels, and the document uses both.

## 2.4 Graph: attention quadratic cost vs. state-space constant cost

A picture to draw on the board. If the sequence length doubles, what happens to the
per-step training cost?

| Sequence length | Self-attention (O(L²)) | Transformer KV-cache mem (O(L)) | Dhara state (O(1)) |
|---|---|---|---|
| 4k | 16M units | 4k × size | fixed |
| 16k | 256M units | 16k × size | fixed |
| 64k | 4B units | 64k × size | fixed |
| 256k | 65B units | 256k × size | fixed |

The point is not that one is "wrong" — attention is unmatched at *retrieval* within a
window — it is that the two families trade different things. Attention pays for precise
look-back; the state-space pays for compactness forever. Dhara's bet: for reasoning over
long documents (a whole repo, a whole record), a controller that *compacts* what matters
into a state beats a cache that keeps everything verbatim.

## 2.5 Dhara vs. a Transformer decoder — head to head

The honest comparison table. "Chosen" = what Dhara does and why; "Trade-off" = what that
costs.

| Axis | Transformer decoder | Dhara (chosen) | Trade-off paid |
|---|---|---|---|
| Core recurrence | attention over all past | fixed-size SSM state | loses exact look-back; retrieval quality depends on the compressed state |
| Training-time complexity | O(L²) compute, O(L) cache | O(L) compute, O(1) memory | scan instead of matmul-heavy attention |
| Generation memory | KV cache grows with context | constant | must recompute/roll the state correctly at each step |
| Top-level structure | one stack | staged cognitive pipeline | much more machinery to supervise |
| Compute routing | uniform depth per token | executive gates + difficulty steps | gates weakly trained today |
| Uncertainty | softmax as-is | dedicated confidence + verification heads + calibration loss | implemented, not yet active |
| Ops complexity | mature ecosystem | custom pipeline | the systems work (Part 9.5) is real engineering |
| Maturity | battle-tested | experimental | the honest answer to "why not Llama" |

**Rule for answering "why not fine-tune a transformer?":** acknowledge that Transformers
are the mature, well-trodden path and predict tokens well; then say the bet is *architectural*:
fixed memory, separable specialization, and trained honesty are properties you cannot buy
by fine-tuning — you either have them in the architecture or you do not. Then list what
would falsify the bet (Part 8 and Part 9.3).

---

Every component in Part 5 traces back to one of these five principles. Each principle is
given with the failure mode it exists to avoid and the mechanism that realizes it.

### Principle 1 — Constant-memory scaling

**Statement.** The model must keep working as sequence length grows, with memory that is
bounded by model size, not sequence length.

**Failure mode avoided.** Self-attention's quadratic compute / linear KV cache makes
long-context reasoning expensive or impossible on fixed hardware. The moment you must
stream an entire repository or record into memory, standard decoders degrade.

**Mechanism.** The SSM/HSSM compression core (Part 5, Layer 3): a fixed-size recurrent
state updated per token, trained with a parallel scan. Memory in sequence length is
O(1) by construction. RoPE with YaRN scaling extends usable context further.

**Honest scope.** Constant memory comes at the cost of the strong retrieval that
attention enjoys within the window; the model's long-context correctness therefore
depends on how well the compressed state captures what matters — an empirical question
the current pretraining will answer.

### Principle 2 — The workspace hypothesis

**Statement.** Reasoning results from several specialized processes sharing intermediate
results through a common buffer, not from one undifferentiated flow.

**Failure mode avoided.** Bespoke point-to-point wiring between modules (planner→QA,
reasoning→decoder, …) that does not scale as modules are added and makes every module
impossible to reuse or reason about in isolation.

**Mechanism.** `CognitiveWorkspace` — a dict-based hub with `read`, `write`, `read_all`,
`clear`, and a learned fusion (`update`) that combines goals + knowledge + state into
one gated residual representation. Every layer writes there; every layer reads from
there.

**Prior work.** Global Workspace Theory (Baars) and the Global Neuronal Workspace
(Dehaene, Sergent): competing specialized processors broadcast into a shared buffer, and
the contents of the workspace are what is "conscious"/available for coordination. We are
adapting a *theory of mind architecture* into a *neural network topology*.

### Principle 3 — Not every problem needs every layer

**Statement.** The model should spend compute in proportion to the difficulty of the
question: trivial queries should be cheap, hard ones deep.

**Failure mode avoided.** Two failure modes: (a) wasted FLOPs — a full planning +
debate + verification pipeline on "what is 2+2"; (b) *degraded simple answers* — heavy
machinery injecting noise into easy questions (over-processing).

**Mechanism.** `ExecutiveController`: per-module soft gates, a hard skip threshold
(below 0.3 a module does not run), a difficulty-driven reasoning step budget, a compute
budget head, and a confidence head. Conceptually this is Adaptive Computation Time
(Graves, 2016) and early-exit / conditional computation (Shazeer et al., 2017) lifted
from per-token routing to the level of whole cognitive stages.

**Honest scope.** Today the main forward only hard-skips the world model and tools;
the remaining gates are recorded and supervisor-ready but do not yet skip in the
forward — see Part 8. The principle is *designed for*; the skip wiring is being extended.

### Principle 4 — Synthetic-to-symbolic spectrum

**Statement.** Neural heads should propose; symbolic executors should verify. Neither
alone suffices.

**Failure mode avoided.** Pure neural stacks produce fluent-but-wrong arithmetic,
unverifiable code, and no way to know an answer is wrong until a human or a test finds
it. Pure symbolic systems cannot generalize to novel phrasing.

**Mechanism.** Three cooperating mechanisms:
- **InternalToolInterface** — a router over a symbolic calculator, a sandboxed Python
  executor, a learned search memory, and a learned database.
- **QualityAssurance** — a fused reflection + verification + self-evaluation +
  correction loop (the "verifier" tradition of process/reward models; Reflexion;
  Self-Refine).
- **SpecialistSandbox** — multi-expert debate (Du et al., 2023) in a differentiable form.

**Prior work lineage.** Tool-augmented LLMs (Toolformer, ReAct, ToolLLM), verifier-based
reasoning, multi-agent debate. We adopt the *ideas* and rebuild the mechanisms inside the
architecture so they are trainable end-to-end.

### Principle 5 — Supervise the cognitive stages, not just the output

**Statement.** A single next-token loss is a very sparse teaching signal for a pipeline
with many intermediate representations; each stage should be held accountable directly.

**Failure mode avoided.** Intermediate modules converging to meaningless or ignored
representations ("dead" modules, as several were found to be during development — e.g.,
the curiosity module was previously computed and then never fed into any loss).

**Mechanism.** `AuxiliaryLossComputer` — a family of auxiliary loss terms (intent,
memory reconstruction, planning, gate supervision, verification, calibration/ECE,
trajectory smoothness, tool selection, entity prediction, decoder, curiosity novelty),
each attached at the corresponding stage and each optional via per-term weights.

**Honest scope.** The mechanism is fully implemented, but the loss executor requires
`aux_targets` to be supplied and **the current training loop does not supply them** —
so today the foundation run trains on the LM loss alone, and the auxiliary family is
being wired in. This split (designed / implemented / actually active) is stated
explicitly per mechanism throughout this document.

# Part 4 — System overview and data flow

## 4.1 The 11-layer pipeline

```
 input_ids (token ids) ───────────────────────────────────────────────┐
                                                                       │
 L1  IntelligentTokenizer  token embeds + semantic metadata (+tags)    │
 L2  AdaptiveSemanticEmbedding  RoPE → task context → context adapter │
 L3  HierarchicalMemoryEngine  working | semantic | HSSM | episodic    │──▶ workspace["memory"]
 L4  IntentUnderstanding       task type | difficulty | reasoning type ─┤──▶ workspace["intent"]
                                        └──▶ AdaptiveDifficultyRouter ─┘      (n_steps ≈ 2..32)
 L5  GlobalPlanner (goal tree, deps, cost, order) ──────────────────────┤──▶ workspace["plan"]
 L6  AdaptiveContinuousReasoning (routed Euler ODE, n_steps) ───────────┤──▶ workspace["reasoning"]
     ExecutiveController  soft gates → skip-set + budget/confidence ────┤──▶ workspace["executive"]
     CognitiveWorkspace.update(goals, memory, reasoning) → fused repr ──┤──▶ workspace["workspace"]
 [gated] WorldModel (entities/relations/events/causality) ──────────────┘
 L8  SpecialistSandbox 7 experts propose → critique → repair →
     consensus ─────────────────────────────────────────────────────────┘──▶ workspace["specialists"]
 [gated] InternalToolInterface  router(calc, python, search, db) ────────┘──▶ workspace["tools"]
 L10 QualityAssurance  reflect→verify→self-eval→correct→converge ─────────┘──▶ workspace["quality_assurance"]
     CuriosityModule  novelty × exploration (× executive confidence) ───────┘
 L11 HierarchicalSparseDecoder  concept → language → adaptive top-k token
      └─ head input = QA-corrected global context + positional memory context
          └─ full-vocab projection (single Linear) → LM loss / sampling
```

## 4.2 What actually runs, every step (code-accurate)

The forward pass (`DharaModel.forward`) executes the following in order. This is the
ground truth for the pipeline; any description of the model should match it.

1. `workspace.reset(batch)` — clears the shared buffer for the new example.
2. `memory.apply_updates(mem_state)` — applies the previous step's pending episodic
   compression / decay / priority writes.
3. Tokenizer → embedding: tokens are embedded; if `categories/languages/doc_roles` are
   provided, per-token semantic metadata is added; RoPE is applied; a task-context
   embedding is added if `task_ids` are provided; then the context adapter refines.
4. `memory(x, mem_state)` → fused memory representation + new memory state + metadata.
   The sequence is pooled (attention-mask-aware) into `h_pooled`.
5. `intent(mem_out)` → task type, difficulty, reasoning type, expected length,
   confidence. `AdaptiveDifficultyRouter` converts difficulty to a step count
   (2…`max_reasoning_steps`).
6. `planner(mem_out, task_type)` → goal tree (embeds, dependencies, order, costs,
   depth weights).
7. `reasoning(state_to_context(h_pooled), memory_to_hidden(h_pooled), n_steps)` →
   trajectory-average hidden state.
8. `executive(h_pooled, intent)` → gates for all 10 modules, budget/depth/confidence/
   value heads, per-module importance; `apply_gates` computes the skip-set
   (gate < 0.3 ⇒ skip).
9. `workspace.update(plan, mem_out, reasoning_out)` → fused workspace representation
   (learned gated residual over goals + knowledge + state).
10. World model (if enabled and not skipped) — writes `world_state`.
11. `specialists(workspace_repr)` — 7 experts propose, then `n_debate_rounds` of
    critique/repair, then critic + consensus fusion. **Always runs** (not gated).
12. Tools (if enabled and not skipped) — router-weighted combination of calculator /
    python / search / database outputs; else `consensus` flows on.
13. `quality_assurance(workspace_repr, final_ws, goal_embeds)` — reflection +
    verification + self-evaluation + correction until convergence; emits corrected
    hidden state, `n_passes`, `final_confidence`.
14. `curiosity(corrected_h, executive_confidence)` — novelty × exploration,
    scaled by executive confidence; its novelty score is injected into the QA
    `eval_out` so the curiosity loss has a target.
15. Final head input: `full_h = task_ctx (QA-corrected) + pos_ctx (memory_to_hidden(mem_out))`
    — i.e., position-wise memory context plus the QA-corrected global representation.
    The decoder stack runs **once** per micro-batch and its single vocabulary projection
    is shared by the LM loss, the entity auxiliary head, and the decoder aux loss
    (previously the stack was decoded a second time inside the loss path, so the
    dominant per-step cost was paid twice — removed in the latest training pass).
16. LM loss: causal-shifted target CE reusing those same logits — position *t* predicts
    token *t+1*, applied via a target mask (`-100` masked). Two modes, config-driven
    (`head_ce`): `dense` (default) = standard full-vocab log-softmax CE; `topk`
    (the foundation run) = CE over the decoder's adaptive top-k candidates ∪ target —
    honest note below in Part 5. If `aux_targets` are provided, auxiliary losses
    are added.

## 4.3 What is gated vs. what always runs (honest version)

| Module | Gated? | Reality in the current forward |
|---|---|---|
| memory, intent, planner, reasoning, workspace, specialists, QA, curiosity, decoder | No | Always run every step |
| WorldModel | Yes (skip if gate < 0.3) | Skippable in the forward today |
| Tools | Yes (skip if gate < 0.3) | Skippable in the forward today |

So the ExecutiveController *records and supervises* gates for all ten modules, but in the
current forward it physically skips only the world model and tools. Making every module
skippable is straightforward and on the roadmap (the `apply_gates` infrastructure already
computes the full skip-set).

## 4.4 Running the model (training / inference)

- **Configs** drive everything: `config_foundation.yaml` (~160M), `config_small.yaml`
  (~173M), `config.yaml` (full scale, `d_hidden` 10240, ~12.2B).
- **Training** runs through a staged training pipeline (`src/training/pipeline.py`):
  staged category groups, per-stage checkpoints, disk cache, async prefetch, and
  `save_steps` checkpoints. The current foundation run is 8 stages × up to 50k total
  steps, single GPU, on the university cluster.
- **Inference** (`DharaModel.generate`) threads the `mem_state` cache across tokens,
  feeding a single token per step with an offset for RoPE, then standard temperature /
  top-k / top-p sampling.

## 4.5 Reading the training telemetry (what each number means)

The training log is the fastest way to convince a class that the project is real. Here is
an actual annotated excerpt from the 2026-09 foundation run (pre-optimization code):

    13:42:45 INFO [TRAIN] step 0/50000 — entering training loop, awaiting first batch...
    13:42:46 INFO [TRAIN] step 0/50000 heartbeat at 0.5s
    step 50  | loss=179.1551 | lr=1.63e-06   | it/s=0.10
    step 100 | loss=173.9732 | lr=3.30e-06   | it/s=0.10
    step 300 | loss=141.5183 | lr=9.97e-06   | it/s=0.10
    step 600 | loss=116.8643 | lr=2.00e-05   | it/s=0.09
    step 800 | loss=110.1507 | lr=2.66e-05   | it/s=0.10
    {'loss': '110.2', 'grad_norm': '11.55', ...}

Teach the columns:
- **`step 0/50000`** — total steps in the run. 50k is the foundation budget over 8 staged
  category groups (code → docs → web → wiki → math → science → books → structured), each
  on one cosine schedule with checkpoints.
- **`loss`** — the LM cross-entropy (Part 1.5.1). It falls from ~179 → ~110 over the
  first 800 steps; *monotone descent, on a log scale* is the pattern you want. The
  absolute magnitude here is larger than a typical "perplexity" because the loss is an
  aggregate over a 64k-vocabulary head and the full sequence (values ≈ the model has
  only just started learning the token distribution — random would be worse).
- **`lr`** — learning rate during warmup. It climbs from ~1.6e-06 toward a scheduled
  peak, then decays on a cosine curve. A lr that *jerks* non-monotonically during
  warmup is a bug smell; a smooth ramp is healthy.
- **`it/s`** — steps per second. On the pre-optimization code this sat at ~0.09–0.10
  (~10.5–11 s/it). After the single-decoder-evaluation + `head_ce: topk` optimization
  (Part 9.5) the target is ~0.14–0.17 (~6–7 s/it).
- **`grad_norm`** — L2 norm of all gradients. Early values of 50–70 then 10–20 as
  training finds a good basin; if it *grows* toward the hundreds, add gradient clipping.
  Our run shows 60.6 → 52.2 → 70.7 → 34.2 → 11.5 — noisy early, settling down. Good.
- **`epoch`** — how many passes through the current stage's dataset have completed.
- **`[TRAIN] ... heartbeat`** — an ops signal (Part 9.5): it proves the training loop is
  alive and tells you the seconds since start even when the progress bar is buffered.
- **Check the first line!!** `awaiting first batch...` then a heartbeat at ~0.5 s means
  the dataloader is healthy. A heartbeat that stays at step 0 forever is the exact
  signature of the fork-lock deadlock fixed in Part 9.5.

**What a healthy long-run looks like.** Loss monotonically down (each -50-step print
smaller than the last), `it/s` roughly constant (±30%), `grad_norm` bounded and trending
down, periodic heartbeats, checkpoints appearing every `save_steps`. Any single symptom
(stuck progress bar, all-NaN loss, grad_norm explosion, loss going up for >500 steps)
takes precedence over everything else.

# Part 5 — Deep dive: components and mechanisms

The template from "How to use this document" is applied to every layer.

---

## Layer 1 — Tokenization: `IntelligentTokenizer`

**What it does.** Maps token ids to embeddings and optionally attaches per-token
*semantic metadata*: token category (8 — e.g., code vs. prose vs. punctuation),
language (16), and document role (8). Metadata, when available, is added to the base
embedding as a learned projection.

**Why it exists.** (Principles 2, 3.) Routing modules need to decide *how* to process a
prompt, and they need the descriptors as part of the representation, not as a sidecar.
"These tokens are an identifier, those are prose, that token closes a block" is exactly
the signal difficulty routing and the planner want — and by placing it at embedding
level, every downstream module receives it for free. Subword ids alone are opaque.

**Key mechanisms.**
- Learned token embedding table with `padding_idx=0` and `sqrt(d)` scaling (A).
- `SemanticMetadataEmbedding`: three small embedding tables (category/language/role,
  each `d_model/4`) concatenated and projected to `d_model` (N as a placement decision;
  tagging embeddings is a known trick in other forms, but tagging every token with three
  semantic axes as a first-class input is our design).
- Metadata is *additive*: the model still works without tags (they are optional), it
  just loses the routing signal.

**Novelty.**
- Component: (S). Tokenizer/embedding is fully standard; metadata design is ours.
- Mechanism: (N) as a first-class, always-available semantic tag channel.

**Honest caveats.**
- The current training loop does **not** pass `categories/languages/doc_roles`, so the
  metadata channel is present but not yet exercised. Wiring it from the data pipeline
  (each dataset knows its category and language) is on the roadmap.

---

## Layer 2 — Embedding: `AdaptiveSemanticEmbedding`

**What it does.** Applies Rotary Position Encoding (RoPE, base θ ≈ 10M), adds a
task-context embedding when a `task_ids` vector is provided, and refines the result with
a stack of context-adapter blocks (LayerNorm → FFN → residual → LayerNorm, the so-called
"pre-norm-in, post-norm-out" block shape fixed during development).

**Why it exists.** (Principles 1, 3.) Position matters in any sequential model; RoPE is
adopted because it bakes relative positions into the representation and extends cleanly
to long contexts via YaRN. The task-context term lets one embedding space specialize the
same weights to code vs. math vs. prose without separate backbones. The adapters give
the otherwise-static embedding some learned transformation capacity before memory.

**Key mechanisms.**
- `RotaryPositionEncoding` (A, Su et al. 2021): per-head half-dimension rotation, with
  `offsets` support so inference can continue at absolute positions during generation.
- YaRN rescaling for context extension (A, Peng et al. 2023) — config surface exists
  (`rope_scaling`), foundation context 4096.
- `TaskContextEmbedding` (S): a learned per-task-type embedding added broadcast over
  the sequence.
- `ContextAdapter`: stack of 4 MLP blocks (A-form residuals; the asymptotic fix of
  normalization is an engineering detail, not a claim).

**Novelty.** (A) with a small (S) extension (task conditioning). RoPE and YaRN are
adopted essentially as published.

**Honest caveats.** Task ids are likewise not passed in the current training loop; and
the base θ = 10M vs. Llama-style 500k is an intentional trade: smaller θ means finer
resolution at short offsets but faster rolloff at long offsets — appropriate for a model
whose compression core removes the need to attend across the whole window.

---

## Layer 3 — Memory: `HierarchicalMemoryEngine`

**What it does.** Fuses four memory systems into a single representation:

- **Working memory** (`WorkingMemory`) — a ring of ~512 slots; the latest token state is
  pushed in, and a learned gate blends the new input with the old state; positional
  slots add stable identity.
- **Semantic memory** (`SemanticMemory`) — a learned bank of ~4096 concept vectors read
  by dot-product attention from the query.
- **Long-context memory** (`LongContextMemory` → **HSSM stack**) — the SSM core that
  compresses the full sequence into a fixed-size state (the O(1)-memory heart).
- **Episodic memory** (`EpisodicMemory`) — a ring of up to ~256 "episodes", each a
  mean-pooled compression of a processed chunk, retrievable by attention.

On top: an **age-conditioned ForgetGate**, a **PriorityScorer**, a **MemoryRetriever**
(multi-head attention with a priority bias), and a **CompressionAE** (autoencoder) whose
MSE feeds the memory supervision.

The four parallel stores are concatenated and fused (`Linear(4·d_model → d_model)`) into
the representation the rest of the pipeline consumes.

**Why it exists.** (Principles 1, 2, 5.)
- O(1) memory: the HSSM gives a fixed-size state no matter the sequence length — this
  is the architectural answer to quadratic attention.
- A *taxonomy* of memory (working/semantic/episodic; cf. Atkinson & Shiffrin, Tulving)
  because different retrieval patterns need different store semantics: working = what I
  am holding right now; semantic = stable facts; episodic = what happened in prior
  context. Attention retrieval across an unbounded past is the standard but memory-heavy
  answer; a compressed bounded store is the alternative this project bets on.
- The extra mechanisms (forget, prioritize, compress) exist so that the bounded stores
  are *managed* rather than blunt FIFO — and so each store gets its own supervisory
  loss (Principle 5).

**Key mechanisms and lineage.**
- Ring-buffer working memory with a learned gating blend and slot position embeddings
  (S — scratchpad/serial memory tradition; the learned gate is ours).
- Learned key-value semantic memory read by attention (A — the DNC/MemNN memory-augmented
  network family; MemGPT-style agents also use semantic+episodic memory).
- **HSSM stack** (S→N, detailed below).
- Episodic mean-pool + FIFO ring + attention retrieval (A — episodic memory buffers in
  memory-augmented agents/RL).
- **ForgetGate**: sigmoid gate conditioned on memory *age*, fading older memories — an
  (N) mechanism (age-conditioned forgetting is not standard in neural memory banks).
- **PriorityScorer + MemoryRetriever**: attention logits biased by a learned priority so
  not all memories compete equally (S — salience/priority bias in attention).
- **CompressionAE**: encoder→decoder with an explicit reconstruction MSE (A —
  autoencoders; wired as a memory *supervisory loss*: ours).

**Novelty.**
- Component: (S) — four stores fused behind one engine is the contribution; each store
  exists in prior work.
- Mechanisms: ForgetGate (N), priority-bias retrieval (S), four-way fusion (N).
- **The HSSM is the most novel single mechanism** (see below).

**Honest caveats.**
- Age tracking is currently a shared counter broadcast over the whole episode buffer,
  not per-episode timestamps — fine for a first version, not a faithful "age".
- The episodic store learns in a *supervised, in-graph* way here only via the pending
  writes; it is closer to a working prototype of episodic memory than a settled design.
- The final fused representation mixes a *whole-sequence pooled* view into every
  position — see the train/serve skew discussion in Part 8. This is by far the most
  important design tension for an autoregressive model.

### The HSSM core (the O(1)-memory claim, in detail)

Each `HierarchicalSSM` layer follows the standard SSM recipe:

- input-dependent **B/C projections** (learned per level),
- a diagonal (state-space) recurrence parameterized by **A** (negative softplus),
- a **dt** learned via softplus from a projection (input-dependent discretization step),
- **SiLU double-gating** (`x_in = silu(Linear(x))`, output gated with `silu(gate(x))`) —
  the Mamba-family motif,
- a **parallel scan** over the sequence implemented with cumulative sums
  (`cumsum` of the exponentiated decay), which is what makes training parallel.

The change we make (this is the **Hierarchical** in HierarchicalSSM): rather than one
state per layer, each layer keeps **three stacked state channels** (`n_levels = 3`), with
separate B/C projections per level, whose representational combination forms a richer
compressed state. This is a small, concrete architectural contribution over single-level
S4/Mamba — the compressed state gets more capacity per parameter.

**Lineage.** (A) S4 (Gu et al.), H3 (Fu et al.), Mamba (Gu & Dao) provide the state-space
recurrence, discretization, gating, and scan. (S/N) the multi-level channel stack is ours.

**Honest caveats.** This is *not* a selective-scan implementation (no hardware-aware
`scan` kernel); the cumulative-sum scan is the simple parallel form, correct for these
sizes but not the optimized Mamba backend. The foundation configs are small enough that
this does not yet matter.

---

## Layer 4 — Intent: `IntentUnderstanding` + `AdaptiveDifficultyRouter`

**What it does.** Pooled sequence → LayerNorm → four heads:
- task **type** (8 classes) — what kind of job this is;
- **difficulty** (5 levels) — how hard the query looks;
- **reasoning type** (8 classes) — which reasoning mode fits;
- expected **length** (regression) and an intent **confidence**.

`AdaptiveDifficultyRouter` turns difficulty into a *step budget*: `difficulty = argmax`,
fraction across 5 levels interpolated to `[2, max_reasoning_steps]` (default 32).

**Why it exists.** (Principle 3.) This is adaptive computation carried by the architecture:
the difficulty prediction determines how many ODE steps reasoning runs, and is designed
to also scale debate rounds and QA passes. Without it, every input pays the same cost and
easy inputs get over-processed.

**Key mechanisms.**
- Classification/regression heads on a pooled representation (A — standard).
- Difficulty→steps interpolation (S — the *routing* is our integration; the concept of
  adaptive compute-time budget is Graves 2016 / ACT).

**Novelty.** (S). Individual classifiers are standard; the supervised difficulty signal
that shapes the rest of the pipeline is the integration.

**Honest caveats.** Difficulty is currently *self-supervised by the LM loss only*
(no task targets in the training loop), so the router is learning from indirect signal
today. Supplying explicit task/difficulty targets from the data pipeline is planned.

---

## Layer 5 — Planning: `GlobalPlanner`

**What it does.** Produces a learned **goal tree**:
- up to 64 subgoal slots, each a learned embedding biased by the input context;
- **graph attention** over the subgoal slots, plus a *hierarchical* pass that masks
  lower tree levels to parent subgoals (mask pattern `2^d`) with soft depth weights
  (4 depth levels);
- predicted **dependencies** between subgoals, an **execution order** (argsort of an
  order head), per-subgoal **cost estimates** (softplus), and an **execution graph**
  embedding;
- per-subgoal node details (cost, depth, type among 8) and a gate that masks inactive
  goals.

**Why it exists.** (Principles 2, 3.) Programming and mathematics decompose
hierarchically; classical hierarchical task networks (HTN — Sacerdoti; Erol et al.) and
chain-of-thought decomposition both show that explicit step structure helps. A
*differentiable* planner lets the plan be consumed by the workspace and later stages and
be given its own supervision (Principle 5) instead of being a prompting trick.

**Key mechanisms.**
- Goal embeddings + context bias (S).
- `GoalGraphAttention` — self-attention over subgoal slots (A — transformer/GAT).
- Hierarchical depth masking — lower levels see only parent subgoals (N-ish).
- Dependency / order / cost / type heads (S — classical planning attributes, learned).

**Novelty.** (S). Classical planning concepts; the differentiable, trainable, loss-
supervised form inside an LM is the contribution.

**Honest caveats.** The planning loss reads a `subgoal_logits` key the planner does not
currently emit, so with current wiring the planning loss is inert (returns 0) until the
planner exposes logits and the data supplies subgoal targets. The planner's *forward*
outputs are used regardless.

---

## Layer 6 — Reasoning: `AdaptiveContinuousReasoning`

**What it does.** Evolves a state vector through **continuous dynamics**:
- `ODETick` — a set of ~5 per-domain MLP dynamics, softly routed by a
  domain router (softmax over domains);
- integration by **Euler steps** with a dt-normalized update
  (`z += dz / sqrt(d_hidden)`, the normalization fixed during development to stop
  magnitude blowup);
- a **variable per-input step count** (from the intent router, default 8, up to 32);
- output = masked trajectory **average** over active steps, so effort scales with
  difficulty while the gradient flows through the whole trajectory.

**Why it exists.** (Principles 1, 3.) Continuous-time dynamics (Neural ODE — Chen et al.,
2018) give a principled view of "reasoning = evolving a trajectory" with an
effort knob, decoupling capacity from discrete layer count, and a smoothness loss can
regularize the path itself (Principle 5).

**Key mechanisms.**
- Neural-ODE-style dynamics (A — Chen et al. 2018; ODE-RNN traditions).
- Euler integrator with normalized dt (A — engineering choice; the adjoint method is
  not implemented — future work).
- Domain routing over dynamics (S — mixture of dynamics, unlike a single ODE field).
- Variable steps + masked trajectory pooling (S — adaptive computation again).

**Novelty.** (A) core, (S) domain routing + adaptive steps.

**Honest caveats.** Euler with one step budget is not a "continuous-depth" network in
the ODE-Net sense; it is a residual-net-with-status with a solver-shaped update. That is
fine and worth stating plainly — the trajectory-average also means the model is averaging
its own intermediate states, a modeling choice that can blunt sharp decisions.

---

## Layer 7 — Executive control: `ExecutiveController`

**What it does.** From the pooled input (plus intent features) it emits:
- **gates** in [0,1] for all ten modules («memory, planner, intent, reasoning, sandbox,
  reflection, verification, decoder, world_model, tools») — gate < 0.3 ⇒ module skipped;
- a **budget** (softmax over modules) for compute allocation;
- a **depth multiplier**, a **confidence** estimate, and a **value** head (RL value);
- a **per-module importance** vector trained by REINFORCE;
- a **ModulePerformanceTracker** (per-module performance history, adaptive per-module
  learning-rate adjustment, a learned performance predictor and a meta-learner).

**Why it exists.** (Principle 3.) Somebody has to decide *which* modules run — that is
the live interface between cost and correctness. It is also an **honesty/interpretability
hook**: the gates are a record of "which steps the model actually took" that can be
surfaced to users ("I used the debate sandbox; I did not verify this product claim").

**Key mechanisms and lineage.**
- Soft per-module gates + hard threshold skip (S — conditional computation / MoE-style
  gating: Shazeer et al., 2017; the difference is *we skip whole stages*, not tokens).
- **REINFORCE** importance update (A — Williams, 1992): mean-baseline advantages over a
  reward buffer, applied every 50 steps; reward = −task_loss − λ·active-gate-count.
  Deliberately decoupled from the main backward pass (update is computed on a detached
  clone so it cannot clobber the outer LM gradients).
- Budget / depth / confidence / value heads (S).
- ModulePerformanceTracker + per-module adaptive LR (N-ish — simple meta-learning).

**Novelty.** (S)/(N): gating and REINFORCE are adopted; *skipping whole cognitive
stages* under a dedicated learned controller, plus budget/depth/confidence predictions,
is a novel composition. Calling it "RL-trained" requires the caveat below.

**Honest caveats (important to be ready for).**
- In the current `DharaModel.forward`, the executive is called *without* a `task_loss`
  argument, so the reward is never computed there and the REINFORCE buffer does not fill
  in normal training. Today the controller trains only via whatever gradients reach the
  gate networks from the outer loss (weak), sparsely via gate-supervision targets when
  they exist, and its `module_importance` stays near its initialization. The full reward
  path is implemented and tested independently, but **it is not yet wired through the
  main forward**. Saying "REINFORCE-trained controller" is a design statement, not a
  current-run fact.
- Only world_model and tools are physically skipped today (Part 4.3).

---

## Layer 8 — Specialist debate: `SpecialistSandbox` (`DebateSandbox`)

**What it does.** Runs a differentiable, in-graph debate:
- 7 domain experts («programming, math, logic, planning, retrieval, creative, safety»),
  each a small network producing a *proposal* and a *confidence*;
- over `n_debate_rounds` (1 in training, up to 3 in inference): each expert reads a
  **cross-examination** summary of the others' proposals, **repairs** its own proposal
  through a repair network gated by a learned repair gate, blending original + repaired;
- a **critic** scores the proposals; final proposals are aggregated by
  `softmax(confidence × critique)` into a **consensus** (learned fusion over all
  experts) and a **best proposal** is selected.

**What it is not.** It is not 7 separate LLMs conversing in natural language — there is
no discrete speech. It is seven parameterized experts ("soft agents") whose "debate" is
vectorized propose→cross-examine→repair inside the forward pass. (Part 8 returns to this
— it is both the honest characterization and the direction for future work.)

**Why it exists.** (Principles 2, 4, and the certainty goal.) Multi-agent debate improves
factuality over a single model (Du et al., 2023). A *sandbox of domain specialists* is
how one model covers medicine, law, science, and code at once and cross-checks claims —
and it is the natural source of the "what could the answer be, not just the top guess"
alternatives the certainty goal wants to surface. Making the debate differentiable means
it can be trained end-to-end instead of orchestrated at prompting time.

**Key mechanisms.**
- Multi-expert proposals + confidence heads (S — mixture-of-experts flavor with a
  semantic identity per expert).
- Cross-examination head + per-expert repair network with a gate (N as a mechanism —
  not standard in debate papers; those exchange text, this is a vectorized loop).
- Critic scoring, consensus fusion, best-proposal selection (S — consensus/aggregation).

**Novelty.** The combination is (S); the *differentiable in-graph debate with repair +
cross-examination + consensus* is the novel part. Multi-agent debate as a concept is (A).

**Honest caveats.** With the training-time round count of 1 and no discrete exchange,
the *goods* of debate (independent evidence correction) is only partially realized;
the "debate" can collapse into a shared soft-expert blend. This is a deliberate
first-version trade and a stated experimental direction, not a claim that the mechanism
is solved.

---

## Layer 9 — Tools and world model: `InternalToolInterface` + `WorldModel`

### Tools
**What it does.** A `SymbolicToolRouter` emits route weights over four tools and fuses
their outputs:

- **SymbolicCalculator** — extracts a numeric expression from text and safely evaluates
  it with a restricted AST/operator allowlist (no builtins);
- **SymbolicPythonExecutor** — runs short code in a sandboxed namespace and returns a
  scalar embedding of the result;
- **SymbolicSearch** — a learned 4096-slot key-value memory read by attention;
- **SymbolicDatabase** — a learned 2048-record store accessed through a learned gate.

Output: `fused = LayerNorm(weighted_tool_sum + input)`.

**Why it exists.** (Principle 4.) A calculator computes what a network only guesses;
executing code (even sandboxed) produces a result the network can condition on. The
router decides *which* tool the current context needs, and the tool-selection CE loss
supervises that choice (Principle 5).

**Novelty.** (A) concept (Toolformer, ReAct, ToolLLM); the learned in-graph router over
four heterogeneous tools is (S). The symbolic executors themselves are deliberately thin
first versions.

**Honest caveats.** In the current wiring the tools receive a *proxy* text vector
(`hidden_text`) rather than the actual prompt/code text, so the symbolic calculator and
executor frequently return zeros in practice. Meaningful tool use needs the real text
plumbed in — a known gap, already tracked.

### World model
**What it does.** From the sequence it extracts:
- **entities** (soft selection from 64 learned embeddings + entity logits),
- **relations** (pairwise scorer over entities, 16 relation types),
- **events** (32 event slots with temporal projection),
- **causality** (multi-head cause→effect attention),
then fuses all four into a `world_state` written to the workspace.

**Why it exists.** (Principles 2, 4.) A structured, *queryable* representation of
entities and causality is what the planner needs for long-horizon reasoning, and the
entity logits give a supervised signal for the entity-prediction loss.

**Novelty.** (S). Neurological/graph world models and scene-graph work exist; the fused
entities/relations/events/causality module is our composition.

**Honest caveats.**
- **Disabled in the current foundation config** (`enable_world_model: false`) — it is a
  parameter sink with no immediate use in code generation.
- Operates on the *whole-sequence* pooled view (train/serve skew again, Part 8).

---

## Layer 10 — Quality assurance: `QualityAssurance`

**What it does.** One multi-pass module fusing the former reflection / verification /
curiosity / self-evaluation paths:

1. **Reflector** — gates: is the question answered? constraints met? any contradiction?
   needs a rethink?
2. **Verifier** — five correctness heads: **syntax**, **compilation probability**,
   **runtime correctness**, **math consistency**, **logic consistency** (a code-centric
   taxonomy, tuned to the current training domain).
3. **SelfEvaluator** — usefulness, **novelty**, **uncertainty** heads.
4. **Corrector** — a gated correction network driven by `error_score × needs_correction`,
   where error is the mean over the five verifier heads against targets.
5. Iterate up to `max_passes` with per-pass embeddings; **early stop** when confidence
   change < threshold *and* confidence > 0.7. Outputs: corrected representation,
   per-pass traces, final confidence.

**Why it exists.** (Principles 4, 5, and the certainty goal.) Unguarded generations look
confident. Reflection (Reflexion — Shinn et al.; Self-Refine — Madaan et al.), and
verification (process/outcome verifiers) are established fixes. The **uncertainty** head
is the direct hook for the certainty-aware objective; the **novelty** head is consumed by
the curiosity loss. Merging them means one loop, shared representations, one convergence
criterion.

**Key mechanisms.**
- Self-critique gates (S — self-critique lineage).
- Code-centric verifier head taxonomy (S — the head *set* is ours).
- Usefulness/novelty/uncertainty self-evaluation (A — self-eval / selective prediction).
- Gated corrector (S — residual/adaptor lineage).
- Per-pass embeddings + convergence-based early stop (N-ish).

**Novelty.** (S). Each head has prior art; the fused single differentiable module with a
shared convergence criterion is the integration.

**Honest caveats.** All five verifier heads are BCE-trained **against `aux_targets`
correctness labels, which are not yet supplied** — so today the verifier heads are
untrained softmax garbage shielded behind the LM loss. The *module runs and gates the
corrected path*, but its heads need supervision to mean anything. This is the single
biggest "implemented but not yet activated" item to be transparent about.

---

## Layer 11 — Decoding: `HierarchicalSparseDecoder`

**What it does.** A three-level conditioning hierarchy ahead of the token head:

1. **Semantic level** — a router over 4096 learned concept embeddings yields a concept
   context vector.
2. **Language level** — a classifier over 8 language groups (with an optional hard hint)
   yields a gated language context; a *soft* language bias `0.1 · W·(language context)`
   is injected into the token logits even without a hint.
3. **Token level** — a difficulty-predictor sets an **adaptive top-k** (32…2048) for
   candidate tokens; learned temperature scales the logits.

A fusion gate combines `hidden + concept_context + language_context` before the token
projection.

**Why it exists.** (Principles 3, 5.) Coarse-to-fine decoding conditions the token head
on semantics and language rather than raw hidden state, echoing Layer 1 metadata.
Adaptive top-k keeps the decision surface wide for hard tokens and narrow for easy ones.

**Key mechanisms.**
- Concept router over shared concept embeddings (S).
- Language classifier + gated context + soft logit bias (A — language heads, here as a
  conditioning layer).
- Adaptive top-k from a difficulty head (A — adaptive top-k).
- Fusion gate (S).

**Honest caveats (important).**
- The decoder **computes a full-vocab projection** (a single `Linear(d_hidden, 128k)`)
  and then applies top-k. The hierarchy *conditions and biases* the logits; it is not
  truly sparse compute. "Sparse" in the current implementation means sparse *sampling*
  candidate sets, not sparse computation. Real sparse decoding is future work.
- The decoder stack runs **once** per micro-batch; the LM loss reuses its logits and
  applies the causal shift with a target mask (position *t* → token *t+1*). Previously
  the stack was re-decoded for the loss, paying the dominant step cost twice
  (`hidden_to_vocab` + `hierarchical_log_prob`).
- The LM loss is a cross-entropy over that projection in one of two modes:
  - `head_ce: dense` (default) — standard full-vocab log-softmax cross-entropy.
  - `head_ce: topk` (the running foundation config) — cross-entropy over the decoder's
    own adaptive top-k candidates ∪ the target. This is a **subset softmax**: the
    partition function is summed over the candidate set only, so probability mass
    outside the top-k is dropped from the loss and its gradients. It is exact only to
    the degree the candidate set covers the model's distribution; the rare-token tail
    is under-penalized. The forward projection is the same in both modes — `topk` is a
    speed/memory trade-off on the *loss*, stated honestly, not a claim of exact sparse
    loss.

**Novelty.** (S): adaptive softmax / coarse-to-fine decoding exist (Grave et al.); the
concept–language–token coupling to the model's own semantic metadata is our composition.

---

## Supporting modules

### `CognitiveWorkspace` (the hub)
A dict of `key → (tensor, metadata, timestamp)` with `read`/`write`/`read_all`/`clear`,
bounded at 32 keys with oldest-first eviction, plus a learned **update** that fuses
`goals + knowledge + state` with a gated residual and a sigmoid *confidence*.
- Theory adopted: Global Workspace Theory (A — Baars; Dehaene).
- Instantiation: explicit dict-buffer API with learned fusion (N).
- Tip for the review: this is where "workspace-centric" is literally true — every module
  in Part 4.2 writes and reads the same buffer.

### `AuxiliaryLossComputer` (the supervision family)
Implements per-term losses keyed to module outputs: intent, memory (reconstruction MSE),
planning, gate (supervision BCE), verification (five-head BCE), calibration (ECE),
trajectory smoothness, tool selection, entity prediction, decoder, curiosity novelty.
Weights are config-driven (`loss_weights`).
- The *family* is a synthesis; **calibration as a training loss** (not merely an eval
  metric) is the least common, most distinctive choice.
- **Honest precision:** the implemented ECE is a simple surrogate
  `|mean(confidence) − mean(accuracy)|`, a single-bin variant of the bucketed
  expected-calibration-error metric (Guo et al., 2017). Stating it as "binned ECE" would
  overclaim; it is a calibration *signal*.
- **Honest precision:** as of now, no training path passes `aux_targets`, so **none of
  these terms fire in the current run**; the terms are tested individually (292 tests)
  and are target-driven by design. See Part 8.

### `CuriosityModule`
SelfEvaluation(unuseful…usefulness, novelty, uncertainty) + an exploration-drive head;
`curiosity_score = novelty × exploration`, scaled by the executive's confidence; the
novelty score is injected into QA's `eval_out` so the *novelty bonus loss*
(`−w·mean(novelty)`, an intrinsic-reward form — Pathak et al. lineage) can fire. When
executive confidence is low, exploration is damped.

# Part 6 — Supervision and training objectives

## 6.1 The primary objective

Next-token prediction (standard LM cross-entropy) through the hierarchical decoder head,
masking `-100` positions. This is the only objective that **fires today**.

## 6.2 The auxiliary family (designed and implemented; not yet activated)

| Term | Source module | Form | Weight (default) |
|---|---|---|---|
| intent | intent heads | CE vs task/difficulty/reasoning targets | 0.05 |
| memory | memory engine | reconstruction MSE (CompressionAE) | 0.01 |
| planning | planner | CE vs subgoal targets | 0.05 |
| gate | executive | BCE vs per-module gate targets | 0.01 |
| verification | QA verifier | BCE vs the 5 correctness heads | 0.02 |
| calibration | executive confidence | \|conf − acc\| · weight | 0.005 |
| trajectory | reasoning | smoothness of ODE trajectory | 0.001 |
| tools | tool router | CE vs tool choice | 0.05 |
| entity | world model | CE vs entity ids | 0.01 |
| decoder | decoder | CE vs decoder targets | 0.01 |
| novelty | curiosity | bonus −w·mean(novelty) | 0.001 |

Activation rule: `AuxiliaryLossComputer` is invoked only when `aux_targets` is provided,
and each term additionally needs its own target. **Today: no caller passes `aux_targets`,
so only the LM loss runs.** This split is intentional (targets come from the data
pipeline, which is being extended to emit task gates, correctness labels, tool choices,
entity/decoder ids). The design claim is: *the supervision family exists, is tested, and
turns on as targets arrive* — it is not a claim that it is active in the current run.

## 6.3 Why calibration as a loss (the certainty goal's training arm)

Ordinary next-token training makes probabilities *maximize-likelihood*, not
*mean-what-they-say*. The `confidence_calibration_loss` (ECE surrogate) directly
penalizes the gap between the executive's predicted confidence and observed accuracy —
an attempt to make "the model knows how sure it is" a trained property rather than a
post-hoc calibration step. The verifier heads are the mechanism that *checks*; the
calibration loss is the mechanism that *tells the truth about confidence*. This is the
training-time machinery behind "say what is correct and tag what might not be."

# Part 7 — Novelty matrix (the meeting cheat-sheet)

| Component | Core mechanism(s) | Component / mechanism status |
|---|---|---|
| SSM core (HSSM) | S4/Mamba recurrence, SiLU gating, cumsum parallel scan; 3 state channels/level | Adopted / **Novel: multi-level stacked channels** |
| O(1) memory in length | fixed-size recurrent state | Adopted (SSM) |
| RoPE + YaRN | rotational positions; rope-scaling | Adopted |
| Memory engine | ring, K-V memory, HSSM, episodic ring, autoencoder, age-forget, priority | **Synthesis**; ForgetGate (N), fusion (N) |
| Intent + difficulty routing | classifiers → ODE steps / top-k budget | Adopted concept / synthesis integration |
| Planner | goal tree, graph attn, deps, order, cost | Adopted concept / differentiable form (S) |
| Continuous reasoning | Euler-routed domain dynamics | Adopted core / S (routing + adaptive steps) |
| Executive | per-module gates + 0.3 skip + REINFORCE | S / **novel pipeline-level skip**; RL not yet wired |
| Specialist debate | proposals, cross-exam, repair, critic, consensus | Adopted concept / **novel in-graph debate** |
| Tools | router over calc/python/search/db | Adopted / S (in-graph learned router) |
| World model | entity/relation/event/causality fusion | S |
| QA module | reflect + verify + self-eval + correct, early-stop | Adopted concepts / **novel fused module** |
| Decoder | concept + language + adaptive top-k conditioning | **S** (still full-vocab projection — honest) |
| 14-loss family | stage supervision; ECE-as-train-loss | **Novel aggregate**; ECE-as-training-signal rare |
| Workspace pipeline | dict hub; Global-Workspace inspiration | Novel instantiation |

# Part 8 — Honest limitations and known considerations

These are the points most likely to come up in review, stated before they are asked.

1. **Train/serve skew (the big one).** During training, `full_h` for the LM head adds a
   global, *whole-sequence-pooled* QA-corrected context to every position. During
   generation, the model sees one token at a time and the pooled context is instead built
   from the recurrent memory state carried across steps. The two conditioning
   distributions do not match, so the model is being trained under a richer condition than
   it serves. This is the most important known gap and the first thing to address in the
   next training iteration (options: causal pooling, training with partial sequence
   exposure, or a two-phase objective that alternates full-context and recurrent
   conditioning).

2. **Auxiliary losses are offline.** The 14-term family is implemented and tested but no
   caller provides `aux_targets`. The verifier heads and the calibration head therefore
   have no targets today.

3. **Executive is partially active.** Gates are computed and recorded for ten modules;
   only world_model and tools are physically skipped; and the REINFORCE reward is not
   wired into the forward (no `task_loss` is passed), so the controller is currently
   trained weakly at best.

4. **Metadata and task context are unused in training.** `categories/languages/doc_roles/
   task_ids` are optional; the current data pipeline does not emit them, so the semantic
   metadata channel and task conditioning are dormant.

5. **Tools get proxy text.** The symbolic calculator/executor receive a numeric feature
   vector instead of real text/code, so they often output zeros. Real tool text like
   code tokens needs to be plumbed in.

6. **Debate is soft.** The sandbox is seven parameterized experts blending vectors — no
   discrete language exchange. Training uses 1 debate round. This realizes only part of
   the multi-agent-debate benefit and is a stated direction rather than a completed one.

7. **Decoder is not truly sparse.** Full-vocab projection is always computed; hierarchy
   only conditions/biases it and narrows sampling. "Sparse decoding" is aspirational.

8. **Reasoning is Euler, not adjoint.** `AdaptiveContinuousReasoning` is a continuous
   residual block with per-input steps. No adjoint/ODE-solver method, no
   continuous-depth across the whole network.

9. **HSSM scan is the simple parallel form.** No hardware-aware selective-scan kernel;
   fine at current sizes, must be revisited at full scale.

10. **The current training run is single-objective.** Foundation training runs the LM
    loss only (see Part 6.2). All "supervision at every stage" claims describe the
    designed/implemented system, not the current run.

11. **No change to the "Methos" lineage in prose.** Some docs still explain the V3/V4
    rename; the project now refers to itself as Dhara everywhere user-facing.

# Part 9 — Where It Is Today

- **Primarily code generation.** The training mix is code-heavy (code 30% of the
  registry weight), and the pipeline optimizes for producing and reasoning about
  programs — hence the code-centric verifier heads (syntax / compilation / runtime /
  math / logic).
- **Foundation pretraining (in progress).** 8 staged category groups (code → docs → web
  → wiki → math → science → books → structured) over 50k total steps on a single cosine
  schedule with per-stage checkpoints, disk cache, async prefetch. Current run:
  `config_foundation.yaml`, ~160M params (`dhara_v3`, hidden 576, 3-level HSSM,
  context 4096, vocab 64k, `head_ce: topk`, single decoder evaluation per micro-batch),
single GPU, running on the university cluster. Before the Story 2 optimization this ran
   ~10.5–11 s/it (~6.3 days ETA); the target after it was ~6–7 s/it (~3.5 days), and the
   measured result is ~8 s/it (~4.6 days ETA) — a real ~25% win, short of the modeled 40%,
   because the forward still computes the full 64k projection (only the loss/backward is
   narrowed).
- **Data.** 54 primary datasets across 8 categories plus 4 fallback-only entries, every
  entry with a fallback chain; 18 built-in web scrapers for a self-hosted docs corpus;
  AST-based code filtering, simhash dedup, contamination filtering.
- **Configs.** `config_foundation.yaml` (~160M), `config_small.yaml` (~173M),
  `config.yaml` (full scale ~12.2B — `dhara_v3`, hidden 10240, ctx 16384, SFT stage;
  FSDP + CPU offload on 4× A100 80GB). Do note: `config.yaml` is the *production* training
  target, not something to run on shared GPU boxes casually.
- **Quality.** 26 test files / 292 tests green, covering config schema, model
  construction/forward, memory, debate, QA, losses, the data pipeline units, and
  alignment.

---

# Part 9.1 — The data factory (how the model gets its training signal)

A model is only as good as its corpus. Dhara's data layer is production-shaped and has
been through two hardening passes (phase-1 crash fixes, then an async/safety pass). What
it does, layer by layer:

**1. A curated registry, not a dump.** 54 primary datasets organized into 8 categories
(code, docs, web, wiki, math, science, books, structured), plus 4 fallback-only entries.
Every entry carries its own **fallback chain** (primary source → mirror → other source),
a per-dataset **quality score**, language/domain distributions, and a registry-wide
category-weighting so the mix can be tuned. In the current run, **code is ~30% of the
weight** — the model is being built to generate and reason about programs first.

**2. Web documentation scraping.** 18 built-in scrapers build a self-hosted docs corpus
(Python docs, PyTorch docs, MDN, Kubernetes docs, and more), each page cleaned,
deduplicated by URL, and validated into JSONL. Downloads are gated HF repositories, so
the pipeline needs an `HF_TOKEN` (set as an environment variable — *never* hardcode tokens
into committed config).

**3. Quality pipeline, per sample:**
- length checks (min/max) and low-quality-marker filtering ("todo", "lorem ipsum");
- language detection via multiline regex features (code vs. prose, which language);
- a **QualityScorer** that scores code higher than boilerplate text (heuristic + v2
  advanced score), so low-quality filler is dropped before it wastes GPU-hours;
- **exact dedup** (hash) and **simhash dedup** (near-duplicate removal) so the model
  is not asked to memorize the same 100 copies of the same blog post;
- a **contamination filter** that removes benchmark-shaped text so evaluation stays
  honest (the model should not have seen HumanEval/GSM8K answers in training).

**4. Streaming with fallbacks.** Datasets stream from the hub with retry/backoff; when a
source stalls or a network error fires, the code falls back to the next mirror silently.
A metadata cache remembers gating state so warm runs skip repeated auth round-trips.

**5. Packing & tokenization.** Cleaned text is tokenized, packed into fixed-length
training windows (context 4096), and cached on disk so a restart does not re-clean
everything. This is where `epoch` in the log (Part 4.5) comes from.

**Why this is lecture-worthy:** modern LLM engineering is mostly data engineering. A
class interested in "how do you actually train a model" should see that half the work is
corpus curation: dedup, contamination control, quality scoring, and per-category
weighting are what separates a real from a toy training run.

---

# Part 9.2 — What we test, and how we keep it green

Evidence quality matters in a review, so the project runs **two** suites:

- **pytest: 292 tests across 26 test files.** Coverage: config schema + validation,
  model construction/forward, memory, debate, QA, the loss family, the data pipeline,
  the async-prefetch hardening regressions, the health reporter, step accounting,
  crash-fix regressions, and the alignment losses. Runs with `python -m pytest tests/ -q`
  (or `python main.py test`).
- **Script suites: 431 checks** not under pytest — the pipeline validator
  (`test_pipeline.py`, 124), the async-pipeline stress (`test_pipeline_async.py`, 43),
  local validation (`test_local_validation.py`, 258), and the async benchmark (6). These
  exercise the data/ops path end-to-end the way unit tests cannot.

The only flaky test is a pre-existing unseeded NSLT convergence test (loss *may* go up on
1-in-3 fresh runs — re-instantiation randomness, not a code bug). Being able to name the
one flaky test and why it flakes is itself a good answer to "how reliable is your
testing?"

---

# Part 9.3 — How we would verify the bet (evaluation & alignment)

The architecture is a bet; here is the scoreboard it will be held to.

**Benchmarks (the planned eval set):**

| Benchmark | Family | Metric | Size |
|---|---|---|---|
| HumanEval | code generation | pass@1 | 164 problems |
| MBPP | code generation | pass@1 | 417 problems |
| MMLU | knowledge (57 subjects) | accuracy | ~14k questions |
| HellaSwag | commonsense NLI | accuracy | 10k |
| ARC | science QA | accuracy | 2,590 (challenge) |
| GSM8K | math word problems | exact match, 8-shot CoT | 1,319 |
| TruthfulQA | factuality | mc1/mc2 | 817 |
| WinoGrande | coreference | accuracy | 1,267 |
| BBH | reasoning (23 tasks) | accuracy, 3-shot CoT | 6,511 |

Code is measured with pass@1 (did the generated program actually *run* on hidden tests) —
a symbolic, verifiable signal that matches Principle 4 (neural propose, symbolic verify).
Knowledge and reasoning are measured to make sure code-trained weights did not regress
general ability.

**Alignment (`src/alignment/`):** DPO, ORPO, SimPO, and KTO trainers are implemented,
each with the standard preference-loss math and a reference-model fallback. These are the
post-pretraining lane for "be helpful and honest"; the calibration loss in Part 6.3 is the
pretraining lane for the same goal.

**What would falsify the bet (say this explicitly in class):**
- If the O(1)-memory core cannot retain factual recall at long context (the compression
  drops what matters), the core advantage evaporates.
- If the stage pipeline does not beat a same-size Transformer on the benchmark table,
  the machinery is not paying for itself.
- If the executive routing (once fully wired) cannot actually *save* compute without
  hurting accuracy, Principle 3 fails.

Naming the falsification conditions up front is what separates a hypothesis from a claim.

---

# Part 9.4 — Configurations and scale

Three configs chart the project's scale ladder. Same code, different size.

| Config | Params | Shape | Purpose |
|---|---|---|---|
| `config_foundation.yaml` | ~160M | `dhara_v3`, hidden 576, d_state 288, 3-level HSSM, ctx 4096, vocab 64k, `head_ce: topk`, 50k steps / 8 groups | the run that is happening now |
| `config_small.yaml` | ~173M | similar, slightly wider | ablations / quick experiments |
| `config.yaml` | ~12.2B | `dhara_v3`, hidden 10240, ctx 16384, with an SFT stage | production target: FSDP + CPU offload across 4× A100 80GB |

Operational details that matter: a GPU auto-detector in `main.py` picks the least-loaded
GPU, and a **reservation system** can hold one before a long job so another process cannot
steal it. A single-GPU run uses FSDP in no-shard mode purely for API compatibility, so the
distributed stack is exercised from day one. The 12.2B config is the real target; the 160M
foundation run exists to validate the whole pipeline cheaply first.

---

# Part 9.5 — The training-systems war stories (and why the run is fast)

This part is the *engineering* argument — the part a professor will respect because it is
not a design claim but a record of things that actually broke and how they were fixed.
All three stories come from the current 50k-step foundation run. Treat them as case
studies: each is a real failure mode, a real diagnosis, and a real fix.

## Story 1 — Training stuck at step 0 forever (the fork-lock deadlock)

**Symptom.** The progress bar sat at `0/50000` indefinitely. The first batch never
arrived; the process was alive (no crash) but made no progress. Hours of GPU time burned
for nothing.

**Diagnosis.** The pipeline uses a **threaded async-prefetch** subsystem (Part 4.4) whose
worker threads hold Python locks while warming the stream. The training DataLoader that
fetches batches was spawning **processes** (num_workers > 0). On Linux, `fork()` copies
the *entire* address space — *including locked mutexes held by the prefetch threads at
that instant*. Each forked worker then tried to acquire an inherited, already-locked lock
on its very first batch fetch, and deadlocked itself forever. The main thread waited on
the workers — a classic futex-wait chain that never resolves. (`futex_wait_queue_me`
forever in the stacks was the signature.)

**Fix.** `dataloader_num_workers=0` for the pretrain stage — batches are fetched on the
main thread, which is fine because the async-prefetch threads already do the streaming
work. No performance cost; workers were only adding a *copying* step on top of an
already-parallel pipeline.

**Lesson to teach.** Any time you mix threads and `fork()`, assume inherited locks. The
Python docs even forbid fork-in-locked-thread scenarios; it is one of those bugs that
"could not happen" until it happens to you at 50k steps scale.

## Story 2 — ~11 s/it and where the time actually went (the double decoder evaluation)

**Symptom.** Healthy training but slow: ~10.5–11 s/it, an ETA of ~6.3 days for 50k steps.

**Diagnosis.** Profile the cost instead of guessing. Two structural facts emerged:
- The decoder + 64k-vocab projection is the single dominant tensor in the step
  (vocab 64000 × sequence × batch of logits). Everything else — SSM, sands, ODE — is
  small by comparison.
- The model forward was running that dominant decoder stack **twice per micro-batch**:
  once to produce `_vocab_logits` for the output, and a second time inside
  `hierarchical_log_prob()` to compute the loss. The single most expensive computation
  was bought twice.

**Fix.** Evaluate the decoder **once** per micro-batch, keep the logits, and apply the
causal shift (position *t* predicts token *t+1*) in the loss via a target mask. A
regression test re-verifies the dense-mode loss is byte-identical to the old path, so the
speedup is math-preserving, not math-changing.

**Second fix (the `head_ce` option).** The loss itself went through a full
`log_softmax` over 64k vocab. A new config flag, `head_ce: dense|topk`, makes the loss an
opt-in **candidate-set cross-entropy**: the CE is computed over the decoder's own adaptive
top-k token candidates ∪ the target instead of the full vocabulary. This is a *subset
softmax* — an approximation (probability mass outside the candidates is dropped from the
loss), which is why it is opt-in and stated honestly. It does not change the forward or
generation; it narrows the loss and its backward target set.

**Expected result.** ~11 s/it → **~6–7 s/it**, ETA from ~6.3 days to ~3.5 days. The
largest, least-controversial win in the whole project was removing a duplicated
computation, not a cleverer algorithm.

**Measured result.** Steady state ~8.0–8.7 s/it (ETA ~110–120 h ≈ 4.6–5 days). A genuine
~25% speedup, short of the modeled 40% — the honest caveat from Part 2.4 held: the forward
pass still evaluates the full 64k head projection, so removing the duplicated decoder pass
and narrowing the loss could only cut part of the wall-clock. Still the single best
cost/benefit change in the project.

## Story 3 — Ops tooling: making stalls visible in seconds (SIGUSR1 + heartbeat)

**Problem.** On the shared university box there is no root/sudo, so `py-spy`/`gdb` (and
even `ptrace`) are unavailable. When something stalls, you were blind.

**Fix 1 — SIGUSR1 stack dump.** `main.py` installs a handler at startup: `kill -USR1 <pid>`
dumps **every** Python thread's stack to the log, with no privileges, and the process
keeps running. A stuck worker's `futex_wait_queue_me` becomes visible in five seconds.

**Fix 2 — the train heartbeat.** A callback logs `[TRAIN] step 0/N — entering training
loop, awaiting first batch...` the instant the loop starts, then a heartbeat every 100
steps. A run that silently stalls is now *loud* within seconds instead of after hours.
The heartbeat line in Part 4.5 — `heartbeat at 0.5s` — is this feature: it proves the
first batch arrived half a second after launch.

**Lesson.** When you cannot attach a profiler, make the software tell you its own story.
A stack-dump signal and a heartbeat turned a multi-hour diagnostic into a one-second
check.

## What else the ops layer does

- **Async prefetch** streams the next batch while the current one trains
  (`UnitPrefetch`, `AsyncCheckpointWriter`), with journal-first scheduling, cooperative
  cancellation on shutdown, retry/backoff, and a watchdog — the hardening that produced
  ~30 of the 292 tests.
- **Checkpointing.** Per-stage checkpoints and disk-cache of cleaned batches mean a
  crashed run resumes like a movie, not like a Rube Goldberg machine.
- **Fresh-start vs. resume** is a first-class CLI flag. `--fresh-start` wipes and begins;
  otherwise a run picks up from the last checkpoint.
- **The numbers.** Running the foundation config: single GPU, ~160M params. Before the
  Story 2 fix: ~11 s/it. After: expected ~6–7 s/it. 50,000 steps is the budget.

---

# Part 9.6 — What specifically changed in the code during this project's recent work

So the class can see the actual diffs behind the war stories (all on `main`):

- `src/dhara/model.py` — single decoder evaluation per micro-batch; `head_ce` mode;
  causal shift via target mask in the loss.
- `src/config/schema.py` — `DharaConfig.head_ce: "dense" | "topk"` (default `dense`).
- `src/models/factory.py` — wires `head_ce` through `create_model`.
- `src/training/pipeline.py` — `dataloader_num_workers=0` for pretrain; the
  `_TrainHeartbeatCallback`.
- `main.py` — the SIGUSR1 stack-dump handler.
- `config_foundation.yaml` — foundation config now runs with `head_ce: topk`.
- `tests/test_head_ce.py` — 5 new tests locking in the dense-vs-reference equivalence,
  top-k bound, gradient flow in both modes, schema default, and factory wiring.

Test count movement: 287 tests / 25 files → **292 tests / 26 files**.

# Part 10 — The Goal: Certainty-Aware Reasoning Beyond Code

*This direction was outlined in a discussion with the professor and is the guiding vision
the architecture is built around.*

Training the model on medical data instead of (or in addition to) code is the test case
for a much deeper capability:

- **Say what is correct, and tag what might not be.** A medical answer cannot just be
  confident — it must flag uncertainty. The model should distinguish what is actually
  correct from what might be incorrect, marking its confidence instead of hiding behind
  a single most-likely output. This is why the verification BCE and the calibration
  (ECE) losses exist: they are the training-time machinery for honesty about uncertainty.
- **Report what could be, not just what is most likely.** Beyond the top guess, surface
  the plausible alternatives — the diagnoses, treatments, or answers that *could* be
  right — so a human (or another agent) can weigh them. The plural agents in the sandbox
  are the natural source of these alternatives.
- **Not only medical.** Finance, law, engineering, research — anywhere a confidently
  wrong answer is worse than an honest uncertain one.
- **The sandbox makes this possible.** The DebateSandbox can hold agents covering many
  domains at once. A medical question can be taken up by a diagnostician agent, a
  pharmacologist agent, and a skeptical reviewer agent that debate, cross-examine each
  other's claims, flag what they are unsure about, and converge on an answer annotated
  with what is verified versus what remains possible.

# Part 11 — Anticipated questions and suggested answers

*(The two professor questions, answered directly.)*

**Q1. Why does each component exist?**
Each component exists because one of the five principles would otherwise be violated.
Memory exists because constant memory requires a state-based core and a memory taxonomy
needs more than one store (P1, P2). Intent + difficulty routing exists because compute
should track difficulty (P3). Executive exists because someone has to decide which
modules run, and that decision should be learned and observable (P3 + interpretability).
Planning exists because multi-step tasks decompose (P2). Reasoning-as-dynamics exists
because a step-count knob gives controllable effort (P3) and a smoothness loss gives
supervision (P5). Debate exists because multiple perspectives correct each other (P4) and
it is the mechanism for surfacing alternative answers. Tools exist because symbolic
execution verifies what networks guess (P4). QA exists because unguarded generations look
confident (P4, P5) and its uncertainty head is the honesty hook. Calibration-as-loss
exists because confidence should be trained to mean something (P5 + certainty goal).
Workspace exists because specialized modules must share state without bespoke wiring
(P2).

**Q2. Which parts are new vs. adopted?**
The short answer: *almost every mechanism is adopted; what is new is the composition.*
Adopted mechanisms: SSM recurrence, RoPE/YaRN, memory-augmented networks, neural-ODE
dynamics, MoE-style gating, REINFORCE, multi-agent debate, reflection/verification,
intrinsic curiosity, adaptive softmax/top-k, HTN planning concepts, global-workspace
theory. New (or new-in-this-form): the workspace-centric pipeline as a trainable topology;
the hierarchical (multi-level) SSM state channels; the differentiable in-graph debate with
repair and cross-examination; the fused single QA module with convergence early-stop; the
concept–language–token decoder conditioning; calibration and 14-term stage supervision as
a coordinated training family; the semantic-metadata token embedding channel; and the
executive's pipeline-level stage skip. The matrix in Part 7 gives the per-mechanism
answer in one line each.

**Q3. (Likely) Then why isn't the training using most of it?**
Because targets arrive from the data pipeline, and the pipeline emits them incrementally.
The architecture and the supervision family are implemented and tested; the training loop
currently exercises the LM objective while aux targets, metadata tags, and correctness
labels are being wired in. Part 8 lists exactly what turns on and what does not.

**Q4. (Likely) What is the language-modeling weakness you worry about most?**
The train/serve skew described in Part 8.1 — the decoder head is conditioned on
whole-sequence pooled context in training but recurrent-cache context at generation.

**Q5. (Likely) If the memory is O(1), how does the model "remember" anything specific?**
It does not keep verbatim copies — it keeps a **compressed state**: whatever survived
compression is what is "remembered." This is the honest core of the trade: exact retrieval
is traded for compactness. The mechanisms that decide what survives (the forget gate, the
priority scorer, the multi-level HSSM channels, the semantic/episodic stores read by
attention) are *learned*, which is exactly why they are trainable components rather than
library calls. Whether the compression retains what matters is the empirical question the
current pretraining is answering (see Part 9.3 — falsification condition #1).

**Q6. (Likely) Why 64,000 tokens and not 128,000 or 32,000?**
Vocabulary size is a knob between compression of the input and embedding cost. 64k is
comfortably larger than the ~32k used by many code-centric tokenizers, giving room for
code tokens and special markers, while keeping the final projection manageable. The head
is the step's dominant tensor (Part 9.5, Story 2) — which is why Dhara makes the head
*configurable* (`head_ce: dense|topk`) instead of treating vocabulary cost as a fixed
constant.

**Q7. (Likely) "Workspace-centric" — isn't that just an attention graph with extra steps?**
Fair, and the honest answer is: the workspace's learned `update` (gated residual over
goals + knowledge + state) *is* a form of attention over module outputs. What the pattern
adds is **structure and convention**: every module has the same interface (`read`/`write`/
`read_all`/`clear`), a bounded 32-key buffer enforces scarcity, and modules can be added,
removed, or supervised in isolation. You could implement the workspace on top of attention;
the point is the discipline it imposes on a 15-module system, and the interpretability it
buys (each key is *named*, e.g. `workspace["specialists"]`).

**Q8. (Likely) What does "the executive is RL-trained" mean if it is not wired in yet?**
It means the REINFORCE machinery exists, is unit-tested, and is the documented training
path for the controller — but in the current pretraining forward the controller's reward
requires a task-loss arm that is not passed, so today the gates train only weakly through
outer losses. Saying it plainly is part of the project's honesty discipline: *designed*,
*implemented*, and *active in the current run* are three different categories, and this
document uses all three accurately (see Part 4.3 and Part 8, #3).

**Q9. (Likely) How does generation work without attention?**
Autoregressively, one token at a time: the current `mem_state` cache is threaded across
steps (with an RoPE offset so position keeps advancing), the decoder head conditions on
the running state, and sampling uses temperature / top-k / top-p as usual. Because there
is no KV cache, the per-step memory is **constant in context**; the cost is that each step
passes through the whole model (no reuse of past states' keys/values — a trade-off
different from a transformer, not a free lunch).

**Q10. (Likely) Why start with code, if the vision is medical?**
Three reasons. (1) **Verifiable signal:** code has an oracle — the tests run or they
don't — so "right" is not subjective; this exercises Principles 2 and 4 with ground truth.
(2) **Hard reasoning:** code forces long-range structure (a function's meaning depends on
identifiers defined hundreds of lines earlier), stressing the O(1)-memory claim hard. (3)
**The machinery is domain-general:** verifier heads, sandboxed specialists, the
calibration loss, and QA loops are being debugged on code, where failures are obvious,
before being pointed at medicine, where failures are expensive. The professor's original
framing — medicine as the ambition, code as the proving ground — is preserved in Part 10.

**Q11. (Likely) What is the single strongest argument for this architecture?**
That the weaknesses of Transformers it targets are *structural*: quadratic attention cost,
one monolithic flow, and uncalibrated confidence. Dhara attacks all three at the level of
the architecture (fixed-size state, staged workspace pipeline, dedicated confidence +
calibration machinery) rather than as patches. The counter-argument is equally structural:
attention's exact retrieval is extremely powerful, and it is not obvious the compressed
state can match it. That tension, stated honestly, is the whole intellectual content of
the project — and it is why Part 9.3 lists explicit falsification conditions.

**Q12. (Likely) What would you tell the class to take away?**
Three sentences. (1) Engineering a real LLM is at least half data engineering and ops —
most of this project's recent work is about deadlocks, time-per-step, and checkpoints, not
algorithm theory. (2) The most effective bottleneck-fix was deleting a *duplicated*
computation — the decoder was being run twice per step, and removing the second pass
should nearly halve training time. (3) Novelty claims decay under scrutiny, so this
document deliberately separates *adopted*, *synthesis*, and *novel* — and separates
*designed*, *implemented*, and *active*, so nothing overclaims what the current run does.

# Status

Pre-release. Data pipeline and model plumbing are production-polished; full pretraining
rollouts are the current milestone. See `project_description/` for architecture, API, and
release-readiness documentation.

---

# Glossary (for the class)

| Term | Meaning (one line) |
|---|---|
| Token / vocab | atomic text units; 64,000 in Dhara |
| Cross-entropy loss | -log P(true token); the training signal for next-token prediction |
| LM head / vocab projection | the final Linear(d_hidden → 64000) producing token logits |
| `head_ce` | config switch: `dense` (full log-softmax) vs `topk` (candidate-set CE over decoder's top-k ∪ target) |
| Scalar shift / causal mask | position *t* predicts token *t+1*; the loss aligns targets to shifted positions |
| SSM (state-space model) | recurrence `h_t = A h_{t-1} + B x_t`, O(1) memory in length |
| HSSM | Dhara's multi-level SSM (3 stacked state channels per layer) |
| Parallel scan | parallel prefix-style evaluation of the linear recurrence |
| RoPE / YaRN | rotary position encoding; YaRN rescales it for longer contexts |
| KV cache | the O(L) per-position key/value cache Transformers keep; Dhara has none |
| Workspace | shared dict buffer all modules read/write (Global Workspace theory) |
| Gate / skip | executive output in [0,1]; gate < 0.3 ⇒ module skipped |
| REINFORCE | policy-gradient method used to train the executive's routing decisions |
| Auxiliary loss | a loss term attached to a specific module (intent, memory, QA, …) |
| ECE (calibration) | expected-calibration-error; Dhara uses a single-bin surrogate as a training signal |
| ODE tick / Euler steps | continuous reasoning dynamics integrated with fixed steps |
| Debate / consensus | specialists propose, cross-examine, repair; fused by confidence × critique |
| Quality score / dedup | data-pipeline filters (length, language, simhash, contamination) |
| FSDP / CPU offload | distributed training strategy; optimizer state offloaded to RAM |
| `it/s`, `grad_norm`, `epoch` | telemetry: steps per second, gradient L2 norm, passes over data |
| SIGUSR1 stack dump | signal that prints all thread stacks without needing root |
| Train heartbeat | periodic "[TRAIN] step N/M …" log proving the loop is alive |

---

# Further reading (all in this workspace)

- `README.md` — the quick overview, badges, features, changelog.
- `ARCHITECTURE.md` — the compact V4 design doc (workspace hub, gates, merged QA).
- `docs/architecture.md` — the deep architecture reference, including the honest decoder
  and head-CE section.
- `docs/async_pipeline_hardening_report.md` — the fork-lock fix, stall diagnostics, and
  the hardening evidence (Part 9.5, Story 1 in depth).
- `project_description/` — 18 further documents: dataset registry, validation, training
  pipeline, configuration guide, debugging guide, performance analysis, release
  readiness, engineer onboarding, and more.
- `config_foundation.yaml` — the live 160M config (see Part 9.4).

*End of document — good luck with the lecture.*