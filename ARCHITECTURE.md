# Dhara V4 — Frontier Cognitive Architecture

## Design Philosophy

- **Workspace-Centric Communication**: Every module reads from and writes to a central CognitiveWorkspace. No module calls another directly.
- **Symbolic Tools with Learned Routing**: External computation (math, code, search, DB) is executed symbolically, not approximated by neural nets. A learned router selects which tool to invoke.
- **Enforced Gating**: The Executive Controller actually skips modules when their gate is below threshold, enforcing compute budgets in the forward pass.
- **Quality Assurance (QA)**: Reflection, verification, self-evaluation, and curiosity are unified into a single multi-pass QA module that scores, reflects, verifies, corrects, and re-verifies.
- **Auxiliary Multi-Objective Training**: Every module contributes a differentiable auxiliary loss. The final loss is a weighted sum of next-token prediction + all auxiliary losses.
- **RL-Trained Executive**: The gate network is trained via REINFORCE with reward = accuracy − λ₁·latency − λ₂·memory − λ₃·energy, making compute allocation a learned skill.

## Architecture Overview

```
Input → Tokenizer → Embedding → Memory → Intent → Workspace
                                                         │
                    ┌────────────────────────────────────┤
                    ▼                                    ▼
            Executive Controller                 Hierarchical Planner
                    │                                    │
                    ▼                                    ▼
            (gate enforcement)               (subgoal tree + deps)
                    │                                    │
                    ▼                                    ▼
            Reasoning ODE ───────────► Workspace ◄─── Plan
                    │                        │
                    ▼                        ▼
            Symbolic Tools Router       Specialist Sandbox
                    │                        │
                    ▼                        ▼
            [calc│python│search│db]     Debate Consensus
                    │                        │
                    └─────────► Workspace ◄──┘
                                    │
                                    ▼
                            Quality Assurance (QA)
                           ┌───────┼───────┐
                           ▼       ▼       ▼
                     Reflection Verify  Self-Eval
                           │       │       │
                           └───► Correct ◄┘
                                    │
                                    ▼
                            Workspace (final)
                                    │
                                    ▼
                            Hierarchical Decoder
                                    │
                                    ▼
                            Output Tokens
```

## Module Specifications

### 1. CognitiveWorkspace (Central Hub)
- **Inputs**: read requests from any module
- **Outputs**: latest written values for any key
- **State**: dict of `{module_name: {output_tensor, metadata, timestamp}}`
- **API**:
  - `read(key)` → tensor or None
  - `write(key, value, metadata)` → None
  - `read_all()` → dict of all current values
  - `broadcast()` → current shared representation
  - `clear(keys)` → reset specified keys
- **Loss**: none (pure communication)
- **Ablation**: replace with residual sum → measure degradation

### 2. Tokenizer + Embedding
- Unchanged from Dhara (IntelligentTokenizer + AdaptiveSemanticEmbedding)

### 3. HierarchicalMemoryEngine
- Unchanged (SSM compression + HSSM multi-scale memory + MemoryManager)
- **Aux Loss**: reconstruction loss between compressed/decompressed states

### 4. IntentUnderstanding
- Unchanged (classification of task type, difficulty, reasoning type)
- **Aux Loss**: cross-entropy for task_type, difficulty, reasoning_type (requires synthetic labels)

### 5. HierarchicalPlanner
- Unchanged (goal tree, dependency graph, execution graph)
- **Aux Loss**: subgoal prediction (predict next subgoal token)

### 6. ExecutiveController (RL-trained, gate-enforcing)
- **Inputs**: workspace state, intent features
- **Outputs**: per-module gates (used to skip modules), compute budget vector, depth estimate
- **State**: running reward buffer for RL update
- **Key Change**: `forward()` returns gates; `apply_gates()` actually skips modules in model.forward()
- **RL Training**: REINFORCE with reward = R = accuracy − λ₁·latency − λ₂·memory − λ₃·energy
- **Aux Loss**: gate prediction (binary cross-entropy with target = 1 for useful modules, 0 for unused)

### 7. AdaptiveContinuousReasoning (ODE)
- Unchanged (LTC liquid time-constant ODE + NSLT neural state)
- **Aux Loss**: trajectory smoothness (penalize large jumps) + terminal state prediction

### 8. SpecialistSandbox + DebateSandbox
- Unchanged (7 specialists with multi-round debate + weighted consensus)

### 9. SymbolicToolRouter (replaces InternalToolInterface)
- **Router**: learned MLP that maps hidden state → tool selection logits
- **Tools**: true symbolic execution:
  - **Calculator**: regex-extract numbers/operators → `eval()` with sandbox
  - **Python**: `exec()` in restricted namespace → capture stdout
  - **Search**: embedding-based retrieval over a key-value store
  - **Database**: structured query over in-memory rows
- **Aux Loss**: tool selection cross-entropy (requires synthetic tool-use labels)
- **No neural approximation**: tools are real computation

### 10. QualityAssurance (merged Reflection + Verification + Curiosity)
- **Unified multi-pass**: reflect → verify → self-evaluate → correct → re-verify → converge
- **Inputs**: workspace content, consensus from specialists
- **Outputs**: corrected_h, confidence, n_passes, all intermediate scores
- **Sub-components**:
  - `Reflector`: checks answered, constraints met, contradictions
  - `Verifier`: checks syntax, compilation, runtime, math consistency, logic
  - `SelfEvaluator`: usefulness, novelty, uncertainty, improvement proposal
  - `Corrector`: generates correction from error signals
  - `ConfidenceTracker`: tracks per-pass confidence, stops on convergence
- **Aux Losses**:
  - correctness prediction (BCE with synthetic correctness labels)
  - confidence calibration (ECE regularization)
  - improvement prediction (did the correction help?)

### 11. WorldModel
- Simplified: entity extraction + relation network only
- Causality and event modeling merged into entity-relation graph
- **Aux Loss**: entity/relation prediction (predict masked entity)

### 12. HierarchicalSparseDecoder
- Unchanged (3-level: Semantic → Language → Token, with adaptive top-k)

## Training Objectives

### Primary Loss
`L = L_nll + Σᵢ wᵢ · Lᵢ_aux`

### Auxiliary Losses (14 total)

| Module | Loss | Target | Weight |
|--------|------|--------|--------|
| Memory | Reconstruction MSE | compressed ≈ original | 0.01 |
| Intent | Task type CE | synthetic task label | 0.05 |
| Planner | Subgoal CE | synthetic subgoal token | 0.05 |
| Executive | Gate BCE + RL reward | gate target + reward | 0.01 |
| Reasoning | Trajectory smoothness | L2 of step-to-step delta | 0.001 |
| Debate | Consensus confidence | calibration | 0.01 |
| Tools | Tool selection CE | synthetic tool label | 0.05 |
| QA: Reflect | Answer coverage BCE | ground-truth answers | 0.02 |
| QA: Verify | Correctness BCE | synthetic correctness | 0.02 |
| QA: SelfEval | Usefulness prediction | human rating proxy | 0.01 |
| QA: Calibration | ECE | confidence = accuracy | 0.005 |
| WorldModel | Entity prediction CE | masked entity | 0.01 |
| Decoder | Group assignment CE | language group label | 0.01 |
| Curiosity | Novelty bonus | negative entropy | 0.001 |

## Curriculum Training (8 Phases)

| Phase | Stages | Losses Active | Dataset Focus |
|-------|--------|---------------|---------------|
| 1 | Core Pretraining | L_nll only | Web text, books |
| 2 | Memory + Intent | +L_mem, +L_intent | Narratives, instructions |
| 3 | Planning + Reasoning | +L_plan, +L_reason | Math, code, logic |
| 4 | Specialists + Debate | +L_debate | Multi-perspective QA |
| 5 | Tool Use | +L_tools, +L_exec | Function calling, code |
| 6 | Quality Assurance | +L_qa_reflect, +L_qa_verify, +L_qa_eval | Error detection, repair |
| 7 | Full System | All losses, RL executive | All datasets |
| 8 | Alignment + Safety | +L_calibration | Constitutional, red-teaming |

## Ablation Plan

Each module can be disabled independently. Evaluation benchmarks:
- MMLU (knowledge), HumanEval (code), GSM8K (math),
- BBH (reasoning), TruthfulQA (factuality),
- Custom: planning accuracy, tool selection accuracy, repair success rate

Measure: Δ score per ablation, Δ inference FLOPs, Δ training time

## Scaling Roadmap

- **7B**: Full Dhara V4, 1× A100 80GB inference, 4× training (FSDP)
- **70B**: MoE variant (8 experts, top-2), 4× A100 inference, 32× training
- **1T**: MoE (64 experts, top-8), distributed inference, curriculum pretraining only

## Engineering Roadmap

1. **Phase 1**: Workspace-as-hub + merged QA + symbolic tools ✓ (current diff)
2. **Phase 2**: Executive gate enforcement + RL reward pipeline
3. **Phase 3**: Curriculum trainer with per-phase loss config
4. **Phase 4**: Synthetic data generation for auxiliary targets
5. **Phase 5**: Ablation harness with automated benchmark runs
6. **Phase 6**: Scaling tests (7B → 70B → 1T)

## Migration Plan from Dhara

1. Replace `InternalToolInterface` with `SymbolicToolRouter` (breaking: different outputs)
2. Replace separate `ReflectionModule`, `VerificationModule`, `CuriosityModule` with single `QualityAssurance` (breaking: different forward signature)
3. Rewrite `CognitiveWorkspace` to be dict-based hub (breaking: different API)
4. Update `ExecutiveController` to enforce gates in model.forward (functional change)
5. Add auxiliary loss computation in model.forward (non-breaking, loss dict)
6. Update config schema with loss weights and curriculum phases (backward compat with defaults)
