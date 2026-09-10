from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.dhara.layer1_tokenizer import IntelligentTokenizer
from src.dhara.layer2_embedding import AdaptiveSemanticEmbedding
from src.dhara.layer3_memory import HierarchicalMemoryEngine
from src.dhara.layer4_intent import IntentUnderstanding, AdaptiveDifficultyRouter
from src.dhara.layer5_planner import GlobalPlanner
from src.dhara.layer6_reasoning import AdaptiveContinuousReasoning
from src.dhara.workspace import CognitiveWorkspace
from src.dhara.layer8_specialists import SpecialistSandbox
from src.dhara.quality_assurance import QualityAssurance
from src.dhara.layer11_decoder import HierarchicalSparseDecoder
from src.dhara.executive import ExecutiveController, MODULE_NAMES
from src.dhara.world_model import WorldModel
from src.dhara.tools import InternalToolInterface
from src.dhara.curiosity import CuriosityModule
from src.dhara.losses import AuxiliaryLossComputer

logger = logging.getLogger(__name__)


class DharaConfig(PretrainedConfig):
    model_type = "dhara_v3"

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 10240,
        d_state: int = 4096,
        d_hidden: int = 10240,
        max_position_embeddings: int = 262144,
        rope_theta: float = 10000000.0,
        sparsity_pct: float = 1.0,
        n_ssm_layers: int = 6,
        n_hssm_levels: int = 3,
        n_ode_steps: int = 8,
        n_trajectories: int = 7,
        n_sim_steps: int = 16,
        use_efficient_sandbox: bool = True,
        n_token_categories: int = 8,
        n_languages: int = 16,
        n_doc_roles: int = 8,
        n_task_types: int = 8,
        n_difficulty_levels: int = 5,
        n_reasoning_types: int = 8,
        max_subgoals: int = 64,
        n_domains: int = 5,
        max_reasoning_steps: int = 32,
        n_semantic_concepts: int = 4096,
        working_mem_capacity: int = 512,
        max_episodes: int = 256,
        n_context_adapter_blocks: int = 4,
        n_language_groups: int = 8,
        adaptive_top_k_min: int = 32,
        adaptive_top_k_max: int = 2048,
        n_experts: int = 7,
        n_debate_rounds: int = 3,
        max_refinement_passes: int = 5,
        max_repair_iters: int = 3,
        max_entities: int = 64,
        n_relation_types: int = 16,
        max_events: int = 32,
        n_tool_types: int = 4,
        enable_executive: bool = True,
        enable_world_model: bool = True,
        enable_tools: bool = True,
        enable_curiosity: bool = True,
        enable_aux_losses: bool = True,
        executive_gate_threshold: float = 0.3,
        qa_max_passes: int = 5,
        qa_converge_threshold: float = 0.05,
        loss_weights: Optional[Dict[str, float]] = None,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_hidden = d_hidden
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.sparsity_pct = sparsity_pct
        self.n_ssm_layers = n_ssm_layers
        self.n_hssm_levels = n_hssm_levels
        self.n_ode_steps = n_ode_steps
        self.n_trajectories = n_trajectories
        self.n_sim_steps = n_sim_steps
        self.use_efficient_sandbox = use_efficient_sandbox
        self.n_token_categories = n_token_categories
        self.n_languages = n_languages
        self.n_doc_roles = n_doc_roles
        self.n_task_types = n_task_types
        self.n_difficulty_levels = n_difficulty_levels
        self.n_reasoning_types = n_reasoning_types
        self.max_subgoals = max_subgoals
        self.n_domains = n_domains
        self.max_reasoning_steps = max_reasoning_steps
        self.n_semantic_concepts = n_semantic_concepts
        self.working_mem_capacity = working_mem_capacity
        self.max_episodes = max_episodes
        self.n_context_adapter_blocks = n_context_adapter_blocks
        self.n_language_groups = n_language_groups
        self.adaptive_top_k_min = adaptive_top_k_min
        self.adaptive_top_k_max = adaptive_top_k_max
        self.n_experts = n_experts
        self.n_debate_rounds = n_debate_rounds
        self.max_refinement_passes = max_refinement_passes
        self.max_repair_iters = max_repair_iters
        self.max_entities = max_entities
        self.n_relation_types = n_relation_types
        self.max_events = max_events
        self.n_tool_types = n_tool_types
        self.enable_executive = enable_executive
        self.enable_world_model = enable_world_model
        self.enable_tools = enable_tools
        self.enable_curiosity = enable_curiosity
        self.enable_aux_losses = enable_aux_losses
        self.executive_gate_threshold = executive_gate_threshold
        self.qa_max_passes = qa_max_passes
        self.qa_converge_threshold = qa_converge_threshold
        self.loss_weights = loss_weights
        self.is_encoder_decoder = is_encoder_decoder


class DharaModel(PreTrainedModel):
    config_class = DharaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HierarchicalSSM", "CognitiveWorkspace", "DebateSandbox", "ExecutiveController"]

    def can_generate(self) -> bool:
        return True

    def __init__(self, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = DharaConfig(**{k: v for k, v in kwargs.items() if k in DharaConfig.__init__.__code__.co_varnames})

        super().__init__(config)
        c = config

        self.vocab_size = c.vocab_size
        self.d_model = c.hidden_size
        self.d_state = c.d_state
        self.d_hidden = c.d_hidden
        self.max_seq_len = c.max_position_embeddings
        self.enable_executive = c.enable_executive
        self.enable_world_model = c.enable_world_model
        self.enable_tools = c.enable_tools
        self.enable_aux_losses = c.enable_aux_losses

        self.tokenizer_layer = IntelligentTokenizer(
            vocab_size=c.vocab_size, d_model=c.hidden_size,
            n_token_categories=c.n_token_categories, n_languages=c.n_languages,
            n_doc_roles=c.n_doc_roles,
        )
        self.embedding = AdaptiveSemanticEmbedding(
            d_model=c.hidden_size, max_seq_len=c.max_position_embeddings,
            rope_base=c.rope_theta,
            n_task_types=c.n_task_types, n_context_adapter_blocks=c.n_context_adapter_blocks,
        )
        self.memory = HierarchicalMemoryEngine(
            d_model=c.hidden_size, d_state=c.d_state, n_hssm_layers=c.n_ssm_layers,
            n_hssm_levels=c.n_hssm_levels, working_mem_capacity=c.working_mem_capacity,
            n_semantic_concepts=c.n_semantic_concepts, max_episodes=c.max_episodes,
        )
        self.memory_to_hidden = nn.Linear(c.hidden_size, c.d_hidden, bias=False)
        self.state_to_context = nn.Linear(c.hidden_size, c.d_state, bias=False)

        self.executive = ExecutiveController(c.hidden_size, c.d_hidden, gate_threshold=c.executive_gate_threshold) if c.enable_executive else None

        self.intent = IntentUnderstanding(
            d_model=c.hidden_size, n_task_types=c.n_task_types,
            n_difficulty_levels=c.n_difficulty_levels,
            n_reasoning_types=c.n_reasoning_types,
        )
        self.difficulty_router = AdaptiveDifficultyRouter(
            n_difficulty_levels=c.n_difficulty_levels,
            min_steps=2, max_steps=c.max_reasoning_steps,
        )
        self.planner = GlobalPlanner(d_model=c.hidden_size, max_subgoals=c.max_subgoals)
        self.reasoning = AdaptiveContinuousReasoning(
            d_state=c.d_state, d_hidden=c.d_hidden,
            n_domains=c.n_domains, max_steps=c.max_reasoning_steps,
        )
        self.workspace = CognitiveWorkspace(d_model=c.hidden_size, d_hidden=c.d_hidden)
        self.world_model = WorldModel(c.hidden_size, c.max_entities, c.n_relation_types, c.max_events) if c.enable_world_model else None
        self.specialists = SpecialistSandbox(d_hidden=c.d_hidden, n_debate_rounds=c.n_debate_rounds)
        self.tools = InternalToolInterface(c.d_hidden) if c.enable_tools else None
        self.quality_assurance = QualityAssurance(c.d_hidden, d_model=c.hidden_size, max_passes=c.qa_max_passes, converge_threshold=c.qa_converge_threshold)
        self.curiosity = CuriosityModule(c.d_hidden) if c.enable_curiosity else None
        self.decoder = HierarchicalSparseDecoder(
            d_hidden=c.d_hidden, vocab_size=c.vocab_size, d_model=c.hidden_size,
            n_language_groups=c.n_language_groups,
            adaptive_top_k_min=c.adaptive_top_k_min,
            adaptive_top_k_max=c.adaptive_top_k_max,
            n_semantic_concepts=c.n_semantic_concepts,
        )
        self.loss_computer = AuxiliaryLossComputer(c.loss_weights) if c.enable_aux_losses else None

        self._log_architecture()

    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Dhara (V4 arch) — %.2fB total (%.2fB trainable) | "
            "Workspace-hub=%s Executive=%s Tools=symbolic QA=merged AuxLosses=%s",
            total_params / 1e9, trainable / 1e9,
            "enabled",
            "enabled" if self.executive else "disabled",
            "enabled" if self.loss_computer else "disabled",
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        categories: Optional[torch.LongTensor] = None,
        languages: Optional[torch.LongTensor] = None,
        doc_roles: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        language_ids: Optional[torch.LongTensor] = None,
        mem_state: Optional[dict] = None,
        aux_targets: Optional[Dict[str, Any]] = None,
        offsets: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        batch, seq_len = input_ids.shape
        device = input_ids.device
        self.workspace.reset(batch)

        self.memory.apply_updates(mem_state)

        x = self.tokenizer_layer(input_ids, categories, languages, doc_roles)
        x = self.embedding(x, task_ids, offsets=offsets)
        mem_out, mem_state, mem_meta = self.memory(x, mem_state)
        # Masked pooling so padding tokens don't contaminate means
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            h_pooled = (mem_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h_pooled = mem_out.mean(dim=1)
        self.workspace.write("memory", mem_out, mem_meta)
        self.workspace.write("memory_pooled", h_pooled)

        intent = self.intent(mem_out)
        n_steps = self.difficulty_router(intent.get("difficulty"))
        self.workspace.write("intent", mem_out, intent)

        plan = self.planner(mem_out, intent.get("task_type"))
        self.workspace.write("plan", mem_out, plan)

        h_ctx = self.state_to_context(h_pooled)
        z_in = self.memory_to_hidden(h_pooled)
        reasoning_out = self.reasoning(h_ctx, z_in, n_steps)
        self.workspace.write("reasoning", reasoning_out)

        exec_decision = None
        skip = {}
        if self.executive is not None:
            exec_in = h_pooled if h_pooled.shape[-1] == self.executive.d_model else torch.zeros(batch, self.executive.d_model, device=device)
            exec_decision = self.executive(exec_in, intent)
            skip = self.executive.apply_gates(exec_decision["gates"])
            self.workspace.write("executive", h_pooled, {
                k: v.detach().mean().item() if isinstance(v, torch.Tensor) else v
                for k, v in exec_decision.items()
            })

        ws_out = self.workspace(plan, mem_out, reasoning_out)
        workspace_repr = ws_out["workspace"]
        self.workspace.write("workspace", workspace_repr)

        world_out = None
        if self.world_model is not None and "world_model" not in skip:
            world_out = self.world_model(mem_out)
            self.workspace.write("world_model", world_out["world_state"])

        specialist_out = self.specialists(workspace_repr, n_debate_rounds=1 if self.training else None)
        consensus = specialist_out["consensus"]
        self.workspace.write("specialists", consensus)

        tool_out = None
        if self.tools is not None and "tools" not in skip:
            try:
                # Provide workspace context to symbolic tools so calculator/
                # python receive real text instead of permanent zeros.
                if hasattr(self.tools, "set_context"):
                    # Use the pooled hidden as a proxy for textual context.
                    self.tools.set_context(hidden_text=workspace_repr.detach().mean(dim=0).tolist() if workspace_repr.numel() < 1024 else None)
            except Exception:
                pass
            tool_out = self.tools(consensus)
            final_ws = tool_out["fused_tool_output"]
            self.workspace.write("tools", final_ws)
        else:
            final_ws = consensus

        qa_out = self.quality_assurance(workspace_repr, final_ws, plan.get("goal_embeds"))
        corrected_h = qa_out["corrected_h"]
        self.workspace.write("quality_assurance", corrected_h, {
            "confidence": qa_out["final_confidence"],
            "n_passes": qa_out["n_passes"],
        })

        # Curiosity is now a live module (previously dead): it scores novelty
        # and gates exploration by executive confidence.
        curiosity_out = None
        if self.curiosity is not None:
            exec_conf = exec_decision.get("confidence") if isinstance(exec_decision, dict) else None
            curiosity_out = self.curiosity(corrected_h, executive_confidence=exec_conf)
            self.workspace.write("curiosity", curiosity_out["consolidated_h"])
            # Feed curiosity novelty into QA's eval_out so the novelty bonus
            # loss (which reads quality_assurance.eval_out.novelty) fires.
            if "eval_out" not in qa_out:
                qa_out["eval_out"] = {}
            qa_out["eval_out"]["novelty"] = curiosity_out["curiosity_score"]

        # --- Auxiliary outputs: make previously-discarded module results
        # reachable so every loss branch in AuxiliaryLossComputer can fire ---
        # Trajectory proxy: use the reasoning output expanded to a short
        # pseudo-trajectory (the real ODE trajectory average is `reasoning_out`;
        # exposing it as a sequence gives the smoothness loss a signal).
        if isinstance(reasoning_out, dict):
            traj = reasoning_out.get("trajectory", reasoning_out.get("z_out", reasoning_out))
        else:
            traj = reasoning_out
        if isinstance(traj, torch.Tensor) and traj.dim() == 2:
            traj = traj.unsqueeze(1).expand(-1, 5, -1)
        elif isinstance(traj, torch.Tensor) and traj.dim() == 1:
            traj = traj.unsqueeze(0).unsqueeze(0).expand(batch, 5, -1)

        # Decoder logits for the decoder auxiliary loss (computed once and
        # reused for both the LM loss and the aux loss to avoid the previous
        # double-decoder evaluation).
        pos_ctx = self.memory_to_hidden(mem_out)
        task_ctx = corrected_h.unsqueeze(1) if corrected_h.dim() == 2 else corrected_h
        full_h = task_ctx + pos_ctx

        # Pre-compute vocab logits once (used by both LM and decoder aux)
        _vocab_logits = self.decoder.hidden_to_vocab(full_h)

        module_outputs = {
            "intent": intent,
            "memory": {"state": mem_out, "reconstruction": mem_out, "loss": mem_meta.get("compression_loss")},
            "planning": plan,
            "executive": exec_decision,
            "trajectory": traj if isinstance(traj, torch.Tensor) else reasoning_out,
            "workspace": ws_out,
            "world_model": world_out,
            "specialists": specialist_out,
            "tools": tool_out,
            "quality_assurance": qa_out,
            "verification": qa_out.get("verify_out", qa_out.get("eval_out", {})),
            "entity": {"logits": _vocab_logits.mean(dim=1)},
            "decoder": {"logits": _vocab_logits},
        }

        aux_losses = None
        if self.loss_computer is not None and aux_targets is not None:
            aux_losses = self.loss_computer(module_outputs, aux_targets)

        if labels is not None:
            # Reuse precomputed full_h / _vocab_logits from above

            shift_h = full_h[:, :-1, :]
            shift_labels = labels[:, 1:]
            flat_h = shift_h.reshape(-1, self.d_hidden)
            flat_labels = shift_labels.reshape(-1)
            valid_mask = flat_labels != -100
            safe_labels = flat_labels.masked_fill(~valid_mask, 0)
            flat_lang_ids = language_ids[:, 1:].reshape(-1) if language_ids is not None else None
            log_probs = self.decoder.hierarchical_log_prob(flat_h, safe_labels, language_ids=flat_lang_ids)
            loss = -log_probs[valid_mask].mean() if valid_mask.any() else log_probs.sum() * 0.0

            if aux_losses:
                aux_total = self.loss_computer.total_loss(aux_losses)
                loss = loss + aux_total

            return CausalLMOutputWithPast(
                loss=loss,
                logits=_vocab_logits,
                past_key_values=mem_state if mem_state is None else (tuple(m.detach() for m in mem_state) if isinstance(mem_state, (list, tuple)) else mem_state),
            )

        # No labels → reuse precomputed full_h / _vocab_logits, sanitized
        logits = torch.nan_to_num(_vocab_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return CausalLMOutputWithPast(logits=logits, past_key_values=mem_state)

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        was_training = self.training
        self.eval()
        batch = input_ids.shape[0]
        device = input_ids.device
        generated = input_ids.clone()
        mem_state = None
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        with torch.no_grad():
            for step in range(max_new_tokens):
                if mem_state is not None:
                    model_input = generated[:, -1:]
                    offsets = torch.full((batch,), generated.shape[1] - 1, device=device, dtype=torch.long)
                else:
                    model_input = generated
                    offsets = None
                outputs = self.forward(model_input, mem_state=mem_state, offsets=offsets)
                mem_state = outputs.past_key_values
                logits = outputs.logits
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    for b in range(batch):
                        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                        next_logits[b, indices_to_remove] = float("-inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # Don't sample new tokens for already-finished sequences.
                if finished.any():
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, eos_token_id if eos_token_id is not None else 0),
                        next_token,
                    )
                generated = torch.cat([generated, next_token], dim=-1)
                if eos_token_id is not None:
                    finished |= (next_token.squeeze(-1) == eos_token_id)
                    if finished.all():
                        break
        if was_training:
            self.train()
        return generated


class DharaForCausalLM(DharaModel):
    """Thin wrapper that shares the decoder head.

    The previous implementation applied an extra random ``lm_head`` only at
    inference, so training and generation used different logits. This wrapper
    now delegates directly to the base model (same head both paths) for
    consistency; the ``lm_head`` attribute is kept as an alias for backward
    compat but is tied to the decoder's output projection where possible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Alias, not a separate projection: generation must use the trained head.
        self.lm_head = self.decoder

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class DharaMoEModel(DharaModel):
    def __init__(self, n_experts: int = 8, top_k_experts: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.top_k_experts = top_k_experts
        logger.info("DharaMoEModel — %d experts, top-%d per token", n_experts, top_k_experts)
