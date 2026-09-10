from __future__ import annotations

import math
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.nslt.layer1_ssm import SSMCompressionEngine
from src.nslt.layer2_ltc import LTCRoutingLayer
from src.nslt.layer3_sandbox import LatentSandbox, LatentSandboxEfficient
from src.nslt.layer4_output import SparseOutputSynthesizer
from src.nslt.moe_ssm import MoE_SSM_Block
from src.nslt.vision_encoder import SigLIPVisionEncoder
from src.nslt.mcts_sandbox import MCTSLatentSandbox, MCTSLatentSandboxEfficient

logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """
    Learned token embeddings with optional tied weights support.

    Args:
        vocab_size: Vocabulary size
        d_model: Embedding dimension
        padding_idx: Index for padding token
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """x: [batch, seq_len] -> [batch, seq_len, d_model]"""
        return self.embedding(x) * self.scale


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — applied to token embeddings for
    position-aware processing before the SSM compression.

    Args:
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        base: Theta base for frequency computation
    """

    def __init__(self, d_model: int, max_seq_len: int = 262144, base: float = 10000000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, offsets: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """x: [batch, seq_len, d_model], offsets: [batch] optionally"""
        batch, seq_len, d_model = x.shape
        device = x.device

        # Compute position indices
        if offsets is not None:
            pos = offsets.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)
        else:
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        # Compute cos/sin for each position and frequency
        inv_freq = self.inv_freq.to(device)  # [d_model/2]
        pos = pos.float()  # [batch, seq_len]
        freqs = torch.einsum("bl,f->blf", pos, inv_freq)  # [batch, seq_len, d_model/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq_len, d_model]

        cos = emb.cos()
        sin = emb.sin()

        # Apply rotary embedding
        x_rotated = self._apply_rotary(x, cos, sin)
        return x_rotated

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embedding to input."""
        d_model = x.shape[-1]
        half = d_model // 2

        x1 = x[..., :half]
        x2 = x[..., half:]

        cos1 = cos[..., :half]
        cos2 = cos[..., half:]
        sin1 = sin[..., :half]
        sin2 = sin[..., half:]

        rotated1 = x1 * cos1 - x2 * sin1
        rotated2 = x1 * sin1 + x2 * cos1

        return torch.cat([rotated1, rotated2], dim=-1)


class NSLTModel(nn.Module):
    """
    Neural-State Liquid Transformer (NSLT) — Full Model

    Combines all 4 architectural layers into a single end-to-end model.

    Architecture pipeline:
        Input IDs [batch, seq_len]
        → Token Embedding [batch, seq_len, d_model]
        → RoPE Encoding [batch, seq_len, d_model]
        → Layer 1: SSM Compression [batch, seq_len, d_model] + [batch, d_state]
        → Layer 1b: Optional stacked SSM blocks (N_ssm_layers)
        → Layer 2: LTC Routing [batch, d_hidden] + trajectory
        → Layer 3: Latent Sandbox [batch, d_hidden]
        → Layer 4: Sparse Output [batch, vocab_size]

    Args:
        vocab_size: Vocabulary size
        d_model: Token embedding / model dimension (default: 7168)
        d_state: SSM compressed state dimension (default: 2048)
        d_hidden: LTC / Sandbox hidden dimension (default: 7168)
        n_ssm_layers: Number of SSM compression layers to stack
        max_seq_len: Maximum sequence length for RoPE
        rope_base: Theta base for RoPE frequencies
        sparsity_pct: Output sparsity percentage (default: 1.0)
        n_ode_steps: Number of ODE integration steps in Layer 2
        n_trajectories: Number of parallel trajectories in Layer 3
        n_sim_steps: Number of energy descent steps in Layer 3
        use_efficient_sandbox: Use memory-efficient LatentSandbox variant
        device: Target device
        dtype: Target dtype
    """

    def __init__(
        self,
        vocab_size: int = 128000,
        d_model: int = 7168,
        d_state: int = 2048,
        d_hidden: int = 7168,
        n_ssm_layers: int = 4,
        max_seq_len: int = 262144,
        rope_base: float = 10000000.0,
        sparsity_pct: float = 1.0,
        n_ode_steps: int = 8,
        n_trajectories: int = 8,
        n_sim_steps: int = 16,
        use_efficient_sandbox: bool = False,
        use_mcts_sandbox: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_hidden = d_hidden
        self.n_ssm_layers = n_ssm_layers
        self.config = SimpleNamespace(
            model_type="nslt",
            is_encoder_decoder=False,
            vocab_size=vocab_size,
            hidden_size=d_model,
            use_cache=True,
        )

        # Token embeddings + RoPE position encoding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.rope = RotaryPositionEncoding(d_model, max_seq_len, rope_base)

        # Layer 1: Stacked SSM Compression Engines
        # Each SSM block compresses the sequence further
        self.ssm_layers = nn.ModuleList([
            SSMCompressionEngine(
                d_model=d_model,
                d_state=d_state,
                dt_rank=max(256, d_model // 32),
                expand_factor=2,
            )
            for _ in range(n_ssm_layers)
        ])

        # SSM output → LTC projection
        self.ssm_to_ltc = nn.Linear(d_model, d_hidden, bias=False)

        # Layer 2: Liquid Time-Constant Routing
        self.ltc = LTCRoutingLayer(
            d_state=d_state,
            d_hidden=d_hidden,
            n_ode_steps=n_ode_steps,
            solver="rk4",
            use_adjoint=True,
        )

        # Layer 3: Latent-Space Sandbox (select variant)
        if use_mcts_sandbox:
            SandboxClass = MCTSLatentSandboxEfficient if use_efficient_sandbox else MCTSLatentSandbox
            self.sandbox = SandboxClass(
                d_hidden=d_hidden,
                d_latent=min(d_hidden // 4, 4096),
                n_simulations=n_trajectories,
                n_directions=4,
                n_sim_steps=n_sim_steps,
                max_depth=3,
            )
        else:
            SandboxClass = LatentSandboxEfficient if use_efficient_sandbox else LatentSandbox
            self.sandbox = SandboxClass(
                d_hidden=d_hidden,
                d_latent=min(d_hidden // 4, 4096),
                n_trajectories=n_trajectories,
                n_sim_steps=n_sim_steps,
                select_every=max(1, n_sim_steps // 4),
            )

        # Layer 4: Sparse Output Synthesizer (shared embedding)
        self.output_synthesizer = SparseOutputSynthesizer(
            d_hidden=d_hidden,
            vocab_size=vocab_size,
            d_model=d_model,
            sparsity_pct=sparsity_pct,
            adaptive_sparsity=False,
            use_shared_embedding=True,
            embedding_matrix=self.token_embedding.embedding.weight,
        )

        # Per-position projection: SSM output [d_model] -> hidden [d_hidden]
        # Replaces the dense LM head O(V·d_model) with O(top_k·d_hidden)
        self.ssm_to_hidden = nn.Linear(d_model, d_hidden, bias=False)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_hidden, eps=1e-6)

        if device.type != "meta":
            self.to(device=device, dtype=dtype)
        self._log_architecture()

    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "NSLTModel — %.2fB total params (%.2fB trainable) | "
            "Vocab=%d, d_model=%d, d_state=%d, d_hidden=%d, "
            "SSM_layers=%d, LTC_steps=%d, Sandbox_traj=%d, "
            "Output_sparsity=%.1f%%",
            total_params / 1e9, trainable / 1e9,
            self.vocab_size, self.d_model, self.d_state, self.d_hidden,
            self.n_ssm_layers, self.ltc.n_ode_steps,
            self.sandbox.n_trajectories, self.output_synthesizer.sparsity_pct,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor,        # [batch, seq_len]
        attention_mask: Optional[torch.LongTensor] = None,  # [batch, seq_len], accepted for HF Trainer compatibility
        labels: Optional[torch.LongTensor] = None,  # [batch, seq_len]
        return_compressed_state: bool = False,
        return_trajectories: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        batch, seq_len = input_ids.shape

        # ── Token Embedding ────────────────────────────────────────────────
        # [batch, seq_len, d_model]
        x = self.token_embedding(input_ids)

        # ── Rotary Position Encoding ────────────────────────────────────────
        # [batch, seq_len, d_model]
        x = self.rope(x)

        # Mask padded positions out of the pipeline entirely: padding tokens
        # must not pollute the SSM compressed state, the pooled context, or
        # the per-position features used for the loss (items 4.1/4.2).
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # ── Layer 1: SSM Compression Stack ─────────────────────────────────
        # Each SSM block compresses and processes the sequence
        compressed_states = []
        for ssm_layer in self.ssm_layers:
            x, h_final = ssm_layer(x, return_state=True)
            # x: [batch, seq_len, d_model], h_final: [batch, d_state]
            compressed_states.append(h_final)

        # Use the last SSM layer's compressed state as the O(1) context
        # [batch, d_state]
        h_compressed = compressed_states[-1]

        # Pool sequence dimension via mean for secondary context
        # [batch, d_model]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x_pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x_pooled = x.mean(dim=1)

        # Project to LTC hidden dimension
        # [batch, d_hidden]
        z_input = self.ssm_to_ltc(x_pooled)

        # ── Layer 2: LTC Routing (Continuous ODE) ──────────────────────────
        # Takes h_compressed (O(1) state) + z_input and evolves through ODE
        # [batch, d_hidden]
        z_ltc, ltc_trajectory = self.ltc(h_compressed, return_trajectory=return_trajectories)

        # Combine pooled input with LTC output via residual
        z_ltc = z_ltc + z_input

        # ── Layer 3: Latent-Space Sandbox ───────────────────────────────────
        # Parallel energy-based reasoning in latent space
        # [batch, d_hidden]
        z_sandbox, energies = self.sandbox(z_ltc, return_trajectories=return_trajectories)

        # Residual from LTC
        z_sandbox = z_sandbox + z_ltc

        # Final normalization
        # [batch, d_hidden]
        z_final = self.final_norm(z_sandbox)

        # ── Layer 4: Ultra-Sparse Output (O(1) memory — no dense LM head) ──
        # Reasoning output — single prediction for the whole sequence
        # [batch, vocab_size]
        main_logits = self.output_synthesizer(z_final)

        # Per-position features from SSM output (no vocabulary projection)
        # [batch, seq_len, d_model] -> [batch, seq_len, d_hidden]
        pos_hidden = self.ssm_to_hidden(x)

        if labels is not None:
            # Causal next-token loss — position t predicts token t+1. The
            # per-position features come ONLY from the causal SSM output
            # (position t depends on tokens <= t). The previous code broadcast
            # the whole-sequence z_final into every position, leaking future
            # tokens into past predictions; z_final is reserved for the
            # sequence-level reasoning head instead (see below).
            shift_hidden = pos_hidden[:, :-1, :]  # [batch, seq_len-1, d_hidden]
            shift_labels = labels[:, 1:]          # [batch, seq_len-1]
            flat_hidden = shift_hidden.reshape(-1, self.d_hidden)
            flat_labels = shift_labels.reshape(-1)
            valid_mask = flat_labels != -100
            safe_labels = flat_labels.masked_fill(~valid_mask, 0)

            # Compute log-probs using sparse output synthesizer
            log_probs = self.output_synthesizer.compute_log_prob(
                flat_hidden, safe_labels
            )  # [batch*seq_len-1]

            # Mask ignore_index tokens (-100)
            if valid_mask.any():
                loss = -log_probs[valid_mask].mean()
            else:
                loss = log_probs.sum() * 0.0

            # Sequence-level reasoning head: predicts the token that follows
            # the full context from the compressed whole-sequence state (no
            # per-position leak — it observes the entire input and emits one
            # prediction). Trained with the final valid label as its target.
            final_valid = labels[:, -1] != -100
            if final_valid.any():
                reason_logits = torch.nan_to_num(
                    main_logits[final_valid], nan=0.0, posinf=50.0, neginf=-50.0
                )
                reason_labels = labels[final_valid, -1]
                loss = loss + 0.1 * F.cross_entropy(reason_logits, reason_labels)

            trajectory_list = [ltc_trajectory, energies] if return_trajectories else None
            return CausalLMOutputWithPast(
                loss=loss,
                logits=main_logits,
                past_key_values=h_compressed if return_compressed_state else None,
            )

        # For inference/generation: materialize per-position logits
        # Only the final position's logits are needed for next-token sampling.
        # We still compute all positions for the generation loop convenience.
        batch_v = pos_hidden.size(0)
        flat_hidden = pos_hidden.view(-1, self.d_hidden)
        flat_logits = []
        chunk_size = 1024
        for i in range(0, flat_hidden.size(0), chunk_size):
            chunk = flat_hidden[i:i+chunk_size]
            chunk_logits = self.output_synthesizer(chunk)
            flat_logits.append(chunk_logits)
        logits = torch.cat(flat_logits, dim=0).view(batch_v, seq_len, self.vocab_size)
        logits = logits + main_logits.unsqueeze(1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=h_compressed if return_compressed_state else None,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,    # [batch, seq_len]
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Autoregressive generation with the NSLT model.

        Note: Unlike standard transformers, NSLT can re-encode the full
        context in O(1) memory each step, avoiding KV-cache entirely.
        """
        self.eval()
        batch = input_ids.shape[0]
        device = input_ids.device

        generated = input_ids.clone()
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Full forward pass (O(1) memory — no KV-cache growth)
                outputs = self.forward(generated)
                logits = outputs.logits

                # Get logits for the last position
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = -float("inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = False

                    for b in range(batch):
                        indices_to_remove = sorted_indices[b][
                            sorted_indices_to_remove[b]
                        ]
                        next_logits[b, indices_to_remove] = -float("inf")

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Concatenate
                generated = torch.cat([generated, next_token], dim=-1)

                # Per-sequence EOS tracking: only stop when every sequence
                # has emitted EOS (previously breaking on *any* EOS truncated
                # the rest of the batch).
                if eos_token_id is not None:
                    finished |= (next_token.squeeze(-1) == eos_token_id)
                    if finished.all():
                        break
                    # For finished sequences, keep the last sampled token but
                    # mask its logits on subsequent steps by forcing eos.
                    # (The full forward still runs; the next iteration's sampled
                    # token for finished rows will be overwritten.)
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )

        return generated

    def get_compressed_state(
        self,
        input_ids: torch.LongTensor,  # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Extract the O(1) compressed state from Layer 1 for a given input.

        This is the core memory mechanism — the compressed state captures the
        entire context in a fixed-size vector, enabling O(1) memory complexity.
        """
        batch, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.rope(x)

        for ssm_layer in self.ssm_layers:
            x, h_final = ssm_layer(x, return_state=True)

        return h_final  # [batch, d_state]

    def forward_multimodal(
        self,
        input_ids: torch.LongTensor,
        images: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with optional visual input.

        Encodes images into the compressed state space and fuses with
        text-based compressed state before the LTC routing layer.

        Args:
            input_ids: [batch, seq_len]
            images: [batch, 3, H, W] — normalized image tensors
            labels: [batch, seq_len] — optional target labels

        Returns:
            logits or loss (same as forward())
        """
        if not hasattr(self, 'vision_encoder'):
            raise AttributeError(
                "vision_encoder not initialized. "
                "Use NSLTModel.from_multimodal_config() to create a multimodal model."
            )

        batch, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.rope(x)

        for ssm_layer in self.ssm_layers:
            x, h_final = ssm_layer(x, return_state=True)
        h_compressed = h_final

        visual_state = self.vision_encoder(images)
        h_compressed = h_compressed + self.vision_proj(visual_state)
        h_compressed = self.vision_norm(h_compressed)

        x_pooled = x.mean(dim=1)
        z_input = self.ssm_to_ltc(x_pooled)
        z_ltc, _ = self.ltc(h_compressed)
        z_ltc = z_ltc + z_input

        z_sandbox, _ = self.sandbox(z_ltc)
        z_sandbox = z_sandbox + z_ltc
        z_final = self.final_norm(z_sandbox)

        main_logits = self.output_synthesizer(z_final)

        pos_hidden = self.ssm_to_hidden(x)

        if labels is not None:
            shift_hidden = pos_hidden[:, :-1, :]
            shift_labels = labels[:, 1:]
            flat_hidden = shift_hidden.reshape(-1, self.d_hidden)
            flat_labels = shift_labels.reshape(-1)
            valid_mask = flat_labels != -100
            safe_labels = flat_labels.masked_fill(~valid_mask, 0)

            log_probs = self.output_synthesizer.compute_log_prob(
                flat_hidden, safe_labels
            )
            loss = -log_probs[valid_mask].mean() if valid_mask.any() else log_probs.sum() * 0.0

            final_valid = labels[:, -1] != -100
            if final_valid.any():
                reason_logits = torch.nan_to_num(
                    main_logits[final_valid], nan=0.0, posinf=50.0, neginf=-50.0
                )
                reason_labels = labels[final_valid, -1]
                loss = loss + 0.1 * F.cross_entropy(reason_logits, reason_labels)
            return CausalLMOutputWithPast(loss=loss, logits=main_logits)

        batch_v = pos_hidden.size(0)
        flat_hidden = pos_hidden.view(-1, self.d_hidden)
        flat_logits = []
        for i in range(0, flat_hidden.size(0), 1024):
            chunk = flat_hidden[i:i+1024]
            flat_logits.append(self.output_synthesizer(chunk))
        logits = torch.cat(flat_logits, dim=0).view(batch_v, seq_len, self.vocab_size)
        logits = logits + main_logits.unsqueeze(1)
        return CausalLMOutputWithPast(logits=logits)

    @classmethod
    def from_multimodal_config(
        cls,
        d_vision: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        vision_d_state: Optional[int] = None,
        **kwargs,
    ):
        model = cls(**kwargs)
        vis_d_state = vision_d_state or model.d_state
        model.vision_encoder = SigLIPVisionEncoder(
            d_state=vis_d_state,
            image_size=image_size,
            patch_size=patch_size,
            d_vision=d_vision,
        )
        model.vision_proj = nn.Linear(vis_d_state, model.d_state, bias=False)
        model.vision_norm = nn.LayerNorm(model.d_state)
        return model

    def get_config(self) -> Dict[str, any]:
        """Return model configuration as a dictionary."""
        config = {
            "architecture": "nslt",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_hidden": self.d_hidden,
            "n_ssm_layers": self.n_ssm_layers,
            "n_ode_steps": self.ltc.n_ode_steps,
            "n_trajectories": self.sandbox.n_trajectories,
            "n_sim_steps": self.sandbox.n_sim_steps,
            "output_sparsity_pct": self.output_synthesizer.sparsity_pct,
            "total_params": sum(p.numel() for p in self.parameters()),
        }
        if hasattr(self, 'vision_encoder'):
            config["vision_encoder"] = self.vision_encoder.get_config()
        return config


class MoENSLTModel(NSLTModel):
    """
    Mixture-of-Experts variant of NSLT.

    Replaces dense SSM blocks with MoE-SSM blocks where each token activates
    only top-k of n_experts SSM experts. Maintains the same O(1) compressed
    state interface.

    Args:
        n_experts: Number of SSM experts per MoE block
        top_k_experts: Number of active experts per token
        expert_d_state: Per-expert state dimension
        Same args as NSLTModel for other parameters.
    """

    def __init__(
        self,
        vocab_size: int = 128000,
        d_model: int = 7168,
        d_state: int = 2048,
        d_hidden: int = 7168,
        n_ssm_layers: int = 4,
        max_seq_len: int = 262144,
        rope_base: float = 10000000.0,
        sparsity_pct: float = 1.0,
        n_ode_steps: int = 8,
        n_trajectories: int = 8,
        n_sim_steps: int = 16,
        use_efficient_sandbox: bool = False,
        use_mcts_sandbox: bool = False,
        n_experts: int = 8,
        top_k_experts: int = 2,
        expert_d_state: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_hidden = d_hidden
        self.n_ssm_layers = n_ssm_layers
        self.config = SimpleNamespace(
            model_type="nslt",
            is_encoder_decoder=False,
            vocab_size=vocab_size,
            hidden_size=d_model,
            use_cache=True,
        )

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.rope = RotaryPositionEncoding(d_model, max_seq_len, rope_base)

        dt_rank = max(256, d_model // 32)
        self.ssm_layers = nn.ModuleList([
            MoE_SSM_Block(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                expand_factor=2,
                n_experts=n_experts,
                top_k=top_k_experts,
                expert_d_state=expert_d_state,
            )
            for _ in range(n_ssm_layers)
        ])

        self.ssm_to_ltc = nn.Linear(d_model, d_hidden, bias=False)

        self.ltc = LTCRoutingLayer(
            d_state=d_state, d_hidden=d_hidden,
            n_ode_steps=n_ode_steps, solver="rk4", use_adjoint=True,
        )

        if use_mcts_sandbox:
            SandboxClass = MCTSLatentSandboxEfficient if use_efficient_sandbox else MCTSLatentSandbox
            self.sandbox = SandboxClass(
                d_hidden=d_hidden, d_latent=min(d_hidden // 4, 4096),
                n_simulations=n_trajectories, n_directions=4,
                n_sim_steps=n_sim_steps, max_depth=3,
            )
        else:
            SandboxClass = LatentSandboxEfficient if use_efficient_sandbox else LatentSandbox
            self.sandbox = SandboxClass(
                d_hidden=d_hidden, d_latent=min(d_hidden // 4, 4096),
                n_trajectories=n_trajectories, n_sim_steps=n_sim_steps,
                select_every=max(1, n_sim_steps // 4),
            )

        self.output_synthesizer = SparseOutputSynthesizer(
            d_hidden=d_hidden, vocab_size=vocab_size, d_model=d_model,
            sparsity_pct=sparsity_pct, adaptive_sparsity=False, use_shared_embedding=True,
            embedding_matrix=self.token_embedding.embedding.weight,
        )

        self.ssm_to_hidden = nn.Linear(d_model, d_hidden, bias=False)

        self.final_norm = nn.LayerNorm(d_hidden, eps=1e-6)
        if device.type != "meta":
            self.to(device=device, dtype=dtype)
        self._log_architecture()

    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "MoENSLTModel — %.2fB total params (%.2fB trainable) | "
            "Vocab=%d, d_model=%d, d_state=%d, d_hidden=%d, "
            "SSM_layers=%d, Experts=%d, TopK=%d, "
            "LTC_steps=%d, Sandbox_traj=%d, Output_sparsity=%.1f%%",
            total_params / 1e9, trainable / 1e9,
            self.vocab_size, self.d_model, self.d_state, self.d_hidden,
            self.n_ssm_layers, self.ssm_layers[0].n_experts if hasattr(self.ssm_layers[0], 'n_experts') else 1,
            self.ssm_layers[0].top_k if hasattr(self.ssm_layers[0], 'top_k') else 1,
            self.ltc.n_ode_steps, self.sandbox.n_trajectories,
            self.output_synthesizer.sparsity_pct,
        )
