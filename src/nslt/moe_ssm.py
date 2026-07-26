from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nslt.layer1_ssm import SSMCompressionEngine
from src.nslt.ssm_scan import selective_scan_vectorized

logger = logging.getLogger(__name__)


class MoE_Router(nn.Module):
    """
    Mixture-of-Experts router for sparse expert selection.

    Selects top-k experts per input token/position using a learned
    gating network with load-balancing loss.

    Args:
        d_model: Model dimension
        n_experts: Total number of experts
        top_k: Number of experts to activate per token
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            gate_weights: [batch, seq_len, top_k] — softmax weights for selected experts
            expert_indices: [batch, seq_len, top_k] — selected expert IDs
            load_balancing_loss: scalar — auxiliary load-balancing loss
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)  # [batch*seq_len, d_model]

        logits = self.gate(x_flat)  # [batch*seq_len, n_experts]
        logits = logits - logits.mean(dim=-1, keepdim=True)

        weights, indices = torch.topk(
            F.softmax(logits, dim=-1), self.top_k, dim=-1
        )

        gate_weights = weights.reshape(*original_shape[:-1], self.top_k)
        expert_indices = indices.reshape(*original_shape[:-1], self.top_k)

        importance = logits.softmax(dim=-1).sum(dim=0)
        load = torch.zeros(self.n_experts, device=x.device)
        for k in range(self.top_k):
            load.scatter_add_(0, indices[:, k], torch.ones_like(indices[:, k], dtype=torch.float32))
        load = load / load.sum()

        load_balancing_loss = self.n_experts * (importance * load).sum()

        return gate_weights, expert_indices, load_balancing_loss


class MoE_SSM_Block(nn.Module):
    """
    Mixture-of-Experts SSM block with sparse expert selection.

    Replaces a single SSM layer with multiple expert SSMs, where
    only top-k experts are activated per input position.

    Args:
        d_model: Model dimension
        d_state: SSM compressed state dimension
        dt_rank: Step-size projection rank
        expand_factor: SSM inner dimension expansion
        n_experts: Number of expert SSMs
        top_k: Number of active experts per token
        expert_d_state: Per-expert state dimension (defaults to d_state // n_experts)
    """

    def __init__(
        self,
        d_model: int = 7168,
        d_state: int = 2048,
        dt_rank: int = 256,
        expand_factor: int = 2,
        n_experts: int = 8,
        top_k: int = 2,
        expert_d_state: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_d_state = expert_d_state or max(1, d_state // n_experts)
        # Ensure total state fits: n_experts * expert_d_state <= d_state
        if self.n_experts * self.expert_d_state > d_state:
            self.expert_d_state = d_state // n_experts

        self.router = MoE_Router(d_model=d_model, n_experts=n_experts, top_k=top_k)

        self.in_proj = nn.Linear(d_model, d_model * expand_factor * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_model * expand_factor,
            out_channels=d_model * expand_factor,
            kernel_size=4,
            groups=d_model * expand_factor,
            padding=3,
            bias=True,
        )

        self.act = nn.SiLU()

        self.dt_proj = nn.Linear(d_model * expand_factor, dt_rank, bias=False)
        self.dt_bias = nn.Parameter(torch.randn(dt_rank) * 0.01)

        expert_A = torch.log(
            torch.rand(n_experts, self.expert_d_state) * 0.5 + 0.001
        )
        self.expert_A_log = nn.Parameter(expert_A)

        self.B_proj = nn.Linear(d_model * expand_factor, dt_rank, bias=False)
        self.C_proj = nn.Linear(d_model * expand_factor, dt_rank, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model * expand_factor, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            return_state: If True, return compressed state

        Returns:
            output: [batch, seq_len, d_model]
            compressed_state: [batch, d_state] (sum of per-expert states)
        """
        batch, seq_len, d_model = x.shape
        device = x.device

        x = self.norm(x)

        x_proj = self.in_proj(x)
        x_inner, x_gate = x_proj.chunk(2, dim=-1)
        d_inner = x_inner.shape[-1]

        x_conv = self.conv1d(x_inner.permute(0, 2, 1))
        x_conv = x_conv[:, :, :seq_len].permute(0, 2, 1)
        x_conv = self.act(x_conv)

        delta = F.softplus(self.dt_proj(x_conv) + self.dt_bias)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        # Route tokens to experts (use normalized input x, not conv features)
        gate_weights, expert_indices, load_loss = self.router(x)

        # Each token activates top_k experts.
        # For each expert, gather its tokens, run SSM, scatter back.
        # expert_d_state is per-expert; total effective state = top_k * expert_d_state
        h_flat = torch.zeros(batch, seq_len, self.d_state, device=device, dtype=x.dtype)
        y_flat = torch.zeros(batch, seq_len, d_inner, device=device, dtype=x.dtype)

        expand_weight = torch.eye(self.expert_d_state, self.dt_rank, device=device, dtype=x.dtype)

        for e_idx in range(self.n_experts):
            mask = (expert_indices == e_idx).any(dim=-1)  # [batch, seq_len]

            if not mask.any():
                continue

            e_id = torch.where(expert_indices == e_idx, gate_weights, torch.zeros_like(gate_weights))
            e_weight = e_id.sum(dim=-1)  # [batch, seq_len]

            A_e = -torch.exp(self.expert_A_log[e_idx])  # [expert_d_state]

            delta_e = F.linear(delta, expand_weight) * e_weight.unsqueeze(-1)
            B_e = F.linear(B, expand_weight) * e_weight.unsqueeze(-1)
            C_e = F.linear(C, expand_weight)

            A_bar_e = torch.exp(delta_e * A_e.unsqueeze(0).unsqueeze(0))
            B_bar_e = (A_bar_e - 1.0) / (A_e + 1e-10) * B_e

            h_e = torch.zeros(batch, self.expert_d_state, device=device, dtype=x.dtype)
            h_e_list = []
            y_e_list = []
            for t in range(seq_len):
                h_e = A_bar_e[:, t, :] * h_e + B_bar_e[:, t, :] * x_conv[:, t, :self.expert_d_state]
                h_e_list.append(h_e.unsqueeze(1))
                y_t = torch.sum(C_e[:, t:t+1, :] * h_e.unsqueeze(1), dim=-1, keepdim=True)
                y_e_list.append(y_t * e_weight[:, t:t+1].unsqueeze(-1))
            h_e_full = torch.cat(h_e_list, dim=1)
            y_e = torch.cat(y_e_list, dim=1)

            h_flat[:, :, e_idx * self.expert_d_state: (e_idx + 1) * self.expert_d_state] += h_e_full
            y_flat = y_flat + y_e

        if d_inner > self.expert_d_state:
            y_flat = y_flat + x_conv[:, :, :d_inner]

        output = y_flat * self.act(x_gate)
        output = self.out_proj(output)

        compressed_state = h_flat[:, -1, :]  # [batch, d_state]

        if return_state:
            return output, compressed_state

        return output, compressed_state.detach()
