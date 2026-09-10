from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nslt.layer1_ssm import SSMCompressionEngine

logger = logging.getLogger(__name__)


class GatedMemoryCell(nn.Module):
    """
    Gated write/erase memory cell for hierarchical state transfer.

    At each update step, receives a new candidate state from the level below
    and decides what to write, what to erase, and what to retain:

        write_gate = sigmoid(W_w * z + b_w)
        erase_gate = sigmoid(W_e * z + b_e)
        h_new = erase_gate * h + write_gate * z

    Args:
        d_state: State dimension
    """

    def __init__(self, d_state: int = 2048):
        super().__init__()
        self.d_state = d_state

        self.write_gate = nn.Linear(d_state, d_state, bias=True)
        self.erase_gate = nn.Linear(d_state, d_state, bias=True)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, d_state] — current memory state
            z: [batch, d_state] — candidate input from lower level

        Returns:
            h_new: [batch, d_state] — updated memory state
        """
        write = torch.sigmoid(self.write_gate(z))
        erase = torch.sigmoid(self.erase_gate(z))
        # Complementary gates so memory stays bounded (previously erase+write
        # could both be 1 → double). Use erase for retention and write for update.
        return erase * h + (1 - erase) * write * z


class MultiScaleSSM(nn.Module):
    """
    Multi-Scale State-Space Model with gated memory hierarchy.

    Maintains three levels of compressed state:
        Level 0 (fast):  Updated every token — local / syntactic context
        Level 1 (medium): Updated every K tokens — paragraph semantics
        Level 2 (slow):   Updated every L tokens — document themes / rare facts

    Each level has its own SSM engine and a gated memory cell that controls
    information flow between levels. The final compressed state is the
    concatenation of all three levels, providing richer O(1) context.

    Args:
        d_model: Token embedding dimension
        d_state: Per-level state dimension (total = d_state * 3)
        dt_rank: Step-size projection rank
        expand_factor: SSM inner dimension expansion
        n_ssm_layers: Number of SSM layers per level
        k_medium: Update medium level every K tokens
        k_slow: Update slow level every L tokens
    """

    def __init__(
        self,
        d_model: int = 7168,
        d_state: int = 2048,
        dt_rank: int = 256,
        expand_factor: int = 2,
        n_ssm_layers: int = 2,
        k_medium: int = 64,
        k_slow: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.n_ssm_layers = n_ssm_layers
        self.k_medium = k_medium
        self.k_slow = k_slow

        # Level 0 (fast): standard SSM stack, updated every token
        self.ssm_fast = nn.ModuleList([
            SSMCompressionEngine(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                expand_factor=expand_factor,
            )
            for _ in range(n_ssm_layers)
        ])

        # Level 1 (medium): integrates level-0 state every K tokens
        self.ssm_medium = SSMCompressionEngine(
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            expand_factor=expand_factor,
        )

        # Level 2 (slow): integrates level-1 state every L tokens
        self.ssm_slow = SSMCompressionEngine(
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            expand_factor=expand_factor,
        )

        # Gated memory cells for hierarchical transfer
        self.gate_fast_to_medium = GatedMemoryCell(d_state=d_state)
        self.gate_medium_to_slow = GatedMemoryCell(d_state=d_state)

        # Compression: lower-level state -> upper-level input
        self.down_medium = nn.Linear(d_state * 2, d_model, bias=False)
        self.down_slow = nn.Linear(d_state * 2, d_model, bias=False)

        # Final mix: combine all levels
        self.mix = nn.Linear(d_state * 3, d_model, bias=False)

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
            x: [batch, seq_len, d_model] — token embeddings + RoPE
            return_state: If True, return final compressed hierarchy state

        Returns:
            output: [batch, seq_len, d_model] — processed sequence
            h_total: [batch, d_state * 3] — concatenated hierarchy state
        """
        batch, seq_len, d_model = x.shape
        device = x.device
        dtype = x.dtype

        if seq_len == 0:
            empty = torch.zeros(batch, 0, d_model, device=device, dtype=dtype)
            h_total = torch.zeros(batch, self.d_state * 3, device=device, dtype=dtype)
            return (empty, h_total) if return_state else (empty, h_total.detach())

        h_med = torch.zeros(batch, self.d_state, device=device, dtype=dtype)
        h_slow = torch.zeros(batch, self.d_state, device=device, dtype=dtype)
        h_fast_t = torch.zeros(batch, self.d_state, device=device, dtype=dtype)

        medium_counter = 0
        slow_counter = 0

        outputs = []

        # NOTE: per-token Python loop is required for the hierarchical schedule
        # (medium/slow levels update every K/L tokens). Batched scan would need
        # ragged scheduling; this is intentionally sequential. For long contexts
        # the fast SSM itself is vectorized (selective_scan) so per-step cost is
        # dominated by SSM, not Python overhead.
        for t in range(seq_len):
            xt = x[:, t:t+1, :]  # [batch, 1, d_model]

            # Level 0 (fast): update every token
            for ssm in self.ssm_fast:
                xt, h_fast_t = ssm(xt, return_state=True)
            outputs.append(xt)

            # Level 1 (medium): update every K tokens
            medium_counter += 1
            if medium_counter >= self.k_medium:
                medium_counter = 0
                med_input = self.down_medium(
                    torch.cat([h_fast_t, h_med], dim=-1)
                ).unsqueeze(1)
                _, h_med_candidate = self.ssm_medium(med_input, return_state=True)
                h_med = self.gate_fast_to_medium(h_med, h_med_candidate)

            # Level 2 (slow): update every L tokens
            slow_counter += 1
            if slow_counter >= self.k_slow:
                slow_counter = 0
                slow_input = self.down_slow(
                    torch.cat([h_med, h_slow], dim=-1)
                ).unsqueeze(1)
                _, h_slow_candidate = self.ssm_slow(slow_input, return_state=True)
                h_slow = self.gate_medium_to_slow(h_slow, h_slow_candidate)

        output = torch.cat(outputs, dim=1)  # [batch, seq_len, d_model]

        # Mix hierarchy into final output
        h_total = torch.cat([h_fast_t, h_med, h_slow], dim=-1)  # [batch, d_state*3]
        bias = self.mix(h_total).unsqueeze(1)  # [batch, 1, d_model]
        output = output + bias

        if return_state:
            return output, h_total

        return output, h_total.detach()

    def get_config(self) -> dict:
        return {
            "type": "MultiScaleSSM",
            "d_model": self.d_model,
            "d_state": self.d_state,
            "dt_rank": self.dt_rank,
            "expand_factor": 2,
            "n_ssm_layers": self.n_ssm_layers,
            "k_medium": self.k_medium,
            "k_slow": self.k_slow,
            "total_state_dim": self.d_state * 3,
        }
