from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SparseGatingUnit(nn.Module):
    """
    Ultra-sparse gating mechanism with adaptive top-k.

    Selects a variable number of vocabulary entries based on the entropy
    of the gate logits. High uncertainty (high entropy) → more vocabulary
    access. Low uncertainty (confident prediction) → extreme sparsity.

    The adaptive k for each batch element is:
        k_b = min_k + (max_k - min_k) * (entropy_b / log(V))

    where entropy_b is the Shannon entropy of the gate distribution.

    Args:
        d_hidden: Hidden dimension from Layer 3
        vocab_size: Total vocabulary size (V)
        min_sparsity_pct: Minimum sparsity (most aggressive, default: 0.5%)
        max_sparsity_pct: Maximum sparsity (most permissive, default: 2.0%)
        adaptive: Enable adaptive top-k based on entropy
    """

    def __init__(
        self,
        d_hidden: int = 7168,
        vocab_size: int = 128000,
        min_sparsity_pct: float = 0.5,
        max_sparsity_pct: float = 2.0,
        adaptive: bool = True,
        sparsity_pct: Optional[float] = None,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        if sparsity_pct is not None:
            adaptive = False
            min_sparsity_pct = sparsity_pct
            max_sparsity_pct = sparsity_pct
        self.min_sparsity_pct = min_sparsity_pct
        self.max_sparsity_pct = max_sparsity_pct
        self.adaptive = adaptive
        self.min_k = max(1, int(vocab_size * min_sparsity_pct / 100.0))
        self.max_k = max(1, int(vocab_size * max_sparsity_pct / 100.0))
        self.default_k = max(1, int(vocab_size * (min_sparsity_pct + max_sparsity_pct) / 200.0))

        self.gate_proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, vocab_size),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.LongTensor, Optional[torch.Tensor]]:
        """
        Compute sparse gate with adaptive top-k and return active indices.

        Returns:
            gate_values: [batch, effective_k] — softmax over selected slots
            top_indices: [batch, effective_k] — indices of active vocabulary entries
            entropy_ratio: [batch] — normalized entropy for each sample (None if not adaptive)
        """
        gate_logits = self.gate_proj(x)
        gate_logits = gate_logits - gate_logits.mean(dim=-1, keepdim=True)

        if self.adaptive:
            probs = F.softmax(gate_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            max_entropy = math.log(self.vocab_size)
            entropy_ratio = (entropy / max_entropy).clamp(0.0, 1.0)

            k_per_batch = (
                self.min_k + (self.max_k - self.min_k) * entropy_ratio
            ).long()
            k_per_batch = k_per_batch.clamp(1, self.vocab_size)
            effective_k = k_per_batch.max().item()
        else:
            effective_k = self.default_k
            entropy_ratio = None

        gate_values, top_indices = torch.topk(
            gate_logits, effective_k, dim=-1
        )

        if self.adaptive:
            k_mask = torch.arange(effective_k, device=x.device).unsqueeze(0) >= k_per_batch.unsqueeze(1)
            gate_values = gate_values.masked_fill(k_mask, float("-inf"))
            top_indices = top_indices.masked_fill(k_mask, 0)

        gate_values = F.softmax(gate_values, dim=-1)

        if self.adaptive:
            return gate_values, top_indices, entropy_ratio
        return gate_values, top_indices, None


class SparseOutputSynthesizer(nn.Module):
    """
    Layer 4: Ultra-Sparse Quantum Output Synthesizer

    Produces the final output logits using extreme sparsity (top-1% of
    vocabulary). This replaces the standard dense LM head with a gated,
    ultra-sparse mixture where only ~1,280 of 128K vocabulary entries
    contribute to each output token.

    Architecture:
        1. Gating: SparseGatingUnit selects top-1% of vocabulary indices
        2. Expert computation: Only the selected vocabulary rows contribute
        3. Output: logits[i] = gate[i] * (W_E[i] @ v) for selected i, 0 otherwise

    This provides a ~100x reduction in output computation vs. standard LM head,
    with the hypothesis that only a tiny fraction of vocabulary is relevant
    for any given context.

    Args:
        d_hidden: Hidden dimension from Layer 3
        vocab_size: Vocabulary size (default: 128000)
        d_model: Model dimension for the embedding matrix (shared)
        sparsity_pct: Percentage of vocabulary to activate (default: 1.0)
        use_shared_embedding: Tie input embedding and output projection
        embedding_matrix: External embedding matrix to share (if tied)
    """

    def __init__(
        self,
        d_hidden: int = 7168,
        vocab_size: int = 128000,
        d_model: int = 7168,
        sparsity_pct: float = 1.0,
        use_shared_embedding: bool = True,
        embedding_matrix: Optional[nn.Parameter] = None,
        adaptive_sparsity: bool = True,
        min_sparsity_pct: float = 0.5,
        max_sparsity_pct: float = 2.0,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.sparsity_pct = sparsity_pct
        self.use_shared_embedding = use_shared_embedding
        self.adaptive_sparsity = adaptive_sparsity

        self.gate = SparseGatingUnit(
            d_hidden=d_hidden,
            vocab_size=vocab_size,
            min_sparsity_pct=min_sparsity_pct,
            max_sparsity_pct=max_sparsity_pct,
            adaptive=adaptive_sparsity,
        )
        self.sparsity_pct = max_sparsity_pct if adaptive_sparsity else sparsity_pct

        # Output embedding matrix (can be shared with input embeddings)
        if embedding_matrix is not None:
            self.output_embedding = embedding_matrix  # [vocab_size, d_model]
        else:
            self.output_embedding = nn.Parameter(
                torch.randn(vocab_size, d_model) * 0.01
            )

        # Project hidden state to model dimension for embedding lookup
        self.hidden_proj = nn.Linear(d_hidden, d_model, bias=False)

        # Learnable temperature for gating sharpness
        self.logit_temperature = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and "embedding" not in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        batch = x.shape[0]

        h = self.hidden_proj(x)
        gate_values, top_indices, _ = self.gate(x)

        logits = torch.zeros(batch, self.vocab_size, device=x.device, dtype=x.dtype)
        for start in range(0, batch, chunk_size):
            end = min(start + chunk_size, batch)
            h_chunk = h[start:end]
            gate_chunk = gate_values[start:end]
            top_chunk = top_indices[start:end]

            selected_embeddings = F.embedding(top_chunk, self.output_embedding)
            selected_logits = torch.sum(
                selected_embeddings * h_chunk.unsqueeze(1), dim=-1
            ) / (self.logit_temperature.abs() + 0.1)
            selected_logits = gate_chunk * selected_logits
            logits[start:end].scatter_(1, top_chunk, selected_logits)

        return logits

    def compute_log_prob(
        self,
        x: torch.Tensor,
        target_ids: torch.LongTensor,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        total = x.shape[0]
        h = self.hidden_proj(x)
        gate_values, top_indices, _ = self.gate(x)

        log_probs_list = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            x_chunk = x[start:end]
            h_chunk = h[start:end]
            target_chunk = target_ids[start:end]
            gate_chunk = gate_values[start:end] if gate_values is not None else None
            top_chunk = top_indices[start:end]

            target_emb = F.embedding(target_chunk, self.output_embedding)
            target_logit = torch.sum(target_emb * h_chunk, dim=-1) / (self.logit_temperature.abs() + 0.1)

            selected_embeddings = F.embedding(top_chunk, self.output_embedding)
            selected_logits = torch.sum(
                selected_embeddings * h_chunk.unsqueeze(1), dim=-1
            ) / (self.logit_temperature.abs() + 0.1)

            all_logits = torch.cat([selected_logits, target_logit.unsqueeze(1)], dim=-1)
            log_probs = F.log_softmax(all_logits, dim=-1)
            log_probs_list.append(log_probs[:, -1])

        return torch.cat(log_probs_list, dim=0)
