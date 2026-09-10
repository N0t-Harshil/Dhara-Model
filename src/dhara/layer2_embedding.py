import torch
import torch.nn as nn
import math


class RotaryPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 262144, base: float = 10000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, offsets: torch.LongTensor = None) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0, f"d_model must be even for RoPE, got {x.shape[-1]}"
        batch, seq_len, d_model = x.shape
        device = x.device
        if offsets is not None:
            assert offsets.shape == (batch,), f"offsets expected [batch], got {tuple(offsets.shape)}"
            pos = offsets.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)
        else:
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        inv_freq = self.inv_freq.to(device)
        freqs = torch.einsum("bl,f->blf", pos.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        half = d_model // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos1, cos2 = cos[..., :half], cos[..., half:]
        sin1, sin2 = sin[..., :half], sin[..., half:]
        rotated1 = x1 * cos1 - x2 * sin1
        rotated2 = x1 * sin1 + x2 * cos1
        return torch.cat([rotated1, rotated2], dim=-1)


class TaskContextEmbedding(nn.Module):
    def __init__(self, d_model: int, n_task_types: int = 16):
        super().__init__()
        self.task_embed = nn.Embedding(n_task_types, d_model)

    def forward(self, task_ids: torch.LongTensor) -> torch.Tensor:
        return self.task_embed(task_ids)


class ContextAdapterBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = residual + x
        # Post-norm (NOT `x + norm(x)`): adding the normalized activation as a
        # residual injected an extra unit-scale signal per block and compounded
        # activation growth across blocks.
        x = self.norm2(x)
        return x


class ContextAdapter(nn.Module):
    def __init__(self, d_model: int, n_blocks: int = 4, d_ff: int = None):
        super().__init__()
        self.blocks = nn.ModuleList([ContextAdapterBlock(d_model, d_ff) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AdaptiveSemanticEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 262144, rope_base: float = 10000000.0,
                 n_task_types: int = 16, n_context_adapter_blocks: int = 4):
        super().__init__()
        self.rope = RotaryPositionEncoding(d_model, max_seq_len, rope_base)
        self.task_context = TaskContextEmbedding(d_model, n_task_types)
        self.context_adapter = ContextAdapter(d_model, n_context_adapter_blocks)

    def forward(self, token_embeds: torch.Tensor, task_ids: torch.LongTensor = None, offsets: torch.LongTensor = None) -> torch.Tensor:
        x = self.rope(token_embeds, offsets=offsets)
        if task_ids is not None:
            task_emb = self.task_context(task_ids)
            x = x + task_emb.unsqueeze(1)
        x = self.context_adapter(x)
        return x
