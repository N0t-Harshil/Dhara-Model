from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkspaceEntry:
    __slots__ = ("tensor", "metadata", "timestamp")

    def __init__(self, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        self.tensor = tensor
        self.metadata = metadata or {}
        self.timestamp = 0


class CognitiveWorkspace(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, max_keys: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.max_keys = max_keys
        self._step = 0
        self._cached_state: Optional[torch.Tensor] = None
        self.goal_proj = nn.Linear(d_model, d_hidden)
        self.knowledge_proj = nn.Linear(d_model, d_hidden)
        self.state_proj = nn.Linear(d_hidden, d_hidden)
        self.fusion = nn.Linear(d_hidden * 3, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.gate = nn.Linear(d_hidden, 1)
        self._store: Dict[str, WorkspaceEntry] = {}

    def _get_device(self) -> torch.device:
        return self.goal_proj.weight.device

    @torch.compiler.disable
    def reset(self, batch: int) -> None:
        self._cached_state = None
        self._store.clear()
        self._step = 0

    def write(self, key: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._store[key] = WorkspaceEntry(tensor.detach().clone(), metadata)
        self._step += 1
        if len(self._store) > self.max_keys:
            oldest = min(self._store.keys(), key=lambda k: self._store[k].timestamp)
            del self._store[oldest]

    def read(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        entry = self._store.get(key)
        if entry is not None:
            return entry.tensor
        return default

    def read_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(key)
        if entry is not None:
            return entry.metadata
        return None

    def read_all(self) -> Dict[str, torch.Tensor]:
        return {k: v.tensor for k, v in self._store.items()}

    def clear(self, keys: Optional[List[str]] = None) -> None:
        if keys is None:
            self._store.clear()
        else:
            for k in keys:
                self._store.pop(k, None)

    def update(self, goals: Dict[str, torch.Tensor], memory_out: torch.Tensor, reasoning_out: torch.Tensor) -> Dict[str, Any]:
        batch = memory_out.shape[0]
        device = memory_out.device
        g = self.goal_proj(goals.get("goal_embeds", memory_out).mean(dim=1))
        k = self.knowledge_proj(memory_out.mean(dim=1))
        s = self.state_proj(reasoning_out)
        fused = self.fusion(torch.cat([g, k, s], dim=-1))
        fused = self.norm(fused)
        gate_val = torch.sigmoid(self.gate(fused))

        if self._cached_state is None or self._cached_state.shape[0] != batch:
            prev_state = torch.zeros(batch, self.d_hidden, device=device)
        else:
            prev_state = self._cached_state
        new_repr = (1 - gate_val) * prev_state + gate_val * fused
        new_confidence = gate_val.squeeze(-1)

        self._cached_state = new_repr.detach()

        result = {
            "workspace": new_repr,
            "confidence": new_confidence,
            "goals": g,
            "knowledge": k,
            "state": s,
        }
        self.write("workspace", new_repr, {"confidence": new_confidence.mean().item()})
        self.write("goals", g)
        self.write("knowledge_pool", k)
        return result

    def broadcast(self) -> torch.Tensor:
        return self._cached_state if self._cached_state is not None else torch.zeros(1, self.d_hidden)

    def forward(self, goals: Dict[str, torch.Tensor], memory_out: torch.Tensor, reasoning_out: torch.Tensor) -> Dict[str, Any]:
        return self.update(goals, memory_out, reasoning_out)
