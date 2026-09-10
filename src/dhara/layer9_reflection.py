from __future__ import annotations

from typing import Optional

import torch

from src.dhara.quality_assurance import QualityAssurance


class RecursiveReflection(QualityAssurance):
    def __init__(self, d_hidden: int, max_refinement_passes: int = 5, converge_threshold: float = 0.05):
        super().__init__(d_hidden=d_hidden, max_passes=max_refinement_passes, converge_threshold=converge_threshold)

    def forward(self, workspace: torch.Tensor, consensus: torch.Tensor,
                goal_embeds: Optional[torch.Tensor] = None) -> dict:
        out = super().forward(workspace, consensus, goal_embeds)
        out["n_refinement_passes"] = out["n_passes"]
        out["final_confidence"] = out.get("final_confidence", 0.5)
        return out


class ReflectionModule(RecursiveReflection):
    pass

__all__ = ["RecursiveReflection", "ReflectionModule"]
