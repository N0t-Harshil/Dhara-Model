from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.dhara.quality_assurance import QualityAssurance


class SelfEvaluation(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.usefulness = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.novelty = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.uncertainty = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))

    def forward(self, h: torch.Tensor) -> dict:
        return {
            "usefulness": torch.sigmoid(self.usefulness(h)).squeeze(-1),
            "novelty": torch.sigmoid(self.novelty(h)).squeeze(-1),
            "uncertainty": torch.sigmoid(self.uncertainty(h)).squeeze(-1),
            "self_evaluation_score": (torch.sigmoid(self.usefulness(h))
                                      + (1 - torch.sigmoid(self.uncertainty(h)))
                                      + torch.sigmoid(self.novelty(h))).squeeze(-1) / 3,
        }


class CuriosityModule(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.evaluator = SelfEvaluation(d_hidden)
        self.exploration_drive = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, h: torch.Tensor, executive_confidence: Optional[torch.Tensor] = None) -> dict:
        eval_out = self.evaluator(h)
        # exploration: [batch] scalar per sample
        exploration = torch.sigmoid(self.exploration_drive(h)).squeeze(-1)

        # novelty: SelfEvaluation returns squeeze(-1) → shape [batch] for 2D h
        # or [batch, seq] for 3D h. Ensure shapes match for the product.
        novelty = eval_out.get("novelty", torch.zeros_like(exploration))
        if novelty.shape != exploration.shape:
            # Flatten to batch-level if novelty has extra dims
            novelty = novelty.mean(dim=tuple(range(1, novelty.dim()))) if novelty.dim() > 1 else novelty

        # Apply executive confidence gating if provided (expand scalar to batch dim)
        if executive_confidence is not None:
            conf = executive_confidence
            if conf.dim() == 0:
                conf = conf.unsqueeze(0).expand(exploration.shape[0])
            elif conf.dim() > 1:
                conf = conf.view(exploration.shape[0], -1).mean(-1)
            if conf.shape[0] != exploration.shape[0]:
                conf = conf.expand(exploration.shape[0])
            exploration = exploration * conf

        return {
            "evaluation": eval_out,
            "exploration_drive": exploration,
            "consolidated_h": self.norm(h),
            "curiosity_score": exploration * novelty,
        }

__all__ = ["SelfEvaluation", "CuriosityModule"]
