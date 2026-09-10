from __future__ import annotations

from typing import Optional

import torch

from src.dhara.quality_assurance import QualityAssurance


class VerificationWithRepair(QualityAssurance):
    def __init__(self, d_hidden: int, max_repair_iters: int = 3):
        super().__init__(d_hidden=d_hidden, max_passes=max_repair_iters)

    def forward(self, h: torch.Tensor) -> dict:
        out = super().forward(h, h)
        return {
            "overall_confidence": out["final_confidence"],
            "verified_representation": out["corrected_h"] - h,
            "corrected_h": out["corrected_h"],
            "n_repair_iters": out["n_passes"],
            "confidence_trace": out["confidence_trace"],
            "code": out.get("verify_out", {}),
            "math": out.get("verify_out", {}),
            "logic": out.get("verify_out", {}),
        }


class VerificationModule(VerificationWithRepair):
    pass

__all__ = ["VerificationWithRepair", "VerificationModule"]
