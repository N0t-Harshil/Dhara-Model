from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reflector(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.answered_check = nn.Linear(d_hidden, 1)
        self.constraint_check = nn.Linear(d_hidden, 1)
        self.contradiction_check = nn.Linear(d_hidden, 1)
        self.rethink_gate = nn.Linear(d_hidden, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "answered": torch.sigmoid(self.answered_check(h)),
            "constraints_met": torch.sigmoid(self.constraint_check(h)),
            "contradiction": torch.sigmoid(self.contradiction_check(h)),
            "needs_rethink": torch.sigmoid(self.rethink_gate(h)),
        }


class Verifier(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.syntax_head = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.compilation_head = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.runtime_head = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.math_consistency = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.logic_consistency = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "syntax_correctness": torch.sigmoid(self.syntax_head(h)),
            "compilation_prob": torch.sigmoid(self.compilation_head(h)),
            "runtime_correctness": torch.sigmoid(self.runtime_head(h)),
            "math_consistency": torch.sigmoid(self.math_consistency(h)),
            "logic_consistency": torch.sigmoid(self.logic_consistency(h)),
        }


class SelfEvaluator(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.usefulness = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.novelty = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.uncertainty = nn.Sequential(nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "usefulness": torch.sigmoid(self.usefulness(h)),
            "novelty": torch.sigmoid(self.novelty(h)),
            "uncertainty": torch.sigmoid(self.uncertainty(h)),
        }


class Corrector(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.error_proj = nn.Linear(d_hidden, d_hidden)
        self.correction_net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden),
        )
        self.correction_gate = nn.Linear(d_hidden, 1)

    def forward(self, h: torch.Tensor, error_signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        error_flat = error_signal.expand(-1, h.shape[-1]) if error_signal.shape[-1] != h.shape[-1] else error_signal
        error = self.error_proj(error_flat)
        correction = self.correction_net(torch.cat([h, error], dim=-1))
        gate = torch.sigmoid(self.correction_gate(correction))
        return gate * correction, gate.squeeze(-1)


class QualityAssurance(nn.Module):
    def __init__(self, d_hidden: int, max_passes: int = 5, converge_threshold: float = 0.05, d_model: Optional[int] = None):
        super().__init__()
        self.max_passes = max_passes
        self.converge_threshold = converge_threshold
        self.reflector = Reflector(d_hidden)
        self.verifier = Verifier(d_hidden)
        self.evaluator = SelfEvaluator(d_hidden)
        self.corrector = Corrector(d_hidden)
        self.confidence_head = nn.Linear(d_hidden, 1)
        self.pass_embed = nn.Embedding(max_passes, d_hidden)
        self.refine_proj = nn.Linear(d_hidden * 3, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.goal_proj: Optional[nn.Linear] = None
        if d_model is not None and d_model != d_hidden:
            self.goal_proj = nn.Linear(d_model, d_hidden)

    def forward(self, workspace: torch.Tensor, consensus: torch.Tensor,
                goal_embeds: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        current = consensus
        all_refined: List[torch.Tensor] = []
        confidence_trace: List[torch.Tensor] = []
        all_verify_scores: List[torch.Tensor] = []
        all_eval_scores: List[torch.Tensor] = []
        reflect_trace: List[Dict[str, torch.Tensor]] = []

        for pass_idx in range(self.max_passes):
            h = self.norm(current)
            pass_info = self.pass_embed.weight[pass_idx].unsqueeze(0).expand_as(h)
            h = h + pass_info

            reflect_out = self.reflector(h)
            verify_out = self.verifier(h)
            eval_out = self.evaluator(h)
            confidence = torch.sigmoid(self.confidence_head(h))

            error_score = 1 - (
                verify_out["syntax_correctness"] + verify_out["compilation_prob"]
                + verify_out["runtime_correctness"] + verify_out["math_consistency"]
                + verify_out["logic_consistency"]
            ).mean(dim=-1, keepdim=True) / 5

            needs_correction = (
                (reflect_out["answered"] < 0.5)
                | (reflect_out["constraints_met"] < 0.5)
                | (reflect_out["contradiction"] > 0.5)
            ).float()

            correction, corr_gate = self.corrector(h, error_score * needs_correction)

            if goal_embeds is not None:
                if goal_embeds.dim() > 2:
                    goal_context = goal_embeds.mean(dim=1)
                else:
                    goal_context = goal_embeds
                if self.goal_proj is not None:
                    goal_context = self.goal_proj(goal_context)
                refine_input = torch.cat([h, workspace, goal_context], dim=-1)
            else:
                refine_input = torch.cat([h, workspace, h], dim=-1)

            refined = self.refine_proj(refine_input)
            needs_reconsider = (needs_correction > 0.5) | (error_score > 0.3)
            current = torch.where(needs_reconsider.expand(-1, current.shape[-1]), refined + correction, current)

            all_refined.append(current.unsqueeze(1))
            confidence_trace.append(confidence)
            all_verify_scores.append(torch.stack([
                verify_out["syntax_correctness"].squeeze(-1), verify_out["compilation_prob"].squeeze(-1),
                verify_out["runtime_correctness"].squeeze(-1), verify_out["math_consistency"].squeeze(-1),
                verify_out["logic_consistency"].squeeze(-1),
            ], dim=-1))
            all_eval_scores.append(torch.stack([
                eval_out["usefulness"].squeeze(-1), eval_out["novelty"].squeeze(-1),
                1 - eval_out["uncertainty"].squeeze(-1),
            ], dim=-1))
            reflect_trace.append(reflect_out)

            if pass_idx > 0 and confidence_trace[-2] is not None:
                conf_change = (confidence - confidence_trace[-2]).abs().mean().item()
                if conf_change < self.converge_threshold and confidence.mean().item() > 0.7:
                    break

        n_passes = len(all_refined)
        verify_stack = torch.stack(all_verify_scores, dim=1)
        eval_stack = torch.stack(all_eval_scores, dim=1)

        return {
            "corrected_h": current,
            "refined": current,
            "final_confidence": confidence_trace[-1].mean().item() if confidence_trace else 0.5,
            "confidence_trace": torch.stack(confidence_trace) if confidence_trace else torch.tensor(0.5),
            "n_passes": n_passes,
            "all_refined": torch.cat(all_refined, dim=1),
            "verify_scores": verify_stack,
            "final_verify": verify_stack[:, -1, :],
            "eval_scores": eval_stack,
            "final_eval": eval_stack[:, -1, :],
            "reflect_out": reflect_trace[-1] if reflect_trace else {},
            "verify_out": {
                "syntax_correctness": verify_stack[:, -1, 0:1],
                "compilation_prob": verify_stack[:, -1, 1:2],
                "runtime_correctness": verify_stack[:, -1, 2:3],
                "math_consistency": verify_stack[:, -1, 3:4],
                "logic_consistency": verify_stack[:, -1, 4:5],
            },
            "eval_out": {
                "usefulness": eval_stack[:, -1, 0:1],
                "novelty": eval_stack[:, -1, 1:2],
                "uncertainty": 1 - eval_stack[:, -1, 2:3],
            },
        }
