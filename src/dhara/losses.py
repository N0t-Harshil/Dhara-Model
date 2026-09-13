from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def intent_loss(intent_out: Dict[str, torch.Tensor],
                task_target: Optional[torch.Tensor] = None,
                difficulty_target: Optional[torch.Tensor] = None,
                reasoning_target: Optional[torch.Tensor] = None,
                weight: float = 0.05) -> torch.Tensor:
    dev = intent_out.get("task_type", intent_out.get("difficulty", intent_out.get("reasoning_type", torch.zeros(1)))).device
    loss = torch.tensor(0.0, device=dev)
    count = 0
    if task_target is not None and "task_type" in intent_out:
        loss = loss + F.cross_entropy(intent_out["task_type"], task_target)
        count += 1
    if difficulty_target is not None and "difficulty" in intent_out:
        loss = loss + F.cross_entropy(intent_out["difficulty"], difficulty_target)
        count += 1
    if reasoning_target is not None and "reasoning_type" in intent_out:
        loss = loss + F.cross_entropy(intent_out["reasoning_type"], reasoning_target)
        count += 1
    return weight * loss / max(count, 1)


def memory_reconstruction_loss(mem_out: torch.Tensor, reconstructed: torch.Tensor,
                               weight: float = 0.01) -> torch.Tensor:
    return weight * F.mse_loss(reconstructed, mem_out.detach())


def planning_loss(plan_out: Dict[str, torch.Tensor],
                  subgoal_targets: Optional[torch.Tensor] = None,
                  weight: float = 0.05) -> torch.Tensor:
    if subgoal_targets is not None and "subgoal_logits" in plan_out:
        logits = plan_out["subgoal_logits"]
        if logits.shape[0] == subgoal_targets.shape[0]:
            return weight * F.cross_entropy(logits.view(-1, logits.size(-1)), subgoal_targets.view(-1))
    return torch.tensor(0.0, device=next(iter(plan_out.values())).device if plan_out else None)


def gate_supervision_loss(gates: Dict[str, torch.Tensor],
                          gate_targets: Dict[str, torch.Tensor],
                          weight: float = 0.01) -> torch.Tensor:
    loss = None
    count = 0
    for name, gate_val in gates.items():
        if name in gate_targets:
            bce = F.binary_cross_entropy(
                gate_val.mean(dim=-1, keepdim=True).clamp(1e-7, 1 - 1e-7),
                gate_targets[name].float().unsqueeze(-1).clamp(1e-7, 1 - 1e-7),
            )
            loss = bce if loss is None else loss + bce
            count += 1
    if loss is None:
        sample = next(iter(gates.values()))
        return torch.tensor(0.0, device=sample.device)
    return weight * loss / max(count, 1)


def verification_loss(verify_out: Dict[str, torch.Tensor],
                      correctness_targets: Dict[str, torch.Tensor],
                      weight: float = 0.02) -> torch.Tensor:
    loss = None
    count = 0
    for key, pred in verify_out.items():
        if key in correctness_targets and pred.dim() == correctness_targets[key].dim():
            bce = F.binary_cross_entropy(
                pred.clamp(1e-7, 1 - 1e-7),
                correctness_targets[key].float().clamp(1e-7, 1 - 1e-7),
            )
            loss = bce if loss is None else loss + bce
            count += 1
    if loss is None:
        sample = next(iter(verify_out.values()))
        return torch.tensor(0.0, device=sample.device)
    return weight * loss / max(count, 1)


def confidence_calibration_loss(confidence: torch.Tensor, accuracy: torch.Tensor,
                                weight: float = 0.005) -> torch.Tensor:
    ece = (confidence - accuracy).abs().mean()
    return weight * ece


def trajectory_smoothness_loss(trajectory: torch.Tensor, weight: float = 0.001) -> torch.Tensor:
    if trajectory.shape[1] < 2:
        return torch.tensor(0.0, device=trajectory.device)
    deltas = trajectory[:, 1:] - trajectory[:, :-1]
    return weight * deltas.pow(2).mean()


def tool_selection_loss(route_weights: torch.Tensor, tool_targets: Optional[torch.Tensor],
                        weight: float = 0.05) -> torch.Tensor:
    if tool_targets is not None:
        return weight * F.cross_entropy(route_weights, tool_targets)
    return torch.tensor(0.0, device=route_weights.device)


def novelty_bonus_loss(novelty: torch.Tensor, weight: float = 0.001) -> torch.Tensor:
    return -weight * novelty.mean()


def entity_prediction_loss(entity_logits: torch.Tensor,
                           entity_targets: Optional[torch.Tensor],
                           weight: float = 0.01) -> torch.Tensor:
    if entity_targets is not None:
        return weight * F.cross_entropy(entity_logits, entity_targets)
    return torch.tensor(0.0, device=entity_logits.device)


def decoder_loss(decoder_logits: torch.Tensor,
                 decoder_targets: Optional[torch.Tensor],
                 weight: float = 0.01) -> torch.Tensor:
    if decoder_targets is not None:
        flat_logits = decoder_logits.view(-1, decoder_logits.size(-1)).float()
        flat_targets = decoder_targets.reshape(-1)
        valid = flat_targets != -100
        if valid.any():
            return weight * F.cross_entropy(
                flat_logits[valid], flat_targets[valid]
            )
    return torch.tensor(0.0, device=decoder_logits.device)


class AuxiliaryLossComputer(nn.Module):
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = weights or {
            "intent": 0.05,
            "memory": 0.01,
            "planning": 0.05,
            "gate": 0.01,
            "verification": 0.02,
            "calibration": 0.005,
            "trajectory": 0.001,
            "tools": 0.05,
            "novelty": 0.001,
            "entity": 0.01,
            "decoder": 0.01,
        }

    def forward(self, module_outputs: Dict[str, Any],
                targets: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        device = self._find_device(module_outputs)

        if "intent" in module_outputs:
            t = (targets or {}).get("intent", {})
            losses["intent"] = intent_loss(
                module_outputs["intent"],
                task_target=t.get("task_type"),
                difficulty_target=t.get("difficulty"),
                reasoning_target=t.get("reasoning_type"),
                weight=self.weights.get("intent", 0.05),
            )

        if "memory" in module_outputs:
            mem = module_outputs["memory"]
            mem_ls = mem.get("loss") if isinstance(mem, dict) else None
            if isinstance(mem_ls, torch.Tensor) and mem_ls.dim() == 0:
                losses["memory"] = self.weights.get("memory", 0.01) * mem_ls
            else:
                mem_out = mem.get("state", mem) if isinstance(mem, dict) else mem
                recon = mem.get("reconstruction", mem_out) if isinstance(mem, dict) else mem_out
                losses["memory"] = memory_reconstruction_loss(
                    mem_out, recon,
                    weight=self.weights.get("memory", 0.01),
                )

        if "planning" in module_outputs:
            plan = module_outputs["planning"]
            if isinstance(plan, dict):
                st = (targets or {}).get("subgoal", {}).get("ids") if targets else None
                losses["planning"] = planning_loss(
                    plan, subgoal_targets=st,
                    weight=self.weights.get("planning", 0.05),
                )

        if "executive" in module_outputs:
            exec_out = module_outputs["executive"]
            if "gates" in exec_out:
                gt = (targets or {}).get("gates", {})
                losses["gate"] = gate_supervision_loss(
                    exec_out["gates"], gt,
                    weight=self.weights.get("gate", 0.01),
                )

        if "verification" in module_outputs:
            verify_out = module_outputs["verification"]
            if isinstance(verify_out, dict):
                ct = (targets or {}).get("correctness", {})
                losses["verification"] = verification_loss(
                    verify_out, ct,
                    weight=self.weights.get("verification", 0.02),
                )

        if "executive" in module_outputs and "confidence" in module_outputs.get("executive", {}):
            if targets and "accuracy" in targets:
                losses["calibration"] = confidence_calibration_loss(
                    module_outputs["executive"]["confidence"],
                    targets["accuracy"],
                    weight=self.weights.get("calibration", 0.005),
                )

        if "trajectory" in module_outputs:
            losses["trajectory"] = trajectory_smoothness_loss(
                module_outputs["trajectory"],
                weight=self.weights.get("trajectory", 0.001),
            )

        if "tools" in module_outputs:
            tools_out = module_outputs["tools"]
            if isinstance(tools_out, dict) and "route_weights" in tools_out:
                tt = (targets or {}).get("tool_type")
                losses["tools"] = tool_selection_loss(
                    tools_out["route_weights"], tt,
                    weight=self.weights.get("tools", 0.05),
                )

        if "entity" in module_outputs:
            ent = module_outputs["entity"]
            ent_logits = ent.get("logits", ent) if isinstance(ent, dict) else ent
            et = (targets or {}).get("entity_ids") if targets else None
            losses["entity"] = entity_prediction_loss(
                ent_logits, et,
                weight=self.weights.get("entity", 0.01),
            )

        if "decoder" in module_outputs:
            dec = module_outputs["decoder"]
            dec_logits = dec.get("logits", dec) if isinstance(dec, dict) else dec
            dt = (targets or {}).get("decoder_ids") if targets else None
            if dt is not None:
                losses["decoder"] = decoder_loss(
                    dec_logits, dt,
                    weight=self.weights.get("decoder", 0.01),
                )

        if "quality_assurance" in module_outputs:
            qa = module_outputs["quality_assurance"]
            if "eval_out" in qa and "novelty" in qa["eval_out"]:
                losses["novelty"] = novelty_bonus_loss(
                    qa["eval_out"]["novelty"],
                    weight=self.weights.get("novelty", 0.001),
                )

        return losses

    def _find_device(self, outputs: Dict[str, Any]) -> torch.device:
        for v in outputs.values():
            if isinstance(v, dict):
                for v2 in v.values():
                    if isinstance(v2, torch.Tensor):
                        return v2.device
            elif isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    def total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(losses.values())).sum() if losses else torch.tensor(0.0)
