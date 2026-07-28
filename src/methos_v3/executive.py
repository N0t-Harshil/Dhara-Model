from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulePerformanceTracker(nn.Module):
    def __init__(self, n_modules: int = 8, d_hidden: int = 1024):
        super().__init__()
        self.n_modules = n_modules
        self.register_buffer("performance_history", torch.zeros(n_modules, 100))
        self.register_buffer("module_lr", torch.ones(n_modules))
        self.register_buffer("update_step", torch.zeros(n_modules))
        self.performance_predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, n_modules),
        )
        self.lr_adjuster = nn.Sequential(
            nn.Linear(n_modules * 2, n_modules), nn.Sigmoid(),
        )

    def update(self, module_idx: int, performance: float, device: torch.device) -> None:
        step = int(self.update_step[module_idx].item())
        idx = step % 100
        self.performance_history[module_idx, idx] = performance
        self.update_step[module_idx] += 1
        if step > 0 and step % 10 == 0:
            recent = self.performance_history[module_idx, :min(step, 100)].mean().item()
            if recent < 0.5:
                val = (self.module_lr[module_idx] * 1.1).clamp(0.1, 3.0)
                self.module_lr.data[module_idx] = val
            elif recent > 0.8:
                val = (self.module_lr[module_idx] * 0.95).clamp(0.1, 3.0)
                self.module_lr.data[module_idx] = val

    def forward(self, h: torch.Tensor, current_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        perf_pred = self.performance_predictor(h)
        if current_loss is not None:
            lr_input = torch.cat([
                F.normalize(perf_pred, dim=-1),
                current_loss.detach().unsqueeze(-1).expand(-1, self.n_modules),
            ], dim=-1)
            lr_adjust = self.lr_adjuster(lr_input)
            val = (self.module_lr * lr_adjust.mean(dim=0)).clamp(0.1, 3.0)
            self.module_lr.data = val
        return {
            "predicted_performance": perf_pred,
            "module_lrs": self.module_lr,
            "performance_history": self.performance_history,
        }


MODULE_NAMES = [
    "memory", "planner", "intent", "reasoning",
    "sandbox", "reflection", "verification", "decoder",
]


class ExecutiveController(nn.Module):
    """
    Executive Controller with resource-budget gating and REINFORCE policy gradient.

    The controller learns to selectively activate/skip computational modules
    using a soft gate mechanism trained via REINFORCE with a composite reward:

        R = accuracy − λ_latency · latency − λ_memory · memory − λ_energy · energy

    Full REINFORCE update (accuracy / latency / memory / energy inputs):
        To enable the full reward formula, pass explicit metrics to `compute_reward()`.
        The minimal working version uses `−task_loss` as an accuracy proxy and
        a compute-penalty on active gate count as a latency proxy.

    Parameters
    ----------
    d_model : int
        Dimension of the input embedding.
    d_hidden : int
        Internal hidden dimension for state encoding.
    n_modules : int
        Number of gated computation modules.
    gate_threshold : float
        Gates below this value cause their module to be skipped at inference.
    rl_lr : float
        Learning rate for the REINFORCE update on `module_importance`.
    lambda_latency : float
        Penalty coefficient λ₁ for latency in the reward.
    lambda_memory : float
        Penalty coefficient λ₂ for memory in the reward.
    lambda_energy : float
        Penalty coefficient λ₃ for energy in the reward.
    """

    def __init__(self, d_model: int, d_hidden: int, n_modules: int = 8,
                 gate_threshold: float = 0.3,
                 rl_lr: float = 1e-4,
                 lambda_latency: float = 0.01,
                 lambda_memory: float = 0.001,
                 lambda_energy: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_modules = n_modules
        self.gate_threshold = gate_threshold
        self.rl_lr = rl_lr
        self.lambda_latency = lambda_latency
        self.lambda_memory = lambda_memory
        self.lambda_energy = lambda_energy
        self.register_buffer("module_state", torch.ones(n_modules))

        self.state_encoder = nn.Linear(d_model * 2, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

        for name in MODULE_NAMES:
            setattr(self, f"{name}_gate", nn.Linear(d_hidden, 1))

        self.budget_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, n_modules),
        )
        self.depth_predictor = nn.Linear(d_hidden, 1)
        self.confidence_estimator = nn.Linear(d_hidden, 1)
        self.value_head = nn.Linear(d_hidden, 1)

        self.tracker = ModulePerformanceTracker(n_modules, d_hidden)
        self.meta_learner = nn.Sequential(
            nn.Linear(d_hidden + n_modules, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, n_modules), nn.Sigmoid(),
        )
        self.module_importance = nn.Parameter(torch.ones(n_modules) / n_modules)
        self.reward_buffer: List[float] = []
        self._log_prob_buffer: List[torch.Tensor] = []  # for REINFORCE

    def compute_reward(
        self,
        task_loss: torch.Tensor,
        gates: Dict[str, torch.Tensor],
        latency: float = 0.0,
        memory_used: float = 0.0,
        energy_used: float = 0.0,
        accuracy: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute composite reward for the REINFORCE update.

        Minimal (always available):
            R = −task_loss − λ_latency · n_active_gates

        Full formula (when accuracy, latency, memory, energy are provided):
            R = accuracy − λ₁ · latency − λ₂ · memory − λ₃ · energy

        Parameters
        ----------
        task_loss : torch.Tensor
            Scalar LM loss for this step.
        gates : dict
            Gate values from forward(), used to compute compute penalty.
        latency : float
            Measured latency in seconds (optional; 0.0 = use compute proxy).
        memory_used : float
            Peak GPU memory delta in GB (optional).
        energy_used : float
            Energy consumption in joules (optional; rarely measurable).
        accuracy : float, optional
            Task accuracy in [0, 1]. If provided, uses full reward formula.
        """
        n_active = torch.stack(list(gates.values())).mean()

        if accuracy is not None:
            # Full reward: R = accuracy − λ₁·latency − λ₂·memory − λ₃·energy
            r = (
                accuracy
                - self.lambda_latency * latency
                - self.lambda_memory * memory_used
                - self.lambda_energy * energy_used
            )
            return torch.tensor(r, device=task_loss.device, dtype=task_loss.dtype)
        else:
            # Minimal proxy: R = −task_loss − λ_latency · n_active_compute_units
            compute_penalty = self.lambda_latency * n_active
            return -task_loss.detach() - compute_penalty

    def _reinforce_update(self) -> None:
        """
        Apply REINFORCE policy gradient update to `module_importance`.

        Uses the reward buffer accumulated over recent steps. The policy gradient
        estimator is:
            ∇J(θ) ≈ E[R · ∇log π(a|s)]
        where π is the softmax over module_importance (the selection policy),
        and R is the cumulative reward.

        Updates `module_importance` in-place via vanilla policy gradient.
        Minimum 10 rewards required before first update.
        """
        if len(self.reward_buffer) < 10:
            return

        # Mean-baseline REINFORCE (reduces variance)
        rewards = torch.tensor(self.reward_buffer, dtype=torch.float32)
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Policy: softmax over module_importance → selection probabilities
        log_probs = F.log_softmax(self.module_importance, dim=0)

        # Policy gradient: maximize E[R · log π] by ascending the gradient
        # Gradient w.r.t. module_importance: advantage-weighted log-probs
        pg_loss = -(advantages.mean() * log_probs.sum())

        # Manual parameter update (no optimizer attached to this sub-network)
        if self.module_importance.grad is not None:
            self.module_importance.grad.zero_()
        pg_loss.backward()
        with torch.no_grad():
            if self.module_importance.grad is not None:
                self.module_importance.data.sub_(
                    self.rl_lr * self.module_importance.grad
                )
                self.module_importance.data.clamp_(1e-6, 1.0)

        self.reward_buffer.clear()

    def forward(self, input_embeds: torch.Tensor, intent: Optional[Dict[str, torch.Tensor]] = None,
                task_loss: Optional[torch.Tensor] = None, latency: float = 0.0,
                memory_used: float = 0.0, energy_used: float = 0.0,
                accuracy: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if input_embeds.dim() == 3:
            pooled = input_embeds.mean(dim=1)
        elif input_embeds.dim() == 2:
            pooled = input_embeds.mean(dim=-1, keepdim=True).expand(-1, self.d_model)
        else:
            pooled = input_embeds
        batch = pooled.shape[0]
        device = pooled.device

        if intent is not None:
            intent_features = torch.stack([
                intent.get("confidence", torch.zeros(batch, device=device)),
                F.softmax(intent.get("task_type", torch.zeros(batch, 8, device=device)), dim=-1).max(dim=-1).values,
                F.softmax(intent.get("difficulty", torch.zeros(batch, 5, device=device)), dim=-1).max(dim=-1).values,
            ], dim=-1)
            intent_padded = torch.zeros(batch, self.d_model, device=device)
            intent_padded[:, :intent_features.shape[-1]] = intent_features
            state = self.state_encoder(torch.cat([pooled, intent_padded], dim=-1))
        else:
            intent_padded = torch.zeros(batch, self.d_model, device=device)
            state = self.state_encoder(torch.cat([pooled, intent_padded], dim=-1))

        h = self.norm(state)
        gates: Dict[str, torch.Tensor] = {}
        for name in MODULE_NAMES:
            gate_net = getattr(self, f"{name}_gate")
            gates[name] = torch.sigmoid(gate_net(h))

        budget_logits = self.budget_head(h)
        budget_weights = F.softmax(budget_logits, dim=-1)
        total_budget = budget_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        budget_weights = budget_weights / total_budget

        depth = torch.sigmoid(self.depth_predictor(h)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_estimator(h)).squeeze(-1)
        value = self.value_head(h).squeeze(-1)

        meta_input = torch.cat([h, self.module_importance.unsqueeze(0).expand(batch, -1)], dim=-1)
        meta_weights = self.meta_learner(meta_input)

        tracker_out = self.tracker(h, task_loss)

        reward = None
        if task_loss is not None:
            reward = self.compute_reward(
                task_loss, gates, latency=latency,
                memory_used=memory_used, energy_used=energy_used,
                accuracy=accuracy,
            )
            self.reward_buffer.append(reward.mean().item())
            if len(self.reward_buffer) > 100:
                self.reward_buffer.pop(0)

            # Apply REINFORCE update every 50 steps
            if len(self.reward_buffer) >= 50:
                self._reinforce_update()

        return {
            "gates": gates,
            "budget_weights": budget_weights,
            "depth_multiplier": depth,
            "confidence": confidence,
            "value": value,
            "meta_weights": meta_weights,
            "predicted_performance": tracker_out["predicted_performance"],
            "module_lrs": tracker_out["module_lrs"],
            "module_importance": self.module_importance,
            "reward": reward,
        }

    def apply_gates(self, gates: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        skip: Dict[str, bool] = {}
        for name in MODULE_NAMES:
            gate_val = gates.get(name, torch.ones(1)).mean().item()
            skip[name] = gate_val < self.gate_threshold
        return skip

    def should_skip_module(self, gate_value: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        t = threshold if threshold is not None else self.gate_threshold
        return gate_value < t
