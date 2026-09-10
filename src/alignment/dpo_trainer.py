from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _sequence_logps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    if logits.dim() != 3:
        raise ValueError(f"Expected sequence logits [batch, seq, vocab], got {tuple(logits.shape)}")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    per_token_logps = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        safe_labels.reshape(-1),
        reduction='none',
    ).reshape_as(shift_labels)
    return (per_token_logps * valid_mask.float()).sum(dim=-1)


def _sequence_logps_mean(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-token-mean sequence log-probs (length-normalized).

    Length normalization is REQUIRED by SimPO and by ORPO's SFT term:
    with summed logps, longer responses dominate and β/γ operate on an
    unintended scale, collapsing training to a length preference.
    """
    summed = _sequence_logps(model, input_ids, attention_mask, labels)
    shift_labels = labels[:, 1:]
    n_tokens = (shift_labels != -100).float().sum(dim=-1).clamp(min=1.0)
    return summed / n_tokens


def _resolve_device(device):
    if device is not None:
        return device
    if torch.cuda.is_available():
        # current_device() returns an int; wrap it explicitly.
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class DPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel],
        tokenizer: PreTrainedTokenizerBase,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = _resolve_device(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def _get_batch_logps(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return _sequence_logps(model, input_ids, attention_mask, labels)

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        policy_chosen_logps = self._get_batch_logps(self.model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        policy_rejected_logps = self._get_batch_logps(self.model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        if self.ref_model is None:
            # DPO against the policy itself zeroes the KL signal: ref ratios
            # cancel and the loss degenerates to -logsigmoid(β·(pc-pr)) —
            # i.e. margin-less SimPO, not DPO. Fail loudly instead.
            raise ValueError(
                "DPOTrainer requires an explicit frozen ref_model. "
                "Use AlignmentPipeline._get_ref_model() to obtain one.")
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
            ref_rejected_logps = self._get_batch_logps(self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        loss, chosen_reward, rejected_reward = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
            "accuracy": (policy_chosen_logps > policy_rejected_logps).float().mean().item(),
        }


class ORPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        beta: float = 0.05,
        learning_rate: float = 5e-7,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = _resolve_device(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def orpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        log_odds = chosen_logps - rejected_logps
        # Canonical ORPO length-normalizes the SFT term (mean per-token NLL);
        # summed logps scale its gradient linearly with response length and
        # overwhelm the odds-ratio term.
        sft_loss = -chosen_logps.mean()
        orpo_loss = sft_loss + self.beta * (-F.logsigmoid(log_odds)).mean()
        return orpo_loss

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        chosen_logps = self._get_mean_logps(self.model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        rejected_logps = self._get_mean_logps(self.model, rejected_input_ids, rejected_attention_mask, rejected_labels)
        loss = self.orpo_loss(chosen_logps, rejected_logps)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "accuracy": (chosen_logps > rejected_logps).float().mean().item()}

    def _get_logps(self, model, input_ids, attention_mask, labels):
        return _sequence_logps(model, input_ids, attention_mask, labels)

    def _get_mean_logps(self, model, input_ids, attention_mask, labels):
        return _sequence_logps_mean(model, input_ids, attention_mask, labels)


class KTOtrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        beta: float = 0.1,
        learning_rate: float = 3e-7,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        ref_model: Optional[PreTrainedModel] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.device = _resolve_device(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def kto_loss(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> torch.Tensor:
        kl = policy_logps - ref_logps
        # The KL baseline must be DETACHED: gradients flowing through it would
        # couple every example in the batch through the baseline term.
        kl_mean = kl.mean().detach()
        losses = torch.where(
            is_desirable,
            self.desirable_weight * -F.logsigmoid(self.beta * (kl - kl_mean)),
            self.undesirable_weight * -F.logsigmoid(self.beta * (kl_mean - kl)),
        )
        return losses.mean()

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        batch_size = chosen_input_ids.shape[0]
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        labels = torch.cat([chosen_labels, rejected_labels], dim=0)
        is_desirable = torch.cat([
            torch.ones(batch_size, device=input_ids.device, dtype=torch.bool),
            torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool),
        ], dim=0)

        policy_logps = self._get_logps(self.model, input_ids, attention_mask, labels)

        if self.ref_model is None:
            # Without a reference model, ref_logps == policy_logps ⇒ kl ≡ 0
            # ⇒ loss is the constant -logsigmoid(0) = log 2 with zero gradient:
            # training would silently do nothing.
            raise ValueError(
                "KTOtrainer requires an explicit frozen ref_model. "
                "Use AlignmentPipeline._get_ref_model() to obtain one.")
        with torch.no_grad():
            ref_logps = self._get_logps(self.ref_model, input_ids, attention_mask, labels).detach()

        loss = self.kto_loss(policy_logps, ref_logps, is_desirable)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def _get_logps(self, model, input_ids, attention_mask, labels):
        return _sequence_logps(model, input_ids, attention_mask, labels)


class SimPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        gamma: float = 0.5,
        beta: float = 2.0,
        learning_rate: float = 5e-7,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = _resolve_device(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def simpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        # Inputs are per-token-MEAN logps (see _sequence_logps_mean): SimPO's
        # β/γ operate on averaged log-probs, not sums.
        logits = self.beta * (chosen_logps - rejected_logps - self.gamma)
        loss = -F.logsigmoid(logits).mean()
        return loss

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        chosen_logps = self._get_mean_logps(self.model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        rejected_logps = self._get_mean_logps(self.model, rejected_input_ids, rejected_attention_mask, rejected_labels)
        loss = self.simpo_loss(chosen_logps, rejected_logps)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "accuracy": (chosen_logps > rejected_logps).float().mean().item()}

    def _get_logps(self, model, input_ids, attention_mask, labels):
        return _sequence_logps(model, input_ids, attention_mask, labels)

    def _get_mean_logps(self, model, input_ids, attention_mask, labels):
        return _sequence_logps_mean(model, input_ids, attention_mask, labels)
