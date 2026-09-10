from __future__ import annotations

import pytest
import torch
from torch import nn

from src.alignment.dpo_trainer import DPOTrainer, KTOtrainer, ORPOTrainer, SimPOTrainer
from src.alignment.pipeline import AlignmentPipeline
from src.config.schema import Config


def _dummy_logps(batch_size: int = 4) -> tuple:
    return (
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
    )


class TestDPO:
    def test_dpo_loss_shape(self):
        chosen_lps = torch.tensor([-2.0, -1.0, -3.0, -0.5])
        rejected_lps = torch.tensor([-4.0, -3.0, -5.0, -2.0])
        ref_chosen = torch.tensor([-2.5, -1.5, -3.5, -1.0])
        ref_rejected = torch.tensor([-3.5, -2.5, -4.5, -1.5])

        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.beta = 0.1

        loss, chosen_reward, rejected_reward = trainer.dpo_loss(
            chosen_lps, rejected_lps, ref_chosen, ref_rejected,
        )
        assert loss.ndim == 0
        assert loss > 0
        assert chosen_reward > rejected_reward

    def test_dpo_prefers_chosen(self):
        chosen_lps = torch.tensor([-1.0, -0.5])
        rejected_lps = torch.tensor([-5.0, -4.0])
        ref_chosen = torch.tensor([-1.5, -1.0])
        ref_rejected = torch.tensor([-4.5, -3.5])

        trainer = DPOTrainer.__new__(DPOTrainer)
        trainer.beta = 0.1
        loss, _, _ = trainer.dpo_loss(chosen_lps, rejected_lps, ref_chosen, ref_rejected)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestORPO:
    def test_orpo_loss_lower_for_preferred(self):
        trainer = ORPOTrainer.__new__(ORPOTrainer)
        trainer.beta = 0.05
        preferred = torch.tensor([-1.0])
        dispreferred = torch.tensor([-5.0])
        loss_good = trainer.orpo_loss(preferred, dispreferred)

        preferred_bad = torch.tensor([-5.0])
        dispreferred_bad = torch.tensor([-1.0])
        loss_bad = trainer.orpo_loss(preferred_bad, dispreferred_bad)
        assert loss_good < loss_bad

    def test_orpo_loss_finite(self):
        trainer = ORPOTrainer.__new__(ORPOTrainer)
        trainer.beta = 0.05
        loss = trainer.orpo_loss(torch.tensor([-2.0, -1.0]), torch.tensor([-4.0, -3.0]))
        assert torch.isfinite(loss).all()


class TestSimPO:
    def test_simpo_loss_lower_for_preferred(self):
        trainer = SimPOTrainer.__new__(SimPOTrainer)
        trainer.gamma = 0.5
        trainer.beta = 2.0
        preferred = torch.tensor([-1.0])
        dispreferred = torch.tensor([-5.0])
        loss_good = trainer.simpo_loss(preferred, dispreferred)

        preferred_bad = torch.tensor([-5.0])
        dispreferred_bad = torch.tensor([-1.0])
        loss_bad = trainer.simpo_loss(preferred_bad, dispreferred_bad)
        assert loss_good < loss_bad

    def test_simpo_loss_finite(self):
        trainer = SimPOTrainer.__new__(SimPOTrainer)
        trainer.gamma = 0.5
        trainer.beta = 2.0
        loss = trainer.simpo_loss(torch.tensor([-2.0]), torch.tensor([-4.0]))
        assert torch.isfinite(loss)


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.proj = nn.Linear(16, 64)

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self.proj(self.embed(input_ids))})


@pytest.fixture
def tiny_alignment_pipeline():
    cfg = Config()
    cfg.training.max_seq_length = 32
    model = _TinyLM()
    tokenizer = type("Tok", (), {})()
    from src.alignment.constitutional import ConstitutionalTrainer
    return AlignmentPipeline.__new__(AlignmentPipeline), model, tokenizer, cfg


class TestRefModelFallback:
    def test_get_ref_model_returns_frozen_distinct_copy(self, tiny_alignment_pipeline):
        pipe, model, tokenizer, cfg = tiny_alignment_pipeline
        pipe.model = model
        pipe.tokenizer = tokenizer
        pipe.cfg = cfg
        pipe.ref_model = None
        pipe._frozen_ref = None
        pipe.device = torch.device("cpu")

        ref = pipe._get_ref_model()
        assert ref is not model, "Reference model must be a separate copy, not the policy"
        assert all(not p.requires_grad for p in ref.parameters()), "Reference must be frozen"
        assert pipe._get_ref_model() is ref, "Frozen reference should be cached/reused"
        same_state = all(
            (a == b).all()
            for a, b in zip(model.state_dict().values(), ref.state_dict().values())
        )
        assert same_state, "Frozen reference should start as a copy of the policy"

    def test_kto_trainer_receives_ref_model(self, tiny_alignment_pipeline):
        pipe, model, tokenizer, cfg = tiny_alignment_pipeline
        pipe.model = model
        pipe.tokenizer = tokenizer
        pipe.cfg = cfg
        pipe.ref_model = None
        pipe._frozen_ref = None
        pipe.device = torch.device("cpu")

        from torch.utils.data import Dataset

        class FakeDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {
                    "chosen_input_ids": torch.zeros(8, dtype=torch.long),
                    "chosen_attention_mask": torch.ones(8, dtype=torch.long),
                    "chosen_labels": torch.zeros(8, dtype=torch.long),
                    "rejected_input_ids": torch.zeros(8, dtype=torch.long),
                    "rejected_attention_mask": torch.ones(8, dtype=torch.long),
                    "rejected_labels": torch.zeros(8, dtype=torch.long),
                }

        results = pipe.run_kto(FakeDataset(), max_steps=1)
        assert "loss" in results
        assert torch.isfinite(torch.tensor(results["loss"]))
