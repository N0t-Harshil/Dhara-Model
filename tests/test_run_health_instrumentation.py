"""Tests for run-health instrumentation: per-branch aux losses, executive gate
visibility, sampled logits/vocab stats on the model, the first-batch stash
installed by the NaN-loss guard, and the _RunHealthCallback held-out ppl +
generation probes."""
import logging

import pytest
import torch
from types import SimpleNamespace

from src.dhara.model import DharaModel, DharaConfig
from src.training.pipeline import (
    TrainingPipeline,
    _LoggingCallback,
    _RunHealthCallback,
)

log = logging.getLogger("src.training.pipeline")


def _tiny_model(head_ce: str = "topk"):
    mc = DharaConfig(
        vocab_size=256, hidden_size=16, d_state=8, d_hidden=32,
        n_ssm_layers=1, n_hssm_levels=1, max_position_embeddings=64,
        adaptive_top_k_min=4, adaptive_top_k_max=16,
        n_semantic_concepts=16, working_mem_capacity=32, max_episodes=8,
        max_reasoning_steps=4, n_domains=3, max_subgoals=8,
        n_task_types=8, n_difficulty_levels=5, n_reasoning_types=4,
        n_language_groups=4, n_token_categories=4, n_languages=4, n_doc_roles=4,
        max_entities=4, n_relation_types=4, max_events=4,
        qa_max_passes=2, qa_converge_threshold=0.05,
        n_debate_rounds=1, head_ce=head_ce,
        enable_executive=True, enable_world_model=False, enable_tools=False,
        enable_curiosity=False, enable_aux_losses=True,
    )
    return DharaModel(config=mc)


def _inputs(batch=2, seq=8):
    ids = torch.randint(4, 128, (batch, seq))
    labels = ids.clone()
    lang = torch.randint(0, 4, (batch, seq))
    aux = {
        "intent": {
            "task_type": torch.randint(0, 4, (batch,)),
            "difficulty": torch.randint(0, 5, (batch,)),
        }
    }
    return ids, labels, lang, aux


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


def test_model_publishes_diagnostics_after_forward():
    model = _tiny_model()
    ids, labels, lang, aux = _inputs()
    out = model(ids, labels=labels, language_ids=lang, aux_targets=aux)

    assert torch.isfinite(out.loss)
    assert isinstance(model._last_aux_losses, dict) and "intent" in model._last_aux_losses
    assert isinstance(model._last_exec_meta, dict)
    assert {"gates", "confidence", "reward", "depth"} <= set(model._last_exec_meta)
    for v in model._last_exec_meta["gates"].values():
        assert isinstance(v, float) and 0.0 <= v <= 1.0
    stats = model._last_logits_stats
    assert stats is not None
    assert stats["vocab"] == 256
    assert stats["logit_mean_abs"] >= 0.0
    assert stats["clip50_frac"] == 0.0
    assert stats["label_max"] < 256 or stats["label_oov_frac"] > 0.0


def test_logging_callback_prints_instrumentation(caplog):
    model = _tiny_model()
    ids, labels, lang, aux = _inputs()
    model(ids, labels=labels, language_ids=lang, aux_targets=aux)

    cb = _LoggingCallback()
    cb.set_model(model)
    args = SimpleNamespace()
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    with caplog.at_level(logging.INFO, logger="src.training.pipeline"):
        cb.on_log(args, state, SimpleNamespace(), logs={"loss": 5.5, "learning_rate": 1e-4})
    msgs = "\n".join(r.message for r in caplog.records)
    assert "aux={" in msgs and "intent=" in msgs
    assert "gates={decoder=" in msgs
    assert "lstats(v=256" in msgs


def _fake_trainer(model):
    class _FT:
        def __init__(self, model):
            self.model = model

        def compute_loss(self, model, inputs, *args, **kwargs):
            return torch.tensor(5.0)

    return _FT(model)


def test_nan_guard_stashes_first_batch():
    model = _tiny_model()
    fake = _fake_trainer(model)
    TrainingPipeline._install_nan_loss_guard(None, fake)

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
        "language_ids": torch.tensor([[0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    fake.compute_loss(fake.model, inputs)
    stash = model._run_health_stash
    assert set(stash) == {"input_ids", "labels", "language_ids"}
    assert stash["input_ids"].tolist() == [[1, 2, 3]]
    assert stash["input_ids"].device.type == "cpu"

    # A later micro-batch must not overwrite the first-batch stash.
    fake.compute_loss(fake.model, {"input_ids": torch.tensor([[9, 9]])})
    assert model._run_health_stash["input_ids"].tolist() == [[1, 2, 3]]


def test_health_callback_ppl_and_generation_on_tiny_model(caplog):
    model = _tiny_model()
    ids, labels, lang, _aux = _inputs(seq=32)
    cb = _RunHealthCallback(tokenizer=_Tok(), eval_every=500, n_ctx=8, n_window=24, n_gen=6)
    cb.set_model(model)
    model._run_health_stash = {
        "input_ids": ids,
        "language_ids": lang,
        "labels": labels,
    }
    model.train()

    with caplog.at_level(logging.INFO, logger="src.training.pipeline"):
        cb._eval_heldout_ppl(model, 500)
        cb._eval_sample(model, 500)

    msgs = "\n".join(r.message for r in caplog.records)
    assert "held-out ppl=" in msgs
    assert "gen[:200]=" in msgs
    assert "failed" not in msgs
    # Training mode must be restored after the eval probes.
    assert model.training is True


def test_health_callback_skips_without_stash(caplog):
    model = _tiny_model()
    cb = _RunHealthCallback(tokenizer=_Tok(), eval_every=500, n_ctx=8, n_window=24, n_gen=6)
    cb.set_model(model)
    with caplog.at_level(logging.INFO, logger="src.training.pipeline"):
        cb._eval_heldout_ppl(model, 500)
        cb._eval_sample(model, 500)
    msgs = "\n".join(r.message for r in caplog.records)
    assert "no stashed batch" in msgs


def test_health_callback_noop_on_non_eval_steps():
    cb = _RunHealthCallback(tokenizer=_Tok(), eval_every=500, n_ctx=8, n_window=24, n_gen=6)
    cb._model = _tiny_model()
    cb.on_log(SimpleNamespace(), SimpleNamespace(global_step=250, is_world_process_zero=True), SimpleNamespace(), logs={})
    cb.on_log(SimpleNamespace(), SimpleNamespace(global_step=0, is_world_process_zero=True), SimpleNamespace(), logs={})