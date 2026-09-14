"""Remaining hardening items from the numeric-stability sweep:

3. Executive REINFORCE/tracker were dead code in pretraining (forward never
   received task_loss).  Now wired behind an opt-in `executive_rl` flag and
   driven by a per-step `update_from_step` hook.
4. AdaptiveDifficultyRouter / reasoning unroll must stay bounded.
5. CognitiveWorkspace must reset per forward (no cross-batch buffering).
7. LR scheduler must continue across stage transitions instead of cold-
   restarting (which caused step-loss spikes at every stage boundary).
"""
import pytest
import torch

from src.training.pipeline import TrainingPipeline, _install_continuous_scheduler

_NAN_GUARD = TrainingPipeline._install_nan_loss_guard


# ---- Item 3: executive_rl wiring ------------------------------------------

def test_schema_default_executive_rl_is_false():
    from src.config.schema import DharaConfig

    assert DharaConfig().executive_rl is False


def test_factory_wires_executive_rl_flag():
    from src.config.schema import Config
    from src.models.factory import ModelFactory
    from unittest.mock import MagicMock

    cfg = Config()
    arch = cfg.model.architecture
    arch.model_type = "dhara_v3"
    arch.hidden_size = 16
    arch.max_position_embeddings = 64
    v3 = arch.dhara_v3
    v3.d_hidden = 32
    v3.d_state = 8
    v3.n_ssm_layers = 1
    v3.n_hssm_levels = 1
    v3.n_semantic_concepts = 16
    v3.working_mem_capacity = 32
    v3.max_episodes = 8
    v3.max_subgoals = 4
    v3.max_reasoning_steps = 4
    v3.adaptive_top_k_max = 16
    v3.enable_aux_losses = False
    v3.executive_rl = True

    tok = MagicMock()
    tok.__len__ = lambda self: 256
    model = ModelFactory.create_model(cfg, tok)
    assert model.executive_rl is True


def test_update_from_step_appends_reward_and_runs_reinforce():
    from src.dhara.executive import ExecutiveController

    exec_ctl = ExecutiveController(d_model=16, d_hidden=32, n_modules=8)
    loss = torch.tensor(3.0)
    gates = {"memory": torch.tensor(0.5), "decoder": torch.tensor(0.5)}
    for _ in range(10):
        exec_ctl.update_from_step(loss, gates)
    assert len(exec_ctl.reward_buffer) == 10
    module_importance_before = exec_ctl.module_importance.detach().clone()
    for _ in range(40):
        exec_ctl.update_from_step(loss, gates)
    assert len(exec_ctl.reward_buffer) == 0  # _reinforce_update clears it
    assert exec_ctl.module_importance.isfinite().all()
    assert (exec_ctl.module_importance >= 1e-6).all()


def test_update_from_step_ignores_nonfinite_loss():
    from src.dhara.executive import ExecutiveController

    exec_ctl = ExecutiveController(d_model=16, d_hidden=32, n_modules=8)
    exec_ctl.update_from_step(torch.tensor(float("nan")), {})
    assert len(exec_ctl.reward_buffer) == 0


def test_model_trains_with_executive_rl_enabled():
    from src.dhara.model import DharaModel, DharaConfig

    mc = DharaConfig(
        vocab_size=256, hidden_size=16, d_state=8, d_hidden=32,
        n_ssm_layers=1, n_hssm_levels=1, max_position_embeddings=32,
        adaptive_top_k_min=4, adaptive_top_k_max=16,
        n_semantic_concepts=16, working_mem_capacity=32, max_episodes=8,
        enable_aux_losses=False, enable_tools=False,
        enable_world_model=False, enable_curiosity=False,
        enable_executive=True, executive_rl=True,
    )
    model = DharaModel(config=mc).train()
    ids = torch.randint(4, 64, (2, 6))
    labels = ids.clone()
    lang = torch.randint(0, 4, (2, 6))
    model.zero_grad()
    out = model(ids, labels=labels, language_ids=lang)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads)
    assert len(model.executive.reward_buffer) >= 1


def test_model_with_executive_rl_off_does_not_touch_reward_buffer():
    from src.dhara.model import DharaModel, DharaConfig

    mc = DharaConfig(
        vocab_size=256, hidden_size=16, d_state=8, d_hidden=32,
        n_ssm_layers=1, n_hssm_levels=1, max_position_embeddings=32,
        adaptive_top_k_min=4, adaptive_top_k_max=16,
        n_semantic_concepts=16, working_mem_capacity=32, max_episodes=8,
        enable_aux_losses=False, enable_tools=False,
        enable_world_model=False, enable_curiosity=False,
        enable_executive=True, executive_rl=False,
    )
    model = DharaModel(config=mc).train()
    ids = torch.randint(4, 64, (2, 6))
    model(ids, labels=ids.clone())
    assert len(model.executive.reward_buffer) == 0


# ---- Item 4: reasoning/router bouns --------------------------------------

def test_difficulty_router_n_steps_bounded():
    from src.dhara.layer4_intent import AdaptiveDifficultyRouter

    router = AdaptiveDifficultyRouter(n_difficulty_levels=5, min_steps=2, max_steps=32)
    for level in range(5):
        logits = torch.full((4, 5), -1e9)
        logits[:, level] = 0.0
        n_steps = router(logits)
        assert n_steps.min().item() >= 2
        assert n_steps.max().item() <= 32


def test_reasoning_unroll_bounded_and_finite_at_max_steps():
    from src.dhara.layer6_reasoning import AdaptiveContinuousReasoning

    reasoner = AdaptiveContinuousReasoning(d_state=16, d_hidden=32, n_domains=3, max_steps=32)
    h = torch.randn(2, 16)
    z = torch.randn(2, 32)
    n_steps = torch.full((2,), 32, dtype=torch.long)
    z_endpoint, z_traj = reasoner(h, z, n_steps)
    assert z_endpoint.shape == (2, 32)
    assert z_traj.shape == (2, 32, 32)
    assert torch.isfinite(z_endpoint).all() and torch.isfinite(z_traj).all()


# ---- Item 5: workspace per-forward isolation -------------------------------

def test_workspace_resets_between_forwards():
    from src.dhara.workspace import CognitiveWorkspace

    ws = CognitiveWorkspace(d_model=8, d_hidden=16, max_keys=4)
    plan2 = {"goal_embeds": torch.randn(2, 4, 8)}
    plan3 = {"goal_embeds": torch.randn(3, 4, 8)}

    ws.reset(2)
    memory = torch.randn(2, 4, 8)
    reasoning = torch.randn(2, 16)
    first = ws.update(plan2, memory, reasoning)
    assert first["workspace"].shape[0] == 2
    assert ws._cached_state.shape[0] == 2

    ws.reset(3)  # model mid-training re=enters with a fresh batch
    assert ws._cached_state is None
    assert len(ws._store) == 0

    second = ws.update(plan3, torch.randn(3, 4, 8), torch.randn(3, 16))
    assert second["workspace"].shape[0] == 3
    assert ws._cached_state.shape[0] == 3
    assert first["workspace"].shape[0] == 2  # first pass output unchanged


def test_workspace_carries_across_eval_forwards_until_reset():
    from src.dhara.workspace import CognitiveWorkspace

    ws = CognitiveWorkspace(d_model=8, d_hidden=16, max_keys=4)
    plan = {"goal_embeds": torch.randn(1, 4, 8)}
    ws.reset(1)
    # No reset between calls (eval / decode): the gated carry must persist.
    first = ws.update(plan, torch.randn(1, 4, 8), torch.randn(1, 16))
    second = ws.update(plan, torch.randn(1, 4, 8), torch.randn(1, 16))
    assert not torch.allclose(second["workspace"], first["workspace"])
    ws.reset(1)
    third = ws.update(plan, torch.randn(1, 4, 8), torch.randn(1, 16))
    assert ws._cached_state is not None
    assert not torch.allclose(third["workspace"], second["workspace"])


# ---- Item 7: LR continuity across stage transitions -------------------------

class _FakeScheduler:
    def __init__(self, lr=8e-5, epoch=0):
        self.last_epoch = epoch
        self.base_lrs = [lr]

    def get_last_lr(self):
        return list(self.base_lrs)


class _FakeState:
    def __init__(self, step=0):
        self.global_step = step


class _FakeTrainer:
    """Mimics the HF Trainer surface used by the continuity patch: a fresh
    scheduler is built on every create_optimizer_and_scheduler call (exactly
    what happens at a stage transition)."""

    def __init__(self):
        self.lr_scheduler = None
        self.state = _FakeState()
        self._calls = 0

    def create_optimizer_and_scheduler(self, num_training_steps):
        self._calls += 1
        self.lr_scheduler = _FakeScheduler(lr=8e-5, epoch=0)


class _FakeLossTrainer:
    """Mimics the HF Trainer.compute_loss surface including the dict-merge at
    trainer.py:3880 that unmasked a MethodType binding bug: when the guard was
    bound without an explicit `self`, the wrapped trainer swallowed `model` and
    `inputs` received the DataParallel-wrapped model, so `{**inputs, **kwargs}`
    raised `TypeError: 'DataParallel' object is not a mapping`."""

    def __init__(self):
        self.nan = False
        self.calls = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        merged = {**inputs, **kwargs}  # mirrors transformers/trainer.py compute_loss
        self.calls.append((model, merged))
        out = torch.tensor(float("nan") if self.nan else 0.5)
        if return_outputs:
            return (out, {"logits": None})
        return out


def test_scheduler_continuity_seeds_stage_restart():
    trainer = _FakeTrainer()
    _install_continuous_scheduler(trainer)

    trainer.create_optimizer_and_scheduler(50000)
    assert trainer.lr_scheduler.last_epoch == 0  # cold start, no seed yet

    trainer.state.global_step = 2600
    trainer.lr_scheduler = _FakeScheduler(lr=5.2e-5, epoch=2600)
    trainer.create_optimizer_and_scheduler(50000)
    assert trainer.lr_scheduler.last_epoch == 2600
    assert trainer.lr_scheduler.base_lrs == [5.2e-5]


def test_nan_loss_guard_binding_passes_model_and_inputs_not_the_trainer():
    from unittest.mock import MagicMock

    fake = _FakeLossTrainer()
    _NAN_GUARD(fake, fake)
    wrapped = MagicMock()  # stands in for the nn.DataParallel-wrapped model
    inputs = {"input_ids": torch.tensor([1, 2, 3])}

    out = fake.compute_loss(wrapped, inputs)
    assert out.item() == 0.5
    assert fake.calls[-1][0] is wrapped
    assert fake.calls[-1][1]["input_ids"].equal(inputs["input_ids"])


def test_nan_loss_guard_returns_zero_loss_for_nonfinite_batch():
    from unittest.mock import MagicMock

    fake = _FakeLossTrainer()
    fake.nan = True
    _NAN_GUARD(fake, fake)

    zero = fake.compute_loss(MagicMock(), {"input_ids": torch.tensor([1])})
    assert zero.item() == 0.0

    fake.nan = False
    out = fake.compute_loss(MagicMock(), {"input_ids": torch.tensor([1])})
    assert out.item() == 0.5  # skipping resets the consecutive counter
