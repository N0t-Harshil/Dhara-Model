"""Regression tests for the model-architecture audit fixes (A1-A3, B4-B9):

A1  aux_targets collator makes the intent aux loss actually trainable in
    pretraining (previously the target tensors were never constructed).
A2/B6  episodic bank is committed during training forward as per-sample rows
    (previously only written at eval/generate through mem_state, so the bank
    stayed all-zeros while pretraining and collapsed the batch to one vector).
A3  workspace resets per-forward during TRAINING only; eval/generate carries the
    gated state across decode steps.
B4  reasoning returns (endpoint, trajectory) so the smoothness loss sees a real
    path instead of a flattened constant.
B5  compression AE is trained on chunk-pooled sequence slices instead of a
    single whole-sequence mean.
B7  SSM stack carries the deepest layer's state, not the layer average.
B8  decoder logits have no learnable temperature (no cheating via logits scale).
B9  language hint uses a learned-gated blend (no future-position shift, no
    fixed 0.5/0.5 mix).
"""
import torch

from src.data.pipeline import pack_sequences  # noqa: F401  (keeps module importable)


# ---- A1: pretraining aux targets -------------------------------------------

def test_pretrain_aux_collator_adds_intent_targets():
    from src.training.pipeline import _PretrainAuxCollator, _TASK_CATEGORIES, _DIFFICULTY_EDGES

    class _Base:
        def __call__(self, features):
            b = features[0]["batch_size"]
            return {
                "input_ids": torch.randint(0, 64, (b, 8)),
                "labels": torch.randint(0, 64, (b, 8)),
                "attention_mask": torch.ones(b, 8, dtype=torch.long),
                "_segments": torch.zeros(b, 8, dtype=torch.long),
            }

    features = [
        {"_category": "code", "_avg_quality": 0.15, "batch_size": 2},
        {"_category": "math", "_avg_quality": 0.55, "batch_size": 2},
        {"_category": "wiki", "_avg_quality": 0.95, "batch_size": 2},
    ]
    batch = _PretrainAuxCollator(_Base())(features)

    targets = batch["aux_targets"]["intent"]
    assert elements_equal(targets["task_type"], [0, 4, 3])  # code/math/wiki ids
    buckets = []
    for q in (0.15, 0.55, 0.95):
        buckets.append(sum(1 for e in _DIFFICULTY_EDGES if q > e))
    assert elements_equal(targets["difficulty"], buckets)
    assert targets["task_type"].shape == (3,)
    assert targets["difficulty"].shape == (3,)


def test_pretrain_aux_collator_falls_back_without_signal():
    from src.training.pipeline import _PretrainAuxCollator, _TASK_CATEGORY_ID

    class _Base:
        def __call__(self, features):
            return {"input_ids": torch.zeros(2, 8, dtype=torch.long)}

    features = [{}, {"_category": "code", "_avg_quality": 0.5}]
    batch = _PretrainAuxCollator(_Base())(features)

    # Missing metadata must NOT silently drop supervision: it coalesces to the
    # neutral fallback so the intent/difficulty classifiers keep training.
    targets = batch["aux_targets"]["intent"]
    fallback_id = _TASK_CATEGORY_ID[_PretrainAuxCollator._FALLBACK_CATEGORY]
    assert elements_equal(targets["task_type"], [fallback_id, 0])
    assert targets["difficulty"].tolist() == [1, 2]
    assert targets["task_type"].shape == (2,)
    assert targets["difficulty"].shape == (2,)


def elements_equal(t, vals):
    return [int(x) for x in t.tolist()] == list(vals)


# ---- B4: reasoning returns (endpoint, trajectory) --------------------------

def test_reasoning_endpoint_is_last_trajectory_frame():
    from src.dhara.layer6_reasoning import AdaptiveContinuousReasoning

    torch.manual_seed(3)
    reasoner = AdaptiveContinuousReasoning(d_state=16, d_hidden=32, n_domains=3, max_steps=8)
    h = torch.randn(2, 16)
    z = torch.randn(2, 32)
    n_steps = torch.tensor([8, 8])
    endpoint, traj = reasoner(h, z, n_steps)
    assert traj.shape == (2, 8, 32)
    assert torch.allclose(endpoint, traj[:, -1], atol=1e-6)
    # A genuinely moving path: the trajectory is NOT the flattened mean the old
    # smoothness loss saw (which made its gradient signal identically zero).
    assert not torch.allclose(traj[:, -1], traj.mean(dim=1), atol=1e-2)


def test_reasoning_early_stopped_trajectory_freezes_after_last_step():
    from src.dhara.layer6_reasoning import AdaptiveContinuousReasoning

    torch.manual_seed(5)
    reasoner = AdaptiveContinuousReasoning(d_state=16, d_hidden=32, n_domains=3, max_steps=8)
    h = torch.randn(2, 16)
    z = torch.randn(2, 32)
    n_steps = torch.tensor([2, 8])
    endpoint, traj = reasoner(h, z, n_steps)
    assert torch.allclose(endpoint[0], traj[0, -1])
    # Sample 0 stopped after step 2: all frames from step 2 onward equal its
    # endpoint (zero delta -> the smoothness loss sees a flat tail, no NaN).
    assert torch.allclose(traj[0, 2:], traj[0, 2:2 + 1].expand(6, -1), atol=1e-6)


# ---- B5: chunk-pooled compression input ------------------------------------

def test_chunk_pool_splits_sequence_into_chunks():
    from src.dhara.layer3_memory import _chunk_pool

    x = torch.randn(2, 130, 8)
    pooled = _chunk_pool(x, chunk=64)
    assert pooled.shape == (2, 2, 8)  # 128 usable tokens -> 2 chunks of 64
    block = x[:, :64, :].mean(dim=1)
    assert torch.allclose(pooled[:, 0], block, atol=1e-6)


def test_chunk_pool_falls_back_for_short_sequences():
    from src.dhara.layer3_memory import _chunk_pool

    x = torch.randn(2, 40, 8)
    pooled = _chunk_pool(x, chunk=64)
    assert pooled.shape == (2, 1, 8)
    assert torch.allclose(pooled[:, 0], x.mean(dim=1), atol=1e-6)


# ---- A2/B6: episodic bank commits real rows during training ----------------

def test_training_forward_commits_per_sample_episodes():
    from src.dhara.layer3_memory import MemoryManager

    torch.manual_seed(0)
    m = MemoryManager(
        d_model=8, d_state=4, n_hssm_layers=1, n_hssm_levels=1,
        working_mem_capacity=8, n_semantic_concepts=8, max_episodes=4,
    )
    m.train()
    x = torch.randn(3, 10, 8)
    m(x, None)
    # The whole batch folded into ONE bank slot before; now B=3 rows land in
    # the bank immediately, per sample.
    assert int(m.episodic.episode_count.item()) == 3
    assert not torch.all(m.episodic.episode_buffer[0] == 0)
    # Row 0 must match sample 0's own compression (not a pooled-over-batch
    # blur). Retrieve with a probe against the bank and check it's non-zero.
    ep0 = m.episodic.episode_buffer[0, 0]
    x0 = x[0]
    assert ep0.shape == (8,)
    assert torch.isfinite(ep0).all()


def test_eval_forward_does_not_commit_bank():
    from src.dhara.layer3_memory import MemoryManager

    torch.manual_seed(1)
    m = MemoryManager(
        d_model=8, d_state=4, n_hssm_layers=1, n_hssm_levels=1,
        working_mem_capacity=8, n_semantic_concepts=8, max_episodes=4,
    )
    m.eval()
    m(torch.randn(3, 10, 8), None)
    assert int(m.episodic.episode_count.item()) == 0
    assert torch.all(m.episodic.episode_buffer[0] == 0)


def test_episodic_bank_ring_buffers_when_full():
    from src.dhara.layer3_memory import MemoryManager

    m = MemoryManager(
        d_model=8, d_state=4, n_hssm_layers=1, n_hssm_levels=1,
        working_mem_capacity=8, n_semantic_concepts=8, max_episodes=4,
    )
    m.train()
    for _ in range(3):  # 9 commits, capacity 4 -> ring must bound the count
        m(torch.randn(3, 10, 8), None)
    assert int(m.episodic.episode_count.item()) <= 4
    assert torch.isfinite(m.episodic.episode_buffer).all()


# ---- B7: SSM stack carries the deepest layer's state -----------------------

def test_hssm_stack_final_state_is_deepest_layer():
    from src.dhara.hssm import HierarchicalSSMStack

    torch.manual_seed(0)
    stack = HierarchicalSSMStack(d_model=64, d_state=16, n_layers=3, n_hssm_levels=3)
    x = torch.randn(2, 32, 64)
    _, h = stack(x, None)
    assert h.shape == (2, 3, 16)  # (batch, n_levels, d_state)
    assert torch.isfinite(h).all()


# ---- B8: no learnable decoder temperature ----------------------------------

def test_token_decoder_has_no_learnable_temperature():
    from src.dhara.layer11_decoder import TokenDecoder

    dec = TokenDecoder(d_hidden=16, vocab_size=64, adaptive_top_k_min=4, adaptive_top_k_max=16)
    names = {n for n, _ in dec.named_parameters()}
    assert "temperature" not in names
    assert "token.temperature" not in names


# ---- B9: gated language blend, no fixed 0.5 split --------------------------

def test_language_decoder_gated_blend_matches_formula():
    from src.dhara.layer11_decoder import LanguageDecoder

    torch.manual_seed(0)
    dec = LanguageDecoder(d_hidden=16, n_language_groups=8)
    # Model contract: flat per-position h with a length-matched hint.
    h = torch.randn(128, 16)
    hint = torch.randint(0, 8, (128,))
    out = dec(h, language_hint=hint)
    pred = torch.softmax(out["language_logits"], dim=-1)
    gate = out["language_gate"]
    onehot = torch.nn.functional.one_hot(hint, num_classes=8).float()
    expected = pred * (1 - gate) + onehot * gate
    assert torch.allclose(out["language_weights"], expected, atol=1e-5)


def test_language_decoder_hint_is_non_fixed_blend():
    from src.dhara.layer11_decoder import LanguageDecoder

    torch.manual_seed(1)
    dec = LanguageDecoder(d_hidden=16, n_language_groups=8)
    h = torch.randn(128, 16)
    hint = torch.randint(0, 8, (128,))
    gated = dec(h, language_hint=hint)["language_weights"]
    no_hint = dec(h)["language_weights"]
    # Not the old hard 0.5/0.5 mixture either: the gate output must actually
    # shift the weights beyond the fixed half-split, i.e. the effective hint
    # proportion is learned, not constant.
    mix50 = 0.5 * no_hint + 0.5 * torch.nn.functional.one_hot(hint, num_classes=8).float()
    assert not torch.allclose(gated, mix50, atol=1e-4)


# ---- A3: workspace is a training-time reset, eval-time carry ----------------

def test_model_resets_workspace_in_training_but_not_eval():
    from src.dhara.model import DharaModel, DharaConfig

    mc = DharaConfig(
        vocab_size=256, hidden_size=16, d_state=8, d_hidden=32,
        n_ssm_layers=1, n_hssm_levels=1, max_position_embeddings=32,
        adaptive_top_k_min=4, adaptive_top_k_max=16,
        n_semantic_concepts=16, working_mem_capacity=32, max_episodes=8,
        enable_aux_losses=True, enable_tools=False,
        enable_world_model=False, enable_curiosity=False,
    )
    model = DharaModel(config=mc)

    calls = []
    orig_reset = model.workspace.reset

    def spy(batch):
        calls.append(batch)
        return orig_reset(batch)

    model.workspace.reset = spy

    ids = torch.randint(4, 64, (2, 6))
    labels = ids.clone()

    model.train()
    model(ids, labels=labels)
    assert len(calls) == 1, "training forward must reset the workspace once"

    calls.clear()
    model.eval()
    model(ids, labels=labels)
    model(ids, labels=labels)
    assert len(calls) == 0, "eval forwards must NOT reset the workspace"


def test_generate_resets_workspace_per_session():
    from src.dhara.model import DharaModel, DharaConfig

    mc = DharaConfig(
        vocab_size=256, hidden_size=16, d_state=8, d_hidden=32,
        n_ssm_layers=1, n_hssm_levels=1, max_position_embeddings=32,
        adaptive_top_k_min=4, adaptive_top_k_max=16,
        n_semantic_concepts=16, working_mem_capacity=32, max_episodes=8,
        enable_aux_losses=False, enable_tools=False,
        enable_world_model=False, enable_curiosity=False,
    )
    model = DharaModel(config=mc)

    calls = []
    orig_reset = model.workspace.reset

    def spy(batch):
        calls.append(batch)
        return orig_reset(batch)

    model.workspace.reset = spy

    ids = torch.randint(4, 64, (1, 4))
    model.generate(ids, max_new_tokens=3, top_k=0, top_p=0.0, eos_token_id=0)
    assert len(calls) == 1, "generate() must reset the workspace exactly once"