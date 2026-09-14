"""Head-CE loss tests: single-eval dense CE regression vs. the reference
hierarchical decoder path, the opt-in top-k candidate CE, and the config wiring
from schema -> model factory -> model."""
import pytest
import torch
from unittest.mock import MagicMock


def _tiny_model(head_ce: str = "dense"):
    from src.dhara.model import DharaModel, DharaConfig

    mc = DharaConfig(
        vocab_size=256, hidden_size=16, d_state=8, d_hidden=32,
        n_ssm_layers=1, n_hssm_levels=1, max_position_embeddings=32,
        adaptive_top_k_min=4, adaptive_top_k_max=16,
        n_semantic_concepts=16, working_mem_capacity=32, max_episodes=8,
        enable_aux_losses=False, enable_tools=False,
        enable_world_model=False, enable_curiosity=False,
        head_ce=head_ce,
    )
    return DharaModel(config=mc)


class TestHeadCe:
    def _inputs(self, batch=2, seq=6, ignore_frac=0.3):
        ids = torch.randint(4, 64, (batch, seq))
        labels = ids.clone()
        mask = torch.rand(batch, seq) < ignore_frac
        labels[torch.nonzero(mask, as_tuple=True)] = -100
        lang = torch.randint(0, 4, (batch, seq))
        return ids, labels, lang

    def test_dense_loss_matches_reference(self):
        model = _tiny_model("dense")
        ids, labels, lang = self._inputs()
        with torch.no_grad():
            captured = {}
            orig = model.decoder.forward

            def spy(h, target_ids=None, language_ids=None):
                captured["h"] = h
                captured["language_ids"] = language_ids
                return orig(h, target_ids=target_ids, language_ids=language_ids)

            model.decoder.forward = spy
            out = model(ids, labels=labels, language_ids=lang)
            model.decoder.forward = orig

            batch, seq = ids.shape
            h = captured["h"].view(batch, seq, -1)
            shift_h = h[:, :-1, :].reshape(-1, 32)
            shift_labels = labels[:, 1:].reshape(-1)
            # No future leak: slot t's decoder got language[t] (not
            # language[t+1]) to predict labels[t+1], so the reference must pair
            # h[:,t] with lang[:,t].
            shift_lang = lang[:, :-1].reshape(-1) if captured["language_ids"] is not None else None
            valid = shift_labels != -100
            safe = shift_labels.masked_fill(~valid, 0)
            lp = model.decoder.hierarchical_log_prob(shift_h, safe, language_ids=shift_lang)
            ref_loss = -lp[valid].mean()

        assert torch.allclose(out.loss.detach().cpu(), ref_loss, atol=1e-5)
        assert torch.isfinite(out.loss)

    def test_topk_loss_runs_finite_and_bounded(self):
        model = _tiny_model("dense")
        ids, labels, lang = self._inputs(ignore_frac=0.0)
        with torch.no_grad():
            dense = model(ids, labels=labels).loss
            model.head_ce = "topk"
            topk = model(ids, labels=labels).loss

        assert torch.isfinite(topk)
        # Candidates = adaptive top-k rows + target (a subset of the vocab), so
        # the candidate logsumexp <= full logsumexp => top-k CE <= dense CE.
        assert topk <= dense + 1e-4
        assert topk < 0.999 * dense + 1e-6

    def test_backward_flows_in_both_modes(self):
        model = _tiny_model("dense")
        ids, labels, _ = self._inputs(ignore_frac=0.0)
        for mode in ("dense", "topk"):
            model.head_ce = mode
            model.zero_grad()
            out = model(ids, labels=labels)
            out.loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(g.isfinite().all() for g in grads)
            assert any(g.abs().sum() > 0 for g in grads)


class TestHeadCeWiring:
    def test_schema_default_is_dense(self):
        from src.config.schema import DharaConfig as SchemaDharaConfig

        assert SchemaDharaConfig().head_ce == "dense"

    def test_factory_wires_head_ce_to_model(self):
        from src.config.schema import Config
        from src.models.factory import ModelFactory

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
        v3.head_ce = "topk"

        tok = MagicMock()
        tok.__len__ = lambda self: 256
        model = ModelFactory.create_model(cfg, tok)

        assert model.head_ce == "topk"