"""MemoryManager.apply_updates hardening: non-finite pending writes must be
rejected so the episodic bank / mem_age / mem_priority / checkpoint never get
poisoned by a NaN entry from a numerically unstable hidden state (this path
runs during generation/eval, where mem_state is supplied). The contract: each
pending fragment ("decayed" -> episode_buffer, "priority" -> mem_priority,
"compressed" -> next episode slot) is applied independently, and mem_age only
advances when at least one fragment was accepted."""
import torch

from src.dhara.layer3_memory import MemoryManager


def _manager():
    return MemoryManager(
        d_model=8, d_state=4, n_hssm_layers=1, n_hssm_levels=1,
        working_mem_capacity=8, n_semantic_concepts=8, max_episodes=4,
    )


def _pending():
    return {
        "pending": {
            "compressed": torch.randn(1, 8),
            "decayed": torch.randn(1, 4, 8),
            "priority": torch.randn(1, 4),
        }
    }


def test_all_nan_pending_is_rejected():
    m = _manager()
    p = _pending()
    for key in p["pending"]:
        p["pending"][key] = torch.full_like(p["pending"][key], float("nan"))
    m.apply_updates(p)
    assert torch.all(m.episodic.episode_buffer == 0)
    assert int(m.mem_age.sum().item()) == 0
    assert torch.all(m.mem_priority == 1)


def test_all_inf_pending_is_rejected():
    m = _manager()
    p = _pending()
    for key in p["pending"]:
        p["pending"][key] = torch.full_like(p["pending"][key], float("inf"))
    m.apply_updates(p)
    assert torch.all(m.episodic.episode_buffer == 0)
    assert int(m.mem_age.sum().item()) == 0


def test_finite_pending_is_applied():
    m = _manager()
    p = _pending()
    m.apply_updates(p)
    assert int(m.mem_age.sum().item()) == m.mem_age.numel()
    assert torch.allclose(m.episodic.episode_buffer, p["pending"]["decayed"])
    assert torch.allclose(m.mem_priority, p["pending"]["priority"])


def test_nan_fragment_skipped_but_finite_fragments_applied():
    m = _manager()
    p = _pending()
    p["pending"]["decayed"] = torch.full_like(p["pending"]["decayed"], float("nan"))
    m.apply_updates(p)
    assert int(m.mem_age.sum().item()) == m.mem_age.numel()
    assert torch.allclose(m.mem_priority, p["pending"]["priority"])
    assert torch.allclose(m.episodic.episode_buffer[0, 0], p["pending"]["compressed"].squeeze(0))
    assert torch.all(m.episodic.episode_buffer[0, 1:] == 0)