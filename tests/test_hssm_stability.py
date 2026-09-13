"""HierarchicalSSM numerical-stability regression tests.

The recurrent scan used to overflow to inf/NaN under bfloat16: when the
cumulative state-decay hits the log_prefix floor, exp_neg = exp(-floor) is
materialized (e.g. exp(+80) ~ 5.5e34 in bf16), multiplied by B_bar*B_stacked,
and summed over the whole sequence. Any state dim whose effective `bb` exceeds
~3 over the clamped tail blows past the bf16 limit (3.4e38) and the resulting
NaN poisons every downstream aux loss (memory reconstruction, intent ...).

The scan now runs in float32 with a floor of -30 so the intermediates stay
bounded (exp(30) ~ 1.1e13) regardless of the model dtype.
"""
import pytest
import torch

from src.dhara.hssm import HierarchicalSSM


def _forced_deep_clamp_model(dtype: torch.dtype):
    """Deterministic switches that pin the scan in the previously-fatal
    regime: strong per-step decay (fast clamp) plus |B_stacked| > 3 so the
    old bf16 cumsum overflowed."""
    torch.manual_seed(0)
    model = HierarchicalSSM(d_model=32, d_state=8, n_levels=1).to(dtype)
    with torch.no_grad():
        model.dt_proj.weight.zero_()
        model.dt_proj.bias.fill_(2.5)          # dt = softplus(2.5) ~ 2.54
        model.A_log.data.fill_(float(torch.log(torch.tensor(2.0))))  # A ~ -1.31
        for proj in model.B_proj:
            proj.weight.zero_()
            proj.bias.fill_(5.0)               # |B_stacked| ~ 5 -> bb ~ 3.7
    return model


def _run(model: torch.nn.Module, batch: int, seq: int, dtype: torch.dtype,
         with_grads: bool):
    x = torch.randn(batch, seq, model.d_model, dtype=dtype)
    out, h = model(x, None)
    assert out.dtype == dtype
    assert h.dtype == dtype
    assert torch.isfinite(out).all() and torch.isfinite(h).all()
    if with_grads:
        out.mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(g.isfinite().all() for g in grads)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_deep_clamp_regime_stays_finite(dtype):
    model = _forced_deep_clamp_model(dtype)
    _run(model, batch=2, seq=2048, dtype=dtype, with_grads=True)


@pytest.mark.parametrize("seq", [64, 512, 2048])
def test_random_scan_finite_and_shapes(seq):
    torch.manual_seed(7)
    model = HierarchicalSSM(d_model=64, d_state=16, n_levels=3)
    x = torch.randn(2, seq, model.d_model)
    out, h = model(x, None)
    assert out.shape == x.shape
    assert h.shape == (2, 3, 16)
    assert torch.isfinite(out).all() and torch.isfinite(h).all()


def test_state_carried_across_chunks_is_finite_and_dtype_stable():
    torch.manual_seed(3)
    model = HierarchicalSSM(d_model=48, d_state=12, n_levels=2)
    x = torch.randn(2, 128, model.d_model)
    _, h = model(x, None)
    x2 = torch.randn(2, 128, model.d_model)
    out, h2 = model(x2, h)
    assert h.dtype == torch.float32
    assert h2.shape == h.shape
    assert torch.isfinite(out).all() and torch.isfinite(h2).all()