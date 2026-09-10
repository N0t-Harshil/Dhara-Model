"""Phase 9 (mandate §9): authoritative pretrain step accounting.

The box-run numbers "5739 samples / 1213 steps" vs the pipeline's
"Est training steps: 1434" came from unlabeled device-batch vs optimizer-batch
estimates. src.utils.steps is the single formula; these tests pin the exact
arithmetic including the 5739/4 = 1434 box anchor.
"""
import src.data.pipeline as dp
from src.utils.steps import estimate_pretrain_steps, format_step_estimate


def test_device_batch_steps_equals_box_anchor():
    """5739 packed @ bs=4 reproduces the box's "Est training steps: 1434"."""
    est = estimate_pretrain_steps(5739, 4)
    assert est.device_batch_steps == 1434
    assert est.optimizer_steps == 1434  # ga=1, world=1 -> same


def test_effective_batch_scales_optimizer_steps():
    """ga=4 + bs=4 -> effective batch 16: 5739 // 16 = 358 optimizer steps,
    with an explicit residual (11) so the estimate is transparent."""
    est = estimate_pretrain_steps(5739, 4, gradient_accumulation_steps=4)
    assert est.gradient_accumulation_steps == 4
    assert est.optimizer_steps == 358
    assert est.remaining_packed == 11


def test_world_size_scales_effective_batch():
    est = estimate_pretrain_steps(8192, 4, gradient_accumulation_steps=4, world_size=2)
    assert est.optimizer_steps == 256  # 8192 // 32
    assert est.device_batch_steps == 2048  # 8192 // 4


def test_guard_against_zero_or_none_inputs():
    est = estimate_pretrain_steps(0, 0, 0, 0)
    assert est.optimizer_steps == 0
    assert est.device_batch_steps == 0
    assert est.batch_size == 1 and est.gradient_accumulation_steps == 1
    assert est.world_size == 1


def test_format_mentions_both_estimates_unambiguously():
    est = estimate_pretrain_steps(5739, 4, gradient_accumulation_steps=4)
    text = format_step_estimate(est)
    assert "device-batch steps=1434" in text
    assert "optimizer steps=358" in text
    assert "eff batch=16" in text


def test_pipeline_summary_uses_the_shared_helper():
    """src.data.pipeline must route its summary estimate through the shared
    helper, not a bespoke total_packed // batch_size line."""
    text = open(dp.__file__, encoding="utf-8").read()
    assert "format_step_estimate" in text
    assert "estimate_pretrain_steps" in text
    assert "src.utils.steps" in text