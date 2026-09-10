"""Authoritative pretrain step accounting (mandate Phase 9).

One formula drives every "Est steps"-style log so the data-pipeline estimate,
the per-unit allocation audit and the training-side status report can never
drift apart again (the box run showed "5739 samples / 1213 steps" against the
pipeline's "Est training steps: 1434" — an unlabeled device-batch vs
optimizer-batch mix-up).

The two numbers are always reported together and explicitly labelled:

* ``device_batch_steps == total_packed // batch_size`` (raw packed batches),
* ``optimizer_steps   == total_packed // effective_batch`` where
  ``effective_batch = batch_size * gradient_accumulation_steps * world_size``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StepEstimate:
    total_packed: int
    batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    device_batch_steps: int
    optimizer_steps: int
    remaining_packed: int


def estimate_pretrain_steps(
    total_packed: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> StepEstimate:
    """Estimate packed-data step counts from a batch-accounting formula.

    ``total_packed`` is the number of packed sequences; ``batch_size`` the
    per-device micro-batch; ``gradient_accumulation_steps`` the optimizer
    accumulation window; ``world_size`` the data-parallel width.
    """
    bs = max(1, int(batch_size or 1))
    ga = max(1, int(gradient_accumulation_steps or 1))
    ws = max(1, int(world_size or 1))
    packed = max(0, int(total_packed or 0))
    eff = bs * ga * ws
    return StepEstimate(
        total_packed=packed,
        batch_size=bs,
        gradient_accumulation_steps=ga,
        world_size=ws,
        device_batch_steps=packed // bs,
        optimizer_steps=packed // eff,
        remaining_packed=packed % eff,
    )


def format_step_estimate(est: StepEstimate) -> str:
    """Single display form so every log line numbers identically.

    e.g. "packed=5739 bs=4 ga=4 world=1 -> device-batch steps=1434, "
         "optimizer steps=358 (eff batch=16, residual=11)"
    """
    eff = est.batch_size * est.gradient_accumulation_steps * est.world_size
    return (
        f"packed={est.total_packed} bs={est.batch_size} "
        f"ga={est.gradient_accumulation_steps} world={est.world_size} -> "
        f"device-batch steps={est.device_batch_steps}, "
        f"optimizer steps={est.optimizer_steps} (eff batch={eff}, "
        f"residual={est.remaining_packed})"
    )