from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import threading
import time
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset, load_from_disk
from transformers import (
    DefaultDataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.alignment.pipeline import AlignmentPipeline
from src.config.schema import Config
from src.data.registry import build_registry
from src.data.pipeline import DEFAULT_MAX_SAMPLES_PER_DATASET, DataPipeline
from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.reporting import EvaluationReport
from src.evaluation.safety import SafetyEvaluator
from src.infrastructure.distributed import DistributedSetup
from src.infrastructure.telemetry import PipelineTelemetry
from src.infrastructure.tracking import ExperimentTracker
from src.models.factory import ARCH_FSDP_LAYER_MAP, ModelFactory
from src.training.asyncprefetch import PrefetchTimeout, UnitPrefetch
from src.training.checkpoint import AsyncCheckpointWriter, verify_checkpoint
from src.utils.reproducibility import set_seed
from src.utils.training import dataloader_num_workers, trainer_tokenizer_kwarg

logger = logging.getLogger(__name__)


def _prefetch_enabled(staging: Any, async_cfg: Any) -> bool:
    """Async prefetch runs only when the staging block opts in AND the global
    ``data.async_pipeline.enabled`` switch is on. A missing/disabled async
    config gives the emergency synchronous fallback (build inline on the
    training thread)."""
    if not bool(getattr(staging, "prefetch", True)):
        return False
    return bool(getattr(async_cfg, "enabled", True))


# Optimizer steps per packed-sample implied by the step allocation beyond
# which a dataset is being replayed so many times that loss reduction is
# almost certainly memorization. Logged as a warning only — never changes
# scheduling (small datasets legitimately get repeated coverage).
_HIGH_REPETITION_EPOCHS = 5.0


def _telemetry_stage_tags(stage_index: int, unit_index: int, n_units: int,
                          dataset_index: int, n_train: int) -> Dict[str, str]:
    """Stage/unit/dataset identifiers for the telemetry log line.

    ``stage_index`` is the 1-based position of the stage group (the loop
    enumerates stages starting at 1), ``unit_index`` the 1-based registry
    position, and ``dataset_index`` the 0-based index into the trainable units
    scheduled for THIS stage (they can differ from the registry position when
    units were skipped). Callers must pass the ACTUAL stage index — the box
    run once reported ``stage=2`` while training stage 1 because the caller
    added one on top of an already 1-based index."""
    return {
        "stage": stage_index,
        "unit": f"{unit_index}/{n_units}",
        "dataset": f"{dataset_index + 1}/{n_train}",
    }


def _install_continuous_scheduler(trainer):
    """Stage transitions call Trainer.train() again, which rebuilds the LR
    scheduler from scratch inside create_optimizer_and_scheduler (the optimizer
    object is reused, but the schedule cold-restarts at base_lr+warmup -> a
    violent loss spike at every stage boundary). Seed each newly-created
    scheduler with the previous stage's last LR and global step so cosine
    annealing continues smoothly instead of restarting. First stage / no prior
    schedule is left untouched; checkpoint resume then overrides anyway.
    """
    state = {"prev_lr": None, "prev_step": None}
    orig = trainer.create_optimizer_and_scheduler

    def seeded_create(self, *args, **kwargs):
        old_sched = getattr(trainer, "lr_scheduler", None)
        if old_sched is not None and old_sched.get_last_lr():
            state["prev_lr"] = float(old_sched.get_last_lr()[0])
            state["prev_step"] = int(getattr(trainer.state, "global_step", 0))
        orig(*args, **kwargs)
        new_sched = getattr(trainer, "lr_scheduler", None)
        if new_sched is not None and state["prev_lr"] is not None:
            new_sched.last_epoch = state["prev_step"]
            new_sched.base_lrs = [state["prev_lr"]]

    trainer.create_optimizer_and_scheduler = MethodType(seeded_create, trainer)
    return trainer


class _FailureJournal:
    """Per-run failure journal. Identical fingerprint on the next launch causes
    the same units to be skipped (deterministic resume) instead of
    re-attempting them."""

    def __init__(self, run_dir: Path, stage_prefix: str) -> None:
        self._path = run_dir / f"failed_units_{stage_prefix}.json"
        self._failed: Dict[str, str] = {}
        if self._path.exists():
            try:
                self._failed = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failure journal unreadable (%s) — starting fresh", e)

    def is_failed(self, key: str) -> bool:
        return key in self._failed

    def reason(self, key: str) -> str:
        return self._failed.get(key, "unknown")

    def mark(self, key: str, reason: str) -> None:
        self._failed[key] = reason
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._failed, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not persist failure journal: %s", e)

    def clear(self, key: str) -> None:
        """Drop a failed marker (a later successful run must not skip this
        unit on resume). No-op when the unit is not in the journal."""
        if key not in self._failed:
            return
        self._failed.pop(key)
        try:
            if not self._failed and self._path.exists():
                self._path.unlink()
            else:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._failed, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not persist failure journal: %s", e)


class _LoggingCallback(TrainerCallback):
    def __init__(self):
        self._step_start = 0.0
        self._model = None

    def set_model(self, model):
        # HF 5.x may wrap the raw model (DataParallel and similar); the module
        # subclasses publish _last_* diagnostics on the INNER module, so unwrap
        # here or getattr() below sees nothing.
        self._model = getattr(model, "module", model)

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            step = state.global_step
            loss = logs.get("loss", logs.get("train_loss", "N/A"))
            lr = logs.get("learning_rate", "N/A")
            loss_val = float(loss) if loss is not None and loss != "N/A" else 0.0
            lr_val = float(lr) if lr is not None and lr != "N/A" else 0.0
            elapsed = time.time() - self._step_start if self._step_start else 0.0
            logger.info("Step %d | loss=%.4f | lr=%.2e | it/s=%.2f", step, loss_val, lr_val, 1.0 / max(elapsed, 1e-8))
            self._log_instrumentation(step)

    def _log_instrumentation(self, step):
        """Append the model's latest auxiliary losses, executive gates and
        logits/vocab stats to the step log so subsystem learning and head
        health are observable without separate eval infrastructure."""
        model = self._model
        if model is None:
            logger.warning("Instrumentation unavailable: _model not wired (set_model never called).")
            return
        try:
            model = getattr(model, "module", model)
            aux = getattr(model, "_last_aux_losses", None)
            aux_meta = getattr(model, "_last_aux_meta", None)
            exec_meta = getattr(model, "_last_exec_meta", None)
            lstats = getattr(model, "_last_logits_stats", None)
            parts = []
            if aux_meta:
                _st = aux_meta.get("status", "?")
                if _st in ("flowing", "braked"):
                    parts.append(
                        f"aux_sum[{_st}](raw={aux_meta['raw']:.3f}->applied={aux_meta['applied']:.3f} "
                        f"cap={aux_meta['cap']:.3f} ce={aux_meta['ce']:.3f})"
                    )
                else:
                    parts.append(f"aux_sum[{_st}]")
            if aux:
                parts.append("aux={" + ", ".join(f"{k}={v:.4f}" for k, v in sorted(aux.items())) + "}")
            if exec_meta:
                gm = exec_meta.get("gates") or {}
                parts.append("gates={" + ", ".join(f"{k}={v:.3f}" for k, v in sorted(gm.items())) + "}")
                if exec_meta.get("confidence") is not None:
                    parts.append(f"conf={exec_meta['confidence']:.4f}")
                if exec_meta.get("reward") is not None:
                    parts.append(f"rew={exec_meta['reward']:.4f}")
            if lstats:
                _extra = ""
                if "label_max" in lstats:
                    _extra = f" lmax={lstats['label_max']} oov={lstats['label_oov_frac']:.4f} tgt={lstats['target_logit_mean']:.4f}"
                parts.append(
                    f"lstats(v={lstats['vocab']} mae={lstats['logit_mean_abs']:.4f} "
                    f"max={lstats['logit_max']:.4f} clip={lstats['clip50_frac']:.6f}{_extra})"
                )
            if parts:
                logger.info("  %s", " | ".join(parts))
            else:
                logger.warning(
                    "Instrumentation empty at step %d (aux=%s exec=%s lstats=%s)",
                    step, aux is not None, exec_meta is not None, lstats is not None,
                )
        except Exception:
            logger.warning("Instrumentation failure at step %d", step, exc_info=True)


class _NaNSafeCallback(TrainerCallback):
    def __init__(self, check_every: int = 100) -> None:
        self.check_every = max(1, check_every)
        self._last_check = -1
        self._model = None

    def set_model(self, model):
        self._model = model

    def on_step_end(self, args, state, control, model=None, **kwargs):
        model = self._model if self._model is not None else model
        if model is None or not state.is_world_process_zero:
            return
        if state.global_step - self._last_check < self.check_every:
            return
        self._last_check = state.global_step
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite parameter at step {state.global_step}: {name}")


class StageBoundaryCallback(TrainerCallback):
    """Stops the Trainer at the end of one pretraining stage.

    Fires in on_step_end (after on_step_begin has reset the control flags):
    requests a full checkpoint save and a training stop, so the main loop's
    _maybe_log_save_evaluate() persists optimizer/scheduler/RNG before the
    step loop breaks. on_train_begin resets the flags for the next train()
    call, which resumes from that checkpoint with global_step unchanged.
    """

    def __init__(self, stage_index: int, total_stages: int, stage_name: str, end_step: int) -> None:
        self.stage_index = stage_index
        self.total_stages = total_stages
        self.stage_name = stage_name
        self.end_step = end_step
        self.triggered = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.triggered = False
        control.should_training_stop = False
        control.should_save = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.triggered and state.global_step >= self.end_step:
            self.triggered = True
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                logger.info(
                    "[STAGE] %d/%d %s — boundary at global_step=%d, saving full checkpoint and stopping stage",
                    self.stage_index, self.total_stages, self.stage_name, state.global_step,
                )


class _LoggingDataCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        return self.collator(features)


# Order matches the staged-registry category list (and n_task_types=8).
_TASK_CATEGORIES = (
    "code", "docs", "web_text", "wiki",
    "math", "science", "books", "structured_knowledge",
)
_TASK_CATEGORY_ID = {name: i for i, name in enumerate(_TASK_CATEGORIES)}
_DIFFICULTY_EDGES = (0.2, 0.4, 0.6, 0.8)


def _difficulty_bucket(quality) -> int:
    q = min(max(float(quality), 0.0), 1.0)
    return min(sum(1 for edge in _DIFFICULTY_EDGES if q > edge), 4)


class _PretrainAuxCollator:
    """Turns signal that DefaultDataCollator silently drops (row-level
    ``_category`` / ``_avg_quality`` are strings, never tensorized) into
    ``aux_targets`` supervision, so AuxiliaryLossComputer actually fires during
    pretraining instead of being permanently inert. task_type is the document
    category; difficulty is bucketed from the row's quality proxy, giving the
    intent classifiers and difficulty router a real learnable signal."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        batch = self.base(features)
        cats = [f.get("_category") for f in features]
        quals = [f.get("_avg_quality") for f in features]
        if all(isinstance(c, str) for c in cats) and all(isinstance(q, (int, float)) for q in quals):
            task_type = torch.tensor([_TASK_CATEGORY_ID.get(c, 0) for c in cats], dtype=torch.long)
            difficulty = torch.tensor([_difficulty_bucket(q) for q in quals], dtype=torch.long)
            batch["aux_targets"] = {"intent": {"task_type": task_type, "difficulty": difficulty}}
        return batch


class _RunHealthCallback(TrainerCallback):
    """Periodic run-health probes: held-out perplexity over a prefix stashed
    from the first training batch, plus a free-form generation sample. Every
    failure degrades to a warning so diagnostics can never kill training."""

    def __init__(self, tokenizer, eval_every: int = 500, n_ctx: int = 128,
                 n_window: int = 384, n_gen: int = 64) -> None:
        self.tokenizer = tokenizer
        self.eval_every = max(1, eval_every)
        self.n_ctx = n_ctx           # context tokens fed before the label window
        self.n_window = n_window     # total tokens fed to the model (ctx + held-out)
        self.n_gen = n_gen
        self._model = None

    def set_model(self, model):
        self._model = model

    def on_train_begin(self, args, state, control, **kwargs):
        # Re-stash under a fresh key each run so a resume picks up a fresh batch.
        model = self._model
        if model is not None:
            try:
                setattr(model, "_run_health_stash", None)
            except Exception:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        step = state.global_step
        if step == 0 or step % self.eval_every != 0:
            return
        model = self._model
        if model is None:
            return
        self._eval_heldout_ppl(model, step)
        self._eval_sample(model, step)

    def _stash(self, model):
        stash = getattr(model, "_run_health_stash", None)
        if not isinstance(stash, dict) or "input_ids" not in stash:
            return None
        if stash["input_ids"].numel() < self.n_window:
            return None
        return stash

    def _eval_heldout_ppl(self, model, step):
        stash = self._stash(model)
        if stash is None:
            logger.warning("[HEALTH] no stashed batch for ppl eval — skipping")
            return
        try:
            device = next(model.parameters()).device
            ids = stash["input_ids"][:1, : self.n_window].to(device)
            lang = stash.get("language_ids")
            if lang is None:
                lang = stash.get("languages")
            lang_slice = lang[:1, : self.n_window].to(device) if lang is not None else None
            labels = ids.clone()
            labels[:, : self.n_ctx] = -100
            was_training = model.training
            model.eval()
            with torch.no_grad():
                out = model(ids, labels=labels, language_ids=lang_slice)
            ce = float(out.loss.detach().float())
            n_tokens = int((labels != -100).sum().item())
            ppl = math.exp(min(ce, 80)) if ce < 80 else float("inf")
            logger.info("  [HEALTH] step=%d held-out ppl=%.3f ce=%.4f tokens=%d",
                        step, ppl, ce, n_tokens)
            if was_training:
                model.train()
        except Exception as e:
            logger.warning("[HEALTH] held-out ppl failed at step %d: %s", step, e)

    def _eval_sample(self, model, step):
        stash = self._stash(model)
        if stash is None:
            logger.warning("[HEALTH] no stashed batch for generation — skipping")
            return
        try:
            device = next(model.parameters()).device
            ids = stash["input_ids"][:1, : self.n_ctx].to(device)
            was_training = model.training
            model.eval()
            old_debate = getattr(model.config, "n_debate_rounds", 1)
            if int(getattr(model.config, "n_experts", 0) or 0) == 0:
                model.config.n_debate_rounds = 0
            with torch.no_grad():
                out_ids = model.generate(
                    ids, max_new_tokens=self.n_gen,
                    temperature=0.6, top_k=40, top_p=0.9,
                )
            prefix = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            logger.info("  [HEALTH] step=%d prefix[:80]=%r", step, prefix[:80])
            logger.info("  [HEALTH] step=%d gen[:200]=%r", step, text[:200])
            if was_training:
                model.train()
            model.config.n_debate_rounds = old_debate
        except Exception as e:
            logger.warning("[HEALTH] generation failed at step %d: %s", step, e)


class _TrainHeartbeatCallback(TrainerCallback):
    """Turn a silent training stall into a visible one.

    HF Trainer's progress bar is the only sign of life, and a stop before the
    first step (e.g. the dataloader fork-deadlock) shows nothing at all. This
    logs "step 0 ... awaiting first batch" the moment train() enters the loop,
    then heartbeats every interval_steps so the log shows elapsed wall time
    even when the tqdm output is buffered/captured.
    """

    def __init__(self, interval_steps: int = 100, total_steps: int = 0) -> None:
        self.interval_steps = max(1, interval_steps)
        self.total_steps = int(total_steps or 0)
        self._t0 = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._t0 = time.time()
        logger.info("[TRAIN] step 0/%d — entering training loop, awaiting first batch...",
                    self.total_steps)

    def on_step_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step % self.interval_steps == 0:
            logger.info("[TRAIN] step %d/%d heartbeat at %.1fs",
                        state.global_step, self.total_steps,
                        time.time() - self._t0)


class TrainingPipeline:
    def __init__(self, cfg: Config, dist_setup: Optional[DistributedSetup] = None) -> None:
        self.cfg = cfg
        self.dist = dist_setup if dist_setup is not None else DistributedSetup(cfg)
        self.tracker = ExperimentTracker(cfg.output.experiment_tracking)
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.ref_model: Optional[PreTrainedModel] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.alignment: Optional[AlignmentPipeline] = None
        self._last_trainer: Optional[Trainer] = None
        self._fresh_start: bool = False
        self._forced_resume: Optional[str] = None
        self._ckpt_writer: Optional[AsyncCheckpointWriter] = None
        if getattr(self.cfg.training, "async_checkpoint", True):
            self._ckpt_writer = AsyncCheckpointWriter(
                checksum=bool(getattr(self.cfg.training, "checkpoint_checksum", True)),
            )
        tcfg = getattr(self.cfg.training, "telemetry", None)
        self.telemetry = PipelineTelemetry(
            enabled=bool(tcfg.enabled if tcfg is not None else True),
            interval_sec=float(tcfg.interval_sec if tcfg is not None else 30.0),
            gpu_util_sampling=bool(tcfg.gpu_util_sampling if tcfg is not None else True),
        )
        set_seed(cfg.project.get("seed", 42) if cfg.project else 42)

    def initialize(self, fresh_start: bool = False, resume_checkpoint: Optional[str] = None) -> None:
        logger.info("Initializing Dhara Class Model pipeline (fresh_start=%s, resume=%s)...", fresh_start, resume_checkpoint)
        self._fresh_start = fresh_start
        if resume_checkpoint is not None:
            ckpt_path = Path(resume_checkpoint)
            self._forced_resume = str(ckpt_path) if (ckpt_path.exists() and ckpt_path.is_dir()) else None
        self.tokenizer = ModelFactory.load_tokenizer(cfg=self.cfg)

        if fresh_start:
            logger.info("Creating model from scratch...")
            self.model = ModelFactory.create_model(self.cfg, self.tokenizer)
        else:
            resume_checkpoint = resume_checkpoint or self._find_resume_checkpoint()
            if resume_checkpoint is not None:
                ckpt_path = Path(resume_checkpoint)
                if ckpt_path.is_dir():
                    try:
                        ok, detail = verify_checkpoint(ckpt_path)
                        if not ok:
                            logger.error(
                                "Resume checkpoint %s failed integrity "
                                "verification (%s) — starting from scratch "
                                "instead of resuming (do not trust a corrupt "
                                "checkpoint).", ckpt_path, detail)
                            resume_checkpoint = None
                    except Exception as e:
                        logger.warning("Checkpoint verification error for %s "
                                       "(%s) — continuing without it.",
                                       ckpt_path, e)
                        resume_checkpoint = None
                if resume_checkpoint is not None and ckpt_path.exists() and ModelFactory.is_compatible(self.cfg, self.tokenizer, ckpt_path):
                    logger.info("Loading model from checkpoint: %s", ckpt_path)
                    self.model, self.tokenizer = ModelFactory.load_model(ckpt_path, self.cfg)
                else:
                    logger.info("Checkpoint incompatible or not found. Creating from scratch.")
                    self.model = ModelFactory.create_model(self.cfg, self.tokenizer)
            else:
                logger.info("No checkpoint found. Creating model from scratch.")
                self.model = ModelFactory.create_model(self.cfg, self.tokenizer)

        if torch.cuda.is_available():
            self.model = self.model.to(self.dist.auto_device())
            torch.cuda.empty_cache()

        self.model.train()
        self.data_pipeline = DataPipeline(self.cfg, self.tokenizer)
        self.alignment = AlignmentPipeline(self.model, self.tokenizer, self.cfg)

        if self.cfg.output.experiment_tracking.enabled:
            self.tracker.init(
                config=self.cfg.model_dump(mode="python"),
                name=self.cfg.model.name,
            )

        param_count = sum(p.numel() for p in self.model.parameters())
        total_params = ModelFactory.estimate_model_size(self.cfg.model.architecture, tokenizer_or_vocab=self.tokenizer)
        logger.info("Model parameters: %s", f"{param_count:,}")
        logger.info("Estimated size: %sB total, %sB active", total_params["total_params_b"], total_params["active_params_b"])
        self.telemetry.start()

    def run_pretrain(self, dataset: Dataset, **overrides) -> Dict[str, float]:
        logger.info("=== PRETRAINING PHASE ===")
        stage = self.cfg.training.pretrain
        return self._train_stage(dataset, "pretrain", stage, **overrides)

    def run_sft(self, dataset: Dataset, **overrides) -> Dict[str, float]:
        logger.info("=== SUPERVISED FINE-TUNING PHASE ===")
        stage = self.cfg.training.sft
        return self._train_stage(dataset, "sft", stage, **overrides)

    def run_instruction_tuning(self, dataset: Dataset, **overrides) -> Dict[str, float]:
        logger.info("=== INSTRUCTION TUNING PHASE ===")
        stage = self.cfg.training.instruction_tuning
        return self._train_stage(dataset, "instruction_tuning", stage, **overrides)

    def run_alignment(self, preference_dataset: Dataset, kto_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        logger.info("=== ALIGNMENT PHASE ===")
        if self.alignment is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.alignment.run_alignment_sequence(preference_dataset, kto_dataset)

    def run_constitutional_alignment(
        self, instructions: List[str], responses: List[str]
    ) -> List[Dict[str, str]]:
        if self.alignment is None:
            raise RuntimeError("Pipeline not initialized.")
        return self.alignment.run_constitutional_alignment(instructions, responses)

    def run_safety_training(self) -> Dict[str, float]:
        logger.info("=== SAFETY TRAINING PHASE ===")
        if self.alignment is None:
            raise RuntimeError("Pipeline not initialized.")
        return self.alignment.run_safety_training(num_steps=self.cfg.training.safety.safety_training_steps)

    def run_red_teaming(self) -> List[Dict[str, Any]]:
        logger.info("=== RED TEAMING PHASE ===")
        if self.alignment is None:
            raise RuntimeError("Pipeline not initialized.")
        return self.alignment.run_red_teaming(num_iters=self.cfg.training.safety.red_teaming_iters)

    def run_evaluation(self) -> Dict[str, Any]:
        logger.info("=== EVALUATION PHASE ===")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        self.model.eval()
        runner = BenchmarkRunner(self.model, self.tokenizer)
        benchmark_results: List[Any] = []
        bench_configs = self.cfg.evaluation.benchmark_configs or {}
        for name in self.cfg.evaluation.benchmarks:
            entry = bench_configs.get(name)
            kwargs: Dict[str, Any] = {}
            if entry is not None and entry.max_samples:
                kwargs["limit"] = entry.max_samples
            if getattr(self.cfg.evaluation, "timeout", 0) > 0:
                kwargs["timeout"] = self.cfg.evaluation.timeout
            benchmark_results.extend(runner.run_benchmarks([name], **kwargs))

        safety_eval = SafetyEvaluator(self.model, self.tokenizer)
        safety_results = safety_eval.full_report()

        reporter = EvaluationReport(self.cfg.evaluation.report_dir)
        report = reporter.generate(
            model_name=self.cfg.model.name,
            benchmark_results=benchmark_results,
            safety_results=safety_results,
        )
        self.tracker.log_metrics({
            **{f"benchmark/{r.name}": r.score for r in benchmark_results},
            "safety_refusal_rate": safety_results.get("safety", {}).get("safety_refusal_rate", 0),
        })
        return report

    def _cache_key(self, ds_info: Any, stage_name: str) -> str:
        # Mirror DataPipeline._get_cache_key: preprocessing/quality settings
        # must invalidate the cache, otherwise changing dedup/filtering
        # silently reuses stale tokenized data (the original bug used only
        # path+max_samples+seq_len+vocab).
        pp = self.cfg.data.preprocessing
        q = self.cfg.data.quality
        raw = "|".join([
            ds_info.path, getattr(ds_info, "name", "") or "",
            getattr(ds_info, "split", "train"),
            str(getattr(ds_info, "max_samples", "")),
            str(self.cfg.training.max_seq_length),
            str(self.cfg.model.architecture.vocab_size),
            q.deduplication.method, f"{q.deduplication.threshold:.4f}",
            str(self.cfg.data.ast_filter.code_filtering),
            str(pp.remove_boilerplate), str(pp.min_text_length),
            "v2",
        ])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cached_dataset_path(self, ds_info: Any, stage_name: str) -> Path:
        ck = self._cache_key(ds_info, stage_name)
        return Path(self.cfg.data.cache_dir) / "tokenized" / stage_name / ck

    def full_training_sequence(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("TRAINING SEQUENCE: Pretrain -> SFT -> Instruction Tuning")
        logger.info("=" * 60)
        results: Dict[str, Any] = {}

        stages = [
            ("pretrain", self.cfg.training.pretrain),
            ("sft", self.cfg.training.sft),
            ("instruction_tuning", self.cfg.training.instruction_tuning),
        ]

        for stage_name, stage_cfg in stages:
            if not stage_cfg.enabled:
                continue

            logger.info("=" * 70)
            logger.info("PHASE: %s", stage_name.upper())
            logger.info("=" * 70)

            if stage_name == "pretrain":
                staging = self.cfg.training.pretrain.staging
                curriculum = self.cfg.data.curriculum
                if staging.enabled and staging.stages:
                    if curriculum.enabled and curriculum.stages:
                        logger.warning(
                            "Both staging and curriculum are enabled — the "
                            "legacy curriculum is ignored in favor of the "
                            "staging stages.")
                    logger.info("Staged pretraining mode: %d stage groups (legacy curriculum ignored)",
                                len(staging.stages))
                    metrics = self._run_staged_pretrain(stage_cfg)
                    results["pretrain"] = metrics
                elif curriculum.enabled and curriculum.stages:
                    prev_max = 0
                    for stage in curriculum.stages:
                        logger.info("")
                        logger.info("--- Curriculum Stage: %s (steps %d-%d, filter=%s) ---",
                                     stage.name, prev_max, stage.max_steps, stage.dataset_filter)
                        steps_this_stage = stage.max_steps - prev_max
                        if steps_this_stage <= 0:
                            continue
                        dataset = self.data_pipeline.build_pretrain_dataset_from_registry(
                            dataset_filter=stage.dataset_filter,
                        )
                        if len(dataset) == 0:
                            logger.warning("Curriculum stage %s produced empty dataset — skipping", stage.name)
                            prev_max = stage.max_steps
                            continue
                        logger.info("Curriculum dataset ready: %d samples", len(dataset))
                        metrics = self._train_stage(dataset, f"pretrain_{stage.name}", stage_cfg,
                                                     max_steps=steps_this_stage)
                        results[f"pretrain/{stage.name}"] = metrics
                        prev_max = stage.max_steps
                else:
                    logger.info("Building weighted mixed pretrain dataset from registry...")
                    t_build = time.perf_counter()
                    dataset = self.data_pipeline.build_pretrain_dataset_from_registry()
                    logger.info("[TIMER] build_pretrain_dataset_from_registry total: %.1fs",
                                time.perf_counter() - t_build)
                    if len(dataset) == 0:
                        logger.error("Pretrain dataset is empty — aborting")
                        return results
                    logger.info("Pretrain dataset ready: %d samples", len(dataset))
                    metrics = self._train_stage(dataset, stage_name, stage_cfg)
                    results[stage_name] = metrics
            else:
                datasets = self.data_pipeline.collector.get_dataset_list()
                if not datasets and getattr(self.cfg.data, "use_registry", False):
                    logger.warning(
                        "No datasets configured for %s and use_registry=true — "
                        "routing the stage through the registry mixture (no "
                        "dedicated instruction corpora).", stage_name)
                    dataset = self.data_pipeline.build_pretrain_dataset_from_registry()
                    if len(dataset) == 0:
                        logger.error("Registry produced no samples for %s — skipping",
                                     stage_name)
                        continue
                    logger.info("%s dataset ready from registry: %d samples",
                                stage_name, len(dataset))
                    metrics = self._train_stage(dataset, stage_name, stage_cfg)
                    results[stage_name] = metrics
                    continue
                valid_count = 0
                skipped_count = 0

                for ds_idx, ds_info in enumerate(datasets, 1):
                    logger.info("")
                    logger.info("--- Dataset [%d/%d]: %s ---", ds_idx, len(datasets), ds_info.path)
                    logger.info("    Category: %s | Max samples: %s", ds_info.category, ds_info.max_samples)

                    cache_path = self._cached_dataset_path(ds_info, stage_name)
                    if cache_path.exists():
                        logger.info("    Loading cached tokenized dataset from %s", cache_path)
                        dataset = load_from_disk(str(cache_path))
                        logger.info("    Loaded %d samples from cache", len(dataset))
                    else:
                        try:
                            samples = list(self.data_pipeline.collector.stream_single_dataset(
                                ds_info, limit=ds_info.max_samples or DEFAULT_MAX_SAMPLES_PER_DATASET))
                        except Exception as e:
                            logger.error("    FAILED: %s — skipping", e)
                            skipped_count += 1
                            continue

                        if not samples:
                            logger.warning("    No valid samples — skipping")
                            skipped_count += 1
                            continue

                        logger.info("    Collected %d samples, tokenizing...", len(samples))
                        dataset = self.data_pipeline.build_stage_dataset(samples)
                        logger.info("    Tokenized dataset: %d samples", len(dataset))

                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        logger.info("    Saving tokenized dataset to %s", cache_path)
                        dataset.save_to_disk(str(cache_path))
                        logger.info("    Cached to disk")

                        del samples
                        gc.collect()

                    metrics = self._train_stage(dataset, stage_name, stage_cfg)
                    results[f"{stage_name}/{ds_info.path}"] = metrics

                    valid_count += 1
                    del dataset
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            logger.info("=" * 70)
            logger.info("Phase %s complete", stage_name.upper())

        if self.cfg.training.alignment.enabled:
            logger.info("=" * 70)
            logger.info("PHASE: ALIGNMENT")
            logger.info("=" * 70)
            pref_ds = self._build_alignment_datasets()
            if pref_ds is not None:
                results["alignment"] = self.run_alignment(pref_ds)
            else:
                logger.warning("Alignment enabled but no usable instructions — skipping alignment phase")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.cfg.training.safety.enabled:
            logger.info("=" * 70)
            logger.info("PHASE: SAFETY TRAINING + RED TEAMING")
            logger.info("=" * 70)
            results["safety_training"] = self.run_safety_training()
            if self.cfg.training.safety.red_teaming_iters > 0:
                results["red_teaming"] = self.run_red_teaming()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.cfg.evaluation.automated_report:
            logger.info("=" * 70)
            logger.info("PHASE: EVALUATION")
            logger.info("=" * 70)
            results["evaluation"] = self.run_evaluation()

        logger.info("Training complete.")
        return results

    def _build_alignment_datasets(self) -> Optional[Dataset]:
        """Collect a bounded set of instruction prompts for preference-pair
        generation, then tokenize them into the preference dataset format.
        Returns None when no usable instruction text can be gathered."""
        instructions: List[str] = []
        try:
            for ds_info in self.data_pipeline.collector.get_dataset_list():
                if len(instructions) >= 256:
                    break
                for sample in self.data_pipeline.collector.stream_single_dataset(
                    ds_info, limit=min(ds_info.max_samples or DEFAULT_MAX_SAMPLES_PER_DATASET, 32)
                ):
                    text = sample.get("prompt") or sample.get("instruction") or sample.get("text") or ""
                    if isinstance(text, str) and len(text) >= 8:
                        instructions.append(text)
                    if len(instructions) >= 256:
                        break
        except Exception as e:
            logger.warning("Instruction collection for alignment failed: %s", e)
            return None
        if not instructions:
            return None
        pairs = self.alignment.generate_preference_data(instructions[:256])
        if not pairs:
            return None
        return self.data_pipeline.build_preference_dataset(pairs)

    def _train_stage(
        self,
        dataset: Dataset,
        stage_name: str,
        stage_cfg: Any,
        **overrides,
    ) -> Dict[str, float]:
        trainer = self._build_trainer(dataset, stage_name, stage_cfg, **overrides)
        self._last_trainer = trainer

        # Verify model is on GPU
        if torch.cuda.is_available():
            model = trainer.model
            for name, param in model.named_parameters():
                if param.device.type == "cpu":
                    logger.warning("WARNING: %s is on CPU", name)
                    break
            logger.info("Model device check: %s", next(model.parameters()).device)

        output_dir = Path(self.cfg.output.model_dir) / stage_name
        resume_checkpoint = None
        if not self._fresh_start and output_dir.exists():
            ckpt_dirs = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
            )
            if ckpt_dirs:
                resume_checkpoint = str(ckpt_dirs[-1])
                logger.info("Resuming from checkpoint: %s", resume_checkpoint)

        t_train = time.perf_counter()
        if hasattr(trainer, "get_train_dataloader"):
            try:
                dl = trainer.get_train_dataloader()
                it = iter(dl)
                t_batch = time.perf_counter()
                next(it)
                logger.info("[TIMER] first batch from dataset: %.1fs (includes lazy mixed-dataset build)",
                            time.perf_counter() - t_batch)
                del it, dl
            except Exception as e:
                logger.warning("First-batch timer skipped: %s", e)
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info("[TIMER] trainer.train total: %.1fs", time.perf_counter() - t_train)
        metrics = result.metrics if hasattr(result, "metrics") else {}
        logger.info("%s complete: %s", stage_name.upper(), metrics)
        self.tracker.log_metrics({f"{stage_name}/{k}": v for k, v in metrics.items()})

        ckpt_dir = Path(self.cfg.output.model_dir)
        checkpoint_path = ckpt_dir / stage_name
        trainer.save_model(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        self._save_checkpoint(checkpoint_path)
        self._flush_checkpoints()
        logger.info("%s checkpoint saved to %s", stage_name, checkpoint_path)
        return metrics

    def _pretrain_output_dir(self) -> Path:
        return Path(self.cfg.output.model_dir) / "pretrain"

    def _latest_pretrain_checkpoint(self) -> Optional[Path]:
        output_dir = self._pretrain_output_dir()
        ckpt_dirs = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
        )
        return ckpt_dirs[-1] if ckpt_dirs else None

    def _pretrain_global_step(self) -> int:
        try:
            state_path = self._pretrain_output_dir() / "trainer_state.json"
            if state_path.exists():
                st = json.loads(state_path.read_text(encoding="utf-8"))
                return int(st.get("global_step", 0))
        except Exception as e:
            logger.warning("Could not read trainer_state.json: %s", e)
        return 0

    # -- dataset-granular checkpoint identity ---------------------------------
    #
    # Every unit checkpoint records its dataset identity so a restart can
    # verify *which* dataset a checkpoint belongs to (never skip by index
    # alone), and a per-run completion manifest is committed atomically only
    # after training + checkpoint + flush succeeded.

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _unit_fingerprint(self, u) -> str:
        """Deterministic sha256 over the dataset identity of a unit. Changing
        the repository/subset/revision/data_dir changes the fingerprint and
        forces a retrain on resume even if the step counter would skip it."""
        ident = {
            "path": getattr(u, "path", ""),
            "name": getattr(u, "name", None) or "",
            "data_dir": str(getattr(u, "data_dir", "") or ""),
            "revision": str(getattr(u, "revision", "") or ""),
            "category": getattr(u, "category", ""),
        }
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()

    def _unit_completions_path(self) -> Path:
        return Path(self.cfg.output.model_dir) / "unit_completions.json"

    def _load_unit_completions(self) -> Dict[str, Any]:
        path = self._unit_completions_path()
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read unit completion manifest (%s): %s", path, e)
        return {}

    def _mark_unit_complete(self, stage_index: int, unit_index: int,
                            unit_key: str, unit_fingerprint: str,
                            global_step: int, total_steps: int) -> None:
        """Atomically commit a dataset-completion record. Only reached after
        training finished AND the per-unit checkpoint flushed to disk."""
        completions = self._load_unit_completions()
        completions[unit_key] = {
            "stage_index": stage_index,
            "unit_index": unit_index,
            "unit_fingerprint": unit_fingerprint,
            "global_step": int(global_step),
            "total_steps": int(total_steps),
            "completed_at": time.time(),
        }
        self._atomic_write_json(self._unit_completions_path(), completions)

    def _clear_stale_run_state(self) -> List[str]:
        """Drop per-run failure journals and the unit completion manifest so a
        --fresh-start run does not inherit skip decisions from an earlier
        crashed run. Returns the paths that were removed."""
        cleared: List[str] = []
        mdir = Path(self.cfg.output.model_dir)
        for _p in sorted(mdir.glob("failed_units_stage*.json")):
            try:
                _p.unlink()
                cleared.append(str(_p))
            except Exception as _e:
                logger.warning("Could not clear failure journal %s: %s", _p, _e)
        _wm = mdir / "unit_completions.json"
        if _wm.exists():
            try:
                _wm.unlink()
                cleared.append(str(_wm))
            except Exception as _e:
                logger.warning("Could not clear completion manifest %s: %s", _wm, _e)
        return cleared

    def _write_unit_identity(self, ckpt_dir: Path, stage_index: int, unit_index: int,
                             u, unit_key: str, end_step: int, total_steps: int) -> None:
        """Persist unit_identity.json inside the latest per-unit checkpoint so
        a resume can verify which dataset the checkpoint belongs to."""
        identity = {
            "stage_index": stage_index,
            "unit_index": unit_index,
            "unit_key": unit_key,
            "unit_fingerprint": self._unit_fingerprint(u),
            "processed_cache_key": hashlib.sha256(
                (unit_key + ":" + self._unit_fingerprint(u)).encode("utf-8")).hexdigest(),
            "global_step": int(end_step),
            "total_steps": int(total_steps),
        }
        self._atomic_write_json(ckpt_dir / "unit_identity.json", identity)

    def _run_staged_pretrain(self, stage_cfg) -> Dict[str, Any]:
        """Run pretraining as N staged groups with a single Trainer.

        One Trainer is built with max_steps = TOTAL across all stages so the
        cosine schedule spans the whole run. Each stage builds (or loads from
        its stage cache) a dataset, swaps it into the trainer, and trains with
        a StageBoundaryCallback that saves a full checkpoint and stops at the
        stage's cumulative step count. Subsequent stages resume from that
        checkpoint (restoring scheduler/RNG state); a crash mid-stage resumes
        from the latest regular checkpoint with global_step unchanged.
        """
        staging = stage_cfg.staging
        stages = list(staging.stages)
        total_stages = len(stages)
        total_steps = sum(s.steps for s in stages)
        if total_steps <= 0:
            raise RuntimeError(
                "Staged pretraining enabled but total steps == 0 — stage.steps must sum to > 0")
        logger.info("=" * 70)
        logger.info("STAGED PRETRAINING: %d stage groups, %d total steps", total_stages, total_steps)
        for i, s in enumerate(stages, 1):
            logger.info("  Stage %d/%d: %s — %d steps, cats=%s",
                        i, total_stages, s.name or f"stage_{i}", s.steps, sorted(s.categories or []))
        logger.info("=" * 70)

        current_step = 0 if self._fresh_start else self._pretrain_global_step()
        if self._fresh_start:
            logger.info("Fresh start requested — restarting all staged pretrain steps from 0")
            cleared_state = self._clear_stale_run_state()
            if cleared_state:
                logger.info("[UNIT] fresh start — cleared stale run state: %s",
                            "; ".join(cleared_state))
        results: Dict[str, Any] = {}
        trainer = None
        run_t0 = time.perf_counter()
        dup_prevented_total = 0
        retries_total = 0
        prefetch_stats: List[Dict[str, Any]] = []

        for i, s in enumerate(stages, 1):
            end_step = sum(x.steps for x in stages[:i])
            name = s.name or f"stage_{i}"
            if current_step >= end_step:
                logger.info("[STAGE] %d/%d %s — already completed (global_step=%d), skipping",
                            i, total_stages, name, current_step)
                continue
            if current_step > 0:
                logger.info("[STAGE] %d/%d %s — resuming (global_step=%d)",
                            i, total_stages, name, current_step)

            logger.info("")
            logger.info("=" * 70)
            logger.info("STAGE %d / %d — %s", i, total_stages, name)
            logger.info("=" * 70)

            if getattr(staging, "mode", "dataset") == "dataset":
                prev_end = end_step - s.steps
                registry = build_registry(exclude=getattr(self.cfg.data, "registry_exclude", None))
                units = registry.all_entries()
                if s.categories:
                    units = [u for u in units if u.category in s.categories]
                if not units:
                    logger.error("Stage %d (%s) has no registry datasets for categories %s — aborting",
                                 i, name, sorted(s.categories or []))
                    return results

                self._warm_stage_metadata(units, i)

                # Dynamic stage sizing: on re-runs where every unit is already
                # packed, allocate steps proportionally to actual cached sample
                # counts; first run falls back to registry weights. Either way
                # the allocation is deterministic for a given cache state.
                cached_counts = [self.data_pipeline.unit_cache_packed_count(u, i, j)
                                 for j, u in enumerate(units, 1)]
                if all(c is not None and c > 0 for c in cached_counts):
                    sizes = [float(c) for c in cached_counts]
                    sizing_basis = "packed"
                else:
                    sizes = [float(u.weight or 1.0) for u in units]
                    sizing_basis = "weight"
                total_sz = sum(sizes) or len(sizes)
                alloc = []
                remaining = s.steps
                for j in range(len(units)):
                    if j == len(units) - 1:
                        alloc.append(remaining)
                    elif remaining - (len(units) - j) >= 1:
                        n = max(1, int(round(s.steps * sizes[j] / total_sz)))
                        n = min(n, remaining - (len(units) - j))
                        alloc.append(n)
                        remaining -= n
                    else:
                        alloc.append(0)  # step budget exhausted — unit gets no steps
                logger.info("[UNIT] stage %d — %d datasets, step allocation basis: %s",
                            i, len(units), sizing_basis)
                # Step accounting audit (mandate Phase 9): the planner must
                # consume the stage budget exactly (sum(alloc) == s.steps), and
                # the data-side packed estimate is reported with the SAME
                # formula (src.utils.steps) so the "Est training steps" numbers
                # from DataPipeline and the per-stage allocation reconcile.
                if sum(alloc) != s.steps:
                    logger.warning("[UNIT] stage %d step allocation does NOT sum to "
                                   "the stage budget (%d != %d) — accounting drift",
                                   i, sum(alloc), s.steps)
                if sizing_basis == "packed":
                    from src.utils.steps import estimate_pretrain_steps, format_step_estimate
                    ga = int(getattr(stage_cfg, "gradient_accumulation_steps", 1) or 1)
                    bs = int(getattr(stage_cfg, "batch_size", 1) or 1)
                    _est = estimate_pretrain_steps(
                        sum(c for c in cached_counts if c), bs,
                        gradient_accumulation_steps=ga,
                        world_size=int(os.environ.get("WORLD_SIZE", "1") or 1))
                    logger.info("[UNIT] stage %d packed-basis step accounting: %s "
                                "| planner optimizer budget=%d (%d units)",
                                i, format_step_estimate(_est), s.steps, len(units))
                plan = []
                for j, u in enumerate(units, 1):
                    unit_end = prev_end + sum(alloc[:j])
                    plan.append({"j": j, "u": u, "end": unit_end,
                                 "steps": alloc[j - 1],
                                 "skip": alloc[j - 1] <= 0 or current_step >= unit_end})

                completions = self._load_unit_completions()
                for item in plan:
                    if item["skip"]:
                        continue
                    ukey = f"{item['u'].path}/{item['u'].name or 'default'}"
                    rec = completions.get(ukey)
                    if rec and rec.get("unit_fingerprint") == self._unit_fingerprint(item["u"]):
                        item["skip"] = True
                        logger.info("[UNIT] %d/%d %s — completion manifest verified "
                                    "(fingerprint match), skipping",
                                    item["j"], len(units), ukey)
                    elif rec:
                        logger.warning("[UNIT] %d/%d %s — completion manifest fingerprint "
                                       "MISMATCH — identity changed, will retrain",
                                       item["j"], len(units), ukey)

                journal = _FailureJournal(Path(self.cfg.output.model_dir),
                                          f"stage{i}")
                # Cull journal-failed units BEFORE scheduling: the prefetch
                # queue must never spawn a background build for a dataset the
                # failure journal already ruled out (previously the skip only
                # fired on the consumer side, after the worker had started).
                if getattr(staging, "skip_failed_units_on_resume", True):
                    for item in plan:
                        if item["skip"]:
                            continue
                        ukey = f"{item['u'].path}/{item['u'].name or 'default'}"
                        if journal.is_failed(ukey):
                            if self.data_pipeline.unit_cache_hit(item["u"], i, item["j"]):
                                logger.info("[UNIT] %d/%d %s — previously failed (%s) but "
                                            "packed cache is warm — scheduling from cache",
                                            item["j"], len(units), ukey,
                                            journal.reason(ukey))
                                journal.clear(ukey)
                                continue
                            item["skip"] = True
                            logger.warning("[UNIT] %d/%d %s — previously failed (%s), "
                                           "skipping per journal",
                                           item["j"], len(units), ukey,
                                           journal.reason(ukey))
                next_trainable = [p for p in plan if not p["skip"]]
                for item in plan:
                    if item["skip"]:
                        if item["steps"] <= 0:
                            logger.info("[UNIT] %d/%d %s — 0 steps allocated, skipping",
                                        item["j"], len(units),
                                        f"{item['u'].path}/{item['u'].name or 'default'}")
                        else:
                            logger.info("[UNIT] %d/%d %s — already completed (global_step=%d), skipping",
                                        item["j"], len(units),
                                        f"{item['u'].path}/{item['u'].name or 'default'}", current_step)

                stage_t0 = time.perf_counter()
                stage_stats = {"units": 0, "hits": 0, "build_sec": 0.0,
                               "train_sec": 0.0, "steps": 0, "tokens": 0,
                               "failed": 0, "empty": 0}
                gb = self._global_batch(stage_cfg)
                seq_len = int(getattr(self.cfg.training, "max_seq_length", 4096) or 4096)
                tokens_per_step = gb * seq_len

                # Memorization-risk visibility (logging only — scheduling is
                # untouched): a small packed dataset that receives many
                # optimizer steps is replayed implied_epochs times. Flag it so
                # the operator can distinguish low-loss-from-memorization from
                # healthy convergence instead of querying it by hand.
                if sizing_basis == "packed":
                    for _uj, (_cc, _aa) in enumerate(zip(cached_counts, alloc), 1):
                        if _cc is not None and _cc > 0 and _aa and _aa > 0:
                            implied_epochs = (_aa * gb) / float(_cc)
                            if implied_epochs > _HIGH_REPETITION_EPOCHS:
                                _uh = plan[_uj - 1]["u"]
                                logger.warning(
                                    "[UNIT] %d/%d %s/%s — %d steps x gb=%d vs %d packed seqs "
                                    "implies ~%.0f dataset repeats — loss on this unit runs "
                                    "down memorization territory (early epochs expected); "
                                    "logged for diagnosis, scheduling unchanged",
                                    _uj, len(units), _uh.path, _uh.name or "default",
                                    _aa, gb, _cc, implied_epochs)

                n_train = len(next_trainable)
                prefetch = None
                async_cfg = getattr(self.cfg.data, "async_pipeline", None)
                if n_train and _prefetch_enabled(staging, async_cfg):
                    # Bounded, ordered, multi-depth prefetch: up to
                    # prefetch_depth datasets are streamed/filtered/tokenized/
                    # packed on worker threads while the GPU trains. Unit 0 is
                    # also fed through the queue (its first get() waits for the
                    # build — nothing can overlap before the Trainer exists).
                    prefetch = UnitPrefetch(
                        build_fn=lambda item, idx, cancel_event=None: self.data_pipeline.build_pretrain_dataset_unit(
                            item["u"], i, item["j"], len(units),
                            cancel_event=cancel_event),
                        total=n_train,
                        depth=self._effective_prefetch_depth(staging, async_cfg),
                        timeout=float(getattr(staging, "prefetch_timeout", 3600.0) or 3600.0),
                        name=f"stage{i}",
                        retries=int(getattr(async_cfg, "retry_count", 0) or 0),
                        retry_backoff_base=float(
                            getattr(async_cfg, "retry_backoff_base", 1.0) or 1.0),
                        retry_backoff_max=float(
                            getattr(async_cfg, "retry_backoff_max", 30.0) or 30.0),
                        max_workers=getattr(async_cfg, "preprocess_workers", None),
                        cache_status=lambda item, idx: self.data_pipeline.unit_cache_hit(
                            item["u"], i, item["j"]),
                    )
                    prefetch.start(next_trainable)

                iter_k = 0
                try:
                    while iter_k < n_train:
                        k = iter_k
                        item = next_trainable[iter_k]
                        j = item["j"]
                        u = item["u"]
                        u_name = f"{u.path}/{u.name or 'default'}"
                        unit_end = item["end"]
                        cache_hit = self.data_pipeline.unit_cache_hit(u, i, j)
                        unit_key = f"{u.path}/{u.name or 'default'}"
                        self.telemetry.record("cache_attempts")
                        if cache_hit:
                            self.telemetry.record("cache_hits")

                        if (getattr(staging, "skip_failed_units_on_resume", True)
                                and journal.is_failed(unit_key)):
                            logger.warning("[UNIT] %d/%d %s — previously failed (%s), "
                                           "skipping per journal",
                                           j, len(units), u_name,
                                           journal.reason(unit_key))
                            if prefetch is not None:
                                prefetch.discard(iter_k)
                            iter_k += 1
                            continue

                        t_unit = time.perf_counter()
                        gpu_wait_sec = 0.0
                        prep_timing = {}
                        try:
                            if prefetch is not None:
                                res = prefetch.get(iter_k)
                                if isinstance(res, tuple) and len(res) == 3:
                                    res_payload, gpu_wait_sec, prep_timing = res
                                    if isinstance(res_payload, tuple):
                                        dataset, umeta = res_payload
                                    else:
                                        dataset, umeta = res_payload, {}
                                else:
                                    dataset, umeta = res, {}
                            else:
                                dataset, umeta = self.data_pipeline.build_pretrain_dataset_unit(
                                    u, i, j, len(units))
                            build_exc = None
                        except Exception as e:
                            dataset, umeta, build_exc = None, None, e
                        build_sec = time.perf_counter() - t_unit
                        logger.info("[ASYNC] GPU wait before dataset %d = %.2f sec", k + 1, gpu_wait_sec)
                        self.telemetry.record("gpu_wait_sec", gpu_wait_sec)

                        if build_exc is not None:
                            stage_stats["failed"] += 1
                            self.telemetry.record("unit_failures")
                            journal.mark(unit_key,
                                         f"{type(build_exc).__name__}: {build_exc}")
                            if isinstance(build_exc, PrefetchTimeout):
                                if prefetch is not None:
                                    prefetch.cancel(iter_k)
                                logger.error(
                                    "[UNIT TIMEOUT] %d/%d %s did not finish "
                                    "within %.0fs — build cancelled, marked "
                                    "failed, continuing", j, len(units), u_name,
                                    build_exc.waited)
                            else:
                                logger.error("[UNIT] %d/%d %s build failed: %s — "
                                             "marked failed, continuing",
                                             j, len(units), u_name, build_exc)
                            if staging.abort_on_unit_error:
                                logger.error("abort_on_unit_error — aborting staged pretraining")
                                return results
                            iter_k += 1
                            continue
                        if dataset is None or len(dataset) == 0:
                            stage_stats["empty"] += 1
                            self.telemetry.record("unit_failures")
                            journal.mark(unit_key, "empty dataset")
                            logger.error("Unit %d/%d (%s) produced an empty dataset — skipped",
                                         j, len(units), u_name)
                            if staging.abort_on_unit_error:
                                logger.error("abort_on_unit_error — aborting staged pretraining")
                                return results
                            iter_k += 1
                            continue

                        stage_stats["build_sec"] += build_sec
                        if cache_hit:
                            stage_stats["hits"] += 1
                            logger.info("[ASYNC] unit %d cache hit — ready immediately", k + 1)
                        ds_drv_stats = getattr(dataset, "_driver_stats", None)
                        if ds_drv_stats:
                            if ds_drv_stats.get("metadata_hits"):
                                self.telemetry.record("metadata_hits",
                                                      ds_drv_stats["metadata_hits"])
                            if ds_drv_stats.get("metadata_attempts"):
                                self.telemetry.record("metadata_attempts",
                                                      ds_drv_stats["metadata_attempts"])
                            if ds_drv_stats.get("builder_hits"):
                                self.telemetry.record("builder_hits",
                                                      ds_drv_stats["builder_hits"])
                            if ds_drv_stats.get("builder_attempts"):
                                self.telemetry.record("builder_attempts",
                                                      ds_drv_stats["builder_attempts"])
                        logger.info("[TIMER] unit %d/%d dataset ready: %.1fs, %d samples (%d steps)%s",
                                    j, len(units), time.perf_counter() - t_unit, len(dataset),
                                    item["steps"], " [cache hit]" if cache_hit else "")

                        if k == n_train - 1 and i < total_stages and _prefetch_enabled(
                                staging, async_cfg):
                            nxt_stage = stages[i]
                            nxt_units = [u for u in registry.all_entries()
                                         if (not nxt_stage.categories or u.category in nxt_stage.categories)]
                            if nxt_units:
                                n_warm = max(1, int(getattr(async_cfg, "metadata_workers", 1) or 1))
                                n_warm = min(n_warm, len(nxt_units))
                                per = (len(nxt_units) + n_warm - 1) // n_warm
                                for w_i in range(n_warm):
                                    chunk = nxt_units[w_i * per:(w_i + 1) * per]
                                    if not chunk:
                                        continue

                                    def _bg(nu=chunk, ns=i + 1, wi=w_i):
                                        try:
                                            self._warm_stage_metadata(nu, ns)
                                        except Exception as e:
                                            logger.warning(
                                                "[META] background warm for stage %d "
                                                "failed: %s", ns, e)
                                    threading.Thread(target=_bg, daemon=True,
                                                     name=f"stage-{i + 1}-warm-{w_i}").start()
                                logger.info("[META] stage %d metadata warming in background "
                                            "(%d worker(s), %d datasets — stage %d still training)",
                                            i + 1, n_warm, len(nxt_units), i)

                        if trainer is None:
                            trainer = self._build_trainer(
                                dataset, "pretrain", stage_cfg,
                                max_steps=total_steps,
                                save_only_model=False,
                                ignore_data_skip=True,
                            )
                            self._last_trainer = trainer
                        else:
                            trainer.train_dataset = dataset

                        resume_checkpoint = None
                        if not self._fresh_start:
                            self._flush_checkpoints()
                            latest = self._latest_pretrain_checkpoint()
                            if latest is not None:
                                resume_checkpoint = str(latest)
                                logger.info("[UNIT] resuming from checkpoint %s", resume_checkpoint)

                        boundary = StageBoundaryCallback(j, len(units), f"{name} — {u_name}", unit_end)
                        trainer.add_callback(boundary)
                        t_train = time.perf_counter()
                        start_global = getattr(getattr(trainer, "state", None), "global_step", 0)
                        logger.info("[ASYNC] dataset %d training start", k + 1)
                        if prefetch is not None:
                            try:
                                prefetch.note_training_state(k, "TRAINING")
                            except Exception:  # noqa: BLE001 — telemetry never breaks training
                                pass
                        try:
                            result = trainer.train(resume_from_checkpoint=resume_checkpoint)
                        finally:
                            trainer.remove_callback(boundary)
                        train_sec = time.perf_counter() - t_train
                        logger.info("[ASYNC] dataset %d training end", k + 1)
                        if prefetch is not None:
                            try:
                                prefetch.note_training_state(k, "TRAINED")
                            except Exception:  # noqa: BLE001 — telemetry never breaks training
                                pass
                        # A dataset that trained to completion clears any stale
                        # journal entry: a later resume must not skip it just
                        # because an earlier run died on it.
                        try:
                            journal.clear(unit_key)
                        except Exception:  # noqa: BLE001 — telemetry never breaks training
                            pass
                        metrics = result.metrics if hasattr(result, "metrics") else {}
                        stage_stats["train_sec"] += train_sec
                        tstate = getattr(trainer, "state", None)
                        end_global = getattr(tstate, "global_step", None)
                        if end_global is None:
                            # Trainer without a real HF state (test doubles) —
                            # fall back to the planned step allocation.
                            steps_taken = item["steps"]
                        else:
                            steps_taken = min(item["steps"], max(0, end_global - start_global))
                        stage_stats["steps"] += steps_taken
                        unit_tokens = steps_taken * tokens_per_step
                        stage_stats["tokens"] += unit_tokens
                        stage_stats["units"] += 1

                        tok_per_sec = unit_tokens / train_sec if train_sec > 0 else 0.0
                        steps_done = unit_end
                        pct = steps_done / total_steps if total_steps > 0 else 0.0
                        avg_sps = (stage_stats["steps"] / stage_stats["train_sec"]
                                   if stage_stats["train_sec"] > 0 else 0.0)
                        eta = (total_steps - steps_done) / avg_sps if avg_sps > 0 else float("inf")
                        bar_len = 20
                        filled = int(pct * bar_len)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        logger.info(
                            "[PROGRESS] stage %d/%d unit %d/%d | [%s] %3.0f%% | %d/%d steps | "
                            "ETA %s | %7.1f tok/s | build %.1fs | cache %s",
                            i, total_stages, k + 1, len(next_trainable), bar, pct * 100,
                            steps_done, total_steps, self._fmt_dur(eta), tok_per_sec,
                            build_sec, "hit" if cache_hit else "miss")
                        logger.info("[UNIT] %d/%d %s complete (steps %d..%d, %.1fs train): %s",
                                    j, len(units), u_name, prev_end, unit_end, train_sec, metrics)
                        self.telemetry.record("tokens", unit_tokens)
                        self.telemetry.record("samples", len(dataset))
                        self.telemetry.record("train_sec", train_sec)
                        self.telemetry.record("units")
                        self.telemetry.set_event("unit_ready_sec", round(build_sec, 1))
                        self.telemetry.set_event("cache_hit_now", int(cache_hit))
                        logger.info("[TELEMETRY] %s", self.telemetry.summary_line(
                            _telemetry_stage_tags(i, j, len(units), k, n_train)))
                        self.tracker.log_metrics({
                            f"pretrain/stage_{i}/unit_{j}/{k}": v for k, v in metrics.items()})
                        self.tracker.log_metrics({
                            f"pretrain/stage_{i}/unit_{j}/build_sec": round(build_sec, 2),
                            f"pretrain/stage_{i}/unit_{j}/train_sec": round(train_sec, 2),
                            f"pretrain/stage_{i}/unit_{j}/tokens_per_sec": round(tok_per_sec, 1),
                            f"pretrain/stage_{i}/unit_{j}/cache_hit": int(cache_hit),
                        })
                        results[f"stage_{i}/unit_{j}"] = metrics

                        self._flush_checkpoints()
                        latest_ckpt = self._latest_pretrain_checkpoint()
                        if latest_ckpt is not None:
                            self._write_unit_identity(
                                latest_ckpt, i, j, u, unit_key, unit_end, total_steps)
                        self._mark_unit_complete(
                            i, j, unit_key, self._unit_fingerprint(u), unit_end, total_steps)

                        iter_k += 1
                        del dataset
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                finally:
                    if prefetch is not None:
                        prefetch.close()
                        dup_prevented_total += prefetch.stats.get(
                            "duplicates_prevented", 0)
                        retries_total += prefetch.stats.get("retries", 0)
                        prefetch_stats.append(prefetch.stats)

                wall = time.perf_counter() - stage_t0
                busy = (stage_stats["train_sec"] / wall) if wall > 0 else 0.0
                hit_rate = (stage_stats["hits"] / stage_stats["units"]
                            if stage_stats["units"] > 0 else 0.0)
                stage_tok_s = stage_stats["tokens"] / wall if wall > 0 else 0.0
                logger.info(
                    "[STAGE SUMMARY] stage %d/%d %s — %d units, cache hit %d/%d (%.0f%%), "
                    "GPU busy %.0f%%, %.1f tok/s, %.2fM tokens, wall %s",
                    i, total_stages, name, stage_stats["units"], stage_stats["hits"],
                    stage_stats["units"], hit_rate * 100, busy * 100, stage_tok_s,
                    stage_stats["tokens"] / 1e6, self._fmt_dur(wall))
                continue

            t_stage = time.perf_counter()
            dataset, meta = self.data_pipeline.build_pretrain_stage_dataset(s, i, total_stages)
            if len(dataset) == 0:
                logger.error("Stage %d (%s) produced an empty dataset — aborting staged pretraining", i, name)
                return results
            logger.info("[TIMER] stage %d dataset ready: %.1fs, %d samples", i, time.perf_counter() - t_stage, len(dataset))

            if trainer is None:
                trainer = self._build_trainer(
                    dataset, "pretrain", stage_cfg,
                    max_steps=total_steps,
                    save_only_model=False,
                    ignore_data_skip=True,
                )
            else:
                trainer.train_dataset = dataset

            resume_checkpoint = None
            if current_step > 0 or i > 1:
                self._flush_checkpoints()
                latest = self._latest_pretrain_checkpoint()
                if latest is not None:
                    resume_checkpoint = str(latest)
                    logger.info("[STAGE] resuming from checkpoint %s", resume_checkpoint)
                else:
                    logger.warning(
                        "[STAGE] expected a checkpoint to resume from (stage %d) but none found — "
                        "optimizer continues in memory, scheduler restarts from zero", i)

            boundary = StageBoundaryCallback(i, total_stages, name, end_step)
            trainer.add_callback(boundary)
            try:
                result = trainer.train(resume_from_checkpoint=resume_checkpoint)
            finally:
                trainer.remove_callback(boundary)
            metrics = result.metrics if hasattr(result, "metrics") else {}
            logger.info("[STAGE] %d/%d %s complete: %s", i, total_stages, name, metrics)
            self.tracker.log_metrics({f"pretrain/stage_{i}/{k}": v for k, v in metrics.items()})
            results[f"stage_{i}"] = metrics

            del dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if trainer is not None:
            checkpoint_path = self._pretrain_output_dir()
            trainer.save_model(str(checkpoint_path))
            self.tokenizer.save_pretrained(str(checkpoint_path))
            self._save_checkpoint(checkpoint_path)
            self._flush_checkpoints()
            logger.info("Final pretrain checkpoint saved to %s", checkpoint_path)

        final_step = self._pretrain_global_step()
        gb = self._global_batch(stage_cfg, trainer)
        seq_len = int(getattr(self.cfg.training, "max_seq_length", 4096) or 4096)
        run_tokens = final_step * gb * seq_len
        logger.info("=" * 70)
        logger.info(
            "[SUMMARY] staged pretraining done: %d/%d steps (%.1f%%) | %.2fM tokens | "
            "global batch %d | wall %s",
            final_step, total_steps,
            (final_step / total_steps * 100) if total_steps > 0 else 0.0,
            run_tokens / 1e6, gb, self._fmt_dur(time.perf_counter() - run_t0))
        logger.info("=" * 70)

        # Final ASYNC PIPELINE REPORT
        tm_snap = self.telemetry.snapshot()
        gpu_wait = tm_snap.get("gpu_wait_sec", 0.0)
        train_time = tm_snap.get("train_sec", 0.0)
        total_time = train_time + gpu_wait
        idle_pct = (gpu_wait / total_time * 100.0) if total_time > 0 else 0.0
        hits = tm_snap.get("cache_hits", 0)
        attempts = tm_snap.get("cache_attempts", 0)
        fails = tm_snap.get("unit_failures", 0)
        units = tm_snap.get("units", 0)

        pf_hits = sum(s.get("prefetch_hits", 0) for s in prefetch_stats)
        pf_misses = sum(s.get("prefetch_misses", 0) for s in prefetch_stats)
        pf_attempts = pf_hits + pf_misses
        pf_hit_pct = (pf_hits / pf_attempts * 100.0) if pf_attempts > 0 else 0.0
        prep_sec = sum(s.get("total_prep_sec", 0.0) for s in prefetch_stats)
        built = sum(s.get("produced", 0) for s in prefetch_stats)
        built = max(built, 1)
        avg_prep = prep_sec / built
        avg_hidden = max(0.0, prep_sec - gpu_wait) / built

        logger.info("\n" + "=" * 60)
        logger.info("ASYNC PIPELINE REPORT")
        logger.info("=" * 60)
        logger.info("datasets scheduled: %d", attempts)
        logger.info("datasets trained: %d", units)
        logger.info("datasets skipped from processed cache: %d", hits)
        logger.info("failed datasets: %d | retried datasets: %d", fails, retries_total)
        timeouts_total = sum(s.get("timeouts", 0) for s in prefetch_stats)
        cancelled_total = sum(s.get("cancelled", 0) for s in prefetch_stats)
        logger.info("timed-out datasets: %d | built-then-cancelled (timeout/shutdown): %d",
                    timeouts_total, cancelled_total)
        logger.info("cache hits: %d | cache misses: %d", hits, max(0, attempts - hits))
        logger.info("metadata cache hits: %d | misses: %d",
                    tm_snap.get("metadata_hits", 0), tm_snap.get("metadata_attempts", 0) - tm_snap.get("metadata_hits", 0))
        logger.info("builder cache hits: %d | misses: %d",
                    tm_snap.get("builder_hits", 0), tm_snap.get("builder_attempts", 0) - tm_snap.get("builder_hits", 0))
        logger.info("prefetch hits: %d | prefetch misses: %d | prefetch hit rate: %.1f%%",
                    pf_hits, pf_misses, pf_hit_pct)
        logger.info("GPU training time: %.2f sec", train_time)
        logger.info("GPU wait time: %.2f sec", gpu_wait)
        logger.info("pipeline idle: %.1f%%", idle_pct)
        logger.info("average dataset preparation: %.2f sec", avg_prep)
        logger.info("average hidden preparation: %.2f sec", avg_hidden)
        logger.info("duplicate builds prevented: %d", dup_prevented_total)
        nonretryable_fails = sum(s.get("nonretryable", 0) for s in prefetch_stats)
        logger.info("non-retryable build failures: %d", nonretryable_fails)
        state_counts: Dict[str, int] = {}
        for _s in prefetch_stats:
            for _st, _n in (_s.get("state_counts") or {}).items():
                state_counts[_st] = state_counts.get(_st, 0) + _n
        if state_counts:
            logger.info("per-dataset final states: %s",
                        ", ".join(f"{k}={v}" for k, v in sorted(state_counts.items())))
        logger.info("=" * 60 + "\n")
        return results

    @staticmethod
    def _effective_prefetch_depth(staging: Any, async_cfg: Any) -> int:
        """Staging ``prefetch_depth`` capped by the global async-pipeline
        bounds ``ready_queue_size`` and ``max_inflight`` (all configurable; a
        missing async section leaves the staging depth unchanged)."""
        depth = max(1, int(getattr(staging, "prefetch_depth", 1) or 1))
        depth = min(depth, int(getattr(async_cfg, "ready_queue_size", depth) or depth))
        depth = min(depth, int(getattr(async_cfg, "max_inflight", depth) or depth))
        return max(1, depth)

    @staticmethod
    def _fmt_dur(secs: float) -> str:
        if not math.isfinite(secs):
            return "∞" if secs > 0 else "0s"
        secs = max(0.0, secs)
        h = int(secs // 3600)
        m = int(secs % 3600 // 60)
        s = int(secs % 60)
        if h:
            return f"{h}h{m:02d}m"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _global_batch(self, stage_cfg, trainer=None) -> int:
        gb = int(getattr(stage_cfg, "batch_size", 1) or 1)
        ga = int(getattr(stage_cfg, "gradient_accumulation_steps", 1) or 1)
        if trainer is not None and getattr(trainer, "args", None) is not None:
            args = trainer.args
            gb = int(getattr(args, "per_device_train_batch_size", gb) or gb)
            ga = int(getattr(args, "gradient_accumulation_steps", ga) or ga)
        ws = 1
        try:
            if torch.distributed.is_initialized():
                ws = torch.distributed.get_world_size()
        except Exception:
            pass
        return max(1, gb * ga * ws)

    def _warm_stage_metadata(self, units, stage_idx: int) -> None:
        """Resolve + cache the file list of every dataset in a stage in
        parallel (no data download). GPU-agnostic: safe to run on a background
        thread while the trainer trains the previous stage."""
        from concurrent.futures import ThreadPoolExecutor
        missing = [u for u in units if not self.data_pipeline.has_metadata_record(u)]
        if not missing:
            logger.info("[META] stage %d — all %d datasets already resolved (metadata cache warm)",
                        stage_idx, len(units))
            return
        logger.info("[META] stage %d — resolving metadata for %d datasets in parallel "
                    "(%d already cached)", stage_idx, len(missing), len(units) - len(missing))
        t0 = time.perf_counter()
        ok = 0
        try:
            with ThreadPoolExecutor(max_workers=min(8, len(missing))) as ex:
                for u, r in zip(missing, ex.map(self.data_pipeline.warm_metadata_cache, missing)):
                    if r:
                        ok += 1
                    else:
                        logger.warning("[META] could not resolve %s/%s — unit build will retry",
                                       u.path, u.name or "default")
        except Exception as e:
            logger.warning("[META] parallel metadata resolution failed: %s", e)
        logger.info("[TIMER] stage %d metadata resolution: %.1fs (%d/%d resolved)",
                    stage_idx, time.perf_counter() - t0, ok, len(missing))

    def _install_nan_loss_guard(self, trainer, max_consecutive_skips: int = 32):
        """Substitute a zero loss for non-finite losses so the offending
        micro-batch is skipped without corrupting weights. bf16 has no
        GradScaler skip, so a NaN loss propagates into the optimizer step and
        infects the weights; catching the loss up front means valid micro-batches
        in the same accumulation window still apply their gradients. Aborts only
        if the model is hopelessly corrupt (too many skips in a row).
        """
        orig = trainer.compute_loss
        stats = {"consecutive": 0, "total_skipped": 0}

        def guarded_compute_loss(self, model, inputs, *args, **kwargs):
            return_outputs = kwargs.get("return_outputs", False)
            if getattr(model, "_run_health_stash", None) is None and isinstance(inputs, dict) and "input_ids" in inputs:
                try:
                    # First batch's prefix feeds the held-out ppl + generation
                    # probes in _RunHealthCallback. Detached + CPU so it never
                    # participates in autograd or holds a GPU allocation.
                    model._run_health_stash = {
                        k: v.detach().cpu() for k, v in inputs.items()
                        if k in ("input_ids", "language_ids", "labels")
                    }
                except Exception:
                    model._run_health_stash = None
            out = orig(model, inputs, *args, **kwargs)
            if return_outputs:
                loss = out[0] if isinstance(out, tuple) else getattr(out, "loss", None)
            else:
                loss = out
            if loss is not None and not torch.isfinite(loss).all():
                stats["consecutive"] += 1
                stats["total_skipped"] += 1
                logger.warning(
                    "[NAN] non-finite loss on micro-batch — skipping (consecutive=%d, total=%d)",
                    stats["consecutive"], stats["total_skipped"],
                )
                if stats["consecutive"] >= max_consecutive_skips:
                    raise RuntimeError(
                        f"{stats['consecutive']} consecutive non-finite losses — aborting"
                    )
                zero = torch.tensor(
                    0.0, dtype=loss.dtype, device=loss.device, requires_grad=True
                )
                if return_outputs:
                    if isinstance(out, tuple):
                        return (zero, out[1])
                    out.loss = zero
                    return out
                return zero
            stats["consecutive"] = 0
            return out

        trainer.compute_loss = MethodType(guarded_compute_loss, trainer)
        return trainer

    def _build_trainer(
        self,
        dataset: Dataset,
        stage_name: str,
        stage_cfg: Any,
        **overrides,
    ) -> Trainer:
        output_dir = Path(self.cfg.output.model_dir) / stage_name

        lr = overrides.get("learning_rate", stage_cfg.learning_rate)
        bs = overrides.get("batch_size", stage_cfg.batch_size)
        gas = overrides.get("gradient_accumulation_steps", getattr(stage_cfg, "gradient_accumulation_steps", 1))
        max_steps = overrides.get("max_steps", stage_cfg.max_steps)
        warmup = overrides.get("warmup_steps", getattr(stage_cfg, "warmup_steps", 200))
        weight_decay = overrides.get("weight_decay", getattr(stage_cfg, "weight_decay", 0.05))
        max_grad_norm = overrides.get("max_grad_norm", getattr(stage_cfg, "max_grad_norm", 1.0))

        base_args = self.dist.get_training_args(str(output_dir), per_device_batch_size=bs,
                                            grad_accum_steps=gas)
        use_bf16 = base_args.get("bf16", False)
        use_fp16 = base_args.get("fp16", False)
        optim_name = self._resolve_optimizer_name(stage_cfg, base_args, overrides)

        num_workers = dataloader_num_workers(dataset)
        if stage_name == "pretrain":
            # Forking DataLoader worker processes while the async prefetch /
            # streaming threads are running is a fork()+threading.Lock deadlock:
            # workers inherit locks held by those threads (which don't exist in
            # the child), block forever on their very first fetch, and the
            # trainer silently pins at step 0 (observed: 0/50000 for hours,
            # main thread futex_wait_queue_me, workers born dead). Read
            # pretrain batches on the main thread instead, at the cost of some
            # prefetch head-room — correctness over peak throughput.
            num_workers = 0


        # Update FSDP transformer layer for MoE models
        model_type = self.cfg.model.architecture.model_type
        fsdp_config = base_args.get("fsdp_config", {})
        if base_args.get("fsdp") and isinstance(fsdp_config, dict):
            fsdp_config["transformer_layer_cls_to_wrap"] = [ARCH_FSDP_LAYER_MAP.get(model_type, "LlamaDecoderLayer")]
 
        eval_strategy = (
            self.cfg.training.eval_strategy
            if stage_name in ("sft", "instruction_tuning")
            else "no"
        )
        eval_steps = self.cfg.training.eval_steps if eval_strategy != "no" else None

        train_dataset_for_trainer = dataset
        eval_dataset = None
        if eval_strategy != "no":
            try:
                n = len(dataset)
                if n >= 10 and hasattr(dataset, "train_test_split"):
                    split = dataset.train_test_split(test_size=max(1, int(n * 0.05)), seed=42)
                    train_dataset_for_trainer = split["train"]
                    eval_dataset = split["test"]
                else:
                    eval_strategy = "no"
                    eval_steps = None
            except (AttributeError, TypeError, ValueError):
                eval_strategy = "no"
                eval_steps = None

        training_args = TrainingArguments(
           output_dir=str(output_dir),
           num_train_epochs=1,
           max_steps=max_steps,
           per_device_train_batch_size=bs,
           gradient_accumulation_steps=gas,
           learning_rate=lr,
           weight_decay=weight_decay,
           warmup_steps=warmup,
           max_grad_norm=max_grad_norm,
           logging_steps=self.cfg.training.logging_steps,
           save_steps=self.cfg.training.save_steps,
           save_total_limit=self.cfg.training.save_total_limit,
           eval_strategy=eval_strategy,
           eval_steps=eval_steps,
           save_strategy="steps",
           save_only_model=overrides.get("save_only_model", False),
           ddp_find_unused_parameters=False,
           fp16=use_fp16,
           bf16=use_bf16,
           optim=optim_name,
           lr_scheduler_type=getattr(stage_cfg, "lr_scheduler_type", "cosine"),
           report_to=self.cfg.output.experiment_tracking.provider if self.cfg.output.experiment_tracking.enabled else "none",
           remove_unused_columns=False,
           load_best_model_at_end=False,
           ignore_data_skip=overrides.get("ignore_data_skip", self.cfg.training.ignore_data_skip),
           dataloader_num_workers=num_workers,
           dataloader_pin_memory=True,
           torch_compile=False,
           gradient_checkpointing=self.cfg.model.architecture.gradient_checkpointing,
           **({"fsdp": base_args["fsdp"]} if base_args.get("fsdp") else {}),
           fsdp_config=fsdp_config,
           **({"deepspeed": base_args["deepspeed"]} if base_args.get("deepspeed") else {}),
        )

        data_collator = _LoggingDataCollator(_PretrainAuxCollator(DefaultDataCollator()))
        logging_cb = _LoggingCallback()
        health_cb = _RunHealthCallback(self.tokenizer, eval_every=500)
        callbacks = [
            logging_cb,
            _NaNSafeCallback(check_every=100),
            _TrainHeartbeatCallback(interval_steps=100, total_steps=max_steps or 0),
            health_cb,
        ]

        # transformers >= 5.0 removed the `tokenizer` kwarg from
        # Trainer.__init__ in favor of `processing_class`.
        tokenizer_kwarg = trainer_tokenizer_kwarg()

        trainer = self._install_nan_loss_guard(
            Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset_for_trainer,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                **{tokenizer_kwarg: self.tokenizer},
            )
        )
        # HF 5.x never forwards `model` to callback hooks; wire the raw model
        # directly so the diagnostics in _LoggingCallback / _NaNSafeCallback /
        # _RunHealthCallback actually observe it.  Unwrap any DataParallel-style
        # wrapper so the INNER module (which publishes _last_* attrs) is seen.
        _model_ref = getattr(trainer, "model", self.model)
        _model_ref = getattr(_model_ref, "module", _model_ref)
        for _cb in callbacks:
            setter = getattr(_cb, "set_model", None)
            if setter is not None:
                setter(_model_ref)
        _install_continuous_scheduler(trainer)
        return trainer

    def _resolve_optimizer_name(
        self,
        stage_cfg: Any,
        base_args: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> str:
        optim = overrides.get("optimizer", getattr(stage_cfg, "optimizer", "adamw_fused"))
        if optim == "adamw_fused":
            if not torch.cuda.is_available() or base_args.get("fsdp") or base_args.get("deepspeed"):
                return "adamw_torch"
        elif optim == "adamw_8bit" and not torch.cuda.is_available():
            logger.warning("adamw_8bit requires CUDA — falling back to adamw_torch")
            return "adamw_torch"
        optim_map = {
            "adamw": "adamw_torch",
            "adamw_8bit": "paged_adamw_8bit",
            "adamw_fused": "adamw_torch_fused",
            "sgd": "sgd",
        }
        candidate = optim_map.get(optim, "adamw_torch")
        # transformers/HF optimizer registry — fall back to adamw_torch when the
        # resolved name is unknown to the installed transformers version.
        try:
            from transformers.optimization import (
                OPTIMIZER_NAME_TO_CLASS as _optimizers,
            )
            if candidate in _optimizers:
                return candidate
        except Exception:
            pass
        try:
            from transformers.optimization import _name_to_optimizer_ctor as _ctors
            if candidate in _ctors:
                return candidate
        except Exception:
            pass
        return "adamw_torch"

    def staged_training(
        self,
        datasets_cfg: List[Any],
        total_steps: int,
    ) -> None:
        is_main = self.dist.is_main_process()
        steps_per_dataset = total_steps // max(len(datasets_cfg), 1)
        ckpt_step = 0

        for stage_idx, ds_entry in enumerate(datasets_cfg, 1):
            ds_name = ds_entry.path if hasattr(ds_entry, "path") else str(ds_entry)
            current_max = ckpt_step + steps_per_dataset
            logger.info("STAGE %d/%d: %s (steps %d-%d)", stage_idx, len(datasets_cfg), ds_name, ckpt_step, current_max)

            stage_cache = Path(self.cfg.data.cache_dir) / "stage_cache" / f"stage_{stage_idx}"
            skip_flag = Path(str(stage_cache) + "_skip")

            if is_main:
                if stage_cache.exists():
                    shutil.rmtree(stage_cache)
                if skip_flag.exists():
                    skip_flag.unlink()

                samples = list(self.data_pipeline.collector.stream_single_dataset(
                    ds_entry, limit=ds_entry.max_samples or DEFAULT_MAX_SAMPLES_PER_DATASET))
                if not samples:
                    skip_flag.touch()
                    logger.warning("No samples in stage %d. Skipping.", stage_idx)
                else:
                    ds = self.data_pipeline.build_stage_dataset(samples)
                    ds.save_to_disk(str(stage_cache))
                    logger.info("Stage %d: %d samples saved.", stage_idx, len(ds))
                    del ds, samples

            if self.dist.is_distributed and dist.is_initialized():
                dist.barrier()

            if skip_flag.exists():
                ckpt_step = current_max
                continue

            stage_dataset = load_from_disk(str(stage_cache))
            self._train_stage(stage_dataset, f"stage_{stage_idx}", self.cfg.training.sft, max_steps=current_max)
            ckpt_step = current_max

            if is_main:
                shutil.rmtree(stage_cache, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("All %d stages complete.", len(datasets_cfg))

    def _trainer_wrapped_model(self) -> Optional[Any]:
        """The HF Trainer's wrapped model (FSDP/DDP), if different from the
        raw model the pipeline holds."""
        trainer = getattr(self, "_last_trainer", None)
        if trainer is not None:
            model = getattr(trainer, "model", None)
            wrapped = getattr(trainer, "model_wrapped", None)
            candidate = wrapped if wrapped is not None and wrapped is not model else model
            if candidate is not None and candidate is not self.model:
                return candidate
        return None

    def _state_dict_for_save(self) -> Optional[Dict[str, Any]]:
        """Full model state dict for the main rank (None on other ranks).

        With FSDP full-shard, the raw model's state_dict holds only the
        rank-local shards — without a FULL_STATE_DICT gather every rank would
        write partial weights into the same path (last-writer-wins = corrupt
        checkpoint). Gather on rank 0, offload to CPU, and let non-main ranks
        skip the write entirely.
        """
        if self.dist.is_distributed and dist.is_initialized() and not self.dist.is_main_process():
            return None
        wrapped = self._trainer_wrapped_model()
        if wrapped is not None:
            try:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    FullStateDictConfig,
                    StateDictType,
                )
                if self.dist.is_distributed and isinstance(wrapped, FSDP):
                    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, cfg):
                        return wrapped.state_dict()
            except ImportError:
                pass
        return self.model.state_dict()

    def save_model(self, path: Optional[str | Path] = None) -> None:
        path = Path(path or self.cfg.output.model_dir)
        state = self._state_dict_for_save()
        if state is None:
            # Non-main ranks own only local shards; the main rank writes the
            # full checkpoint (HF trainer.save_model already gathers for us).
            return
        path.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path / "pytorch_model.bin"))
        arch = self.cfg.model.architecture
        config = {
            "model_type": arch.model_type,
            "architecture": arch.model_type,
            "vocab_size": len(self.tokenizer) if self.tokenizer is not None else arch.vocab_size,
            "hidden_size": arch.hidden_size,
            "max_position_embeddings": arch.max_position_embeddings,
            "rope_theta": arch.rope_theta,
            "nslt": arch.nslt.model_dump(mode="python") if arch.model_type == "nslt" else None,
            "dhara_v3": arch.dhara_v3.model_dump(mode="python") if arch.model_type == "dhara_v3" else None,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(path))
        logger.info("Model saved to %s", path)

    def _find_resume_checkpoint(self) -> Optional[Path]:
        model_dir = Path(self.cfg.output.model_dir)
        # 1) Root-level final save (weights present) — the most recent durable
        #    artifact and the canonical resume point.
        if (model_dir / "pytorch_model.bin").exists():
            return model_dir
        # 2) Newest model weights anywhere under the model dir — stage dirs,
        #    HF checkpoint-* subfolders, etc.
        candidates = sorted(
            model_dir.rglob("pytorch_model.bin"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].parent
        # 3) Config-only artifact (weights may still be written by the caller).
        if (model_dir / "config.json").exists():
            return model_dir
        # 4) Explicit checkpoint_dir fallback.
        checkpoint_dir = Path(self.cfg.output.checkpoint_dir)
        if checkpoint_dir.exists():
            candidates = sorted(
                checkpoint_dir.rglob("pytorch_model.bin"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0].parent
        return None

    def _save_checkpoint(self, path: Path) -> None:
        state = self._state_dict_for_save()
        if state is None:
            # Non-main ranks hold only FSDP shards — the main rank owns the
            # checkpoint (matches trainer.save_model's gather behavior, so the
            # last write to pytorch_model.bin is always the FULL state).
            return
        path.mkdir(parents=True, exist_ok=True)
        arch = self.cfg.model.architecture
        config = {
            "model_type": arch.model_type,
            "architecture": arch.model_type,
            "vocab_size": len(self.tokenizer) if self.tokenizer is not None else arch.vocab_size,
            "hidden_size": arch.hidden_size,
            "max_position_embeddings": arch.max_position_embeddings,
            "rope_theta": arch.rope_theta,
            "nslt": arch.nslt.model_dump(mode="python") if arch.model_type == "nslt" else None,
            "dhara_v3": arch.dhara_v3.model_dump(mode="python") if arch.model_type == "dhara_v3" else None,
        }

        def _write(target: Path) -> None:
            (target / "config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8")
            torch.save(state, str(target / "pytorch_model.bin"))

        if self._ckpt_writer is not None and self._ckpt_writer.submit(path, _write):
            logger.info("Checkpoint %s queued for async write", path)
            return
        _write(path)

    def _flush_checkpoints(self) -> None:
        """Durability boundary: block until queued async checkpoints are on
        disk. Called before any resume lookup / stage switch so the next
        train() always sees a complete checkpoint."""
        if self._ckpt_writer is not None:
            if not self._ckpt_writer.flush():
                logger.error("Async checkpoint flush failed — checkpoints may be incomplete")

    def cleanup(self) -> None:
        self.telemetry.finish()
        if self.data_pipeline is not None:
            self.data_pipeline.close()
        if self._ckpt_writer is not None:
            self._ckpt_writer.close()
        self.tracker.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
