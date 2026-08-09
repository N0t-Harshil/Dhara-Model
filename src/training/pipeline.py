from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import os
import random
import shutil
import threading
import time
from pathlib import Path
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
from src.training.asyncprefetch import UnitPrefetch
from src.training.checkpoint import AsyncCheckpointWriter
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


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


class _LoggingCallback(TrainerCallback):
    def __init__(self):
        self._step_start = 0.0

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


class _NaNSafeCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or not state.is_world_process_zero:
            return
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
        logger.info("Initializing Methos Class Model pipeline (fresh_start=%s, resume=%s)...", fresh_start, resume_checkpoint)
        self.tokenizer = ModelFactory.load_tokenizer(cfg=self.cfg)

        if fresh_start:
            logger.info("Creating model from scratch...")
            self.model = ModelFactory.create_model(self.cfg, self.tokenizer)
        else:
            resume_checkpoint = resume_checkpoint or self._find_resume_checkpoint()
            if resume_checkpoint is not None:
                ckpt_path = Path(resume_checkpoint)
                if ckpt_path.exists() and ModelFactory.is_compatible(self.cfg, self.tokenizer, ckpt_path):
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
        benchmark_results = runner.run_benchmarks(self.cfg.evaluation.benchmarks)

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
        raw = f"{ds_info.path}_{ds_info.max_samples}_{self.cfg.training.max_seq_length}_{self.cfg.model.architecture.vocab_size}"
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
                        filtered_datasets = self.data_pipeline.get_active_datasets_for_stage(stage)
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

        logger.info("Training complete.")
        return results

    def _train_stage(
        self,
        dataset: Dataset,
        stage_name: str,
        stage_cfg: Any,
        **overrides,
    ) -> Dict[str, float]:
        trainer = self._build_trainer(dataset, stage_name, stage_cfg, **overrides)

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
        if output_dir.exists():
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

        current_step = self._pretrain_global_step()
        results: Dict[str, Any] = {}
        trainer = None
        run_t0 = time.perf_counter()

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
                registry = build_registry()
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
                plan = []
                for j, u in enumerate(units, 1):
                    unit_end = prev_end + sum(alloc[:j])
                    plan.append({"j": j, "u": u, "end": unit_end,
                                 "steps": alloc[j - 1],
                                 "skip": alloc[j - 1] <= 0 or current_step >= unit_end})

                journal = _FailureJournal(Path(self.cfg.output.model_dir),
                                          f"stage{i}")
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

                n_train = len(next_trainable)
                prefetch = None
                if n_train and getattr(staging, "prefetch", True):
                    # Bounded, ordered, multi-depth prefetch: up to
                    # prefetch_depth datasets are streamed/filtered/tokenized/
                    # packed on worker threads while the GPU trains. Unit 0 is
                    # also fed through the queue (its first get() waits for the
                    # build — nothing can overlap before the Trainer exists).
                    prefetch = UnitPrefetch(
                        build_fn=lambda item, idx: self.data_pipeline.build_pretrain_dataset_unit(
                            item["u"], i, item["j"], len(units)),
                        total=n_train,
                        depth=max(1, int(getattr(staging, "prefetch_depth", 1) or 1)),
                        timeout=float(getattr(staging, "prefetch_timeout", 900.0) or 900.0),
                        name=f"stage{i}",
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
                        try:
                            dataset, umeta = prefetch.get(iter_k) if prefetch is not None \
                                else self.data_pipeline.build_pretrain_dataset_unit(
                                    u, i, j, len(units))
                            build_exc = None
                        except Exception as e:
                            dataset, umeta, build_exc = None, None, e
                        build_sec = time.perf_counter() - t_unit

                        if build_exc is not None:
                            stage_stats["failed"] += 1
                            self.telemetry.record("unit_failures")
                            journal.mark(unit_key,
                                         f"{type(build_exc).__name__}: {build_exc}")
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
                        logger.info("[TIMER] unit %d/%d dataset ready: %.1fs, %d samples (%d steps)%s",
                                    j, len(units), time.perf_counter() - t_unit, len(dataset),
                                    item["steps"], " [cache hit]" if cache_hit else "")

                        if k == n_train - 1 and i < total_stages and getattr(staging, "prefetch", True):
                            nxt_stage = stages[i]
                            nxt_units = [u for u in registry.all_entries()
                                         if (not nxt_stage.categories or u.category in nxt_stage.categories)]
                            if nxt_units:
                                def _bg(nu=nxt_units, ns=i + 1):
                                    try:
                                        self._warm_stage_metadata(nu, ns)
                                    except Exception as e:
                                        logger.warning("[META] background warm for stage %d failed: %s", ns, e)
                                threading.Thread(target=_bg, daemon=True,
                                                 name=f"stage-{i + 1}-warm").start()
                                logger.info("[META] stage %d metadata warming in background "
                                            "(stage %d still training)", i + 1, i)

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
                        self._flush_checkpoints()
                        latest = self._latest_pretrain_checkpoint()
                        if latest is not None:
                            resume_checkpoint = str(latest)
                            logger.info("[UNIT] resuming from checkpoint %s", resume_checkpoint)

                        boundary = StageBoundaryCallback(j, len(units), f"{name} — {u_name}", unit_end)
                        trainer.add_callback(boundary)
                        t_train = time.perf_counter()
                        try:
                            result = trainer.train(resume_from_checkpoint=resume_checkpoint)
                        finally:
                            trainer.remove_callback(boundary)
                        train_sec = time.perf_counter() - t_train
                        metrics = result.metrics if hasattr(result, "metrics") else {}
                        stage_stats["train_sec"] += train_sec
                        stage_stats["steps"] += item["steps"]
                        unit_tokens = item["steps"] * tokens_per_step
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
                            {"stage": i + 1, "unit": f"{j}/{len(units)}"}))
                        self.tracker.log_metrics({
                            f"pretrain/stage_{i}/unit_{j}/{k}": v for k, v in metrics.items()})
                        self.tracker.log_metrics({
                            f"pretrain/stage_{i}/unit_{j}/build_sec": round(build_sec, 2),
                            f"pretrain/stage_{i}/unit_{j}/train_sec": round(train_sec, 2),
                            f"pretrain/stage_{i}/unit_{j}/tokens_per_sec": round(tok_per_sec, 1),
                            f"pretrain/stage_{i}/unit_{j}/cache_hit": int(cache_hit),
                        })
                        results[f"stage_{i}/unit_{j}"] = metrics

                        iter_k += 1
                        del dataset
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                finally:
                    if prefetch is not None:
                        prefetch.close()

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
        return results

    @staticmethod
    def _fmt_dur(secs: float) -> str:
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
        optim = overrides.get("optimizer", getattr(stage_cfg, "optimizer", "adamw_fused"))

        def _has_fused_adamw() -> bool:
            if not torch.cuda.is_available():
                return False
            try:
                from transformers.trainer_utils import is_torch_fused_available
                return is_torch_fused_available()
            except ImportError:
                return hasattr(torch.optim, "AdamW") and torch.cuda.is_bf16_supported()

        optim_fused = "adamw_torch_fused" if _has_fused_adamw() else "adamw_torch"
        optim_map = {
            "adamw": "adamw_torch",
            "adamw_8bit": "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            "adamw_fused": optim_fused,
            "sgd": "sgd",
        }
        optim_name = optim_map.get(optim, "adamw_torch")

        is_iterable = isinstance(dataset, IterableDataset)
        num_workers = 0 if is_iterable else 8

        base_args = self.dist.get_training_args(str(output_dir))
        use_bf16 = base_args.get("bf16", False)
        use_fp16 = base_args.get("fp16", False)
        optim_name = self._resolve_optimizer_name(stage_cfg, base_args, overrides)

        is_iterable = isinstance(dataset, IterableDataset)
        num_workers = 0 if is_iterable else min(4, os.cpu_count() or 4)

        # Update FSDP transformer layer for MoE models
        model_type = self.cfg.model.architecture.model_type
        fsdp_config = base_args.get("fsdp_config", {})
        if isinstance(fsdp_config, dict):
            fsdp_config["transformer_layer_cls_to_wrap"] = [ARCH_FSDP_LAYER_MAP.get(model_type, "LlamaDecoderLayer")]

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
            eval_strategy=self.cfg.training.eval_strategy if stage_name in ("sft", "instruction_tuning") else "no",
            eval_steps=self.cfg.training.eval_steps if stage_name in ("sft", "instruction_tuning") else 0,
            save_strategy="steps",
            save_only_model=overrides.get("save_only_model", True),
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
            **({"fsdp": base_args["fsdp"]} if base_args.get("fsdp") else {}),
            **({"fsdp_config": fsdp_config} if fsdp_config else {}),
            **({"deepspeed": base_args["deepspeed"]} if base_args.get("deepspeed") else {}),
        )

        data_collator = _LoggingDataCollator(DefaultDataCollator())
        callbacks = [_LoggingCallback(), _NaNSafeCallback()]

        # transformers >= 5.0 removed the `tokenizer` kwarg from
        # Trainer.__init__ in favor of `processing_class`.
        tokenizer_kwarg = (
            "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
        )

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            **{tokenizer_kwarg: self.tokenizer},
        )

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
        optim_map = {
            "adamw": "adamw_torch",
            "adamw_8bit": "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            "adamw_fused": "adamw_torch_fused" if hasattr(torch.optim, "AdamW") else "adamw_torch",
            "sgd": "sgd",
        }
        return optim_map.get(optim, "adamw_torch")

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

    def save_model(self, path: Optional[str | Path] = None) -> None:
        path = Path(path or self.cfg.output.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path / "pytorch_model.bin"))
        arch = self.cfg.model.architecture
        config = {
            "model_type": arch.model_type,
            "architecture": arch.model_type,
            "vocab_size": len(self.tokenizer) if self.tokenizer is not None else arch.vocab_size,
            "hidden_size": arch.hidden_size,
            "max_position_embeddings": arch.max_position_embeddings,
            "rope_theta": arch.rope_theta,
            "nslt": arch.nslt.model_dump(mode="python") if arch.model_type == "nslt" else None,
            "methos_v3": arch.methos_v3.model_dump(mode="python") if arch.model_type == "methos_v3" else None,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(path))
        logger.info("Model saved to %s", path)

    def _find_resume_checkpoint(self) -> Optional[Path]:
        model_dir = Path(self.cfg.output.model_dir)
        if (model_dir / "pytorch_model.bin").exists() or (model_dir / "config.json").exists():
            return model_dir

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
            "methos_v3": arch.methos_v3.model_dump(mode="python") if arch.model_type == "methos_v3" else None,
        }
        state = self.model.state_dict()

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
