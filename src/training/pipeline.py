from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.data.pipeline import DataPipeline
from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.reporting import EvaluationReport
from src.evaluation.safety import SafetyEvaluator
from src.infrastructure.distributed import DistributedSetup
from src.infrastructure.tracking import ExperimentTracker
from src.models.factory import ARCH_FSDP_LAYER_MAP, ModelFactory
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


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


class _LoggingDataCollator:
    def __init__(self, collator):
        self.collator = collator
        self._logged = False

    def __call__(self, features):
        batch = self.collator(features)
        if not self._logged:
            self._logged = True
            if "input_ids" in batch:
                logger.info("Batch input_ids shape: %s", batch["input_ids"].shape)
        return batch


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
                        samples = list(self.data_pipeline.collector.stream_single_dataset(ds_info))
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
            logger.info("Phase %s complete: %d datasets OK, %d skipped", stage_name.upper(), valid_count, skipped_count)

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

        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        metrics = result.metrics if hasattr(result, "metrics") else {}
        logger.info("%s complete: %s", stage_name.upper(), metrics)
        self.tracker.log_metrics({f"{stage_name}/{k}": v for k, v in metrics.items()})

        ckpt_dir = Path(self.cfg.output.model_dir)
        checkpoint_path = ckpt_dir / stage_name
        trainer.save_model(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        self._save_checkpoint(checkpoint_path)
        logger.info("%s checkpoint saved to %s", stage_name, checkpoint_path)
        return metrics

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
            logging_steps=self.cfg.training.logging_steps,
            save_steps=self.cfg.training.save_steps,
            save_total_limit=self.cfg.training.save_total_limit,
            eval_strategy=self.cfg.training.eval_strategy if stage_name in ("sft", "instruction_tuning") else "no",
            eval_steps=self.cfg.training.eval_steps if stage_name in ("sft", "instruction_tuning") else 0,
            save_strategy="steps",
            save_only_model=True,
            ddp_find_unused_parameters=False,
            fp16=use_fp16,
            bf16=use_bf16,
            optim=optim_name,
            lr_scheduler_type=getattr(stage_cfg, "lr_scheduler_type", "cosine"),
            report_to=self.cfg.output.experiment_tracking.provider if self.cfg.output.experiment_tracking.enabled else "none",
            remove_unused_columns=False,
            load_best_model_at_end=False,
            ignore_data_skip=self.cfg.training.ignore_data_skip,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            torch_compile=False,
            **({"fsdp": base_args["fsdp"]} if base_args.get("fsdp") else {}),
            **({"fsdp_config": fsdp_config} if fsdp_config else {}),
            **({"deepspeed": base_args["deepspeed"]} if base_args.get("deepspeed") else {}),
        )

        data_collator = _LoggingDataCollator(DefaultDataCollator())
        callbacks = [_LoggingCallback()]

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=self.tokenizer,
        )

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

                samples = list(self.data_pipeline.collector.stream_single_dataset(ds_entry))
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
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save(self.model.state_dict(), str(path / "pytorch_model.bin"))

    def cleanup(self) -> None:
        self.tracker.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
