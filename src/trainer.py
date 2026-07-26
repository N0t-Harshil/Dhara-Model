from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, IterableDataset
from transformers import DefaultDataCollator, Trainer, TrainerCallback, TrainingArguments

from src.config.schema import Config, load_config
from src.data.pipeline import DataPipeline

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class _LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            step = state.global_step
            loss = logs.get("loss", logs.get("train_loss"))
            lr = logs.get("learning_rate")
            if loss is None:
                return
            logger.info("Step %d | loss=%.4f | lr=%.2e", step, float(loss), float(lr) if lr is not None else 0)


class ModelTrainer:
    def __init__(
        self,
        model: Any,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
        prompt_style: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cfg: Config = self._resolve_config(config_path)
        self.prompt_style: str = (
            prompt_style
            or self.cfg.alignment.prompt_style
            or "standard"
        )
        self._train_dataset: Optional[Dataset] = None
        self._eval_dataset: Optional[Dataset] = None
        self._trainer: Optional[Trainer] = None
        self._data_pipeline: Optional[DataPipeline] = None

    def _resolve_config(self, path: str | Path) -> Config:
        try:
            return load_config(path)
        except Exception:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            return Config.model_validate(raw)

    @property
    def _unwrap_model(self):
        if hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
            return self.model.model
        return self.model

    @property
    def data_pipeline(self) -> DataPipeline:
        if self._data_pipeline is None:
            self._data_pipeline = DataPipeline(self.cfg, getattr(self.model, "tokenizer", None))
        return self._data_pipeline

    def _constitution_block(self) -> str:
        principles = self.cfg.alignment.constitution or []
        lines = [f"- {p.strip()}" for p in principles if p.strip()]
        return "### Constitution\n" + "\n".join(lines) + "\n\n" if lines else ""

    def _format_prompt(self, language: str, problem: str) -> str:
        return self.data_pipeline.format_prompt(
            language, problem,
            self.prompt_style,
            self.cfg.alignment.constitution if self.prompt_style == "constitutional" else None,
        )

    def _tokenize_supervised_batch(self, texts: List[Tuple[str, str]]) -> Dict[str, Any]:
        return self.data_pipeline.tokenize_supervised(texts, self.cfg.training.max_seq_length)

    def _examples_to_prompt_response(self, examples: Dict[str, List[Any]]) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        if "text" in examples:
            for text in examples["text"]:
                text = str(text)
                marker = "### Response\n"
                if marker in text:
                    prompt, response = text.split(marker, 1)
                    rows.append((f"{prompt}{marker}", response))
                else:
                    rows.append(("", text))
            return rows
        output_count = len(examples.get("output", []))
        for i in range(output_count):
            languages = examples.get("language")
            language = str(languages[i] if languages and i < len(languages) else "python")
            instructions = examples.get("instruction")
            problem = str(instructions[i] if instructions and i < len(instructions) else "")
            inputs = examples.get("input")
            input_text = str(inputs[i] if inputs and i < len(inputs) else "")
            solution = str(examples["output"][i] or "")
            if not solution:
                logger.warning("Skipping example %d: empty output/solution", i)
                continue
            if input_text:
                problem = f"{problem}\n{input_text}"
            rows.append((self._format_prompt(language, problem), solution))
        return rows

    def prepare_data(self, dataset_file: str | Path, eval_split: float = 0.15) -> None:
        import json
        dataset_file = Path(dataset_file)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")
        raw = json.loads(dataset_file.read_text(encoding="utf-8"))
        logger.info("Loaded %d raw examples from %s", len(raw), dataset_file)

        pairs: List[Tuple[str, str]] = []
        for ex in raw:
            lang = ex.get("language", "python")
            prob = ex.get("problem") or ex.get("instruction", "")
            inp = ex.get("input", "")
            sol = ex.get("solution") or ex.get("output", "")
            if inp:
                prob = f"{prob}\n{inp}"
            pairs.append((self._format_prompt(lang, prob), sol))

        def tokenize(examples: Dict[str, List[int]]) -> Dict[str, Any]:
            batch_pairs = [(pairs[idx][0], pairs[idx][1]) for idx in examples["idx"]]
            return self._tokenize_supervised_batch(batch_pairs)

        ds = Dataset.from_dict({"idx": list(range(len(pairs)))})
        ds = ds.map(tokenize, batched=True, remove_columns=["idx"])
        split = ds.train_test_split(test_size=eval_split, seed=42)
        self._train_dataset = split["train"]
        self._eval_dataset = split["test"]
        logger.info("Prepared data: %d train, %d eval", len(self._train_dataset), len(self._eval_dataset))

    def _preprocess_function(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        pairs = self._examples_to_prompt_response(examples)
        if not pairs:
            logger.warning("No standard examples found in batch; returning empty tokenization.")
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return self._tokenize_supervised_batch(pairs)

    def train(
        self,
        streaming_collector: Optional[Any] = None,
        max_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        theme: str = "all",
        stages: Optional[list] = None,
        resume_from_checkpoint: Optional[bool | str] = None,
        _prebuilt_dataset: Optional[Any] = None,
    ) -> Dict[str, float]:
        if callable(getattr(self.model, "is_ready", False)):
            if not self.model.is_ready:
                raise RuntimeError("Model is not loaded.")
        train_dataset = self._train_dataset
        eval_dataset = self._eval_dataset

        if _prebuilt_dataset is not None:
            cols = _prebuilt_dataset.column_names if hasattr(_prebuilt_dataset, "column_names") else []
            if "input_ids" in cols:
                train_dataset = _prebuilt_dataset
            else:
                logger.info("Tokenizing pre-built dataset...")
                raw_cols = [c for c in ["instruction", "input", "output", "language"] if c in cols]
                map_kwargs = dict(batched=True, remove_columns=raw_cols) if raw_cols else dict(batched=True)
                train_dataset = _prebuilt_dataset.map(self._preprocess_function, **map_kwargs)
            if max_steps is None:
                max_steps = 1000

        elif streaming_collector:
            if stages:
                train_iterable = IterableDataset.from_generator(
                    streaming_collector.stream_samples, gen_kwargs={"stages": stages}
                )
            else:
                train_iterable = IterableDataset.from_generator(
                    streaming_collector.stream_samples, gen_kwargs={"theme": theme}
                )
            train_dataset = train_iterable.map(
                self._preprocess_function, batched=True,
                remove_columns=["instruction", "input", "output", "language"],
            )
            if max_steps is None:
                max_steps = 1000

        if train_dataset is None:
            raise RuntimeError("No training dataset. Call prepare_data() or provide a streaming_collector.")

        tcfg = self.cfg.training
        _epochs = epochs or 1
        _bs = batch_size or tcfg.sft.batch_size or 4
        _out = str(output_dir or self.cfg.output.model_dir)

        use_bf16 = self.cfg.model.dtype == "bfloat16" and torch.cuda.is_available()
        use_fp16 = self.cfg.model.dtype == "float16" and torch.cuda.is_available() and not use_bf16

        is_iterable = isinstance(train_dataset, IterableDataset)
        num_workers = 0 if is_iterable else min(4, os.cpu_count() or 4)

        fsdp_cfg = self.cfg.distributed.fsdp
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_fsdp = self.cfg.distributed.strategy == "fsdp" and world_size > 1

        fsdp_arg = ""
        fsdp_config_arg = {}
        if use_fsdp:
            fsdp_arg = "full_shard auto_wrap"
            fsdp_config_arg = {
                "transformer_layer_cls_to_wrap": [fsdp_cfg.transformer_layer_cls],
                "backward_prefetch": fsdp_cfg.backward_prefetch,
                "forward_prefetch": fsdp_cfg.forward_prefetch,
                "activation_checkpointing": True,
                "use_orig_params": True,
                "sync_module_states": True,
                "limit_all_gathers": True,
                "fsdp_mixed_precision": fsdp_cfg.mixed_precision,
            }
            logger.info("FSDP enabled: %d processes", world_size)

        training_args = TrainingArguments(
            output_dir=_out,
            num_train_epochs=_epochs,
            max_steps=max_steps if max_steps else -1,
            per_device_train_batch_size=_bs,
            per_device_eval_batch_size=_bs,
            gradient_accumulation_steps=tcfg.sft.gradient_accumulation_steps,
            learning_rate=tcfg.sft.learning_rate,
            weight_decay=tcfg.sft.weight_decay,
            warmup_steps=tcfg.sft.warmup_steps,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            eval_steps=tcfg.eval_steps,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            save_total_limit=tcfg.save_total_limit,
            save_only_model=True,
            gradient_checkpointing=not use_fsdp,
            ddp_find_unused_parameters=False,
            fp16=use_fp16,
            bf16=use_bf16,
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            lr_scheduler_type=tcfg.sft.lr_scheduler_type,
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=False if use_fsdp else bool(eval_dataset),
            metric_for_best_model="eval_loss" if (eval_dataset and not use_fsdp) else None,
            ignore_data_skip=self.cfg.training.ignore_data_skip,
            fsdp=fsdp_arg if fsdp_arg else None,
            fsdp_config=fsdp_config_arg if fsdp_config_arg else None,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
        )

        self._trainer = Trainer(
            model=self._unwrap_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DefaultDataCollator(),
            callbacks=[_LoggingCallback()],
            tokenizer=getattr(self.model, "tokenizer", self._data_pipeline.tokenizer if self._data_pipeline else None),
        )

        result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = result.metrics
        logger.info("Training complete: %s", metrics)
        return metrics

    def evaluate(self) -> Dict[str, float]:
        import math
        if self._trainer is None:
            raise RuntimeError("Call train() before evaluate().")
        if self._eval_dataset is None:
            raise RuntimeError("No evaluation dataset available.")
        metrics = self._trainer.evaluate()
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def save_checkpoint(self, path: Optional[str | Path] = None) -> None:
        save_dir = Path(path or self.cfg.output.checkpoint_dir)
        if hasattr(self.model, "save_model"):
            self.model.save_model(save_dir)
        else:
            self._unwrap_model.save_pretrained(save_dir)
        logger.info("Checkpoint saved to %s", save_dir)
