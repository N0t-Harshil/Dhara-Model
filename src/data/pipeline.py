from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.config.schema import Config, DatasetEntryConfig
from src.data.quality import (
    ContaminationFilter,
    ExactDeduplicator,
    MinHashDeduplicator,
    QualityFilter,
    QualityScorer,
)
from src.data.streaming import MassiveDataCollector

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(
        self,
        cfg: Config,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dedup = MinHashDeduplicator(threshold=cfg.data.quality.deduplication.threshold)
        self.contamination = ContaminationFilter(benchmarks=cfg.data.quality.contamination.benchmarks)
        self.scorer = QualityScorer()
        self._collector: Optional[MassiveDataCollector] = None

    @property
    def collector(self) -> MassiveDataCollector:
        if self._collector is None:
            self._collector = MassiveDataCollector(self.cfg.data.datasets)
        return self._collector

    def format_prompt(
        self,
        language: str,
        problem: str,
        style: str = "standard",
        constitution: Optional[List[str]] = None,
    ) -> str:
        if language.strip().lower() == "text":
            prompt = f"### Instruction\nAnswer the following request:\n{problem}\n\n"
        else:
            prompt = f"### Instruction\nWrite a {language} solution for the following problem:\n{problem}\n\n"

        if style == "constitutional" and constitution:
            block = "\n".join(f"- {p}" for p in constitution if p.strip())
            prompt = f"### Constitution\n{block}\n\n{prompt}### Alignment Note\nFollow the constitution above.\n\n### Response\n"
        else:
            prompt += "### Response\n"

        return prompt

    def tokenize_supervised(
        self,
        texts: List[Tuple[str, str]],
        max_length: int,
    ) -> Dict[str, Any]:
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        for prompt, response in texts:
            full_text = f"{prompt}{response}{self.tokenizer.eos_token or ''}"
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=max_length,
            )["input_ids"]
            full_ids = self.tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length,
            )["input_ids"]

            attention_mask = [1] * len(full_ids)
            labels = list(full_ids)
            prompt_len = len(prompt_ids)
            labels[:prompt_len] = [-100] * prompt_len

            pad_len = max_length - len(full_ids)
            if pad_len > 0:
                full_ids.extend([pad_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                labels.extend([-100] * pad_len)

            batch_input_ids.append(full_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    def build_sft_dataset(
        self,
        data: List[Dict[str, str]],
        style: str = "standard",
        constitution: Optional[List[str]] = None,
        apply_quality_filter: bool = True,
        apply_dedup: bool = True,
        apply_contamination: bool = True,
    ) -> Dataset:
        max_len = self.cfg.training.max_seq_length
        pairs: List[Tuple[str, str]] = []

        for ex in data:
            lang = ex.get("language", "python")
            prob = ex.get("instruction") or ex.get("problem") or ""
            inp = ex.get("input", "")
            sol = ex.get("output") or ex.get("solution") or ""

            if not sol:
                continue
            if apply_quality_filter:
                content = f"{prob} {inp} {sol}"
                if not QualityFilter.check_length(content, self.cfg.data.quality.min_length, self.cfg.data.quality.max_length):
                    continue
                if not QualityFilter.is_high_quality_content(content):
                    continue
            if apply_contamination and self.contamination.is_contaminated(f"{prob} {sol}"):
                continue

            if inp:
                prob = f"{prob}\n{inp}"
            prompt = self.format_prompt(lang, prob, style, constitution)
            pairs.append((prompt, sol))

        if apply_dedup:
            unique: List[Tuple[str, str]] = []
            for p, r in pairs:
                if not self.dedup.is_duplicate(r):
                    unique.append((p, r))
            pairs = unique

        if not pairs:
            logger.warning("No valid samples after filtering!")
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

        ds = Dataset.from_dict({"idx": list(range(len(pairs)))})
        ds = ds.map(
            lambda examples: self.tokenize_supervised(
                [(pairs[i][0], pairs[i][1]) for i in examples["idx"]],
                max_length=max_len,
            ),
            batched=True,
            remove_columns=["idx"],
        )
        logger.info("Built SFT dataset: %d samples", len(ds))
        return ds

    def build_preference_dataset(
        self,
        data: List[Dict[str, Any]],
    ) -> Dataset:
        max_len = self.cfg.training.max_seq_length
        chosen_list, rejected_list = [], []

        for ex in data:
            chosen = ex.get("chosen") or ex.get("response") or ex.get("output") or ""
            rejected = ex.get("rejected") or ex.get("response_rejected") or ""
            if not chosen or not rejected:
                continue
            prompt_text = ex.get("prompt") or ex.get("instruction") or ""
            chosen_list.append((prompt_text, chosen))
            rejected_list.append((prompt_text, rejected))

        if not chosen_list:
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

        chosen_ds = Dataset.from_dict({"idx": list(range(len(chosen_list)))})
        chosen_ds = chosen_ds.map(
            lambda ex: self.tokenize_supervised(
                [(chosen_list[i][0], chosen_list[i][1]) for i in ex["idx"]],
                max_len,
            ),
            batched=True, remove_columns=["idx"],
        )

        rejected_ds = Dataset.from_dict({"idx": list(range(len(rejected_list)))})
        rejected_ds = rejected_ds.map(
            lambda ex: self.tokenize_supervised(
                [(rejected_list[i][0], rejected_list[i][1]) for i in ex["idx"]],
                max_len,
            ),
            batched=True, remove_columns=["idx"],
        )

        combined = Dataset.from_dict({
            "chosen_input_ids": chosen_ds["input_ids"],
            "chosen_attention_mask": chosen_ds["attention_mask"],
            "chosen_labels": chosen_ds["labels"],
            "rejected_input_ids": rejected_ds["input_ids"],
            "rejected_attention_mask": rejected_ds["attention_mask"],
            "rejected_labels": rejected_ds["labels"],
        })
        logger.info("Built preference dataset: %d pairs", len(combined))
        return combined

    def build_stage_dataset(
        self,
        raw_samples: List[Dict[str, str]],
    ) -> Dataset:
        if not raw_samples:
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
        ds = Dataset.from_list(raw_samples)
        ds = ds.map(
            self._preprocess_batch,
            batched=True,
            batch_size=256,
            remove_columns=[c for c in ["instruction", "input", "output", "language"] if c in ds.column_names],
            desc="Tokenizing stage dataset",
        )
        return ds

    def _preprocess_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        pairs: List[Tuple[str, str]] = []
        for i in range(len(examples.get("output", []))):
            lang = str(examples.get("language", ["python"] * len(examples["output"]))[i] or "python")
            problem = str(examples.get("instruction", [""] * len(examples["output"]))[i] or "")
            inp = str(examples.get("input", [""] * len(examples["output"]))[i] or "")
            sol = str(examples["output"][i] or "")
            if inp:
                problem = f"{problem}\n{inp}"
            prompt = self.format_prompt(lang, problem, self.cfg.alignment.prompt_style, self.cfg.alignment.constitution)
            pairs.append((prompt, sol))
        if not pairs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return self.tokenize_supervised(pairs, self.cfg.training.max_seq_length)
