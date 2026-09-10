"""Tests for supervised training preprocessing."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset

from src.trainer import ModelTrainer
from src.training.pipeline import TrainingPipeline
from src.infrastructure.distributed import DistributedSetup
from src.config.schema import Config


class _TinyTokenizer:
    eos_token = "</s>"
    eos_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict:
        ids = [(ord(ch) % 97) + 3 for ch in text]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}


class TestTrainerPreprocessing(unittest.TestCase):
    def test_supervised_tokenization_masks_prompt_labels(self):
        model = MagicMock()
        model.tokenizer = _TinyTokenizer()
        trainer = ModelTrainer(model)
        trainer.cfg.training.max_seq_length = 32

        batch = trainer._tokenize_supervised_batch([("prompt:", "answer")])
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]

        prompt_len = len(model.tokenizer("prompt:")["input_ids"])
        self.assertTrue(all(label == -100 for label in labels[:prompt_len]))
        self.assertEqual(labels[prompt_len], input_ids[prompt_len])
        self.assertEqual(len(input_ids), 32)
        self.assertEqual(len(batch["attention_mask"][0]), 32)

    def test_constitutional_prompt_includes_alignment_block(self):
        model = MagicMock()
        model.tokenizer = _TinyTokenizer()
        trainer = ModelTrainer(model)

        prompt = trainer._format_prompt("python", "sort a list")

        self.assertIn("### Constitution", prompt)
        self.assertIn("### Alignment Note", prompt)
        self.assertIn("Follow the constitution above", prompt)

    def test_text_prompt_uses_general_assistant_format(self):
        model = MagicMock()
        model.tokenizer = _TinyTokenizer()
        trainer = ModelTrainer(model)

        prompt = trainer._format_prompt("text", "Explain gradient descent.")

        self.assertIn("Answer the following request", prompt)
        self.assertNotIn("Write a text solution", prompt)

    def test_resolves_adamw_fused_to_safe_optimizer_for_fsdp(self):
        cfg = Config()
        pipeline = TrainingPipeline(cfg)
        stage_cfg = cfg.training.pretrain
        base_args = {"fsdp": "full_shard auto_wrap"}

        optim_name = pipeline._resolve_optimizer_name(stage_cfg, base_args, {})

        self.assertEqual(optim_name, "adamw_torch")

    def test_pretrain_trainer_disables_eval_steps(self):
        cfg = Config()
        pipeline = TrainingPipeline(cfg)
        pipeline.model = torch.nn.Linear(4, 2)
        pipeline.tokenizer = MagicMock()
        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
            "labels": [[1, 2, 3, 4]],
        })

        trainer = pipeline._build_trainer(dataset, "pretrain", cfg.training.pretrain)

        self.assertEqual(trainer.args.eval_strategy, "no")
        self.assertIsNone(trainer.args.eval_steps)

    @patch("torch.cuda.is_available", return_value=False)
    def test_fsdp_cpu_does_not_force_bf16(self, cuda_available):
        cfg = Config()
        cfg.model.dtype = "bfloat16"
        cfg.distributed.strategy = "fsdp"
        dist_setup = DistributedSetup(cfg)
        args = dist_setup.get_training_args("out")

        self.assertFalse(args.get("bf16", False))
        self.assertFalse(args.get("fp16", False))


if __name__ == "__main__":
    unittest.main()
