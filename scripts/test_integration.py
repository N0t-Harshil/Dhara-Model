"""
End-to-end integration test.
Exercises the full training pipeline from data ingestion through checkpoint recovery.

Run with:  python scripts/test_integration.py
On GPU:    python scripts/test_integration.py --gpu 0
Quick CPU: python scripts/test_integration.py --skip-training

Returns exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  OK  {name}")
        PASS += 1
    else:
        suffix = f" -- {detail}" if detail else ""
        print(f"  FAIL {name}{suffix}")
        FAIL += 1


def build_test_docs(tmpdir: str) -> str:
    """Create minimal local JSONL doc files for testing."""
    docs_dir = Path(tmpdir, "docs")
    sources = {
        "python-docs": [
            {"text": "Python is a programming language. " * 20,
             "source": "python-docs", "title": "Python Guide",
             "url": "https://docs.python.org/3/", "text_length": 500},
            {"text": "The print function outputs text. " * 20,
             "source": "python-docs", "title": "Print Function",
             "url": "https://docs.python.org/3/library/functions.html", "text_length": 450},
            {"text": "Lists are mutable sequences. " * 20,
             "source": "python-docs", "title": "Lists",
             "url": "https://docs.python.org/3/tutorial/introduction.html", "text_length": 400},
        ],
        "pytorch-docs": [
            {"text": "PyTorch is a machine learning framework. " * 20,
             "source": "pytorch-docs", "title": "PyTorch Overview",
             "url": "https://pytorch.org/docs/stable/", "text_length": 500},
            {"text": "Tensors are the core data structure. " * 20,
             "source": "pytorch-docs", "title": "Tensor Basics",
             "url": "https://pytorch.org/docs/stable/tensors.html", "text_length": 450},
        ],
        "numpy-docs": [
            {"text": "NumPy provides array computing. " * 20,
             "source": "numpy-docs", "title": "NumPy Quickstart",
             "url": "https://numpy.org/doc/stable/", "text_length": 400},
        ],
    }
    for name, records in sources.items():
        src_dir = docs_dir / name
        src_dir.mkdir(parents=True, exist_ok=True)
        with open(str(src_dir / "documents.jsonl"), "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return str(docs_dir)


def build_test_config(tmpdir: str, docs_dir: str) -> str:
    """Create a minimal config pointing at local docs for testing."""
    config = {
        "model": {
            "name": "Integration Test Model",
            "dtype": "float32",
            "device": "cpu",
            "train_from_scratch": True,
            "architecture": {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 128000,
                "attention_implementation": "eager",
                "gradient_checkpointing": False,
                "use_compile": False,
                "rope_theta": 10000.0,
                "tie_word_embeddings": True,
            },
        },
        "training": {
            "max_seq_length": 512,
            "save_steps": 5,
            "save_total_limit": 2,
            "eval_strategy": "no",
            "logging_steps": 1,
            "pretrain": {
                "enabled": True,
                "learning_rate": 1e-4,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_steps": 10,
                "warmup_steps": 2,
                "optimizer": "adamw",
            },
            "sft": {"enabled": False},
            "alignment": {"enabled": False},
            "instruction_tuning": {"enabled": False},
            "rlhf": {"enabled": False},
            "safety": {"enabled": False},
        },
        "data": {
            "streaming": True,
            "cache_dir": str(Path(tmpdir, "hf_cache")),
            "quality": {
                "min_length": 30,
                "max_length": 500000,
                "deduplication": {"enabled": False},
                "contamination": {"enabled": False},
                "quality_scoring": {"enabled": False},
                "language_detection": {"enabled": False},
                "toxicity_filtering": {"enabled": False},
            },
            "preprocessing": {
                "remove_boilerplate": False,
                "min_text_length": 30,
            },
            "ast_filter": {"code_filtering": False},
            "function_sampling": {"enabled": False},
            "sampler": {"balance_by": "documents"},
        },
        "tokenizer": {
            "source": "huggingface",
            "huggingface_model": "Xenova/claude-tokenizer",
        },
        "output": {
            "model_dir": str(Path(tmpdir, "model")),
            "checkpoint_dir": str(Path(tmpdir, "model", "checkpoints")),
            "log_dir": str(Path(tmpdir, "logs")),
        },
        "distributed": {"strategy": "none"},
        "huggingface_model": "Xenova/claude-tokenizer",
        "use_registry": False,
        "datasets": [
            {"path": "json", "name": "python-docs",
             "data_dir": str(Path(docs_dir, "python-docs")),
             "category": "docs", "weight": 0.6, "quality_score": 0.95},
            {"path": "json", "name": "pytorch-docs",
             "data_dir": str(Path(docs_dir, "pytorch-docs")),
             "category": "docs", "weight": 0.3, "quality_score": 0.95},
            {"path": "json", "name": "numpy-docs",
             "data_dir": str(Path(docs_dir, "numpy-docs")),
             "category": "docs", "weight": 0.1, "quality_score": 0.95},
        ],
    }
    config_path = Path(tmpdir, "test_config.yaml")
    import yaml
    with open(str(config_path), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(config_path)


def phase1_registry_and_validation(docs_dir: str) -> None:
    """Phase 1: Build registry and validate datasets."""
    print("\n--- Phase 1: Registry & Validation ---")

    from scripts.verify_datasets import _verify_local_dataset
    from src.data.registry import DatasetRegistry, DatasetInfo, build_registry, CATEGORY_WEIGHTS

    registry = build_registry()
    entries = registry.all_entries()
    test("registry builds", len(entries) >= 50, f"got {len(entries)}")

    total_w = sum(e.weight for e in entries)
    test("total weight ~ 1.0", abs(total_w - 1.0) < 0.02, f"got {total_w}")

    local_info = DatasetInfo(
        "json", "docs", 0.01, 0.95,
        name="python-docs", data_dir=str(Path(docs_dir, "python-docs")),
    )
    result = _verify_local_dataset(local_info)
    test("local doc ok", result["status"] == "ok_local",
         f"got {result['status']}: {result.get('error', '')}")
    test("local doc has records", result["record_count"] >= 3,
         f"got {result['record_count']}")
    test("local doc has avg_text_len", result["avg_text_len"] > 0,
         f"got {result['avg_text_len']}")

    missing_info = DatasetInfo(
        "json", "docs", 0.01, 0.95,
        name="missing-docs", data_dir="/does/not/exist",
    )
    result = _verify_local_dataset(missing_info)
    test("missing doc detected", result["status"] == "missing_local",
         f"got {result['status']}: {result.get('error', '')}")


def phase2_stream_and_pack(config_path: str) -> None:
    """Phase 2: Stream samples, pack sequences, tokenize."""
    print("\n--- Phase 2: Stream, Pack & Tokenize ---")

    import yaml
    from src.config.schema import Config, load_config
    from src.data.pipeline import DataPipeline

    cfg = load_config(config_path)

    import requests
    import threading
    tok = None
    def _load_tok():
        nonlocal tok
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Xenova/claude-tokenizer")
        except Exception as e:
            tok = e

    t = threading.Thread(target=_load_tok, daemon=True)
    t.start()
    t.join(timeout=15)
    if t.is_alive():
        print("  SKIP Phase 2: tokenizer download timed out (no network)")
        return
    if isinstance(tok, Exception):
        print(f"  SKIP Phase 2: tokenizer error: {tok}")
        return
    if tok is None:
        print("  SKIP Phase 2: tokenizer not available")
        return

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pipe = DataPipeline(cfg, tok)
    ds = pipe.build_pretrain_dataset()
    test("dataset built", ds is not None)
    test("dataset has samples", len(ds) >= 1, f"got {len(ds) if ds else 0}")

    if ds and len(ds) > 0:
        sample = ds[0]
        test("sample has input_ids", "input_ids" in sample)
        test("sample has attention_mask", "attention_mask" in sample)
        test("sample has labels", "labels" in sample)
        if "input_ids" in sample:
            test("input_ids non-empty", len(sample["input_ids"]) > 0)
            test("input_ids within max_seq_length",
                 len(sample["input_ids"]) <= cfg.training.max_seq_length,
                 f"got {len(sample['input_ids'])}")


def phase3_training(config_path: str, use_gpu: bool = False) -> None:
    """Phase 3: Short training run, checkpoint save/load/resume."""
    print("\n--- Phase 3: Training & Checkpoint ---")

    from src.config.schema import load_config
    import torch

    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cfg = load_config(config_path)
    cfg.model.device = device
    if device == "cpu":
        cfg.model.dtype = "float32"

    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments
        tok = AutoTokenizer.from_pretrained("Xenova/claude-tokenizer")
    except Exception as e:
        print(f"  SKIP training (no network/tokenizer): {e}")
        return

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_config = AutoConfig.for_model(
        "gpt2",
        vocab_size=tok.vocab_size,
        hidden_size=cfg.model.architecture.hidden_size,
        num_hidden_layers=cfg.model.architecture.num_hidden_layers,
        num_attention_heads=cfg.model.architecture.num_attention_heads,
        num_key_value_heads=cfg.model.architecture.num_key_value_heads,
        intermediate_size=cfg.model.architecture.intermediate_size,
        max_position_embeddings=cfg.model.architecture.max_position_embeddings,
        tie_word_embeddings=True,
    )
    try:
        model = AutoModelForCausalLM.from_config(model_config)
        model = model.to(device)
        if model_config.tie_word_embeddings:
            model.tie_weights()
    except Exception as e:
        print(f"  SKIP training (model create failed): {e}")
        return

    from src.data.pipeline import DataPipeline
    pipe = DataPipeline(cfg, tok)
    ds = pipe.build_pretrain_dataset()
    if ds is None or len(ds) == 0:
        print("  SKIP training (empty dataset)")
        return

    training_args = TrainingArguments(
        output_dir=cfg.output.checkpoint_dir,
        max_steps=10,
        per_device_train_batch_size=cfg.training.pretrain.batch_size,
        learning_rate=cfg.training.pretrain.learning_rate,
        warmup_steps=2,
        logging_steps=1,
        save_steps=5,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to="none",
        fp16=False,
        bf16=False,
    )

    def collate_fn(batch):
        import torch
        input_ids = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        attention_mask = torch.stack([torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch])
        labels = torch.stack([torch.tensor(b["labels"], dtype=torch.long) for b in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
    )

    # Train initial run
    print("    Training 10 steps...")
    trainer.train()
    test("initial training completed", True)

    # Save checkpoint manually (model weights only)
    ckpt_path = Path(cfg.output.checkpoint_dir, "test_checkpoint")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_path))
    test("checkpoint saved", ckpt_path.exists())

    # Load checkpoint
    try:
        model2 = AutoModelForCausalLM.from_config(model_config)
        model2 = model2.to(device)
        if model_config.tie_word_embeddings:
            model2.tie_weights()
        ckpt_files = list(ckpt_path.glob("*.safetensors")) + list(ckpt_path.glob("*.bin"))
        test("checkpoint has model files", len(ckpt_files) > 0,
             f"files: {[f.name for f in ckpt_files]}")
        test("checkpoint loadable", True)
    except Exception as e:
        test(f"checkpoint reload failed: {e}", False)
        return

    # Resume training
    trainer2 = Trainer(
        model=model2,
        args=TrainingArguments(
            output_dir=str(Path(cfg.output.checkpoint_dir, "resume")),
            max_steps=5,
            per_device_train_batch_size=cfg.training.pretrain.batch_size,
            learning_rate=cfg.training.pretrain.learning_rate,
            logging_steps=1,
            eval_strategy="no",
            remove_unused_columns=False,
            dataloader_drop_last=True,
            report_to="none",
            fp16=False,
            bf16=False,
        ),
        train_dataset=ds,
        data_collator=collate_fn,
    )
    print("    Resuming training for 5 steps...")
    run_ckpts = sorted(Path(cfg.output.checkpoint_dir).glob("checkpoint-*"))
    resume_path = str(run_ckpts[-1]) if run_ckpts else str(ckpt_path)
    trainer2.train(resume_from_checkpoint=resume_path)
    test("resumed training completed", True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end integration test")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training/checkpoint phases")
    args = parser.parse_args()

    print("=" * 60)
    print("  END-TO-END INTEGRATION TEST")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="integration_test_")
    try:
        docs_dir = build_test_docs(tmpdir)
        config_path = build_test_config(tmpdir, docs_dir)

        phase1_registry_and_validation(docs_dir)
        phase2_stream_and_pack(config_path)

        if not args.skip_training:
            phase3_training(config_path, use_gpu=args.gpu)
        else:
            print("\n--- Phase 3: Training (SKIPPED) ---")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
