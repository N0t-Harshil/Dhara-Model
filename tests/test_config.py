from __future__ import annotations

import tempfile
import yaml
from pathlib import Path
from typing import Any, Dict

import pytest

from src.config.schema import Config, load_config


SAMPLE_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "test-model",
        "dtype": "bfloat16",
        "architecture": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
        },
    },
    "training": {
        "max_seq_length": 1024,
        "sft": {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "max_steps": 100,
        },
    },
    "data": {
        "datasets": [
            {"path": "test/dataset", "max_samples": 100},
        ],
    },
}


def _write_config(data: Dict[str, Any]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


class TestConfigValidation:
    def test_load_valid_config(self):
        path = _write_config(SAMPLE_CONFIG)
        cfg = load_config(path)
        assert cfg.model.name == "test-model"
        assert cfg.model.architecture.hidden_size == 512
        assert cfg.training.sft.learning_rate == 1e-4
        path.unlink(missing_ok=True)

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.model.name == "Methos Class Model"
        assert cfg.training.sft.batch_size == 4
        assert len(cfg.alignment.constitution) == 10

    def test_config_migration_v1_to_v2(self):
        v1_config = {
            "model": {"name": "test", "dtype": "bfloat16", "architecture": {"hidden_size": 512}},
            "training": {"max_seq_length": 1024},
            "data_collection": {
                "datasets": [{"path": "test/ds", "max_samples": 50}],
                "streaming": True,
            },
            "fsdp": {"enabled": True, "sharding_strategy": "full_shard"},
        }
        path = _write_config(v1_config)
        cfg = load_config(path)
        assert len(cfg.data.datasets) == 1
        assert cfg.data.datasets[0].path == "test/ds"
        assert cfg.distributed.fsdp.enabled is True
        path.unlink(missing_ok=True)

    def test_invalid_dtype_rejected(self):
        bad = dict(SAMPLE_CONFIG)
        bad["model"]["dtype"] = "invalid_dtype"
        path = _write_config(bad)
        with pytest.raises(Exception):
            load_config(path)
        path.unlink(missing_ok=True)

    def test_architecture_defaults(self):
        cfg = Config()
        arch = cfg.model.architecture
        assert arch.attention_implementation == "flash_attention_2"
        assert arch.tie_word_embeddings is False
        assert arch.hidden_act == "silu"

    def test_estimate_model_size_accepts_tokenizer_or_vocab(self):
        from src.models.factory import ModelFactory

        cfg = Config()
        arch = cfg.model.architecture
        size_by_arch = ModelFactory.estimate_model_size(arch)
        assert "total_params_b" in size_by_arch

        # Ensure backward-compatible keyword argument is accepted.
        dummy_vocab_size = 50000
        estimate_with_vocab = ModelFactory.estimate_model_size(
            arch,
            tokenizer_or_vocab=dummy_vocab_size,
        )
        assert estimate_with_vocab["total_params_b"] < size_by_arch["total_params_b"]

    def test_distributed_config(self):
        cfg = Config()
        assert cfg.distributed.strategy == "fsdp"
        assert cfg.distributed.fsdp.sharding_strategy == "full_shard"

    def test_evaluation_config(self):
        cfg = Config()
        assert "human_eval" in cfg.evaluation.benchmarks
        assert "hellaswag" in cfg.evaluation.benchmarks
        assert cfg.evaluation.timeout == 60
