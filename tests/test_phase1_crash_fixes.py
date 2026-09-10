"""Regression tests for the Phase-1 crash-fix audit (A1-A10)."""
import inspect
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import BatchEncoding


# ---------------------------------------------------------------- A1
class TestCollectorSignature:
    def test_stream_single_dataset_accepts_raw_text(self):
        from src.massive_data_collector import MassiveDataCollector

        sig = inspect.signature(MassiveDataCollector.stream_single_dataset)
        assert "raw_text" in sig.parameters

    def test_legacy_shim_accepts_raw_text(self):
        import src.data.streaming as streaming

        # The module-level re-export must be callable with raw_text either way.
        sig = inspect.signature(streaming.MassiveDataCollector.stream_single_dataset)
        assert "raw_text" in sig.parameters


# ---------------------------------------------------------------- A2
class TestAuxLossToolsNone:
    def test_tools_none_does_not_crash(self):
        from src.dhara.losses import AuxiliaryLossComputer

        comp = AuxiliaryLossComputer()
        losses = comp.forward({"tools": None}, targets={})
        assert isinstance(losses, dict)
        assert "tools" not in losses


# ---------------------------------------------------------------- A3
class TestEmptyPreferenceDataset:
    def _make_pipeline(self):
        from src.alignment.pipeline import AlignmentPipeline
        from src.config.schema import Config

        cfg = Config()
        model = torch.nn.Linear(4, 4)
        tok = MagicMock()
        return AlignmentPipeline(model=model, tokenizer=tok, cfg=cfg)

    def test_empty_dataset_returns_early(self):
        pipeline = self._make_pipeline()

        class EmptyDS(torch.utils.data.Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, i):  # pragma: no cover - never called
                raise AssertionError

        result = pipeline._run_preference_training(
            trainer=MagicMock(), dataset=EmptyDS(), name="dpo")
        assert result["steps"] == 0


# ---------------------------------------------------------------- A4
class TestGenerationBudget:
    def _mock_tokenizer(self, seq_len):
        tok = MagicMock()
        tok.return_value = BatchEncoding({
            "input_ids": torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        })
        tok.decode.return_value = "ok"
        return tok

    def test_critique_long_prompt_positive_budget(self):
        from src.alignment.constitutional import ConstitutionalTrainer

        model = MagicMock()
        model.generate.return_value = torch.ones(1, 3000, dtype=torch.long)
        ct = ConstitutionalTrainer(
            model=model, tokenizer=self._mock_tokenizer(3000),
            constitution=["be nice"], max_length=2048,
        )
        out = ct.critique("inst", "resp")
        assert out == "ok"
        kwargs = model.generate.call_args.kwargs
        assert kwargs["max_new_tokens"] >= 1

    def test_generate_preference_pairs_long_prompt(self):
        from src.alignment.constitutional import ConstitutionalTrainer

        model = MagicMock()
        model.generate.return_value = torch.ones(1, 3000, dtype=torch.long)
        ct = ConstitutionalTrainer(
            model=model, tokenizer=self._mock_tokenizer(3000),
            constitution=["be nice"], max_length=2048,
        )
        pairs = ct.generate_preference_pairs(["long instruction" * 500])
        assert len(pairs) == 1
        assert model.generate.call_args.kwargs["max_new_tokens"] >= 1

    def test_safety_max_len_floor(self):
        from src.evaluation.safety import SafetyEvaluator

        model = MagicMock()
        model.config.max_position_embeddings = 128
        model.device = torch.device("cpu")
        model.generate.return_value = torch.ones(1, 130, dtype=torch.long)
        tok = MagicMock()
        tok.return_value = BatchEncoding({"input_ids": torch.ones(1, 10, dtype=torch.long)})
        tok.decode.return_value = "hello"
        ev = SafetyEvaluator(model=model, tokenizer=tok, max_new_tokens=256)
        out = ev._query("probe")
        assert out == "hello"


# ---------------------------------------------------------------- A5/A6
class TestFactoryLoad:
    @pytest.fixture()
    def tiny_setup(self, tmp_path):
        from src.config.schema import Config

        cfg = Config()
        arch = cfg.model.architecture
        if arch.model_type == "nslt":
            from src.nslt import NSLTModel

            model = NSLTModel(vocab_size=64, d_model=16, d_state=8,
                              d_hidden=32, n_ssm_layers=1, max_seq_len=32)
        else:
            from src.dhara.model import DharaModel, DharaConfig

            mc = DharaConfig(vocab_size=64, hidden_size=16, d_state=8,
                                d_hidden=32, n_ssm_layers=1, n_hssm_levels=1,
                                max_position_embeddings=32)
            model = DharaModel(config=mc)
        tok = MagicMock()
        tok.__len__ = lambda self: 64
        return cfg, model, tok

    def test_missing_weights_raises_clear_error(self, tiny_setup, tmp_path):
        from src.models.factory import ModelFactory

        cfg, _, tok = tiny_setup
        (tmp_path / "config.json").write_text('{"model_type": "%s"}'
                                              % cfg.model.architecture.model_type)
        with pytest.raises(FileNotFoundError, match="[Nn]o model weights"):
            ModelFactory.load_model(tmp_path, cfg, tokenizer=tok)

    def test_strict_raises_on_shape_mismatch(self, tiny_setup, tmp_path):
        from src.models.factory import ModelFactory

        cfg, model, tok = tiny_setup
        sd = {k: v for k, v in model.state_dict().items()}
        # Corrupt one tensor's shape
        first_key = next(iter(sd))
        sd[first_key] = torch.zeros(1)
        torch.save(sd, tmp_path / "pytorch_model.bin")
        with pytest.raises(RuntimeError, match="incompatible|missing"):
            ModelFactory.load_model(tmp_path, cfg, tokenizer=tok, strict=True)

    def test_lenient_load_skips_mismatch(self, tiny_setup, tmp_path):
        from src.models.factory import ModelFactory

        cfg, model, tok = tiny_setup
        sd = {k: v for k, v in model.state_dict().items()}
        first_key = next(iter(sd))
        sd[first_key] = torch.zeros(1)
        torch.save(sd, tmp_path / "pytorch_model.bin")
        loaded, _ = ModelFactory.load_model(tmp_path, cfg, tokenizer=tok, strict=False)
        assert loaded is not None


# ---------------------------------------------------------------- A7
class TestTelemetryMultiGpu:
    def test_gpu_memory_sums_across_gpus(self, monkeypatch):
        from src.infrastructure import telemetry

        fake = MagicMock()
        fake.returncode = 0
        fake.stdout = "4096, 8192\n2048, 8192\n"
        monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: fake)
        res = telemetry._gpu_memory()
        assert res is not None
        free_gb, total_gb = res
        assert abs(free_gb - 6.0) < 1e-6
        assert abs(total_gb - 16.0) < 1e-6

    def test_gpu_memory_single_gpu(self, monkeypatch):
        from src.infrastructure import telemetry

        fake = MagicMock()
        fake.returncode = 0
        fake.stdout = "1024, 4096\n"
        monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: fake)
        free_gb, total_gb = telemetry._gpu_memory()
        assert abs(free_gb - 1.0) < 1e-6
        assert abs(total_gb - 4.0) < 1e-6


# ---------------------------------------------------------------- A8
class TestGzipDetection:
    def test_jsonl_gz_detected_as_json(self):
        from src.data.metadata_cache import _loader_for_files

        assert _loader_for_files(["part-000.jsonl.gz"]) == "json"
        assert _loader_for_files(["a.json.gz", "b.jsonl.gz"]) == "json"

    def test_plain_extensions_still_work(self):
        from src.data.metadata_cache import _loader_for_files

        assert _loader_for_files(["a.parquet", "b.parquet"]) == "parquet"
        assert _loader_for_files(["a.jsonl"]) == "json"
        assert _loader_for_files([]) is None


# ---------------------------------------------------------------- A9
class TestColdDriverFallback:
    def test_non_datasetinfo_info_gets_valid_cold_driver(self):
        """Resolution failure with a non-DatasetInfo info must still yield a
        driver whose stream() can run (not a silent None-info no-op)."""
        from src.data.drivers import detect_driver, DRIVER_KIND_STREAMING

        info = SimpleNamespace(path="definitely/not/a/real/dataset_xyz",
                               name=None, split="train", data_dir=None,
                               category="general", weight=1.0,
                               quality_score=0.5, max_samples=None)
        driver, diag = detect_driver(info, meta_cache=None, builder_cache=None,
                                     preprocess_sig="", token_sig="")
        assert diag["driver_kind"] == DRIVER_KIND_STREAMING
        assert getattr(driver, "info", None) is not None


# ---------------------------------------------------------------- A10
class TestPrefetchDuplicateFailFast:
    def test_duplicate_fails_fast_when_first_times_out(self):
        from src.training.asyncprefetch import UnitPrefetch, PrefetchTimeout

        pf = UnitPrefetch(build_fn=lambda u, i: time.sleep(30), total=2,
                          depth=2, timeout=0.25)
        pf.start(["dup-a", "dup-a"])
        with pytest.raises(PrefetchTimeout):
            pf.get(0)
        t0 = time.monotonic()
        with pytest.raises(PrefetchTimeout):
            pf.get(1)
        assert time.monotonic() - t0 < 0.2

    def test_duplicate_ok_path_unaffected(self):
        from src.training.asyncprefetch import UnitPrefetch

        pf = UnitPrefetch(build_fn=lambda u, i: {"v": i}, total=2,
                          depth=2, timeout=5.0)
        pf.start(["x", "x"])
        r0 = pf.get(0)
        r1 = pf.get(1)
        assert r0[0]["v"] == 0
        assert r1[0]["v"] == 0  # shares first payload
