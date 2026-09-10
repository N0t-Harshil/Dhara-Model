"""Phase 1 (mandate §1): cache-aware tokenizer acquisition.

Acceptance: a warm, hash-verified tokenizer is never re-downloaded or
re-trained; only a missing cache, a corrupt/unverified cache, or an explicit
force triggers (re)acquisition; the ModelFactory manifest is the single source
of hash truth. main.cmd_full_training must not force a re-download on
--fresh-start.
"""
import importlib
import types
from pathlib import Path

import pytest


def _make_valid_cache(p) -> None:
    (p / "tokenizer.json").write_bytes(b"vocab-v1")
    (p / "tokenizer_config.json").write_bytes(b"cfg-v1")
    from src.models.factory import ModelFactory
    ModelFactory._write_tokenizer_manifest(p)


def _make_config(tmp_path, source: str = "huggingface") -> "Path":
    cfg = tmp_path / "tok.yaml"
    cfg.write_text(
        "tokenizer:\n"
        f"  source: {source}\n"
        "  huggingface_model: fake/claude-tokenizer\n",
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def _module(monkeypatch):
    tt = importlib.import_module("src.tokenizer_trainer")
    return tt


def _patch_download(_module, monkeypatch, recorder):
    def fake_download(output_dir, model_id, force):
        recorder.append((output_dir, model_id, force))
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_bytes(f"vocab-{len(recorder)}".encode())
        (p / "tokenizer_config.json").write_bytes(b"cfg")
    monkeypatch.setattr(_module, "download_tokenizer", fake_download)


def _patch_train(_module, monkeypatch, recorder):
    def fake_train(output_dir, vocab_size, max_samples, force):
        recorder.append((output_dir, vocab_size, max_samples, force))
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_bytes(f"custom-{len(recorder)}".encode())
        (p / "vocab.json").write_bytes(b"{}")
        (p / "merges.txt").write_bytes(b"")
    monkeypatch.setattr(_module, "train_custom_tokenizer", fake_train)


def test_warm_verified_cache_skips_download(tmp_path, _module, monkeypatch):
    from src.tokenizer_trainer import ensure_tokenizer
    cdir = tmp_path / "tok"
    cdir.mkdir()
    _make_valid_cache(cdir)
    cfg = _make_config(tmp_path)

    calls = []
    _patch_download(_module, monkeypatch, calls)

    ensure_tokenizer(str(cdir), str(cfg))
    assert calls == [], "warm verified cache must never re-download"


def test_unverified_present_cache_reaquires_with_force(tmp_path, _module, monkeypatch):
    from src.tokenizer_trainer import ensure_tokenizer
    cdir = tmp_path / "tok"
    cdir.mkdir()
    _make_valid_cache(cdir)
    (cdir / "tokenizer.json").write_bytes(b"tampered")  # manifest mismatch
    cfg = _make_config(tmp_path)

    calls = []
    _patch_download(_module, monkeypatch, calls)

    ensure_tokenizer(str(cdir), str(cfg))
    assert len(calls) == 1, "unverified cache must be re-acquired exactly once"
    assert calls[0][2] is True, "re-acquire must force the sub-acquirer"
    manifest = cdir / "tokenizer_version.json"
    assert manifest.exists(), "manifest must be written after re-acquire"
    from src.models.factory import ModelFactory
    assert ModelFactory._verify_tokenizer_cache(cdir) is not None


def test_missing_cache_downloads(tmp_path, _module, monkeypatch):
    from src.tokenizer_trainer import ensure_tokenizer
    cdir = tmp_path / "tok"
    cdir.mkdir()
    cfg = _make_config(tmp_path)

    calls = []
    _patch_download(_module, monkeypatch, calls)

    ensure_tokenizer(str(cdir), str(cfg))
    assert len(calls) == 1
    assert calls[0][2] is True, "unverified state must use the re-acquire path"
    from src.models.factory import ModelFactory
    assert ModelFactory._verify_tokenizer_cache(cdir) is not None


def test_explicit_force_always_reaquires(tmp_path, _module, monkeypatch):
    from src.tokenizer_trainer import ensure_tokenizer
    cdir = tmp_path / "tok"
    cdir.mkdir()
    _make_valid_cache(cdir)
    cfg = _make_config(tmp_path)

    calls = []
    _patch_download(_module, monkeypatch, calls)

    ensure_tokenizer(str(cdir), str(cfg), force=True)
    assert len(calls) == 1


def test_custom_source_warm_cache_skips_retrain(tmp_path, _module, monkeypatch):
    from src.tokenizer_trainer import ensure_tokenizer
    cdir = tmp_path / "tok"
    cdir.mkdir()
    _make_valid_cache(cdir)
    cfg = _make_config(tmp_path, source="custom")

    calls = []
    _patch_train(_module, monkeypatch, calls)

    ensure_tokenizer(str(cdir), str(cfg))
    assert calls == [], "warm verified custom cache must never retrain"


def test_main_full_training_does_not_force_tokenizer_on_fresh_start(
        tmp_path, monkeypatch):
    """--fresh-start must NOT trigger a tokenizer re-download: the cache is
    verified separately and force is only for explicit override."""
    import main as main_mod
    from src.config.schema import Config

    calls = []

    def fake_ensure(config_path, force=False):
        calls.append((config_path, force))

    def fake_load(path):
        return Config()

    class FakeDist:
        def __init__(self, cfg=None):
            self.cfg = cfg

        def is_main_process(self):
            return True
        is_distributed = False

    class FakePipe:
        def __init__(self, *a, **k):
            pass

        def initialize(self, fresh_start=False):
            pass

        def full_training_sequence(self):
            return {}

        def save_model(self):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr("src.tokenizer_trainer.ensure_tokenizer", fake_ensure)
    monkeypatch.setattr("src.config.schema.load_config", fake_load)
    monkeypatch.setattr("src.infrastructure.distributed.DistributedSetup", FakeDist)
    monkeypatch.setattr("src.training.pipeline.TrainingPipeline", FakePipe)
    monkeypatch.setattr(main_mod, "_apply_hf_token_env", lambda cfg: None)
    monkeypatch.setattr("src.utils.hf_auth.report_hf_auth", lambda cfg: None)

    args = types.SimpleNamespace(config="nonexistent.yaml", fresh_start=True)
    main_mod.cmd_full_training(args)

    assert calls, "ensure_tokenizer must be called"
    assert calls[0][1] is False, "fresh_start must not force a tokenizer re-download"