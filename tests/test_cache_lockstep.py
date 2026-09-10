"""Phase 11 (mandate §11): cache-invalidation lockstep.

Metadata-cache fingerprints and packed-cache keys (unit + registry + stage +
legacy tokenized) are derived from the SAME two guards —
``processing_signature`` and ``tokenizer_signature`` — so a preprocessing or
tokenizer change invalidates both families together. The only remaining
divergence risk is a metadata-format/fingerprint bump: the unit key now
carries both an ``mcs<metadata-schema>`` tag and a family-wide
``pcf<format-epoch>`` tag so every packed-cache key flips together.
"""
import hashlib
from types import SimpleNamespace

import pytest

import src.data.pipeline as pipeline_mod
from src.data.metadata_cache import compute_fingerprint
from src.data.pipeline import DataPipeline, PACKED_CACHE_FORMAT_VERSION
from src.data.registry import DatasetInfo
from src.utils.steps import estimate_pretrain_steps


def _fake_pipe():
    fake = object.__new__(DataPipeline)
    fake.cfg = SimpleNamespace(
        data=SimpleNamespace(
            preprocessing=SimpleNamespace(
                remove_boilerplate=True, min_text_length=10,
                boilerplate_file_patterns=["*.html"],
                autogen_patterns=[]),
            quality=SimpleNamespace(
                deduplication=SimpleNamespace(method="simhash",
                                              threshold=0.90)),
            ast_filter=SimpleNamespace(code_filtering=True),
            function_sampling=SimpleNamespace(enabled=False),
            sampler=SimpleNamespace(balance_by="weights"),
            language_balancing=SimpleNamespace(enabled=False,
                                               target_distribution=None),
            domain_balancing=SimpleNamespace(enabled=False, include=None),
            max_samples_per_dataset=1000,
            use_packed_cache=True,
            cache_dir="cache",
        ),
        training=SimpleNamespace(
            max_seq_length=2048,
            pretrain=SimpleNamespace(batch_size=4,
                                     gradient_accumulation_steps=2,
                                     staging=SimpleNamespace(stage_cache_dir="staging")),
        ),
        model=SimpleNamespace(architecture=SimpleNamespace(vocab_size=65536)),
    )
    fake.tokenizer = SimpleNamespace(name_or_path="tok", vocab_size=65536)
    fake.meta_cache = SimpleNamespace(fingerprint_version=1)
    return fake


def _unit_info():
    return DatasetInfo(path="org/repo", name="sub", category="code",
                       weight=0.1, quality_score=0.9)


def _unit_raw_key(ppsig, toksig, info, ds_limit=1000):
    """Raw byte-identical unit key input (mirrors unit_cache_key)."""
    return hashlib.sha256("|".join([
        "unit-v1", info.path, info.name or "", info.split,
        info.category, str(ds_limit), ppsig, toksig,
        f"mcs{pipeline_mod.METADATA_CACHE_SCHEMA_VERSION}",
        f"pcf{PACKED_CACHE_FORMAT_VERSION}",
    ]).encode()).hexdigest()[:16]


def test_preprocess_change_invalidates_fingerprint_and_unit_key():
    """A preprocessing change flips the metadata fingerprint (same signatures)
    AND the raw unit-key input, i.e. both caches invalidate together."""
    fake = _fake_pipe()
    info = _unit_info()
    ppsig_a = fake.processing_signature()
    toksig = fake.tokenizer_signature
    fake.cfg.data.preprocessing.min_text_length = 999
    ppsig_b = fake.processing_signature()
    assert ppsig_a != ppsig_b
    fp_a = compute_fingerprint("org/repo", "sub", "train", None,
                               ["a.parquet"], "rev1", ppsig_a, toksig)
    fp_b = compute_fingerprint("org/repo", "sub", "train", None,
                               ["a.parquet"], "rev1", ppsig_b, toksig)
    assert fp_a != fp_b
    assert (_unit_raw_key(ppsig_a, toksig, info)
            != _unit_raw_key(ppsig_b, toksig, info))


def test_format_epoch_bump_invalidates_every_packed_key(monkeypatch):
    """Bumping the family-wide epoch (a packed-format change) flips the unit,
    registry and stage keys together."""
    fake = _fake_pipe()
    info = _unit_info()
    stage_cfg = SimpleNamespace(name="s0", categories=["code"], weights={},
                                steps=100, max_samples_per_dataset=10)
    k_unit_v1 = fake.unit_cache_key(info)
    k_reg_v1 = fake._registry_cache_key(info, 1000)
    k_stage_v1 = fake._stage_cache_key(stage_cfg, 0)
    k_legacy_v1 = fake._get_cache_key(
        SimpleNamespace(path="org/repo", name="sub", split="train",
                        max_samples=1000),
        "pretrain")

    monkeypatch.setattr(pipeline_mod, "PACKED_CACHE_FORMAT_VERSION", 2)
    assert fake.unit_cache_key(info) != k_unit_v1
    assert fake._registry_cache_key(info, 1000) != k_reg_v1
    assert fake._stage_cache_key(stage_cfg, 0) != k_stage_v1
    assert fake._get_cache_key(
        SimpleNamespace(path="org/repo", name="sub", split="train",
                        max_samples=1000),
        "pretrain") != k_legacy_v1


def test_metadata_schema_bump_invalidates_unit_key(monkeypatch):
    """A metadata-cache schema bump alone (no packed-format change) must still
    invalidate unit caches via the mcs tag."""
    fake = _fake_pipe()
    info = _unit_info()
    k_v1 = fake.unit_cache_key(info)
    monkeypatch.setattr(pipeline_mod, "METADATA_CACHE_SCHEMA_VERSION", 2)
    assert fake.unit_cache_key(info) != k_v1


def test_tokenizer_change_invalidates_unit_key():
    """tokenizer_signature is shared, so a tokenizer change invalidates the
    unit key (and, via the same toksig in compute_fingerprint, the metadata
    record)."""
    fake = _fake_pipe()
    info = _unit_info()
    k_a = fake.unit_cache_key(info)
    fake.tokenizer = SimpleNamespace(name_or_path="tok", vocab_size=65535)
    assert fake.unit_cache_key(info) != k_a


def test_phase9_helper_still_importable():
    est = estimate_pretrain_steps(5739, 4, gradient_accumulation_steps=4)
    assert est.optimizer_steps == 358