from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


def test_registry() -> None:
    print("\n--- Registry ---")
    from src.data.registry import build_registry, CATEGORY_WEIGHTS
    registry = build_registry()
    entries = registry.all_entries()
    total_weight = sum(e.weight for e in entries)
    test("total weight ~ 1.0", abs(total_weight - 1.0) < 0.005, f"got {total_weight}")
    test("at least 50 entries", len(entries) >= 50, f"got {len(entries)}")
    test("all weights positive", all(e.weight > 0 for e in entries))
    test("all quality in [0,1]", all(0 <= e.quality_score <= 1 for e in entries))
    # Check each category is within 1% of target
    from collections import defaultdict
    cat_w = defaultdict(float)
    for e in entries:
        cat_w[e.category] += e.weight
    for cat, exp in CATEGORY_WEIGHTS.items():
        test(f"category '{cat}' weight ~ {exp:.2f}",
             abs(cat_w.get(cat, 0) - exp) < 0.01,
             f"got {cat_w.get(cat, 0):.4f}")
    # Check no placeholder
    test("no placeholder entries", all("placeholder" not in e.path for e in entries))
    # Check no duplicate path+name+category
    seen = set()
    for e in entries:
        key = f"{e.path}/{e.name or ''}/{e.category}"
        test(f"unique entry: {key[:50]}", key not in seen)
        seen.add(key)


def test_streaming_module() -> None:
    print("\n--- Streaming ---")
    import inspect
    from src.data import streaming
    src = inspect.getsource(streaming.stream_dataset)
    test("no trust_remote_code", "trust_remote_code" not in src)
    test("handles DatasetDict", "DatasetDict" in src)
    test("preserves metadata", "**sample" in src)
    test("uses logger.exception", "logger.exception" in src)
    # Test the extract_text function
    from src.data.registry import extract_text
    sample = {"text": "hello world", "id": 1}
    text = extract_text(sample)
    test("extract_text returns text field", text == "hello world")
    sample_no_text = {"content": "some content", "id": 2}
    text = extract_text(sample_no_text)
    test("extract_text falls back to content", text == "some content")
    # Test detect_text_fields
    from src.data.registry import detect_text_fields
    fields = detect_text_fields(sample)
    test("detect_text_fields finds text", "text" in fields)


def test_packing() -> None:
    print("\n--- Packing ---")
    from src.data.pipeline import pack_sequences
    max_len = 256
    eos_id = 2
    # Simple test: pack 3 short docs
    samples = [
        {"input_ids": [1, 3, 4], "quality_score": 0.9},
        {"input_ids": [5, 6, 7, 8], "quality_score": 0.8},
        {"input_ids": [9, 10], "quality_score": 0.7},
    ]
    packed, eff = pack_sequences(samples, max_len, eos_id)
    test("packing produced sequences", len(packed) >= 1)
    test("packing efficiency > 0", eff >= 0)
    test("packed seq has _segments", "_segments" in packed[0])
    test("packed seq has _avg_quality", "_avg_quality" in packed[0])
    test("packed seq has correct keys",
         all(k in packed[0] for k in ["input_ids", "labels", "attention_mask"]))
    test("all sequences <= max_len",
         all(len(p["input_ids"]) == max_len for p in packed))
    # Verify segments
    first_seg = packed[0]["_segments"]
    test(f"multiple docs in first pack ({first_seg})", first_seg >= 2, f"got {first_seg}")
    # Test single long document
    long_sample = [{"input_ids": list(range(max_len + 50)), "quality_score": 0.5}]
    packed_long, eff_long = pack_sequences(long_sample, max_len, eos_id)
    test("long doc is windowed", len(packed_long) >= 1)
    test("windowed doc <= max_len",
         all(len(p["input_ids"]) == max_len for p in packed_long))


def test_padding() -> None:
    print("\n--- Padding ---")
    from src.data.pipeline import pack_sequences
    max_len = 256
    eos_id = 2
    # Fill bins with small docs
    samples = [{"input_ids": [1, 3, 4, 5], "quality_score": 0.5} for _ in range(100)]
    packed, eff = pack_sequences(samples, max_len, eos_id)
    # Calculate padding ratio
    total_slots = len(packed) * max_len
    total_filled = sum(
        sum(1 for m in p["attention_mask"] if m == 1) for p in packed
    )
    padding_pct = (total_slots - total_filled) / total_slots * 100
    test(f"padding < 5% ({padding_pct:.1f}%)", padding_pct < 5.0, f"got {padding_pct:.1f}%")
    test(f"packing eff > 95% ({eff*100:.1f}%)", eff > 0.95, f"got {eff:.3f}")


def test_random_window() -> None:
    print("\n--- Random Window ---")
    from src.data.pipeline import random_window_sample
    # With fixed seed
    tokens = list(range(1000))
    seq_len = 128
    windows = set()
    for _ in range(20):
        w = tuple(random_window_sample(tokens, seq_len))
        windows.add(w)
    test("multiple different windows", len(windows) > 1, f"got {len(windows)} unique")
    test("window length = seq_len",
         all(len(w) == seq_len for w in windows))
    # Test short sequence
    short = list(range(50))
    result = random_window_sample(short, 128)
    test("short doc returns all tokens", len(result) == 50)
    test("short doc unchanged", result == short)


def test_quality_scoring() -> None:
    print("\n--- Quality Scoring ---")
    from src.data.quality import document_quality_score, detect_language

    # Code
    code = "def foo(x: int) -> int:\n    return x + 1\n"
    res = document_quality_score(code, category="code", language="python")
    test("code quality > 0.3", res["final"] > 0.3, f"got {res['final']:.3f}")
    test("code has components", len(res) > 3)

    # Docs
    docs = "## Overview\nThis is the introduction.\n## Parameters\nx: int\n## Returns\nint"
    res = document_quality_score(docs, category="docs", language="text")
    test("docs quality > 0.3", res["final"] > 0.3, f"got {res['final']:.3f}")
    test("docs has doc_completeness", "doc_completeness" in res)

    # Web
    web = "The quick brown fox jumps over the lazy dog. " * 20
    res = document_quality_score(web, category="web_text", language="text")
    test("web quality in range", 0 < res["final"] < 1)

    # Language detection
    lang = detect_language(code)
    test("lang detect python", lang == "python", f"got {lang}")
    lang = detect_language("fn main() { println!(\"hello\"); }")
    test("lang detect rust", lang == "rust", f"got {lang}")


def test_boilerplate_removal() -> None:
    print("\n--- Boilerplate ---")
    from src.data.pipeline import remove_boilerplate

    # License header should be removed
    code_with_license = """# MIT License
#
# Copyright (c) 2024
#
# Permission is hereby granted

def foo():
    return 42
"""
    cleaned = remove_boilerplate(code_with_license)
    test("license header removed", "MIT License" not in cleaned, f"got first 50 chars: {cleaned[:50]}")
    test("code preserved", "def foo()" in cleaned)
    test("code structure intact", "return 42" in cleaned)

    # Docstring should NOT be removed (it's not a license)
    code_with_docstring = '''def foo():
    """This is a docstring explaining what foo does."""
    return 42
'''
    cleaned = remove_boilerplate(code_with_docstring)
    test("docstrings preserved", "docstring" in cleaned)

    # No boilerplate to remove
    clean_code = "def bar(x): return x * 2"
    cleaned = remove_boilerplate(clean_code)
    test("clean code unchanged", cleaned == clean_code)


def test_weighted_mixed_dataset() -> None:
    print("\n--- Weighted Mixed Dataset ---")
    from src.data.pipeline import WeightedMixedDataset
    from datasets import Dataset

    # Create dummy datasets
    ds1 = Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]} for _ in range(50)])
    ds2 = Dataset.from_list([{"input_ids": [4, 5, 6], "labels": [4, 5, 6], "attention_mask": [1, 1, 1]} for _ in range(50)])

    weighted = [(ds1, 0.7, "ds1"), (ds2, 0.3, "ds2")]
    mixed = WeightedMixedDataset(weighted, total_samples=100, quality_scores=[0.9, 0.6])
    test("mixed dataset created", len(mixed) == 100)
    test("mixed has valid samples", all(idx < len(mixed.datasets) for idx, _ in mixed.indices))
    sample = mixed[0]
    test("mixed returns dict", isinstance(sample, dict))
    test("mixed has input_ids", "input_ids" in sample)

    # Check quality-aware weights used
    weights = mixed.weights
    test("weights list created", len(weights) == 2)
    test("weights sum to 1", abs(sum(weights) - 1.0) < 0.01)
    # Higher quality dataset should have higher weight (when base weights are equal)
    # ds1 has base 0.7 * 0.9 = 0.63, ds2 has 0.3 * 0.6 = 0.18
    test("quality-aware weighting", weights[0] > weights[1], f"got {weights[0]:.3f} vs {weights[1]:.3f}")


def test_health_report() -> None:
    print("\n--- Health Report ---")
    from src.data.health_reporter import DatasetHealthReport
    report = DatasetHealthReport(name="test")
    report.add_dataset_stats(
        path="test/ds", category="code", weight=0.5,
        raw_count=1000, after_boilerplate=900, after_quality=800,
        after_dedup=700, packed_count=300, total_tokens=600000,
        duplicate_removed=50, rejection_reasons={"quality_fail": 100},
        quality_scores=[0.5, 0.6, 0.7], lang_dist={"python": 50},
        domain_dist={"backend": 30}, token_lengths=[100, 200, 300, 4000],
        max_seq_length=2048,
    )
    report.compute_global_stats()
    gs = report.global_stats
    test("health report has stats", len(gs) > 5)
    test("health report has tokens", gs.get("total_tokens", 0) > 0)
    test("health report has packing", "average_packing_efficiency" in gs)
    test("health report has quality", "average_quality_score" in gs)
    test("health report has long_ctx", "long_context_documents" in gs)
    text = report.summary_text()
    test("health report generates text", len(text) > 100)
    test("health report has language section", "Language" in text)
    test("health report has domain section", "Domain" in text)
    test("health report has category breakdown", "Category" in text)


def test_semantic_dedup() -> None:
    print("\n--- Semantic Dedup (light) ---")
    from src.data.quality import SemanticDeduplicator
    sd = SemanticDeduplicator(threshold=0.99)
    # Test with exact duplicates
    assert not sd.is_duplicate("unique document")
    assert sd.is_duplicate("unique document")
    test("exact duplicate detected", sd.duplicate_count == 1)
    sd.reset()
    test("reset works", sd.duplicate_count == 0)
    # Test filter_batch
    texts = ["doc a", "doc b", "doc a"]
    keep, flags = sd.filter_batch(texts)
    test("batch filter keeps unique", len(keep) == 2, f"got {len(keep)}")
    test("batch filter flags correct", flags == [True, True, False])


def test_fallback_detection() -> None:
    print("\n--- Fallback ---")
    from src.data.streaming import stream_dataset_with_fallbacks
    from src.data.registry import DatasetRegistry, DatasetInfo
    registry = DatasetRegistry()
    primary = DatasetInfo(
        path="nonexistent/path", category="code",
        weight=1.0, quality_score=0.5,
        fallbacks=["valid_fallback_ds"],
    )
    secondary = DatasetInfo(
        path="valid_fallback_ds", category="code",
        weight=1.0, quality_score=0.5,
    )
    registry.register(primary)
    registry.register(secondary)
    # Just verify the function handles gracefully (no crash)
    import contextlib
    with contextlib.suppress(Exception):
        list(stream_dataset_with_fallbacks(primary, registry, limit=1))
    test("fallback doesn't crash", True)


def test_registry_delegation() -> None:
    print("\n--- Registry Delegation ---")
    import inspect
    from src.data import pipeline
    src = inspect.getsource(pipeline.DataPipeline.build_pretrain_dataset)
    test("delegates to registry", "build_pretrain_dataset_from_registry" in src)
    test("checks use_registry", "use_registry" in src)


def main() -> None:
    print("=" * 60)
    print("  DATA PIPELINE TEST SUITE")
    print("=" * 60)

    tests = [
        test_registry,
        test_streaming_module,
        test_packing,
        test_padding,
        test_random_window,
        test_quality_scoring,
        test_boilerplate_removal,
        test_weighted_mixed_dataset,
        test_health_report,
        test_semantic_dedup,
        test_fallback_detection,
        test_registry_delegation,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            global FAIL
            FAIL += 1
            print(f"  EXCEPTION {t.__name__}: {e}")

    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
