from __future__ import annotations

import inspect
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from src.data.registry import build_registry, DatasetRegistry, DatasetInfo, CATEGORY_WEIGHTS

FAILURES: List[str] = []
WARNINGS: List[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    logger.error("FAIL: %s", msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    logger.warning("WARN: %s", msg)


def check_total_weight(registry: DatasetRegistry) -> None:
    entries = registry.all_entries()
    total = sum(e.weight for e in entries)
    if abs(total - 1.0) < 0.005:
        logger.info("  PASS: total weight = %.4f", total)
    else:
        fail("total weight = %.4f (expected 1.000)" % total)


def check_category_weights(registry: DatasetRegistry) -> None:
    cat_weights: Dict[str, float] = {}
    for e in registry.all_entries():
        cat_weights[e.category] = cat_weights.get(e.category, 0.0) + e.weight
    for cat, expected in CATEGORY_WEIGHTS.items():
        actual = cat_weights.get(cat, 0.0)
        diff = abs(actual - expected)
        status = "OK" if diff < 0.01 else "MISMATCH"
        logger.info("  %-25s expected=%.3f actual=%.3f [%s]", cat, expected, actual, status)
        if diff >= 0.01:
            fail("category '%s' weight %.4f differs from target %.4f" % (cat, actual, expected))


def check_no_duplicate_paths(registry: DatasetRegistry) -> None:
    seen: Set[str] = set()
    for e in registry.all_entries():
        key = f"{e.path}/{e.name or ''}/{e.category}"
        if key in seen:
            warn("duplicate entry: %s (weight %.4f)" % (key, e.weight))
        seen.add(key)
    logger.info("  PASS: no duplicate entries")


def check_fallbacks_exist(registry: DatasetRegistry) -> None:
    all_paths = {(e.path, e.category) for e in registry.all_entries()}
    for e in registry.all_entries():
        for fb in e.fallbacks:
            if (fb, e.category) not in all_paths:
                logger.info("  NOTE: fallback %s/%s not in same category", fb, e.category)


def check_quality_scores_in_range(registry: DatasetRegistry) -> None:
    for e in registry.all_entries():
        if not (0.0 <= e.quality_score <= 1.0):
            fail("%s quality_score=%.2f out of [0,1]" % (e.path, e.quality_score))
    logger.info("  PASS: all quality scores in [0, 1]")


def check_positive_weights(registry: DatasetRegistry) -> None:
    for e in registry.all_entries():
        if e.weight <= 0:
            fail("%s has non-positive weight %.4f" % (e.path, e.weight))
    logger.info("  PASS: all weights positive")


def check_no_placeholder(registry: DatasetRegistry) -> None:
    for e in registry.all_entries():
        if "placeholder" in e.path.lower():
            fail("placeholder dataset found: %s" % e.path)
    logger.info("  PASS: no placeholder datasets")


def check_trust_remote_code() -> None:
    import inspect
    from src.data import streaming
    src = inspect.getsource(streaming)
    if "trust_remote_code" in src:
        fail("trust_remote_code found in streaming.py")
    else:
        logger.info("  PASS: no trust_remote_code in streaming")


def check_metadata_preserved() -> None:
    from src.data import streaming
    src = inspect.getsource(streaming.stream_dataset)
    if "**sample" in src:
        logger.info("  PASS: streaming preserves metadata ({**sample, ...})")
    else:
        fail("streaming does not preserve metadata (missing {**sample, ...})")


def check_datasets_api() -> None:
    from src.data import streaming
    src = inspect.getsource(streaming.stream_dataset)
    for check in ["DatasetDict", "IterableDatasetDict", "isinstance(ds, (DatasetDict, IterableDatasetDict))"]:
        if check in src:
            break
    else:
        fail("streaming does not handle DatasetDict/IterableDatasetDict")
    logger.info("  PASS: streaming handles DatasetDict/IterableDatasetDict")


def check_logger_exception() -> None:
    from src.data import streaming
    src = inspect.getsource(streaming)
    if "logger.exception" in src:
        logger.info("  PASS: uses logger.exception()")
    else:
        fail("streaming does not use logger.exception()")


def check_registry_delegation() -> None:
    from src.data import pipeline
    src = inspect.getsource(pipeline.DataPipeline.build_pretrain_dataset)
    if "use_registry" in src and "build_pretrain_dataset_from_registry" in src:
        logger.info("  PASS: build_pretrain_dataset delegates when use_registry=true")
    else:
        fail("build_pretrain_dataset does not delegate to registry")


def print_summary(registry: DatasetRegistry) -> None:
    entries = registry.all_entries()
    cats = defaultdict(lambda: {"count": 0, "weight": 0.0})
    for e in entries:
        cats[e.category]["count"] += 1
        cats[e.category]["weight"] += e.weight

    print()
    print("=" * 72)
    print("  REGISTRY VALIDATION REPORT")
    print("=" * 72)
    print(f"  Total entries: {len(entries)}")
    print(f"  Total weight:  {sum(e.weight for e in entries):.4f}")
    print()
    print(f"  {'Category':20s} {'Count':5s} {'Weight':8s} {'Target':8s} {'Status':10s}")
    print("  " + "-" * 51)
    for cat, expected in CATEGORY_WEIGHTS.items():
        info = cats.get(cat, {"count": 0, "weight": 0.0})
        diff = abs(info["weight"] - expected)
        status = "OK" if diff < 0.01 else "MISMATCH"
        print(f"  {cat:20s} {info['count']:5d} {info['weight']:8.4f} {expected:8.4f} {status:10s}")
    print()

    print("  Fallback chains:")
    for e in entries:
        if e.fallbacks:
            print(f"    {e.path+'/'+(e.name or ''):40s} -> {', '.join(e.fallbacks)}")
    print()

    if FAILURES:
        print(f"  FAILURES ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"    [FAIL] {f}")
    else:
        print("  [PASS] All checks passed")
    if WARNINGS:
        print(f"  WARNINGS ({len(WARNINGS)}):")
        for w in WARNINGS:
            print(f"    [WARN] {w}")
    print("=" * 72)


def main() -> None:
    logger.info("Building registry...")
    registry = build_registry()

    print("")
    print("=" * 72)
    print("  REGISTRY VALIDATION CHECKS")
    print("=" * 72)

    check_total_weight(registry)
    check_category_weights(registry)
    check_no_duplicate_paths(registry)
    check_fallbacks_exist(registry)
    check_quality_scores_in_range(registry)
    check_positive_weights(registry)
    check_no_placeholder(registry)
    check_trust_remote_code()
    check_metadata_preserved()
    check_datasets_api()
    check_logger_exception()
    check_registry_delegation()

    print_summary(registry)

    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
