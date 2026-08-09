#!/usr/bin/env python3
"""
Pipeline Validation Report — validates dataset configuration, preprocessing,
quality thresholds, deduplication, and curriculum staging.

Usage: python scripts/validate_pipeline.py [--config path/to/config.yaml]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import DatasetEntryConfig, load_config


def validate_datasets(datasets: list, config_path: Path) -> list:
    errors = []
    categories = defaultdict(list)
    total_weight = 0.0
    for i, ds in enumerate(datasets):
        if not ds.path:
            errors.append(f"  Dataset #{i+1}: missing 'path'")
        if not ds.category:
            errors.append(f"  Dataset #{i+1} ({ds.path}): missing 'category'")
        else:
            categories[ds.category].append(ds.path)
        if ds.weight <= 0:
            errors.append(f"  Dataset #{i+1} ({ds.path}): weight {ds.weight} must be > 0")
        if ds.quality_score is not None and not (0.0 <= ds.quality_score <= 1.0):
            errors.append(f"  Dataset #{i+1} ({ds.path}): quality_score {ds.quality_score} must be in [0, 1]")
        total_weight += ds.weight

    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"  Total dataset weight {total_weight:.4f} != 1.0 (consider normalizing)")

    return errors


def generate_report(cfg, config_path: Path) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"  PIPELINE VALIDATION REPORT — {cfg.model.name}")
    lines.append(f"  Config: {config_path}")
    lines.append("=" * 72)

    data = cfg.data
    ds_list = data.datasets

    # 1. Dataset overview
    lines.append("")
    lines.append("── Dataset Mixture ──────────────────────────────────────────────")
    lines.append(f"  Total dataset entries: {len(ds_list)}")
    lines.append(f"  Streaming mode:        {data.streaming}")
    lines.append(f"  Cache dir:             {data.cache_dir}")
    lines.append("")

    if not ds_list:
        lines.append("  [ERROR] No datasets configured!")
    else:
        total_w = sum(ds.weight for ds in ds_list)
        lines.append(f"  {'Path':<45} {'Cat':<12} {'Wt':>6} {'QS':>5} {'Max':>8}")
        lines.append(f"  {'-'*45} {'-'*12} {'-'*6} {'-'*5} {'-'*8}")
        for ds in ds_list:
            qs = f"{ds.quality_score:.2f}" if ds.quality_score is not None else "auto"
            mx = f"{ds.max_samples or '∞':>8}"
            w = f"{ds.weight / total_w * 100 if total_w > 0 else 0:5.1f}%"
            name = ds.path[:44]
            lines.append(f"  {name:<45} {ds.category or '?':<12} {w:>6} {qs:>5} {mx:>8}")
        lines.append(f"  {'Total weight':>45} {total_w:6.3f}")

    # 2. Per-category breakdown
    lines.append("")
    lines.append("── Category Breakdown ────────────────────────────────────────────")
    by_cat = defaultdict(list)
    for ds in ds_list:
        by_cat[ds.category or "unknown"].append(ds)
    for cat, entries in sorted(by_cat.items()):
        w = sum(ds.weight for ds in entries)
        n = len(entries)
        lines.append(f"  {cat:<15} {n:2d} datasets, total weight {w:5.3f} ({w / total_w * 100:.1f}%)")

    # 3. Preprocessing config
    pp = data.preprocessing
    lines.append("")
    lines.append("── Preprocessing ────────────────────────────────────────────────")
    lines.append(f"  Remove boilerplate:    {pp.remove_boilerplate}")
    lines.append(f"  Min text length:       {pp.min_text_length}")
    lines.append(f"  Max text length:       {pp.max_text_length}")
    lines.append(f"  License keywords:      {len(pp.license_keywords)} patterns")
    lines.append(f"  Boilerplate patterns:  {len(pp.boilerplate_file_patterns)} patterns")

    # 4. Language balancing
    lb = data.language_balancing
    lines.append("")
    lines.append("── Language Balancing ───────────────────────────────────────────")
    lines.append(f"  Enabled:               {lb.enabled}")
    if lb.enabled and lb.target_distribution:
        lines.append(f"  Target distribution:   {lb.target_distribution}")
        td_total = sum(lb.target_distribution.values())
        if abs(td_total - 1.0) > 0.01:
            lines.append(f"  [WARNING] Target distribution sums to {td_total:.3f} (expected 1.0)")

    # 5. Synthetic reasoning
    sr = data.synthetic_reasoning
    lines.append("")
    lines.append("── Synthetic Reasoning ──────────────────────────────────────────")
    lines.append(f"  Enabled:               {sr.enabled}")
    if sr.enabled:
        lines.append(f"  Max chains:            {sr.max_chains}")
        lines.append(f"  Chain range:           [{sr.chain_length_min}, {sr.chain_length_max}]")

    # 6. Quality pipeline
    q = data.quality
    lines.append("")
    lines.append("── Quality Pipeline ─────────────────────────────────────────────")
    lines.append(f"  Length range:          [{q.min_length}, {q.max_length}]")
    lines.append(f"  Dedup method:          {q.deduplication.method} (threshold={q.deduplication.threshold})")
    lines.append(f"  Contamination:         {q.contamination.enabled}")
    if q.contamination.enabled:
        lines.append(f"  Benchmarks:            {', '.join(q.contamination.benchmarks)}")

    # 7. Curriculum
    lines.append("")
    lines.append("── Curriculum ───────────────────────────────────────────────────")
    lines.append(f"  Enabled:               {data.curriculum.enabled}")
    if data.curriculum.enabled:
        for s in data.curriculum.stages:
            filt = ", ".join(s.dataset_filter) if s.dataset_filter else "ALL"
            lines.append(f"  Stage '{s.name}': max_steps={s.max_steps}, filter=[{filt}]")
            if s.description:
                lines.append(f"    └─ {s.description}")

    # 8. Dataset validation errors
    errors = validate_datasets(ds_list, config_path)
    if errors:
        lines.append("")
        lines.append("── Validation Errors ───────────────────────────────────────────")
        lines.extend(errors)
    else:
        lines.append("")
        lines.append("  [OK] No validation errors.")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  Validation complete.")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline configuration")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        for candidate in ["config_foundation.yaml", "config_small.yaml", "config.yaml"]:
            p = PROJECT_ROOT / candidate
            if p.exists():
                config_path = p
                break
        else:
            print("No config file found. Provide one with --config.")
            sys.exit(1)

    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    report = generate_report(cfg, config_path)
    print("\n" + report)

    out_file = PROJECT_ROOT / "docs" / "pipeline_validation.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_file}")


if __name__ == "__main__":
    main()
