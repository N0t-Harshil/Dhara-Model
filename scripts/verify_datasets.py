from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from src.data.registry import build_registry, DatasetRegistry

REQUIRED_DOC_FIELDS = {"text", "source", "title", "url"}


def try_load_sample(path: str, name: Optional[str] = None, split: str = "train",
                    timeout_sec: int = 60, max_samples: int = 5, token: Optional[str] = None) -> Dict[str, Any]:
    from datasets import load_dataset
    import os

    if token:
        os.environ["HF_TOKEN"] = token

    result = {"path": path, "name": name, "split": split, "status": "unknown", "error": "",
              "loaded": 0, "columns": [], "avg_text_len": 0, "time_sec": 0}

    start = time.time()
    try:
        kwargs = {"path": path, "split": split, "streaming": True}
        if name:
            kwargs["name"] = name
        ds = load_dataset(**kwargs)
        samples = []
        timeout = start + timeout_sec
        for i, s in enumerate(ds):
            if time.time() > timeout:
                result["status"] = "timeout"
                break
            if i >= max_samples:
                break
            samples.append(s)
        elapsed = time.time() - start
        result["time_sec"] = round(elapsed, 2)
        result["loaded"] = len(samples)
        if samples:
            result["columns"] = list(samples[0].keys())
            texts = []
            for s in samples:
                for v in s.values():
                    if isinstance(v, str) and len(v) > 20:
                        texts.append(v)
                        break
            if texts:
                result["avg_text_len"] = round(sum(len(t) for t in texts) / len(texts))
            result["status"] = "ok"
        else:
            result["status"] = "empty"
            result["error"] = "no samples returned (empty split?)"
    except Exception as e:
        elapsed = time.time() - start
        result["time_sec"] = round(elapsed, 2)
        err_str = str(e)
        if "gated" in err_str.lower() or "access" in err_str.lower() or "permission" in err_str.lower():
            result["status"] = "gated"
            result["error"] = f"GATED — request access at https://huggingface.co/{path}"
        elif "not found" in err_str.lower() or "404" in err_str:
            result["status"] = "not_found"
            result["error"] = f"NOT FOUND — {path} does not exist"
        elif "split" in err_str.lower():
            result["status"] = "no_split"
            result["error"] = f"Split '{split}' not available"
        elif "name" in err_str.lower() or "configuration" in err_str.lower():
            result["status"] = "no_config"
            result["error"] = f"Config '{name}' not available"
        else:
            result["status"] = "error"
            result["error"] = err_str[:200]
    return result


def _verify_local_dataset(info: DatasetInfo) -> Dict[str, Any]:
    base = {
        "path": info.path, "name": info.name, "category": info.category,
        "error": "", "loaded": 0, "columns": [], "avg_text_len": 0,
        "time_sec": 0, "weight": info.weight, "record_count": 0,
        "valid_ratio": 0.0, "total_lines": 0,
    }
    if not info.data_dir:
        base["status"] = "invalid_local"
        base["error"] = "no data_dir specified"
        return base

    data_path = Path(info.data_dir)
    if not data_path.exists():
        base["status"] = "missing_local"
        base["error"] = f"directory not found: {data_path}"
        return base

    if not data_path.is_dir():
        base["status"] = "invalid_local"
        base["error"] = f"not a directory: {data_path}"
        return base

    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        base["status"] = "missing_local"
        base["error"] = f"no .jsonl files in {data_path}"
        return base

    start = time.time()
    total_records = 0
    total_text_len = 0
    total_lines = 0
    text_count = 0
    sample_cols: set = set()
    errors: list = []
    file_read_errors = 0

    for jf in jsonl_files:
        if not jf.is_file():
            errors.append(f"not a file: {jf}")
            file_read_errors += 1
            continue
        try:
            content = jf.read_bytes()
            content.decode("utf-8")
        except UnicodeDecodeError as e:
            errors.append(f"invalid UTF-8 in {jf.name}: {e}")
            file_read_errors += 1
            continue
        except Exception as e:
            errors.append(f"unreadable {jf.name}: {e}")
            file_read_errors += 1
            continue

        with open(str(jf), "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"invalid JSON at {jf.name}:{line_no}: {e}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"non-dict record at {jf.name}:{line_no}")
                    continue
                if total_records == 0:
                    sample_cols = set(record.keys())
                    missing = REQUIRED_DOC_FIELDS - sample_cols
                    if missing:
                        errors.append(f"missing required fields at {jf.name}: {missing}")
                total_records += 1
                text_val = record.get("text", "")
                if isinstance(text_val, str):
                    total_text_len += len(text_val)
                    text_count += 1

    elapsed = time.time() - start
    valid_ratio = total_records / max(total_lines, 1)

    base["time_sec"] = round(elapsed, 2)
    base["loaded"] = min(total_records, 5)
    base["columns"] = list(sample_cols)
    base["record_count"] = total_records
    base["total_lines"] = total_lines
    base["valid_ratio"] = round(valid_ratio, 4)
    if total_text_len > 0 and text_count > 0:
        base["avg_text_len"] = total_text_len // text_count

    # Classify based on valid_record_ratio
    if total_lines == 0 and file_read_errors > 0:
        base["status"] = "invalid_local"
        err_detail = "; ".join(errors[:3]) if errors else f"{file_read_errors} file(s) failed"
        base["error"] = f"0 of {len(jsonl_files)} file(s) readable: {err_detail}"
    elif total_lines == 0:
        base["status"] = "empty_local"
        base["error"] = f"0 non-empty lines across {len(jsonl_files)} file(s)"
    elif valid_ratio < 0.50:
        base["status"] = "invalid_local"
        base["error"] = f"valid_ratio={valid_ratio:.1%} ({total_records}/{total_lines} records)"
    elif valid_ratio < 0.95:
        base["status"] = "warning_local"
        base["error"] = f"valid_ratio={valid_ratio:.1%} ({total_records}/{total_lines} records)"
    elif errors:
        base["status"] = "warning_local"
        base["error"] = "; ".join(errors[:3])
    else:
        base["status"] = "ok_local"

    if errors and total_lines > 0 and valid_ratio >= 0.50 and "valid_ratio" not in base.get("error", ""):
        err_detail = "; ".join(errors[:3])
        if base["error"]:
            base["error"] += f" ({err_detail})"
        else:
            base["error"] = err_detail

    return base


def verify_registry(registry: DatasetRegistry, token: Optional[str] = None,
                    max_per_dataset: int = 5) -> List[Dict[str, Any]]:
    entries = registry.all_entries()
    results = []
    seen_paths = set()
    for info in entries:
        if info.path.startswith("json"):
            result = _verify_local_dataset(info)
            results.append(result)
            wt_str = f"{info.weight:.3f}".ljust(6)
            name_str = f"{info.name or '?'}"[:55].ljust(55)
            if result["status"] == "ok_local":
                logger.info("  OK_LOCAL %s %s | %d records, ratio=%.1f%%, cols=%d, avg_len=%d",
                            wt_str, name_str, result["record_count"],
                            result.get("valid_ratio", 1.0) * 100,
                            len(result["columns"]), result["avg_text_len"])
            elif result["status"] == "warning_local":
                logger.warning("  WARN_LOCAL %s %s | %s", wt_str, name_str, result["error"])
            elif result["status"] == "missing_local":
                logger.warning("  MISS_LOCAL %s %s | %s", wt_str, name_str, result["error"])
            elif result["status"] == "empty_local":
                logger.warning("  EMPTY_LOCAL %s %s | %s", wt_str, name_str, result["error"])
            else:
                logger.warning("  INVALID_LOCAL %s %s | %s", wt_str, name_str, result["error"])
            continue
        dedup_key = f"{info.path}/{info.name or ''}"
        if dedup_key in seen_paths:
            results.append({
                "path": info.path, "name": info.name, "category": info.category,
                "status": "skipped_dup", "error": "same path/name already verified",
                "loaded": 0, "columns": [], "avg_text_len": 0, "time_sec": 0,
                "weight": info.weight,
            })
            continue
        seen_paths.add(dedup_key)
        result = try_load_sample(info.path, info.name, info.split, max_samples=max_per_dataset, token=token)
        result["category"] = info.category
        result["weight"] = info.weight
        result["fallbacks"] = info.fallbacks
        results.append(result)
        status_str = result["status"].ljust(12)
        wt_str = f"{info.weight:.3f}".ljust(6)
        name_str = f"{info.path}/{info.name or ''}"[:55].ljust(55)
        if result["status"] == "ok":
            logger.info("  OK  %s %s | %d samples, cols=%d, avg_len=%d",
                        wt_str, name_str, result["loaded"], len(result["columns"]), result["avg_text_len"])
        elif result["status"] == "gated":
            logger.warning("  GATED %s %s | %s", wt_str, name_str, result["error"])
        elif result["status"] == "not_found":
            logger.warning("  MISS  %s %s | %s (fallbacks: %s)", wt_str, name_str,
                           result["error"], info.fallbacks)
        else:
            logger.warning("  FAIL  %s %s | %s (fallbacks: %s)", wt_str, name_str,
                           result["error"], info.fallbacks)
    return results


def report(results: List[Dict[str, Any]]):
    categories: Dict[str, Dict[str, float]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, {"total": 0, "ok": 0, "gated": 0, "missing": 0, "error": 0, "local": 0, "warn": 0, "weight_ok": 0.0, "weight_fail": 0.0})
        categories[cat]["total"] += 1
        categories[cat][r["status"]] = categories[cat].get(r["status"], 0) + 1
        if r["status"] in ("ok", "skipped_dup", "ok_local"):
            categories[cat]["weight_ok"] += r.get("weight", 0)
        else:
            categories[cat]["weight_fail"] += r.get("weight", 0)

    print("\n" + "=" * 80)
    print("  DATASET AVAILABILITY REPORT")
    print("=" * 80)
    print(f"\n  {'Category':20s} {'Total':6s} {'OK+Dup':6s} {'Warn':6s} {'Gated':6s} {'Missing':8s} {'LocalOK':7s} {'Wt OK':8s}")
    print("  " + "-" * 72)
    total_ok = total_fail = 0.0
    for cat in sorted(categories.keys()):
        c = categories[cat]
        ok_count = c.get('ok', 0) + c.get('skipped_dup', 0)
        local_ok = c.get('ok_local', 0)
        local_warn = c.get('warning_local', 0)
        missing = c.get('not_found', 0) + c.get('no_split', 0) + c.get('no_config', 0) + c.get('error', 0) + c.get('missing_local', 0) + c.get('empty_local', 0) + c.get('invalid_local', 0)
        print(f"  {cat:20s} {c['total']:6d} {ok_count:6d} {local_warn:6d} {c.get('gated',0):6d} {missing:8d} {local_ok:7d} {c['weight_ok']:8.3f}")
        total_ok += c["weight_ok"]
        total_fail += c["weight_fail"]

    print(f"\n  Total weight available: {total_ok:.3f}")
    print(f"  Total weight at risk:   {total_fail:.3f}")
    if total_fail > 0.01:
        print(f"\n  WARNING: {total_fail:.1%} of corpus weight may fail during training!")
        print("  Review entries marked GATED, MISS, or FAIL above.")
        print("  Add fallbacks or replace with available alternatives.")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify all registry datasets are accessible")
    parser.add_argument("--token", default=None, help="HF token for gated datasets")
    parser.add_argument("--max-samples", type=int, default=5, help="Samples to load per dataset")
    parser.add_argument("--report", default=None, help="Save report JSON to path")
    args = parser.parse_args()

    logger.info("Building registry...")
    registry = build_registry()
    logger.info("Verifying %d entries...", len(registry.all_entries()))
    results = verify_registry(registry, token=args.token, max_per_dataset=args.max_samples)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Report saved to %s", args.report)

    report(results)

    fail_count = sum(1 for r in results if r["status"] not in ("ok", "ok_local", "skipped_dup"))
    if fail_count > 0:
        logger.warning("\n%d entries may fail during training. Review and fix.", fail_count)
        sys.exit(1)


if __name__ == "__main__":
    main()
