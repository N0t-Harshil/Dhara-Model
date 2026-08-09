from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def audit_jsonl(path: Path) -> Dict[str, Any]:
    result = {
        "source": path.parent.name,
        "path": str(path),
        "status": "PASS",
        "total_lines": 0,
        "valid_records": 0,
        "empty_docs": 0,
        "duplicates": 0,
        "html_leakage": 0,
        "avg_length": 0,
        "median_length": 0,
        "max_length": 0,
        "min_length": 0,
        "total_tokens": 0,
        "errors": [],
    }
    if not path.exists():
        result["status"] = "MISSING"
        result["errors"].append("File not found")
        return result

    try:
        content = path.read_bytes()
        content.decode("utf-8")
    except UnicodeDecodeError as e:
        result["status"] = "CORRUPT"
        result["errors"].append(f"Invalid UTF-8: {e}")
        return result
    except Exception as e:
        result["status"] = "CORRUPT"
        result["errors"].append(f"Unreadable: {e}")
        return result

    lengths: List[int] = []
    texts_seen: set = set()
    with open(str(path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            result["total_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                result["errors"].append(f"Line {line_no}: invalid JSON: {e}")
                continue
            if not isinstance(record, dict):
                result["errors"].append(f"Line {line_no}: non-dict record")
                continue
            result["valid_records"] += 1

            text = record.get("text", "")
            if not isinstance(text, str) or not text.strip():
                result["empty_docs"] += 1
            if isinstance(text, str):
                text_len = len(text)
                lengths.append(text_len)
                token_est = text_len // 4
                result["total_tokens"] += token_est
                text_lower = text.lower()
                if "<html" in text_lower or "<!doctype" in text_lower or "<div" in text_lower or "<p>" in text_lower:
                    result["html_leakage"] += 1
                if text in texts_seen:
                    result["duplicates"] += 1
                texts_seen.add(text)

    if lengths:
        lengths.sort()
        n = len(lengths)
        result["min_length"] = lengths[0]
        result["max_length"] = lengths[-1]
        result["avg_length"] = sum(lengths) // n
        if n % 2 == 1:
            result["median_length"] = lengths[n // 2]
        else:
            result["median_length"] = (lengths[n // 2 - 1] + lengths[n // 2]) // 2

    dup_ratio = result["duplicates"] / max(result["valid_records"], 1)
    empty_ratio = result["empty_docs"] / max(result["valid_records"], 1)
    html_ratio = result["html_leakage"] / max(result["valid_records"], 1)
    err_ratio = (result["total_lines"] - result["valid_records"]) / max(result["total_lines"], 1)

    if result["valid_records"] == 0:
        result["status"] = "CORRUPT"
        result["errors"].append("No valid records")
    if err_ratio > 0.05:
        result["status"] = "CORRUPT"
        result["errors"].append(f"{err_ratio:.1%} parse errors")
    if html_ratio > 0.01:
        result["status"] = "HTML_LEAKAGE"
        result["errors"].append(f"{html_ratio:.1%} records contain HTML tags")
    if dup_ratio > 0.50:
        result["status"] = "HIGH_DUP"
        result["errors"].append(f"{dup_ratio:.1%} duplicate documents")
    if empty_ratio > 0.05:
        if result["status"] in ("PASS",):
            result["status"] = "PASS_WARN"
        result["errors"].append(f"{empty_ratio:.1%} empty documents")
    if result["valid_records"] > 0 and result["avg_length"] < 100:
        if result["status"] in ("PASS",):
            result["status"] = "PASS_WARN"
        result["errors"].append(f"Avg length {result['avg_length']} chars - very short documents")

    return result


def main():
    data_dirs = [
        Path("data/docs"),
        Path("data/web_text"),
        Path("data/math"),
        Path("data/science"),
        Path("data/books"),
    ]
    all_results: List[Dict[str, Any]] = []
    total_valid = 0
    total_tokens = 0

    for dd in data_dirs:
        if not dd.exists():
            logger.info("  %s: no data directory", dd)
            continue
        for jsonl_file in sorted(dd.rglob("*.jsonl")):
            result = audit_jsonl(jsonl_file)
            all_results.append(result)
            total_valid += result["valid_records"]
            total_tokens += result["total_tokens"]

    print("\n" + "=" * 90)
    print("  CORPUS QUALITY AUDIT")
    print("=" * 90)
    print(f"\n  {'Source':20s} {'Status':15s} {'Records':8s} {'Avg Len':8s} {'Med Len':8s} {'Max Len':8s} {'Empty':6s} {'Dup%':6s} {'HTML%':6s}")
    print("  " + "-" * 90)
    for r in sorted(all_results, key=lambda x: x["source"]):
        status = r["status"]
        avg = str(r["avg_length"])
        med = str(r["median_length"])
        mx = str(r["max_length"])
        dup_pct = f"{r['duplicates']/max(r['valid_records'],1)*100:.0f}%"
        html_pct = f"{r['html_leakage']/max(r['valid_records'],1)*100:.0f}%"
        print(f"  {r['source']:20s} {status:15s} {r['valid_records']:8d} {avg:>8s} {med:>8s} {mx:>8s} {r['empty_docs']:6d} {dup_pct:>6s} {html_pct:>6s}")

    print(f"\n  Total valid records: {total_valid}")
    print(f"  Total estimated tokens: {total_tokens:,}")

    blocking = [r for r in all_results if r["status"] in ("CORRUPT", "HTML_LEAKAGE", "HIGH_DUP")]
    if blocking:
        print(f"\n  BLOCKING ISSUES:")
        for r in blocking:
            print(f"    {r['source']:20s}: {'; '.join(r['errors'][:3])}")
    else:
        print(f"\n  No blocking issues found.")

    warn = [r for r in all_results if r["status"] == "PASS_WARN"]
    if warn:
        print(f"\n  WARNINGS:")
        for r in warn:
            print(f"    {r['source']:20s}: {'; '.join(r['errors'][:3])}")

    has_errors = any(r["status"] not in ("PASS", "PASS_WARN") for r in all_results)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
