from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.data.registry import build_registry


def try_entry(path: str, name: str | None, data_dir: str | None, timeout_sec: int = 30) -> dict:
    result = {"path": path, "name": name, "status": "?", "detail": "", "time": 0}
    kwargs = {"path": path, "split": "train", "streaming": True}
    if name:
        kwargs["name"] = name
    if data_dir:
        kwargs["data_dir"] = data_dir

    start = time.time()
    try:
        ds = load_dataset(**kwargs)
        deadline = start + timeout_sec
        count = 0
        text_fields = []
        for s in ds:
            if time.time() > deadline:
                result["status"] = "timeout"
                break
            if count >= 3:
                break
            if count == 0:
                text_fields = [k for k, v in s.items() if isinstance(v, str) and len(v) > 50]
            count += 1
        elapsed = round(time.time() - start, 1)
        result["time"] = elapsed
        if result["status"] != "timeout":
            result["status"] = "ok" if count > 0 else "empty"
            result["detail"] = f"{count} samples, text_fields={text_fields[:5]}"
            result["text_fields"] = text_fields[:5]
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        result["time"] = elapsed
        err = str(e)[:150]
        if "gated" in err.lower() or "access" in err.lower():
            result["status"] = "gated"
        elif "not found" in err.lower() or "404" in err:
            result["status"] = "not_found"
        elif "cast" in err.lower() or "schema" in err.lower():
            result["status"] = "schema_mismatch"
        else:
            result["status"] = "error"
        result["detail"] = err
    return result


def main():
    registry = build_registry()
    seen: set = set()
    results = []
    for e in registry.all_entries():
        key = f"{e.path}/{e.name or ''}/{e.data_dir or ''}"
        if key in seen:
            continue
        seen.add(key)
        if e.path == "json":
            results.append({"path": e.path, "name": e.name, "status": "json", "detail": "local jsonl (skipped)", "time": 0})
            continue
        r = try_entry(e.path, e.name, e.data_dir)
        r["category"] = e.category
        r["weight"] = e.weight
        results.append(r)

    fail_w = 0.0
    ok_w = 0.0
    print()
    print("=" * 80)
    print("  EVERY REGISTRY ENTRY VERIFICATION")
    print("=" * 80)
    for r in sorted(results, key=lambda x: (x.get("category", ""), x["path"])):
        p = f"{r['path']}/{r.get('name') or ''}"
        s = r["status"]
        d = r.get("detail", "")
        w = r.get("weight", 0)
        t = r.get("time", 0)
        if s == "ok":
            ok_w += w
            print(f"  OK   [{t:4.0f}s] {p:55s} {d}")
        elif s == "json":
            ok_w += w
            print(f"  JSON [{t:4.0f}s] {p:55s} (local JSONL - skipped)")
        elif s == "gated":
            fail_w += w
            print(f"  GATE [{t:4.0f}s] {p:55s} needs HF_TOKEN")
        elif s in ("not_found", "schema_mismatch", "error", "timeout"):
            fail_w += w
            print(f"  FAIL [{t:4.0f}s] {p:55s} {d}")
        else:
            fail_w += w
            print(f"  ?    [{t:4.0f}s] {p:55s} {s}: {d}")

    print()
    print(f"  OK weight:     {ok_w:.3f}")
    print(f"  FAIL weight:   {fail_w:.3f}")
    if fail_w > 0:
        print(f"\n  WARNING: {fail_w:.3f} weight in failed entries.")
        print("  These entries must be fixed before training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
