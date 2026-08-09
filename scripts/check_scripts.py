from __future__ import annotations

import os, sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HF_TOKEN"] = ""

import requests
from src.data.registry import build_registry


def check_script(path: str, name: str | None) -> dict:
    """Check if a dataset uses a Python script (broken on datasets 2.20+) or is Parquet/JSONL."""
    result = {"path": path, "name": name, "script": False, "parquet": False, "exists": False, "gated": False}
    try:
        r = requests.get(f"https://huggingface.co/api/datasets/{path}", timeout=15)
        if r.status_code != 200:
            if r.status_code == 401 or r.status_code == 403:
                result["gated"] = True
            return result
        data = r.json()
        result["exists"] = True
        # Check configs
        configs = data.get("configs", [])
        if not configs:
            configs = [{"config": {"name": name or "default"}}] if name else [{}]
        found_script = False
        found_parquet = False
        for cfg in configs:
            cfg_name = cfg.get("config", {}).get("name", "default")
            if name and cfg_name != name:
                continue
            # Check if this config has parquet files
            try:
                r2 = requests.get(
                    f"https://huggingface.co/api/datasets/{path}/tree/main/data",
                    timeout=10
                )
                if r2.status_code == 200:
                    files = r2.json()
                    for f in files:
                        fn = f.get("path", "")
                        if fn.endswith(".parquet"):
                            found_parquet = True
                        if fn.endswith(".py") or fn.endswith(".txt"):
                            pass  # config files, not data
            except Exception:
                pass
            # Check for dataset script in root
            try:
                r3 = requests.head(
                    f"https://huggingface.co/{path}/resolve/main/{path.split('/')[-1]}.py",
                    timeout=10
                )
                if r3.status_code == 200:
                    found_script = True
            except Exception:
                pass
        result["script"] = found_script and not found_parquet
        result["parquet"] = found_parquet
    except Exception as e:
        result["error"] = str(e)[:100]
    return result


def main():
    registry = build_registry()
    seen: set = set()
    results = []
    for e in registry.all_entries():
        key = f"{e.path}/{e.name or ''}"
        if key in seen:
            continue
        seen.add(key)
        if e.path == "json":
            results.append({"path": e.path, "name": e.name, "status": "json"})
            continue
        if e.path in ("HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-edu", "cerebras/SlimPajama-627B",
                      "code-search-net/code_search_net", "codeparrot/codeparrot-clean"):
            results.append({"path": e.path, "name": e.name, "status": "known_parquet"})
            continue
        r = check_script(e.path, e.name)
        results.append(r)

    print("=" * 80)
    print("  DATASET SCRIPT CHECK (datasets 2.20+ compatibility)")
    print("=" * 80)
    for r in sorted(results, key=lambda x: x.get("status", "") + x["path"]):
        p = f"{r['path']}/{r.get('name') or ''}"
        st = r.get("status", "")
        if st == "json":
            print(f"  JSONL  {p}")
        elif st == "known_parquet":
            print(f"  PARQ   {p}")
        elif r.get("gated"):
            print(f"  GATED  {p}")
        elif r.get("script") and not r.get("parquet"):
            print(f"  SCRIPT {p}  <-- WILL FAIL on datasets 2.20+")
        elif r.get("parquet"):
            print(f"  PARQ   {p}")
        else:
            print(f"  ?      {p}  {r.get('error', 'unknown')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
