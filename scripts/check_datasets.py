from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from src.data.registry import build_registry, DatasetRegistry

REAL_HF_DATASETS = {
    "HuggingFaceFW/fineweb",
    "HuggingFaceFW/fineweb-edu",
    "cerebras/SlimPajama-627B",
    "code-search-net/code_search_net",
    "codeparrot/codeparrot-clean",
    "deepmind/code_contests",
    "deepmind/pg19",
    "wikimedia/wikipedia",
    "open-web-math/open-web-math",
    "AI-MO/NuminaMath-CoT",
    "AI-MO/NuminaMath-1.5",
    "GAIR/MathPile",
    "lean-dojo/lean4",
    "google/wit",
    "bigcode/the-stack-v2-dedup",
    "HuggingFaceFW/fineweb-code",
}

GATED_DATASETS = {
    "bigcode/the-stack-v2-dedup",
    "HuggingFaceFW/fineweb-code",
}

def check_hf_exists(path: str) -> Optional[str]:
    try:
        import requests
        url = f"https://huggingface.co/api/datasets/{path}"
        r = requests.head(url, timeout=10, allow_redirects=True)
        if r.status_code == 200:
            return "exists"
        elif r.status_code == 401:
            return "gated"
        else:
            body = requests.get(url, timeout=10).json()
            if "gated" in body.get("private", False) or body.get("disabled", False):
                return "gated"
            return "unknown"
    except Exception:
        return "error"

def check_entry(path: str, name: Optional[str]) -> str:
    if path == "json":
        return "local_jsonl"
    dedup_key = f"{path}/{name or ''}"
    if dedup_key in {
        "bigcode/the-stack-v2-dedup/Python",
        "bigcode/the-stack-v2-dedup/C++",
        "bigcode/the-stack-v2-dedup/JavaScript",
        "bigcode/the-stack-v2-dedup/TypeScript",
        "bigcode/the-stack-v2-dedup/Java",
        "bigcode/the-stack-v2-dedup/Rust",
        "bigcode/the-stack-v2-dedup/Go",
        "bigcode/the-stack-v2-dedup/SQL",
        "bigcode/the-stack-v2-dedup/Shell",
        "bigcode/the-stack-v2-dedup/C#",
        "bigcode/the-stack-v2-dedup/PHP",
        "bigcode/the-stack-v2-dedup/Kotlin",
        "bigcode/the-stack-v2-dedup/Swift",
        "bigcode/the-stack-v2-dedup/R",
        "bigcode/the-stack-v2-dedup/Julia",
        "bigcode/the-stack-v2-dedup/Scala",
        "bigcode/the-stack-v2-dedup/Lua",
        "bigcode/the-stack-v2-dedup/C",
        "bigcode/the-stack-v2-dedup/Objective-C",
    }:
        return "gated (Stack v2)"
    base = path.split("/")[0] if "/" in path else path
    if path in REAL_HF_DATASETS:
        if path in GATED_DATASETS:
            return "gated"
        return "verified_real"
    likely_fake = {
        "proof-pile", "proofwiki", "openalex", "pubmed",
        "gutenberg", "dbpedia", "conceptnet", "wordnet",
        "arxiv", "acl_anthology", "libretexts", "openstax",
        "wikidata", "algorithmicresearchgroup",
    }
    if base in likely_fake or any(f in path for f in likely_fake):
        return "likely_fake"
    return "unknown"


def main():
    registry = build_registry()
    entries = registry.all_entries()
    print("=" * 80)
    print("  REGISTRY DATASET VERIFICATION")
    print("=" * 80)
    rows = []
    ok_w, fail_w, gated_w, json_w = 0.0, 0.0, 0.0, 0.0
    for e in sorted(entries, key=lambda x: (x.category, x.path)):
        status = check_entry(e.path, e.name)
        rows.append((e.category, e.path, e.name or "", e.weight, status))
        if status == "verified_real":
            ok_w += e.weight
        elif status in ("gated", "gated (Stack v2)"):
            gated_w += e.weight
        elif status == "local_jsonl":
            json_w += e.weight
        elif status == "likely_fake":
            fail_w += e.weight
        else:
            fail_w += e.weight

    print(f"\n  {'Category':20s} {'Path':40s} {'Name':20s} {'Weight':8s} {'Status'}")
    print("  " + "-" * 100)
    for cat, path, name, weight, status in rows:
        status_mark = {
            "verified_real": "OK",
            "gated": "GATED",
            "gated (Stack v2)": "GATED",
            "likely_fake": "FAKE",
            "local_jsonl": "JSONL",
            "unknown": "?",
        }.get(status, "?")
        print(f"  {cat:20s} {path:40s} {name:20s} {weight:<8.4f} {status_mark}")

    print()
    print(f"  Verified real:      {ok_w:.3f} ({ok_w*100:.1f}%)")
    print(f"  Gated (needs token): {gated_w:.3f} ({gated_w*100:.1f}%)")
    print(f"  Local JSONL:        {json_w:.3f} ({json_w*100:.1f}%)")
    print(f"  Likely fake:        {fail_w:.3f} ({fail_w*100:.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:              {ok_w+gated_w+json_w+fail_w:.3f}")
    print()
    if fail_w > 0.001:
        print(f"  WARNING: {fail_w:.3f} weight ({fail_w*100:.1f}%) from entries that will")
        print("  always trigger fallbacks. These weight entries redirect to fallback")
        print("  datasets (FineWeb-Edu, FineWeb, etc.), changing the actual training mixture.")
        print("  Consider replacing with real datasets or accepting the mixture shift.")
    print("=" * 80)


if __name__ == "__main__":
    main()
