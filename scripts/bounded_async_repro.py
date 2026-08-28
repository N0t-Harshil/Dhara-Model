#!/usr/bin/env python3
"""
Bounded live repro of the async TypeError hunt (Phase 19).

Drives the real DataPipeline through the real UnitPrefetch worker exactly like
the dataset-granular training loop (training/pipeline.py), but with only the
first TWO registry datasets of stage 0 and a tiny per-dataset cap, so a full
build stays within a few minutes on CPU.

Usage: python scripts/bounded_async_repro.py [--config config_foundation.yaml]

Any build failure is surfaced with its phase classification + context, so the
exact handling site of a TypeError (or any other error) can be pinned down
without greping stderr text.
"""

import argparse
import logging
import sys
import time
import traceback
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

from src.config.schema import load_config  # noqa: E402
from src.data.pipeline import DataPipeline  # noqa: E402
from src.data.registry import build_registry  # noqa: E402
from src.models.factory import ModelFactory  # noqa: E402
from src.training.asyncprefetch import (  # noqa: E402
    UnitPrefetch,
    _phase_of,
    _retryable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_foundation.yaml")
    parser.add_argument("--n-units", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--cache", dest="cache", action="store_true",
                        help="mirror production: enable the packed dataset cache")
    parser.add_argument("--no-cache", dest="cache", action="store_false",
                        help="disable the packed dataset cache (hermetic repro)")
    parser.add_argument("--metadata", dest="metadata", action="store_true",
                        help="enable HF metadata caching (mirrors production)")
    parser.add_argument("--no-metadata", dest="metadata", action="store_false",
                        help="disable HF metadata caching")
    parser.set_defaults(cache=True, metadata=True)
    args = parser.parse_args()

    cfg = load_config(str(PROJECT_ROOT / args.config))
    cfg.data.max_samples_per_dataset = args.max_samples
    cfg.data.use_packed_cache = args.cache
    cfg.data.metadata_cache.enabled = args.metadata

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.tokenizer_trainer import download_tokenizer

    tok_dir = PROJECT_ROOT / "models" / "tokenizer"
    download_tokenizer(output_dir=str(tok_dir),
                       model_id=cfg.tokenizer.huggingface_model, force=False)
    tokenizer = ModelFactory.load_tokenizer(path=str(tok_dir), cfg=cfg)

    stg = getattr(cfg.training.pretrain, "stages", [None])[0]
    registry = build_registry()
    units = registry.all_entries()
    cats = getattr(stg, "categories", None)
    if cats:
        units = [u for u in units if u.category in cats]
    if args.n_units:
        units = units[: args.n_units]
    print("=" * 72)
    print("PHASE 19 — bounded async build repro")
    print("=" * 72)
    print(f"stage 0 categories : {cats}")
    print(f"units to build     : {len(units)} (cap {args.max_samples} samples each)")
    print(f"packed cache       : {'ON' if args.cache else 'OFF'}   "
          f"HF metadata cache : {'ON' if args.metadata else 'OFF'}")
    for i, u in enumerate(units, 1):
        print(f"  [{i}] {getattr(u, 'path', '?')}/{getattr(u, 'name', 'default')} "
              f"cat={getattr(u, 'category', '?')} "
              f"split={getattr(u, 'split', '?')}")

    dp = DataPipeline(cfg, tokenizer)
    plan = [{"j": j, "u": u} for j, u in enumerate(units, 1)]
    total = len(plan)

    prefetch = UnitPrefetch(
        build_fn=lambda item, idx: dp.build_pretrain_dataset_unit(
            item["u"], 0, item["j"], total),
        total=total,
        depth=args.depth,
        timeout=600.0,
        name="p19",
        retries=0,
        max_workers=2,
        cache_status=lambda item, idx: dp.unit_cache_hit(item["u"], 0, item["j"]),
    )
    prefetch.start(plan)

    ok = 0
    t0 = time.perf_counter()
    for i in range(total):
        item = plan[i]
        tag = f"{item['u'].path}/{item['u'].name or 'default'}"
        try:
            res = prefetch.get(i)
            if isinstance(res, tuple) and len(res) == 2:
                dataset, _meta = res
            elif isinstance(res, tuple):
                dataset = res[0]
            else:
                dataset = res
            n = len(dataset)
            ok += 1
            elapsed = time.perf_counter() - t0
            print(f"[P19] [{item['j']}/{total}] OK  {tag}  samples={n} ({elapsed:.1f}s)")
        except Exception as e:  # noqa: BLE001
            phase, ctx = _phase_of(e)
            elapsed = time.perf_counter() - t0
            print("!" * 72)
            print(f"[P19] [{item['j']}/{total}] FAILED  {tag}  after {elapsed:.1f}s")
            print(f"[P19] phase    : {phase!r}   retryable={_retryable(e)}")
            if ctx:
                print(f"[P19] context  : {ctx}")
            print("!" * 72)
            traceback.print_exc()
    prefetch.close()

    print("=" * 72)
    print(f"PHASE 19 RESULT: {ok}/{total} built cleanly")
    print("=" * 72)
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())