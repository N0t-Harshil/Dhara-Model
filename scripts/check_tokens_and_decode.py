from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from src.config.schema import load_config
from src.data.pipeline import remove_boilerplate, random_window_sample, pack_sequences
from src.data.quality import document_quality_score
from src.data.registry import build_registry, extract_text
from src.data.streaming import stream_dataset_with_fallbacks


def main():
    tok = AutoTokenizer.from_pretrained("Xenova/claude-tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = load_config("config_foundation.yaml")
    max_len = cfg.training.max_seq_length or 2048
    eos_id = tok.eos_token_id or 0
    random.seed(42)

    registry = build_registry()
    seen_cats: set = set()
    entries = []
    for e in registry.all_entries():
        if e.category not in seen_cats:
            entries.append(e)
            seen_cats.add(e.category)

    print(f"Testing {len(entries)} entries (one per category)")
    all_packed = []
    cat_tokens: dict = defaultdict(int)

    for info in entries:
        cat = info.category
        path_str = f"{info.path}/{info.name or ''}"
        print(f"  Streaming {cat} ({path_str})...", end=" ", flush=True)
        try:
            samples = list(stream_dataset_with_fallbacks(info, registry, limit=10))
            print(f"{len(samples)} raw samples", end="", flush=True)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        cleaned = []
        quality_scores = []
        for s in samples:
            t = extract_text(s, info.text_fields)
            if not t:
                continue
            t = remove_boilerplate(t)
            if len(t) < 50:
                continue
            cleaned.append(t)
            qs = document_quality_score(t, category=cat, language=info.language)
            quality_scores.append(qs["final"])

        if not cleaned:
            print(" (no valid text after cleaning)")
            continue

        tokenized = []
        for idx, text in enumerate(cleaned):
            tok_ids = tok.encode(text, add_special_tokens=False)
            tok_ids = random_window_sample(tok_ids, max_len)
            doc_qs = quality_scores[idx] if idx < len(quality_scores) else 0.5
            tokenized.append({"input_ids": tok_ids, "quality_score": doc_qs})

        packed, eff = pack_sequences(tokenized, max_len, eos_id)
        for p in packed:
            p["_category"] = cat

        all_packed.extend(packed)
        tcount = sum(len(p["input_ids"]) for p in packed)
        cat_tokens[cat] += tcount
        print(f" -> {len(packed)} packed seqs, {tcount} tokens, eff={eff:.1%}")

    total = sum(cat_tokens.values())
    print()
    print("=" * 60)
    print("  PACKED TOKEN DISTRIBUTION")
    print("=" * 60)
    for cat, tokens in sorted(cat_tokens.items()):
        print(f"  {cat:25s}: {tokens:6d} tokens ({tokens/total*100:5.1f}%)")
    print(f"  {'':25s}  ------")
    print(f"  {'Total':25s}: {total:6d} tokens")

    print()
    print("=" * 60)
    print("  DECODED PACKED SAMPLES")
    print("=" * 60)

    mixed = sorted(all_packed, key=lambda x: hash(f"{x.get('_category', '')}_{x.get('input_ids', [None])[0]}"))
    for i in range(min(10, len(mixed))):
        s = mixed[i]
        cat = s.get("_category", "?")
        segs = s.get("_segments", 1)
        avg_q = s.get("_avg_quality", 0.0)
        text = tok.decode(s["input_ids"], skip_special_tokens=False)
        print(f"--- Packed {i} | cat={cat} | {segs} docs | q={avg_q:.3f} | {len(s['input_ids'])} tokens ---")
        print(text[:1000])
        if len(text) > 1000:
            print("...[truncated]")
        print()


if __name__ == "__main__":
    main()
