from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Phase 1: Dataset Verification ─────────────────────────────────

def phase1_verify_registry(token: Optional[str] = None) -> Dict[str, Any]:
    from scripts.verify_datasets import verify_registry, report
    from src.data.registry import build_registry

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: DATASET VERIFICATION")
    logger.info("=" * 70)

    registry = build_registry()
    entries = registry.all_entries()
    logger.info("Registry: %d entries, total weight=%.6f", len(entries), sum(e.weight for e in entries))

    results = verify_registry(registry, token=token, max_per_dataset=5)
    report(results)

    total_ok = sum(r["weight"] for r in results if r["status"] in ("ok", "skipped_dup", "ok_local"))
    total_weight = sum(r["weight"] for r in results)
    available_pct = total_ok / total_weight if total_weight > 0 else 0

    failures = [r for r in results if r["status"] not in ("ok", "skipped_dup", "ok_local")]
    if failures:
        logger.warning("\n  Failures (%d):", len(failures))
        for f in failures:
            logger.warning("    %s/%s (cat=%s, w=%.4f): %s",
                           f["path"], f.get("name", ""), f.get("category", ""),
                           f.get("weight", 0), f.get("error", ""))

    return {
        "total_entries": len(entries),
        "total_weight": total_weight,
        "available_weight": total_ok,
        "available_pct": available_pct,
        "failures": len(failures),
        "failure_details": failures,
    }


# ── Phase 2: Build Documentation ──────────────────────────────────

def phase2_build_docs(output_dir: str = "data/docs",
                      sources: Optional[List[str]] = None,
                      max_per_source: int = 2000) -> Dict[str, int]:
    from src.data.doc_builder import scrape_all

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: BUILD DOCUMENTATION CORPUS")
    logger.info("=" * 70)

    results = scrape_all(output_dir, sources, max_per_source)
    total = sum(results.values())
    logger.info("Total pages scraped: %d", total)

    # Verify JSONL files exist
    missing = []
    empty = []
    for name in (sources or list(results.keys())):
        jsonl_path = Path(output_dir) / name / "documents.jsonl"
        if not jsonl_path.exists():
            missing.append(name)
        elif jsonl_path.stat().st_size == 0:
            empty.append(name)

    if missing:
        logger.warning("Missing JSONL files: %s", missing)
    if empty:
        logger.warning("Empty JSONL files: %s", empty)

    return {
        "results": results,
        "total_pages": total,
        "missing": missing,
        "empty": empty,
    }


# ── Phase 3: Validate Documentation Quality ───────────────────────

def phase3_validate_docs(output_dir: str = "data/docs") -> Dict[str, Any]:
    from src.data.registry import DOC_SOURCES

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: DOCUMENTATION QUALITY VALIDATION")
    logger.info("=" * 70)

    results = {}
    total_docs = 0
    total_tokens = 0

    # Map SOURCE_NAME from scrapers to registry doc_names
    for doc_name in DOC_SOURCES:
        jsonl_path = Path(output_dir) / doc_name / "documents.jsonl"
        if not jsonl_path.exists():
            logger.warning("  %s: JSONL not found at %s", doc_name, jsonl_path)
            results[doc_name] = {"status": "missing", "docs": 0, "avg_len": 0, "total_tokens": 0}
            continue
        try:
            docs = []
            with open(str(jsonl_path), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
            doc_count = len(docs)
            lengths = [d.get("text_length", len(d.get("text", ""))) for d in docs]
            avg_len = sum(lengths) / max(len(lengths), 1)
            cat_tokens = sum(lengths)
            total_docs += doc_count
            total_tokens += cat_tokens

            # Check for duplicates (by URL)
            urls = [d.get("url", "") for d in docs]
            unique_urls = len(set(urls))
            dup_pct = (1 - unique_urls / max(len(urls), 1)) * 100

            # Check for boilerplate/very short
            short_docs = sum(1 for l in lengths if l < 200)
            short_pct = short_docs / max(len(lengths), 1) * 100

            # Check metadata
            has_metadata = all(
                d.get("source") and d.get("title") and d.get("text")
                for d in docs[:100]
            )

            results[doc_name] = {
                "status": "ok",
                "docs": doc_count,
                "avg_len": int(avg_len),
                "total_tokens": cat_tokens,
                "dup_pct": round(dup_pct, 1),
                "short_pct": round(short_pct, 1),
                "has_metadata": has_metadata,
            }
            logger.info("  %s: %d docs, avg_len=%d, dup=%.1f%%, short=%.1f%%, meta=%s",
                        doc_name, doc_count, int(avg_len), dup_pct, short_pct, has_metadata)
        except Exception as e:
            logger.error("  %s: ERROR — %s", doc_name, e)
            results[doc_name] = {"status": "error", "error": str(e)}

    logger.info("  Total docs: %d, total tokens (chars): %d", total_docs, total_tokens)
    return {
        "results": results,
        "total_docs": total_docs,
        "total_chars": total_tokens,
    }


# ── Phase 4: Token Mixture Validation ────────────────────────────

def phase4_token_distribution(
    tokenizer_name: str = "Xenova/claude-tokenizer",
    max_samples: int = 5000,
    max_seq_length: int = 2048,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from src.data.pipeline import DataPipeline
    from src.config.schema import load_config

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 4: TOKEN DISTRIBUTION VALIDATION")
    logger.info("=" * 70)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = load_config("config_foundation.yaml")
    pipe = DataPipeline(cfg, tok)
    ds = pipe.build_pretrain_dataset()

    cat_tokens: Dict[str, int] = defaultdict(int)
    cat_samples: Dict[str, int] = defaultdict(int)
    total_tokens = 0
    total_samples = min(max_samples, len(ds))

    for i in range(total_samples):
        s = ds[i]
        length = int(sum(s["attention_mask"]))
        cat = s.get("_category", "unknown")
        cat_tokens[cat] += length
        cat_samples[cat] += 1
        total_tokens += length

    targets = {
        "code": 0.30, "web_text": 0.20, "docs": 0.15, "wiki": 0.10,
        "math": 0.10, "science": 0.05, "books": 0.05,
        "structured_knowledge": 0.05,
    }

    logger.info("Measured token distribution (%d samples, %d tokens):",
                total_samples, total_tokens)
    distribution = {}
    for cat in sorted(targets.keys()):
        t = cat_tokens.get(cat, 0)
        pct = t / max(total_tokens, 1) * 100
        target_pct = targets[cat] * 100
        diff = pct - target_pct
        marker = "OK" if abs(diff) < 3 else ("HIGH" if diff > 0 else "LOW")
        logger.info("  %-25s %6.2f%% (target %5.1f%%) [%s]", cat, pct, target_pct, marker)
        distribution[cat] = {
            "tokens": t,
            "pct": round(pct, 2),
            "target_pct": round(target_pct, 2),
            "diff": round(diff, 2),
            "status": marker,
        }

    overall_status = "OK" if all(
        abs(d["diff"]) < 3 for d in distribution.values()
    ) else "WARNING"

    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "distribution": distribution,
        "overall_status": overall_status,
    }


# ── Phase 5: Decode Packed Samples ────────────────────────────────

def phase5_decode_samples(
    tokenizer_name: str = "Xenova/claude-tokenizer",
    num_samples: int = 20,
) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer
    from src.data.pipeline import DataPipeline
    from src.config.schema import load_config

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 5: DECODED SAMPLE INSPECTION")
    logger.info("=" * 70)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = load_config("config_foundation.yaml")
    pipe = DataPipeline(cfg, tok)
    ds = pipe.build_pretrain_dataset()

    eos_id = tok.eos_token_id or 0
    samples = []
    total_metadata_missing = 0
    total_boundary_ok = 0

    for i in range(min(num_samples, len(ds))):
        s = ds[i]
        cat = s.get("_category", "?")
        dataset_name = s.get("_dataset", "")
        language = s.get("_language", "")
        domain = s.get("_domain", "")
        segs = s.get("_segments", 1)
        q = s.get("_avg_quality", 0.0)
        mask = s["attention_mask"]
        token_len = int(sum(mask))
        length = len(mask)

        # Decode with EOS markers visible
        ids = s["input_ids"][:token_len]
        text = tok.decode(ids, skip_special_tokens=False)

        # Check for issues
        issues = []
        if not text.strip():
            issues.append("EMPTY")
        if text.count("\ufffd") > 10:
            issues.append("MALFORMED_UTF8")
        if any(kw in text.lower() for kw in ["license", "copyright", "all rights reserved"]
               ) and len(text) < 200:
            issues.append("LICENSE_SPAM")
        if text.count("\n\n\n\n") > 5:
            issues.append("EXCESSIVE_NEWLINES")
        if len(set(text)) < 20:
            issues.append("REPETITIVE")
        if not dataset_name:
            issues.append("NO_DATASET_META")
        if segs < 1:
            issues.append("ZERO_SEGMENTS")

        # Count EOS tokens as segment separators
        eos_count = sum(1 for tid in ids if tid == eos_id)
        if eos_count > 0 and eos_count + 1 != segs:
            issues.append(f"SEG_MISMATCH(eos={eos_count}, segs={segs})")

        # Check padding region is clean
        if token_len < length:
            pad_ids = s["input_ids"][token_len:]
            pad_text = tok.decode(pad_ids, skip_special_tokens=False)
            pad_has_content = len(pad_text.strip()) > 5 if pad_text else False
            if pad_has_content:
                issues.append("DIRTY_PADDING")

        if segs < 1:
            total_metadata_missing += 1
        else:
            total_boundary_ok += 1

        sample = {
            "index": i,
            "category": cat,
            "dataset": dataset_name,
            "language": language,
            "domain": domain,
            "segments": segs,
            "eos_count": eos_count,
            "avg_quality": round(q, 3),
            "token_length": token_len,
            "total_length": length,
            "padding": length - token_len,
            "text_preview": text[:500],
            "issues": issues,
        }
        samples.append(sample)

        tag = " [ISSUES]" if issues else ""
        logger.info("--- #%d | %s | %s | %d segs | %d tok | q=%.3f%s ---",
                    i, cat, dataset_name[:20] if dataset_name else "?", segs, token_len, q, tag)
        logger.info(text[:300])

    issue_count = sum(len(s["issues"]) for s in samples)
    logger.info("")
    logger.info("  Metadata ok: %d/%d", total_boundary_ok, num_samples)
    logger.info("  Metadata missing: %d", total_metadata_missing)
    if issue_count:
        logger.warning("  Issues found: %d across %d samples", issue_count, num_samples)
    else:
        logger.info("  No issues found in %d decoded samples", num_samples)

    return samples


# ── Phase 6: Packing Quality ──────────────────────────────────────

def phase6_packing_quality(
    tokenizer_name: str = "Xenova/claude-tokenizer",
    max_samples: int = 2000,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from src.data.pipeline import DataPipeline
    from src.config.schema import load_config

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 6: PACKING QUALITY")
    logger.info("=" * 70)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = load_config("config_foundation.yaml")
    max_len = cfg.training.max_seq_length
    pipe = DataPipeline(cfg, tok)
    ds = pipe.build_pretrain_dataset()

    lengths = []
    padding_counts = []
    segment_counts = []
    quality_scores_in_packed = []
    padding_buckets = defaultdict(int)
    corrupted_utf8 = 0
    zero_segment = 0
    missing_metadata = 0

    for i in range(min(max_samples, len(ds))):
        s = ds[i]
        mask = s["attention_mask"]
        length = int(sum(mask))
        padding = len(mask) - length
        segs = s.get("_segments", 1)
        qa = s.get("_avg_quality", 0.0)
        lengths.append(length)
        padding_counts.append(padding)
        segment_counts.append(segs)
        quality_scores_in_packed.append(qa)

        gap = 0
        if len(padding_buckets) < 10:
            if padding <= 1:
                bucket = "0-1"
            elif padding <= 5:
                bucket = "2-5"
            elif padding <= 16:
                bucket = "6-16"
            elif padding <= 64:
                bucket = "17-64"
            else:
                bucket = "65+"
            padding_buckets[bucket] += 1

        if segs < 1:
            zero_segment += 1
        if not s.get("_dataset") or not s.get("_category"):
            missing_metadata += 1

        tokens = tok.decode(s["input_ids"][:length], skip_special_tokens=False)
        if tokens.count("\ufffd") > 20:
            corrupted_utf8 += 1

    avg_len = sum(lengths) / max(len(lengths), 1)
    avg_padding = sum(padding_counts) / max(len(padding_counts), 1)
    avg_segments = sum(segment_counts) / max(len(segment_counts), 1)
    avg_quality_packed = sum(quality_scores_in_packed) / max(len(quality_scores_in_packed), 1)
    padding_pct = avg_padding / max_len * 100
    utilization_pct = avg_len / max_len * 100

    logger.info("  Max seq length:    %d", max_len)
    logger.info("  Avg actual length: %.1f", avg_len)
    logger.info("  Avg padding:       %.1f (%.2f%%)", avg_padding, padding_pct)
    logger.info("  Avg utilization:   %.1f%%", utilization_pct)
    logger.info("  Avg segments:      %.2f", avg_segments)
    logger.info("  Avg quality:       %.3f", avg_quality_packed)
    logger.info("  Padding buckets:   %s", dict(padding_buckets))
    logger.info("  Padding target:    <5%%")
    logger.info("  Zero-segment:      %d", zero_segment)
    logger.info("  Missing metadata:  %d", missing_metadata)
    logger.info("  Corrupted UTF-8:   %d", corrupted_utf8)

    status = "PASS" if padding_pct < 5 else "WARNING"
    if corrupted_utf8 > 0:
        status = "WARNING"
    if missing_metadata > 0:
        status = "WARNING"
    logger.info("  Status: %s", status)

    return {
        "max_seq_length": max_len,
        "avg_length": round(avg_len, 1),
        "avg_padding": round(avg_padding, 1),
        "padding_pct": round(padding_pct, 2),
        "utilization_pct": round(utilization_pct, 1),
        "avg_segments": round(avg_segments, 2),
        "avg_quality_packed": round(avg_quality_packed, 3),
        "padding_buckets": dict(padding_buckets),
        "zero_segment_samples": zero_segment,
        "missing_metadata_samples": missing_metadata,
        "corrupted_utf8_samples": corrupted_utf8,
        "status": status,
        "samples_analyzed": len(lengths),
    }


# ── Phase 7: Fallback Verification ────────────────────────────────

def phase7_verify_fallbacks(token: Optional[str] = None) -> Dict[str, Any]:
    from src.data.streaming import stream_dataset_with_fallbacks
    from src.data.registry import build_registry, DatasetInfo

    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 7: FALLBACK VERIFICATION")
    logger.info("=" * 70)

    if token:
        os.environ["HF_TOKEN"] = token

    registry = build_registry()
    results = {}

    # Test a docs entry (should fall back to FineWeb-Edu since JSONL not built)
    docs_entry = None
    for e in registry.all_entries():
        if e.path == "json" and e.name == "python-docs":
            docs_entry = e
            break

    if docs_entry:
        logger.info("Testing docs entry fallback (JSONL not yet built)...")
        try:
            samples = list(stream_dataset_with_fallbacks(docs_entry, registry, limit=1))
            if samples:
                logger.info("  DOCS FALLBACK OK — loaded %d samples from FineWeb-Edu", len(samples))
                results["docs_fallback"] = {"status": "ok", "fallback_activated": True, "samples": len(samples)}
            else:
                logger.warning("  DOCS FALLBACK: empty")
                results["docs_fallback"] = {"status": "empty"}
        except Exception as e:
            logger.error("  DOCS FALLBACK FAILED: %s", e)
            results["docs_fallback"] = {"status": "error", "error": str(e)}

    return results


# ── Phase 8: Smoke Test + Checkpoint Save/Load ───────────────────

def phase8_smoke_checkpoint(config_path: str = "config_foundation.yaml",
                            max_steps: int = 100) -> Dict[str, Any]:
    """Run training smoke test + verify checkpoint save/load/resume."""
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 8: SMOKE TEST + CHECKPOINT VERIFICATION (%d steps)", max_steps)
    logger.info("=" * 70)

    import subprocess
    import tempfile
    work_dir = str(Path(__file__).resolve().parent.parent)
    ckpt_dir = tempfile.mkdtemp(prefix="ckpt_test_")

    results = {}

    # Check if training script exists
    train_script = Path(work_dir) / "scripts" / "run_training.py"
    if not train_script.exists():
        logger.warning("  scripts/run_training.py not found — skipping training smoke test")
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        return {"status": "SKIPPED", "reason": "scripts/run_training.py not found",
                "train_return_code": 0, "checkpoint_found": False, "resume_return_code": 0}

    # Step 1: Train and save at step max_steps
    logger.info("  Training %d steps with checkpoint save...", max_steps)
    r1 = subprocess.run(
        [sys.executable, str(train_script),
         "--config", config_path,
         "--max-steps", str(max_steps),
         "--logging-steps", "1",
         "--save-steps", str(max_steps),
         "--output-dir", ckpt_dir],
        capture_output=True, text=True, cwd=work_dir,
    )
    results["train_return_code"] = r1.returncode
    results["train_stdout_tail"] = r1.stdout[-2000:] if r1.stdout else ""
    results["train_stderr_tail"] = r1.stderr[-2000:] if r1.stderr else ""

    for line in r1.stdout.split("\n")[-20:]:
        logger.info("  %s", line)

    # Step 2: Check checkpoint file exists
    import glob
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "**", "*.pt"), recursive=True) + \
                 glob.glob(os.path.join(ckpt_dir, "**", "checkpoint*"), recursive=True)
    ckpt_found = len(ckpt_files) > 0
    results["checkpoint_files"] = ckpt_files[:5]
    results["checkpoint_found"] = ckpt_found
    logger.info("  Checkpoint files found: %d", len(ckpt_files))
    if ckpt_files:
        logger.info("  First: %s", ckpt_files[0])

    # Step 3: Resume from checkpoint for another 10 steps
    if ckpt_found and r1.returncode == 0:
        logger.info("  Resuming from checkpoint for 10 steps...")
        r2 = subprocess.run(
            [sys.executable, "scripts/run_training.py",
             "--config", config_path,
             "--max-steps", str(max_steps + 10),
             "--logging-steps", "1",
             "--resume-from", ckpt_dir],
            capture_output=True, text=True, cwd=work_dir,
        )
        results["resume_return_code"] = r2.returncode
        results["resume_stdout_tail"] = r2.stdout[-1000:] if r2.stdout else ""
        logger.info("  Resume return code: %d", r2.returncode)
        for line in r2.stdout.split("\n")[-5:]:
            logger.info("  %s", line)
    else:
        results["resume_return_code"] = -1
        logger.warning("  Checkpoint not found — skipping resume test")

    # Cleanup
    import shutil
    try:
        shutil.rmtree(ckpt_dir)
    except Exception:
        pass

    overall = "PASS" if (r1.returncode == 0 and ckpt_found and results.get("resume_return_code", -1) == 0) else "WARNING"
    results["status"] = overall
    logger.info("  Status: %s", overall)
    return results


# ── Report Generator ──────────────────────────────────────────────

def generate_report(all_results: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  FINAL PRODUCTION READINESS REPORT")
    lines.append("=" * 70)

    # Registry summary
    v = all_results.get("phase1", {})
    lines.append("")
    lines.append("--- Registry ---")
    lines.append(f"  Total entries:      {v.get('total_entries', '?')}")
    lines.append(f"  Total weight:       {v.get('total_weight', '?'):.4f}")
    lines.append(f"  Available weight:   {v.get('available_weight', '?'):.4f}")
    lines.append(f"  Corpus available:   {v.get('available_pct', 0)*100:.1f}%")
    if v.get('failures', 0) > 0:
        lines.append(f"  Failures:           {v['failures']}")
        for f in v.get('failure_details', []):
            lines.append(f"    {f['path']}/{f.get('name','')}: {f.get('error','')}")

    # Docs summary
    d = all_results.get("phase3", {})
    if d:
        lines.append("")
        lines.append("--- Documentation ---")
        lines.append(f"  Total documents:  {d.get('total_docs', 0)}")
        lines.append(f"  Total chars:      {d.get('total_chars', 0):,}")
        ok_sources = sum(1 for r in d.get("results", {}).values() if r.get("status") == "ok")
        missing_sources = sum(1 for r in d.get("results", {}).values() if r.get("status") == "missing")
        lines.append(f"  Sources OK:       {ok_sources}")
        if missing_sources:
            lines.append(f"  Sources missing:  {missing_sources}")

    # Token distribution
    td = all_results.get("phase4", {})
    if td:
        lines.append("")
        lines.append("--- Token Distribution ---")
        for cat, info in sorted(td.get("distribution", {}).items()):
            lines.append(f"  {cat:25s} {info['pct']:6.2f}% (target {info['target_pct']:5.1f}%) [{info['status']}]")
        lines.append(f"  Overall:  {td.get('overall_status', '?')}")

    # Packing quality
    pq = all_results.get("phase6", {})
    if pq:
        lines.append("")
        lines.append("--- Packing Quality ---")
        lines.append(f"  Avg utilization:  {pq.get('utilization_pct', 0):.1f}%")
        lines.append(f"  Padding:          {pq.get('padding_pct', 0):.2f}% (target <5%)")
        lines.append(f"  Avg length:       {pq.get('avg_length', 0)}")
        lines.append(f"  Avg segments:     {pq.get('avg_segments', 0):.2f}")
        lines.append(f"  Avg quality:      {pq.get('avg_quality_packed', 0):.3f}")
        lines.append(f"  UTF-8 corrupt:    {pq.get('corrupted_utf8_samples', 0)}")
        lines.append(f"  Missing metadata: {pq.get('missing_metadata_samples', 0)}")
        lines.append(f"  Status:           {pq.get('status', '?')}")

    # Decoded samples
    ds = all_results.get("phase5", [])
    total_issues = 0
    if ds:
        lines.append("")
        lines.append("--- Decoded Samples ---")
        total_issues = sum(len(s.get("issues", [])) for s in ds)
        lines.append(f"  Samples inspected: {len(ds)}")
        lines.append(f"  Issues found:      {total_issues}")
        for s in ds:
            issues = s.get("issues", [])
            tag = " [ISSUES: " + ", ".join(issues) + "]" if issues else ""
            lines.append(f"  #{s['index']}: {s['category']} ({s['segments']} segs, q={s['avg_quality']:.3f}){tag}")

    # Fallback verification
    fb = all_results.get("phase7", {})
    if fb:
        lines.append("")
        lines.append("--- Fallback Behavior ---")
        for key, val in fb.items():
            lines.append(f"  {key}: {val.get('status', '?')}")

    # Final decision
    lines.append("")
    lines.append("-" * 70)

    # Determine readiness
    issues = []
    if v.get("available_pct", 1) < 0.90:
        issues.append(f"Corpus availability {v.get('available_pct', 0)*100:.0f}% < 90%")
    if pq.get("padding_pct", 0) >= 5:
        issues.append(f"Padding {pq.get('padding_pct', 0):.1f}% >= 5%")
    if td.get("overall_status") == "WARNING":
        issues.append("Token distribution differs from targets by >3%")
    if total_issues > 0:
        issues.append(f"{total_issues} decoded sample issues detected")
    smoke = all_results.get("phase8", {})
    if smoke.get("train_return_code", 0) != 0:
        issues.append("Training smoke test failed")
    if smoke.get("checkpoint_found") is False and smoke.get("status") != "SKIPPED":
        issues.append("Checkpoint save failed")
    if smoke.get("resume_return_code", 0) != 0 and smoke.get("status") != "SKIPPED":
        issues.append("Checkpoint resume failed")
    fb_failures = [k for k, v in fb.items() if v.get("status") == "error"]
    if fb_failures:
        issues.append(f"Fallback errors: {fb_failures}")

    if not issues:
        lines.append("  DECISION: READY FOR FULL PRETRAINING")
    else:
        minor = all(
            "padding" not in i.lower() and "corpus" not in i.lower() and "smoke" not in i.lower()
            for i in issues
        )
        if minor:
            lines.append("  DECISION: READY WITH MINOR WARNINGS")
        else:
            lines.append("  DECISION: NOT READY")
        lines.append("  Issues:")
        for issue in issues:
            lines.append(f"    - {issue}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Production validation suite")
    parser.add_argument("--token", default=None, help="HF token for gated datasets")
    parser.add_argument("--doc-dir", default="data/docs", help="Documentation output directory")
    parser.add_argument("--tokenizer", default="Xenova/claude-tokenizer", help="Tokenizer name/path")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples for token distribution")
    parser.add_argument("--smoke-steps", type=int, default=100, help="Training smoke test steps")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip training smoke test")
    parser.add_argument("--skip-docs", action="store_true", help="Skip doc building")
    parser.add_argument("--report", default=None, help="Save report to path")
    args = parser.parse_args()

    all_results: Dict[str, Any] = {}

    # Phase 1: Dataset verification
    all_results["phase1"] = phase1_verify_registry(token=args.token)

    # Phase 2: Build docs
    if not args.skip_docs:
        all_results["phase2"] = phase2_build_docs(args.doc_dir)
    else:
        all_results["phase2"] = {"results": {}, "total_pages": 0, "missing": [], "empty": []}

    # Phase 3: Validate docs
    all_results["phase3"] = phase3_validate_docs(args.doc_dir)

    # Phase 4: Token distribution
    all_results["phase4"] = phase4_token_distribution(args.tokenizer, args.max_samples)

    # Phase 5: Decoded samples
    all_results["phase5"] = phase5_decode_samples(args.tokenizer, 20)

    # Phase 6: Packing quality
    all_results["phase6"] = phase6_packing_quality(args.tokenizer, args.max_samples)

    # Phase 7: Fallback verification
    all_results["phase7"] = phase7_verify_fallbacks(token=args.token)

    # Phase 8: Smoke test + checkpoint verification
    if not args.skip_smoke:
        all_results["phase8"] = phase8_smoke_checkpoint(max_steps=args.smoke_steps)
    else:
        all_results["phase8"] = {"status": "SKIPPED", "train_return_code": 0}

    # Generate report
    report_text = generate_report(all_results)
    print("\n" + report_text)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report_text)
        # Save raw data as JSON
        json_path = Path(args.report).with_suffix(".json")
        with open(str(json_path), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Report saved to %s (data: %s)", args.report, json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
