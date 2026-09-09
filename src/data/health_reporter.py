from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LONG_CONTEXT_THRESHOLD = 4096


class DatasetHealthReport:
    def __init__(self, name: str, config_path: str = ""):
        self.name = name
        self.config_path = config_path
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.datasets: List[Dict[str, Any]] = []
        self.global_stats: Dict[str, Any] = {}
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.languages: Dict[str, int] = {}
        self.domains: Dict[str, int] = {}
        self.errors: List[str] = []

    def add_dataset_stats(
        self,
        path: str,
        category: str,
        weight: float,
        raw_count: int,
        after_boilerplate: int,
        after_quality: int,
        after_dedup: int,
        packed_count: int,
        total_tokens: int,
        duplicate_removed: int,
        rejection_reasons: Optional[Dict[str, int]] = None,
        quality_scores: Optional[List[float]] = None,
        lang_dist: Optional[Dict[str, int]] = None,
        domain_dist: Optional[Dict[str, int]] = None,
        token_lengths: Optional[List[int]] = None,
        max_seq_length: int = 2048,
    ) -> None:
        retention = (packed_count / max(raw_count, 1)) * 100
        avg_seq_tokens = total_tokens / max(packed_count, 1)
        # Packing efficiency: how well sequences fill the max_seq_length
        packing_eff = avg_seq_tokens / max(max_seq_length, 1) if packed_count > 0 else 0.0
        # Padding ratio: wasted tokens due to padding
        ideal_tokens = packed_count * max_seq_length
        padding_ratio = 1.0 - (total_tokens / max(ideal_tokens, 1))
        # Long-context ratio: docs with >4K tokens (pre-packing)
        long_ctx_count = sum(1 for t in (token_lengths or []) if t > LONG_CONTEXT_THRESHOLD)
        long_ctx_ratio = long_ctx_count / max(len(token_lengths or []), 1)

        entry = {
            "path": path,
            "category": category,
            "weight": weight,
            "raw_samples": raw_count,
            "after_boilerplate": after_boilerplate,
            "after_quality": after_quality,
            "after_dedup": after_dedup,
            "packed_sequences": packed_count,
            "total_tokens": total_tokens,
            "duplicates_removed": duplicate_removed,
            "retention_rate": round(retention, 2),
            "avg_tokens_per_seq": round(avg_seq_tokens, 1),
            "packing_efficiency": round(packing_eff, 4),
            "padding_ratio": round(padding_ratio, 4),
            "long_context_count": long_ctx_count,
            "long_context_ratio": round(long_ctx_ratio, 4),
            "max_seq_length": max_seq_length,
        }
        if rejection_reasons:
            entry["rejection_reasons"] = rejection_reasons
        if quality_scores:
            entry["quality_scores"] = {
                "mean": round(sum(quality_scores) / max(len(quality_scores), 1), 4),
                "min": round(min(quality_scores), 4),
                "max": round(max(quality_scores), 4),
                "median": round(sorted(quality_scores)[len(quality_scores) // 2], 4) if quality_scores else 0,
                "count": len(quality_scores),
            }
        if lang_dist:
            for k, v in lang_dist.items():
                # A dataset with no single detected language yields a None
                # label (e.g. the packed-cache path uses
                # `meta["lang_d"] or info.language`); render it explicitly
                # instead of letting None reach the summary formatter, where
                # format specs like `:<15` raise
                # TypeError: unsupported format string passed to NoneType.
                label = k if k is not None else "unknown"
                if v is None:
                    continue
                self.languages[label] = self.languages.get(label, 0) + int(v)
        if domain_dist:
            for k, v in domain_dist.items():
                label = k if k is not None else "unknown"
                if v is None:
                    continue
                self.domains[label] = self.domains.get(label, 0) + int(v)
        self.datasets.append(entry)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def compute_global_stats(self) -> None:
        # "Generated:" must reflect the aggregation time, not the pipeline's
        # construction time (the report can be computed and re-saved hours
        # after the DataPipeline was created on a warm cache).
        self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.datasets:
            self.global_stats = {"error": "no datasets"}
            return
        total_raw = sum((d.get("raw_samples") or 0) for d in self.datasets)
        total_packed = sum((d.get("packed_sequences") or 0) for d in self.datasets)
        total_tokens = sum((d.get("total_tokens") or 0) for d in self.datasets)
        total_dup = sum((d.get("duplicates_removed") or 0) for d in self.datasets)
        avg_retention = sum((d.get("retention_rate") or 0) for d in self.datasets) / max(len(self.datasets), 1)
        avg_packing = sum((d.get("packing_efficiency") or 0) for d in self.datasets) / max(len(self.datasets), 1)
        avg_padding = sum((d.get("padding_ratio") or 0) for d in self.datasets) / max(len(self.datasets), 1)
        avg_tokens = sum((d.get("avg_tokens_per_seq") or 0) for d in self.datasets) / max(len(self.datasets), 1)

        # Category breakdown
        cats: Dict[str, Dict[str, Any]] = {}
        for d in self.datasets:
            c = d["category"]
            if c not in cats:
                cats[c] = {"dataset_count": 0, "packed_count": 0, "token_count": 0,
                           "weight": 0.0, "long_ctx_count": 0}
            cats[c]["dataset_count"] += 1
            cats[c]["packed_count"] += d["packed_sequences"]
            cats[c]["token_count"] += d["total_tokens"]
            cats[c]["weight"] += d["weight"]
            cats[c]["long_ctx_count"] += d["long_context_count"]
        self.categories = cats

        # Long-context ratio uses token_lengths counts, not quality_scores.
        total_long_ctx = sum(d["long_context_count"] for d in self.datasets)
        # Sum over actual document counts per dataset (not quality_scores).
        # add_dataset_stats stores token_lengths-derived counts; fall back to
        # quality_scores count when token_lengths was absent.
        total_docs_for_ratio = 0
        for d in self.datasets:
            cnt = d.get("long_context_count", 0)
            # Prefer stored total docs if available via raw_samples.
            total_docs_for_ratio += d.get("raw_samples", d.get("quality_scores", {}).get("count", 0))
        # Fallback: at least count datasets with any data
        if total_docs_for_ratio == 0:
            total_docs_for_ratio = max(len(self.datasets), 1)

        # Quality aggregation
        all_qs = []
        for d in self.datasets:
            qs = d.get("quality_scores", {})
            if (qs.get("count", 0) or 0) > 0 and qs.get("mean") is not None:
                all_qs.append(qs["mean"])
        avg_quality = sum(all_qs) / max(len(all_qs), 1) if all_qs else 0.0

        # Language/domain distribution. Report BOTH the labeled-sample count
        # AND the distinct-label count so the aggregate can never be mistaken
        # for a label count (the box run read "Languages Detected 44548" as if
        # all 44548 samples were one label: they were — 'unknown').
        total_lang = sum(self.languages.values()) if self.languages else 0
        total_domain = sum(self.domains.values()) if self.domains else 0
        distinct_langs = len(self.languages) if self.languages else 0
        distinct_domains = len(self.domains) if self.domains else 0

        self.global_stats = {
            "total_datasets": len(self.datasets),
            "total_raw_samples": total_raw,
            "total_packed_sequences": total_packed,
            "total_tokens": total_tokens,
            "total_duplicates_removed": total_dup,
            "average_retention_pct": round(avg_retention, 2),
            "average_packing_efficiency": round(avg_packing, 4),
            "average_padding_ratio": round(avg_padding, 4),
            "average_tokens_per_sequence": round(avg_tokens, 1),
            "average_quality_score": round(avg_quality, 4),
            "long_context_documents": total_long_ctx,
            "long_context_ratio_global": round(total_long_ctx / max(total_docs_for_ratio, 1), 4),
            "categories": len(cats),
            # Legacy keys kept for JSON backward compatibility: they are the
            # labeled-SAMPLE counts (sum of per-label counts), not label counts.
            "languages_detected": total_lang,
            "domains_detected": total_domain,
            # Accurate, non-ambiguous names (source of truth for the report).
            "language_labeled_samples": total_lang,
            "domain_labeled_samples": total_domain,
            "distinct_languages": distinct_langs,
            "distinct_domains": distinct_domains,
        }

    def summary_text(self) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append(f"DATASET HEALTH REPORT — {self.name}")
        lines.append(f"Generated: {self.timestamp}")
        lines.append("=" * 72)
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  ! {e}")
        lines.append(f"\n── Global Statistics ────────────────────────────────────────")
        # Accurate display names: 'languages_detected'/'domains_detected' are
        # legacy JSON keys that mean labeled-SAMPLE counts — never ambiguous.
        display_names = {
            "languages_detected": "Languages (labeled samples)",
            "domains_detected": "Domains (labeled samples)",
            "language_labeled_samples": "Language Labeled Samples",
            "domain_labeled_samples": "Domain Labeled Samples",
            "distinct_languages": "Distinct Languages",
            "distinct_domains": "Distinct Domains",
        }
        for k, v in self.global_stats.items():
            if isinstance(v, float):
                lines.append(f"  {display_names.get(k, k).replace('_', ' ').title():45s} {v:.4f}")
            else:
                lines.append(f"  {display_names.get(k, k).replace('_', ' ').title():45s} {v}")
        lines.append(f"\n── Per Dataset ─────────────────────────────────────────────")
        header = (f"{'Dataset':<30} {'Cat':<10} {'Raw':>7} {'Packed':>7} "
                  f"{'Tok':>9} {'Ret%':>5} {'PackEff':>7} {'Pad':>5} {'AvgQS':>6} {'LgCtx':>5}")
        lines.append(header)
        lines.append("-" * len(header))
        for d in sorted(self.datasets, key=lambda x: (x.get("retention_rate") or 0)):
            name = d["path"][:29]
            qs_mean = d.get("quality_scores", {}).get("mean") or 0
            lines.append(
                f"{name:<30} {d['category']:<10} {(d['raw_samples'] or 0):>7} "
                f"{(d['packed_sequences'] or 0):>7} {(d['total_tokens'] or 0):>9} "
                f"{(d['retention_rate'] or 0):>4.0f}% "
                f"{(d['packing_efficiency'] or 0):>6.0%} "
                f"{(d['padding_ratio'] or 0):>4.0%} "
                f"{qs_mean:>5.2f} "
                f"{(d['long_context_count'] or 0):>5}"
            )
        if self.categories:
            lines.append(f"\n── Category Token Distribution ──────────────────────────────")
            total_tok = self.global_stats.get("total_tokens") or 1
            try:
                from src.data.registry import CATEGORY_WEIGHTS
                _cat_targets = {k: round(v * 100) for k, v in CATEGORY_WEIGHTS.items()}
            except Exception:
                _cat_targets = {
                    "code": 30, "web_text": 20, "docs": 15, "wiki": 10,
                    "math": 10, "science": 5, "books": 5, "structured_knowledge": 5, "long_context": 5,
                }
            for cat, info in sorted(self.categories.items(), key=lambda x: -(x[1].get("token_count") or 0)):
                pct = (info.get("token_count") or 0) / max(total_tok, 1) * 100
                target_pct = _cat_targets.get(cat, 0)
                status = "OK" if abs(pct - target_pct) < 5 else "MISMATCH"
                lines.append(f"  {cat:<20} {(info.get('token_count') or 0):>10} tokens ({pct:5.1f}%) "
                           f"[target {target_pct}%] [{status}]")
        if self.languages:
            lines.append(f"\n── Language Distribution ───────────────────────────────────")
            total_l = sum((v or 0) for v in self.languages.values())
            for lang, count in sorted(self.languages.items(), key=lambda x: -(x[1] or 0)):
                lang_label = "unknown" if lang is None else str(lang)
                count_v = count if isinstance(count, (int, float)) else None
                pct = None if total_l <= 0 or count_v is None else (count_v / total_l) * 100
                pct_text = f"{pct:5.1f}" if pct is not None else "N/A"
                count_text = f"{count_v:>8}" if count_v is not None else "       -"
                lines.append(f"  {lang_label:<15} {count_text} ({pct_text}%)")
        if self.domains:
            lines.append(f"\n── Domain Distribution ─────────────────────────────────────")
            total_d = sum((v or 0) for v in self.domains.values())
            for domain, count in sorted(self.domains.items(), key=lambda x: -(x[1] or 0)):
                domain_label = "unknown" if domain is None else str(domain)
                count_v = count if isinstance(count, (int, float)) else None
                pct = None if total_d <= 0 or count_v is None else (count_v / total_d) * 100
                pct_text = f"{pct:5.1f}" if pct is not None else "N/A"
                count_text = f"{count_v:>8}" if count_v is not None else "       -"
                lines.append(f"  {domain_label:<15} {count_text} ({pct_text}%)")
        lines.append(f"\n── Per-Dataset Quality Breakdown ─────────────────────────────")
        for d in self.datasets:
            qs = d.get("quality_scores", {})
            if (qs.get("count", 0) or 0) > 0:
                mean = qs.get("mean") or 0.0
                minimum = qs.get("min") or 0.0
                maximum = qs.get("max") or 0.0
                lines.append(f"  {d['path'][:35]:35s} mean={mean:.3f} "
                             f"min={minimum:.3f} max={maximum:.3f} n={qs.get('count', 0)}")
        lines.append("\n" + "=" * 72)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "config_path": self.config_path,
            "timestamp": self.timestamp,
            "global_stats": self.global_stats,
            "categories": self.categories,
            "languages": self.languages,
            "domains": self.domains,
            "datasets": self.datasets,
            "errors": self.errors,
        }, indent=2, default=str)

    def save(self, output_dir: str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "dataset_health_report.json"
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Health report saved to %s", path)
        return path
