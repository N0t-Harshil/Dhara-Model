from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LONG_CONTEXT_THRESHOLD = 4096


class DatasetHealthReport:
    def __init__(self, name: str, config_path: str = ""):
        self.name = name
        self.config_path = config_path
        self.timestamp = datetime.utcnow().isoformat()
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
            self.languages.update(lang_dist)
        if domain_dist:
            self.domains.update(domain_dist)
        self.datasets.append(entry)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def compute_global_stats(self) -> None:
        if not self.datasets:
            self.global_stats = {"error": "no datasets"}
            return
        total_raw = sum(d["raw_samples"] for d in self.datasets)
        total_packed = sum(d["packed_sequences"] for d in self.datasets)
        total_tokens = sum(d["total_tokens"] for d in self.datasets)
        total_dup = sum(d["duplicates_removed"] for d in self.datasets)
        avg_retention = sum(d["retention_rate"] for d in self.datasets) / max(len(self.datasets), 1)
        avg_packing = sum(d["packing_efficiency"] for d in self.datasets) / max(len(self.datasets), 1)
        avg_padding = sum(d["padding_ratio"] for d in self.datasets) / max(len(self.datasets), 1)
        avg_tokens = sum(d["avg_tokens_per_seq"] for d in self.datasets) / max(len(self.datasets), 1)
        total_long_ctx = sum(d["long_context_count"] for d in self.datasets)
        total_docs_with_lengths = sum(d.get("quality_scores", {}).get("count", 0) for d in self.datasets)

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

        # Quality aggregation
        all_qs = []
        for d in self.datasets:
            qs = d.get("quality_scores", {})
            if qs.get("count", 0) > 0:
                all_qs.append(qs["mean"])
        avg_quality = sum(all_qs) / max(len(all_qs), 1) if all_qs else 0.0

        # Language/domain distribution
        total_lang = sum(self.languages.values()) if self.languages else 0
        total_domain = sum(self.domains.values()) if self.domains else 0

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
            "long_context_ratio_global": round(total_long_ctx / max(total_docs_with_lengths, 1), 4),
            "categories": len(cats),
            "languages_detected": total_lang,
            "domains_detected": total_domain,
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
        for k, v in self.global_stats.items():
            if isinstance(v, float):
                lines.append(f"  {k.replace('_', ' ').title():45s} {v:.4f}")
            else:
                lines.append(f"  {k.replace('_', ' ').title():45s} {v}")
        lines.append(f"\n── Per Dataset ─────────────────────────────────────────────")
        header = (f"{'Dataset':<30} {'Cat':<10} {'Raw':>7} {'Packed':>7} "
                  f"{'Tok':>9} {'Ret%':>5} {'PackEff':>7} {'Pad':>5} {'AvgQS':>6} {'LgCtx':>5}")
        lines.append(header)
        lines.append("-" * len(header))
        for d in sorted(self.datasets, key=lambda x: x["retention_rate"]):
            name = d["path"][:29]
            qs_mean = d.get("quality_scores", {}).get("mean", 0)
            lines.append(
                f"{name:<30} {d['category']:<10} {d['raw_samples']:>7} "
                f"{d['packed_sequences']:>7} {d['total_tokens']:>9} "
                f"{d['retention_rate']:>4.0f}% "
                f"{d['packing_efficiency']:>6.0%} "
                f"{d['padding_ratio']:>4.0%} "
                f"{qs_mean:>5.2f} "
                f"{d['long_context_count']:>5}"
            )
        if self.categories:
            lines.append(f"\n── Category Token Distribution ──────────────────────────────")
            total_tok = self.global_stats.get("total_tokens", 1)
            for cat, info in sorted(self.categories.items(), key=lambda x: -x[1]["token_count"]):
                pct = info["token_count"] / max(total_tok, 1) * 100
                target_pct = {
                    "code": 30, "web_text": 20, "docs": 15, "wiki": 10,
                    "math": 8, "science": 5, "books": 5, "structured_knowledge": 2, "long_context": 5,
                }.get(cat, 0)
                status = "OK" if abs(pct - target_pct) < 5 else "MISMATCH"
                lines.append(f"  {cat:<20} {info['token_count']:>10} tokens ({pct:5.1f}%) "
                           f"[target {target_pct}%] [{status}]")
        if self.languages:
            lines.append(f"\n── Language Distribution ───────────────────────────────────")
            total_l = sum(self.languages.values())
            for lang, count in sorted(self.languages.items(), key=lambda x: -x[1]):
                pct = count / max(total_l, 1) * 100
                lines.append(f"  {lang:<15} {count:>8} ({pct:5.1f}%)")
        if self.domains:
            lines.append(f"\n── Domain Distribution ─────────────────────────────────────")
            total_d = sum(self.domains.values())
            for domain, count in sorted(self.domains.items(), key=lambda x: -x[1]):
                pct = count / max(total_d, 1) * 100
                lines.append(f"  {domain:<15} {count:>8} ({pct:5.1f}%)")
        lines.append(f"\n── Per-Dataset Quality Breakdown ─────────────────────────────")
        for d in self.datasets:
            qs = d.get("quality_scores", {})
            if qs.get("count", 0) > 0:
                lines.append(f"  {d['path'][:35]:35s} mean={qs['mean']:.3f} "
                           f"min={qs['min']:.3f} max={qs['max']:.3f} n={qs['count']}")
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
