from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.benchmarks import BenchmarkResult

logger = logging.getLogger(__name__)


class EvaluationReport:
    def __init__(self, report_dir: str | Path = "./eval_reports") -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model_name: str,
        benchmark_results: List[BenchmarkResult],
        safety_results: Optional[Dict[str, Any]] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmarks": {
                r.name: {
                    "score": r.score,
                    "passed": r.passed,
                    "total": r.total,
                    "time_seconds": r.time_seconds,
                }
                for r in benchmark_results
            },
            "safety": safety_results or {},
            "additional_metrics": additional_metrics or {},
        }

        if benchmark_results:
            avg_score = sum(r.score for r in benchmark_results) / len(benchmark_results)
        else:
            avg_score = float("nan")
        report["summary"] = {
            "average_benchmark_score": avg_score,
            "total_benchmarks": len(benchmark_results),
        }

        # Microsecond + pid to avoid same-second collisions.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        import os as _os
        report_path = self.report_dir / f"eval_{timestamp}_{_os.getpid()}.json"
        # Use allow_nan=False so NaNs are caught (should not happen after above).
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

        self._print_summary(report)
        return report

    def _print_summary(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print(f"  EVALUATION REPORT: {report.get('model_name', 'unknown')}")
        print(f"  Time: {report['timestamp']}")
        print("=" * 60)

        # Per-benchmark PASS thresholds reflect task difficulty/ random baseline.
        _thresholds = {
            "HumanEval": 0.10, "MBPP": 0.10, "MMLU": 0.25, "HellaSwag": 0.25,
            "ARC": 0.25, "GSM8K": 0.10, "TruthfulQA": 0.30, "WinoGrande": 0.50, "BBH": 0.20,
        }
        for bm_name, bm_data in report.get("benchmarks", {}).items():
            thresh = _thresholds.get(bm_name, 0.5)
            tag = "smoke" if bm_data.get("smoke") else ("PASS" if bm_data["score"] >= thresh else "FAIL")
            smoke_note = " (toy data)" if bm_data.get("smoke") else ""
            print(f"  [{tag:5s}] {bm_name:20s}: {bm_data['score']:.4f}  ({bm_data['passed']}/{bm_data['total']}){smoke_note}")

        if "safety" in report and report["safety"]:
            safety = report["safety"]
            sr = safety.get("safety", {}).get("safety_refusal_rate", "N/A")
            hr = safety.get("honesty", {}).get("honesty_uncertainty_rate", "N/A")
            print(f"  Safety refusal rate: {sr}")
            print(f"  Honesty uncertainty rate: {hr}")

        print(f"  Average benchmark score: {report.get('summary', {}).get('average_benchmark_score', 0):.4f}")
        print("=" * 60 + "\n")
