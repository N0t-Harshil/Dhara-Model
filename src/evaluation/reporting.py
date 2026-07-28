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

        avg_score = sum(r.score for r in benchmark_results) / max(len(benchmark_results), 1)
        report["summary"] = {
            "average_benchmark_score": avg_score,
            "total_benchmarks": len(benchmark_results),
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"eval_{timestamp}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self._print_summary(report)
        return report

    def _print_summary(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print(f"  EVALUATION REPORT: {report.get('model_name', 'unknown')}")
        print(f"  Time: {report['timestamp']}")
        print("=" * 60)

        for bm_name, bm_data in report.get("benchmarks", {}).items():
            status = "PASS" if bm_data["score"] >= 0.5 else "FAIL"
            print(f"  [{status}] {bm_name:20s}: {bm_data['score']:.4f}  ({bm_data['passed']}/{bm_data['total']})")

        if "safety" in report and report["safety"]:
            safety = report["safety"]
            sr = safety.get("safety", {}).get("safety_refusal_rate", "N/A")
            hr = safety.get("honesty", {}).get("honesty_uncertainty_rate", "N/A")
            print(f"  Safety refusal rate: {sr}")
            print(f"  Honesty uncertainty rate: {hr}")

        print(f"  Average benchmark score: {report.get('summary', {}).get('average_benchmark_score', 0):.4f}")
        print("=" * 60 + "\n")
