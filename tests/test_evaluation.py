from __future__ import annotations

from src.evaluation.safety import SafetyEvaluator
from src.evaluation.benchmarks import (
    HumanEvalBenchmark,
    BenchmarkResult,
    _try_load_hf,
)


class TestTryLoadHf:
    def test_positional_subset_name_is_forwarded(self, monkeypatch):
        import datasets

        calls = {}

        def fake_load_dataset(dataset, **kwargs):
            calls["dataset"] = dataset
            calls["kwargs"] = dict(kwargs)
            return object()

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
        _try_load_hf("cais/mmlu", "college_mathematics", split="test")
        assert calls["dataset"] == "cais/mmlu"
        assert calls["kwargs"]["name"] == "college_mathematics"
        assert calls["kwargs"]["split"] == "test"

    def test_no_subset_passes_kwargs_only(self, monkeypatch):
        import datasets

        calls = {}

        def fake_load_dataset(dataset, **kwargs):
            calls["kwargs"] = dict(kwargs)
            return object()

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
        _try_load_hf("openai_humaneval", split="test")
        assert calls["kwargs"] == {"split": "test"}

    def test_offline_fallback_returns_none(self, monkeypatch):
        import datasets

        def failing_load_dataset(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr(datasets, "load_dataset", failing_load_dataset)
        assert _try_load_hf("cais/mmlu", "abstract_algebra", split="test") is None


class TestSafetyEvaluator:
    def test_refusal_keywords_detected(self):
        keywords = SafetyEvaluator.REFUSAL_KEYWORDS
        assert "cannot" in keywords
        assert "I'm sorry" in keywords
        assert len(keywords) > 5

    def test_safety_probes_exist(self):
        assert len(SafetyEvaluator.SAFETY_PROBES) > 0
        categories = {p["category"] for p in SafetyEvaluator.SAFETY_PROBES}
        assert "harmful_code" in categories
        assert "refusal_test" in categories

    def test_honesty_probes_exist(self):
        assert len(SafetyEvaluator.HONESTY_PROBES) > 0


class TestBenchmarkResult:
    def test_representation(self):
        r = BenchmarkResult("test", 0.75, {"passed": 3, "total": 4, "time_seconds": 10.0})
        assert "test" in repr(r)
        assert "0.75" in repr(r)
        assert r.score == 0.75
        assert r.passed == 3
        assert r.total == 4


class TestHumanEvalBenchmark:
    def test_code_extraction(self):
        extracted = HumanEvalBenchmark._extract_code("```python\nprint('hello')\n```")
        assert extracted == "print('hello')"

        extracted2 = HumanEvalBenchmark._extract_code("print('no fences')")
        assert extracted2 == "print('no fences')"
