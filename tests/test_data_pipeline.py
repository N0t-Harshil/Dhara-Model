from __future__ import annotations

from typing import List

import pytest

from src.config.schema import Config
from src.data.streaming import MassiveDataCollector
from src.data.quality import QualityFilter, ExactDeduplicator


SAMPLE_ENTRIES = [
    {"instruction": "def foo", "output": "def foo():\n    pass", "language": "python"},
    {"instruction": "add", "output": "def add(a, b): return a + b", "language": "python"},
]


class TestStreamingCollector:
    def test_clean_text(self):
        assert MassiveDataCollector._clean_text(" hello ") == "hello"
        assert MassiveDataCollector._clean_text(None) == ""
        assert MassiveDataCollector._clean_text(["a", "b"]) == "a\nb"

    def test_normalize_language(self):
        assert MassiveDataCollector._normalize_language("py") == "python"
        assert MassiveDataCollector._normalize_language("js") == "javascript"
        assert MassiveDataCollector._normalize_language("c++") == "cpp"
        assert MassiveDataCollector._normalize_language("python") == "python"

    def test_is_valid_sample(self):
        long_code = "def foo():" + " x = 1\n" * 20
        assert MassiveDataCollector._is_valid_sample(long_code, "python")
        assert not MassiveDataCollector._is_valid_sample("hi", "python")
        assert not MassiveDataCollector._is_valid_sample("lorem ipsum dolor sit amet, consectetur adipiscing elit", "python")

    def test_detect_language(self):
        assert MassiveDataCollector._detect_language("def foo(): import os") == "python"
        assert MassiveDataCollector._detect_language("fn main() { let x = 1; }") == "rust"
        assert MassiveDataCollector._detect_language("const hello = () => { return 1; }") == "javascript"
        assert MassiveDataCollector._detect_language("The quick brown fox is jumping") == "text"
        assert MassiveDataCollector._detect_language("public class Hello { void main() {} }") == "java"

    def test_extract_standard_fields(self):
        collector = MassiveDataCollector([])
        entry = {"instruction": "write a function", "input": "add two numbers", "output": "def add(a,b): return a+b"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "write a function"
        assert inp == "add two numbers"
        assert out == "def add(a,b): return a+b"

    def test_extract_problem_solution(self):
        collector = MassiveDataCollector([])
        entry = {"problem": "solve x+2=5", "solution": "x=3"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "solve x+2=5"
        assert out == "x=3"

    def test_extract_question_answer(self):
        collector = MassiveDataCollector([])
        entry = {"question": "what is 2+2?", "answer": "4"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "what is 2+2?"
        assert out == "4"

    def test_extract_dolly_format(self):
        collector = MassiveDataCollector([])
        entry = {"context": "math context", "instruction": "compute 2+2", "response": "4"}
        inst, inp, out = collector._extract_fields(entry)
        assert "math context" in inst
        assert "compute 2+2" in inst
        assert out == "4"

    def test_extract_flan_format(self):
        collector = MassiveDataCollector([])
        entry = {"inputs": "translate to french: hello", "targets": "bonjour"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "translate to french: hello"
        assert out == "bonjour"

    def test_extract_sharegpt_fields(self):
        collector = MassiveDataCollector([])
        human = {"from": "human", "value": "hello there"}
        gpt = {"from": "gpt", "value": "hi how can I help"}
        inst1, _, _ = collector._extract_sharegpt_fields(human)
        assert inst1 == "hello there"
        _, _, out2 = collector._extract_sharegpt_fields(gpt)
        assert out2 == "hi how can I help"

    def test_extract_chat_messages(self):
        collector = MassiveDataCollector([])
        entry = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "hello"
        assert out == "hi there"

    def test_extract_orca_style(self):
        collector = MassiveDataCollector([])
        entry = {"system_prompt": "you are a math tutor", "question": "what is 2+2?", "response": "4"}
        inst, inp, out = collector._extract_fields(entry)
        assert "you are a math tutor" in inst
        assert "what is 2+2?" in inst
        assert out == "4"

    def test_extract_tool_use(self):
        collector = MassiveDataCollector([])
        entry = {"tool_definition": "fn add(a,b) -> int", "instruction": "call add(1,2)", "response": "3"}
        inst, inp, out = collector._extract_fields(entry)
        assert "fn add(a,b) -> int" in inst
        assert "call add(1,2)" in inst
        assert out == "3"

    def test_extract_code_explanation(self):
        collector = MassiveDataCollector([])
        entry = {"code": "print('hello')", "explanation": "prints hello"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "prints hello"
        assert out == "print('hello')"

    def test_extract_text_fallback(self):
        collector = MassiveDataCollector([])
        entry = {"text": "some long article text about programming in python"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == ""
        assert out == "some long article text about programming in python"

    def test_extract_natural_question(self):
        collector = MassiveDataCollector([])
        entry = {"sentence1": "the cat sat", "sentence2": "the cat sat on mat", "label": "entailment"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "the cat sat"
        assert inp == "the cat sat on mat"
        assert out == "entailment"

    def test_valid_sample_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        assert MassiveDataCollector._is_valid_sample(text, "text")

    def test_valid_sample_rejects_too_short(self):
        assert not MassiveDataCollector._is_valid_sample("short", "text")


class TestQualityFilter:
    def test_valid_content(self):
        assert QualityFilter.check_length("hello world " * 10, min_len=50, max_len=250000)
        assert not QualityFilter.check_length("short", min_len=50)
        assert QualityFilter.is_high_quality_content("def valid_function(): pass")

    def test_contamination_detection(self):
        from src.data.quality import ContaminationFilter
        cf = ContaminationFilter(benchmarks=["human_eval"])
        assert cf.is_contaminated("def check_solution(): assert result == 42")
        assert not cf.is_contaminated("def normal(x): return x")


class TestDeduplicator:
    def test_exact_dedup(self):
        dedup = ExactDeduplicator()
        assert not dedup.is_duplicate("unique text")
        assert dedup.is_duplicate("unique text")
        assert not dedup.is_duplicate("other text")

    def test_exact_dedup_reset(self):
        dedup = ExactDeduplicator()
        dedup.is_duplicate("test")
        dedup.reset()
        assert not dedup.is_duplicate("test")


class TestCollectorProductionPathGaps:
    """_process_entry must cover every format _extract_fields does (it used
    to silently fall back to raw-text extraction for dolly/flan/orca/tool
    formats)."""

    def test_process_entry_dolly_format(self):
        collector = MassiveDataCollector([])
        entry = {
            "context": "A long blog post about testing frameworks.",
            "instruction": "Summarize the post.",
            "response": "The post argues testing matters for reliability.",
        }
        sample = collector._process_entry(entry, "all", [])
        assert sample is not None
        assert "blog post" in sample["instruction"]
        assert "reliability" in sample["output"]

    def test_process_entry_tool_use_format(self):
        collector = MassiveDataCollector([])
        entry = {
            "tool_definition": '{"name": "calculator", "args": ["a", "b"]}',
            "instruction": "Add 1 and 2.",
            "response": "The result is 3.",
        }
        inst, _, out = collector._extract_fields(entry)
        assert "calculator" in inst
        assert out == "The result is 3."

    def test_process_entry_orca_format(self):
        collector = MassiveDataCollector([])
        entry = {
            "system_prompt": "You are a helpful assistant that answers with precision and clarity.",
            "question": "What is two plus two? Explain your reasoning step by step.",
            "response": "Four. Two plus two equals four: 2 + 2 = 4, the sum of the two addends.",
        }
        sample = collector._process_entry(entry, "all", [])
        assert sample is not None
        assert "helpful assistant" in sample["instruction"]
        assert sample["output"].startswith("Four")

    def test_process_entry_flan_format(self):
        collector = MassiveDataCollector([])
        entry = {
            "inputs": "Translate the following English sentence into French: 'The weather today is beautiful and sunny.'",
            "targets": "Le temps aujourd'hui est beau et ensoleillé.",
        }
        sample = collector._process_entry(entry, "all", [])
        assert sample is not None
        assert sample["instruction"].startswith("Translate")
        assert "ensoleillé" in sample["output"]


class TestWeightedMixedDatasetGaps:
    def test_zero_weights_fall_back_to_uniform(self):
        from datasets import Dataset
        from src.data.pipeline import WeightedMixedDataset

        ds1 = Dataset.from_list([{"x": 1}] * 2)
        ds2 = Dataset.from_list([{"x": 2}] * 2)
        wm = WeightedMixedDataset(
            [(ds1, 0.0, "a"), (ds2, 0.0, "b")], total_samples=3,
        )
        assert wm.weights == [0.5, 0.5]
        assert [wm[i]["x"] for i in range(3)]

    def test_exhausted_datasets_start_fresh_pass(self):
        from datasets import Dataset
        from src.data.pipeline import WeightedMixedDataset

        ds1 = Dataset.from_list([{"x": 1}])
        ds2 = Dataset.from_list([{"x": 2}])
        wm = WeightedMixedDataset(
            [(ds1, 1.0, "a"), (ds2, 1.0, "b")], total_samples=4,
        )
        seen = [wm[i]["x"] for i in range(4)]
        # Total capacity is 2 — after the first pass a fresh pass must start
        # (the old code re-sampled consumed indices, fabricating an epoch of
        # ~100% duplicates).
        assert len(seen) == 4
        assert seen.count(1) >= 2
        assert seen.count(2) >= 2


class TestRegistryFallbackOnly:
    def test_fallback_entries_resolvable_but_never_streamed(self):
        from src.data.registry import build_registry

        registry = build_registry()
        assert registry.get_by_path_category("allenai/dolma", "web_text") is not None
        assert registry.get_by_path_category(
            "togethercomputer/RedPajama-Data-1T", "web_text") is not None
        assert registry.get_by_path_category("b-mc2/sql-create-context", "code") is not None
        streamed = [e.path for e in registry.all_entries()]
        assert "allenai/dolma" not in streamed
        assert "b-mc2/sql-create-context" not in streamed


class _FakeStreamer:
    """Iterable stand-in for ShardedStreamIterator with the ``.stats`` surface
    ``build_pretrain_dataset_from_registry`` and its diagnostics read/write."""

    def __init__(self, records):
        from types import SimpleNamespace

        self._records = iter(records)
        self.stats = SimpleNamespace(
            timings={}, raw_rows=len(records), gated=0, plan=None, record=None,
            shard_stats={}, shard_streamed={}, shard_accepted={},
            driver_kind=None, metadata_hit=False, builder_hit=False,
            repo_resolution_skipped=False, first_resolution=False,
            script_resumed=False,
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._records)

    def close(self):
        pass


class _FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, truncation=False, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 50000) + 3 for c in str(text)][:64]}


class TestRegistryBuildHealthAggregation:
    """Audit regressions on the registry build path.

    * avg_qs / quality stats must come from the ACCEPTED texts only. The old
      code indexed the all-candidates ``quality_scores`` list with the
      accepted-text position, so any callback/reject between two accepted
      texts shifted doc_qs and biased avg_qs low (mean over every scored
      candidate, rejections included).
    * add_dataset_stats must receive each dataset's OWN lang/domain
      distribution — the old code passed the run-global cumulative dict, so a
      multi-dataset build doubled every label's count in the health report.
    """

    def _fake_pipe(self, health_report):
        from types import SimpleNamespace

        from src.data.pipeline import DataPipeline

        pipe = DataPipeline.__new__(DataPipeline)
        pipe.cfg = SimpleNamespace(
            data=SimpleNamespace(
                preprocessing=SimpleNamespace(
                    remove_boilerplate=False, min_text_length=1,
                    license_keywords=[], boilerplate_file_patterns=[]),
                quality=SimpleNamespace(
                    deduplication=SimpleNamespace(method="exact", threshold=0.85)),
                ast_filter=SimpleNamespace(code_filtering=False),
                function_sampling=SimpleNamespace(enabled=False),
                sampler=SimpleNamespace(balance_by="samples"),
                language_balancing=SimpleNamespace(enabled=False,
                                                   target_distribution=None),
                domain_balancing=SimpleNamespace(enabled=False, include=None),
                use_packed_cache=False,
                cache_dir=".",
                health_reporting=SimpleNamespace(enabled=False, output_dir=""),
                sanity_checks=SimpleNamespace(enabled=False),
                dataset_policies=[],
                shard_workers=4,
                max_samples_per_dataset=1000,
                bottleneck_threshold_sec=None,
                acceptance_investigation_threshold=0,
            ),
            training=SimpleNamespace(
                max_seq_length=64,
                pretrain=SimpleNamespace(batch_size=8,
                                         gradient_accumulation_steps=1)),
            model=SimpleNamespace(architecture=SimpleNamespace(vocab_size=65536)),
        )
        pipe.health_report = health_report
        pipe.tokenizer = _FakeTokenizer()
        pipe.contamination = SimpleNamespace(is_contaminated=lambda text: False)
        pipe.exact_dedup = SimpleNamespace(is_duplicate=lambda text: False)
        pipe._prefetch_driver = lambda *a, **k: None
        pipe._get_cleanup_pool = lambda *a, **k: None
        return pipe

    def _records_for(self, texts):
        return [{"output": t, "_shard": 0, "file_path": ""} for t in texts]

    def _run_build(self, monkeypatch, records_by_name, score_map):
        from src.data.health_reporter import DatasetHealthReport
        from src.data.registry import build_registry

        health = DatasetHealthReport("audit-registry")
        pipe = self._fake_pipe(health)

        def fake_qs(text, cat, lang):
            return score_map[text]

        monkeypatch.setattr("src.data.pipeline._pool_quality_score", fake_qs)
        pipe._stream_sharded = (
            lambda info, registry, limit, policy: _FakeStreamer(
                self._records_for(records_by_name[info.name])))
        result = pipe.build_pretrain_dataset_from_registry(
            include=[("bigcode/the-stack-v2-dedup", "Python"),
                     ("bigcode/the-stack-v2-dedup", "C++")])
        health.compute_global_stats()
        return result, health

    def test_avg_qs_reflects_accepted_texts_only(self, monkeypatch):
        text_p = ["def add(a, b): return a + b",  # accepted (0.95)
                  "x",                            # too low quality -> rejected
                  "def j(): return 2"]            # accepted (0.85)
        text_c = ["int main(){return 0;}",        # accepted (0.80)
                  "y"]                            # rejected (0.20)
        result, health = self._run_build(
            monkeypatch,
            records_by_name={"Python": text_p, "C++": text_c},
            score_map={
                "def add(a, b): return a + b": 0.95,
                "x": 0.10,
                "def j(): return 2": 0.85,
                "int main(){return 0;}": 0.80,
                "y": 0.20,
            })
        metas = result._dataset_metas
        assert len(metas) == 2
        # Accepted-only quality, in dataset order (Python unit then C++ unit).
        assert len(metas[0]["quality_scores"]) == 2
        assert sum(metas[0]["quality_scores"]) == pytest.approx(1.80)
        assert metas[0]["avg_qs"] == pytest.approx(0.90)
        assert len(metas[1]["quality_scores"]) == 1
        assert metas[1]["avg_qs"] == pytest.approx(0.80)
        # Packed per-pack means use each accepted text's own score. Before the
        # fix, the rejected text's score was mis-indexed into the second
        # accepted text (pack1 mean = (0.95 + 0.10)/2 = 0.525, not 0.9).
        py_rows = result._entries[0][0].to_list()
        assert [float(r["_avg_quality"]) for r in py_rows] == pytest.approx([0.9])
        cpp_rows = result._entries[1][0].to_list()
        assert [float(r["_avg_quality"]) for r in cpp_rows] == pytest.approx([0.8])
        # Health report per-dataset means match the accepted texts.
        means = sorted(e["quality_scores"]["mean"] for e in health.datasets)
        assert means == pytest.approx([0.80, 0.90])
        assert health.global_stats["average_quality_score"] == pytest.approx(0.85)

    def test_health_lang_domain_dist_is_per_dataset(self, monkeypatch):
        # All accepted — the regression is that the run-global cumulative
        # dict was passed per dataset, doubling every label's count on a
        # multi-dataset build (2 datasets -> 2x inflation).
        text_p = ["def add(a, b): return a + b",
                  "def sub(a, b): return a - b",
                  "def mul(a, b): return a * b"]
        text_c = ["int main(){return 0;}",
                  "int f(int x){return x + 1;}"]
        all_texts = text_p + text_c
        _, health = self._run_build(
            monkeypatch,
            records_by_name={"Python": text_p, "C++": text_c},
            score_map={t: 0.9 for t in all_texts})
        expected = len(all_texts)
        assert sum(health.languages.values()) == expected
        assert sum(health.domains.values()) == expected
        assert health.global_stats["languages_detected"] == expected
        assert health.global_stats["domains_detected"] == expected
