"""Tests for dataset normalization in the massive data collector."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.massive_data_collector import MassiveDataCollector


def _collector() -> MassiveDataCollector:
    collector = MassiveDataCollector.__new__(MassiveDataCollector)
    collector.config = {}
    collector.datasets = []
    collector.target_libraries = []
    return collector


class TestMassiveDataCollector(unittest.TestCase):
    def test_problem_solution_schema_uses_language_hint(self):
        collector = _collector()

        sample = collector._process_entry(
            {
                "problem": "Write a C++ function that returns the square of an integer.",
                "solution": "int square(int x) { return x * x; }",
                "lang": "C++",
            },
            theme="all",
            target_libs=[],
        )

        self.assertIsNotNone(sample)
        self.assertEqual(sample["language"], "cpp")
        self.assertIn("square", sample["output"])

    def test_chat_schema_can_be_used_as_text_alignment_data(self):
        collector = _collector()

        sample = collector._process_entry(
            {
                "messages": [
                    {"role": "user", "content": "Explain why clear error messages matter."},
                    {
                        "role": "assistant",
                        "content": (
                            "Clear error messages help people understand what failed, "
                            "what action to take next, and whether the system state is safe."
                        ),
                    },
                ]
            },
            theme="all",
            target_libs=[],
            ds_info={"language": "text"},
        )

        self.assertIsNotNone(sample)
        self.assertEqual(sample["language"], "text")
        self.assertIn("error messages", sample["instruction"])

    def test_stream_single_dataset_honors_dataset_max_samples(self):
        collector = _collector()
        rows = [
            {
                "instruction": "Write a Python function that returns one.",
                "output": "def one():\n    return 1",
                "language": "python",
            },
            {
                "instruction": "Write a Python function that returns two.",
                "output": "def two():\n    return 2",
                "language": "python",
            },
            {
                "instruction": "Write a Python function that returns three.",
                "output": "def three():\n    return 3",
                "language": "python",
            },
        ]

        with patch("src.massive_data_collector.load_dataset", return_value=rows):
            samples = list(
                collector.stream_single_dataset(
                    {"path": "local/test", "max_samples": 2},
                    theme="all",
                )
            )

        self.assertEqual(len(samples), 2)

    def test_stream_single_dataset_accepts_attribute_style_entry(self):
        """Training pipeline passes pydantic DatasetEntryConfig models — the
        collector must handle non-dict entries instead of crashing and
        silently skipping every dataset."""
        from types import SimpleNamespace

        collector = _collector()
        rows = [
            {
                "instruction": "Write a Python function that returns one.",
                "output": "def one():\n    return 1",
                "language": "python",
            },
        ]
        ds_info = SimpleNamespace(
            path="local/test", max_samples=1, name=None,
            split="train", data_dir=None, languages=None,
        )
        with patch("src.massive_data_collector.load_dataset", return_value=rows):
            samples = list(collector.stream_single_dataset(ds_info, theme="all"))

        self.assertEqual(len(samples), 1)
        self.assertIn("one", samples[0]["output"])

    def test_stream_single_dataset_skip_is_single_pass(self):
        """skip_samples must not re-load/re-iterate the whole dataset per
        skipped sample (was: O(skip) full re-streams + infinite-loop risk)."""
        collector = _collector()
        rows = [
            {
                "instruction": f"Write a Python function that returns {i} and documents it.",
                "output": f"def f{i}():\n    \"\"\"Returns {i}.\"\"\"\n    return {i}",
                "language": "python",
            }
            for i in range(5)
        ]
        with patch("src.massive_data_collector.load_dataset", return_value=rows) as ld:
            samples = list(
                collector.stream_single_dataset(
                    {"path": "local/test", "max_samples": 2},
                    theme="all",
                    skip_samples=2,
                )
            )

        ld.assert_called_once()
        self.assertEqual(len(samples), 2)
        self.assertIn("def f2", samples[0]["output"])


if __name__ == "__main__":
    unittest.main()
