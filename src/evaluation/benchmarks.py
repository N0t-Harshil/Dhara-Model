from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _try_load_hf(dataset: str, name: Optional[str] = None, **kwargs):
    """Best-effort HF load with offline fallback (no network → toy data)."""
    try:
        from datasets import load_dataset
        load_kwargs = dict(kwargs)
        if name is not None:
            load_kwargs["name"] = name
        return load_dataset(dataset, **load_kwargs)
    except Exception as e:
        logger.warning("Could not load HF dataset %s (%s) — using built-in toy fallback.", dataset, e)
        return None


class BenchmarkResult:
    def __init__(self, name: str, score: float, details: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.score = score
        self.details = details or {}
        self.passed = self.details.get("passed", 0)
        self.total = self.details.get("total", 0)
        self.time_seconds = self.details.get("time_seconds", 0.0)
        self.smoke = self.details.get("smoke", False)

    def __repr__(self) -> str:
        tag = " [smoke]" if self.smoke else ""
        return f"{self.name}: {self.score:.4f} ({self.passed}/{self.total}) in {self.time_seconds:.1f}s{tag}"


class BaseBenchmark(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        ...


# ── Code Generation Benchmarks ──────────────────────────────────────────────

class HumanEvalBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__("HumanEval")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, max_new_tokens: int = 512, **kwargs) -> BenchmarkResult:
        problems = self._get_problems(limit=kwargs.get("limit"))
        passed = 0
        start = time.time()

        for problem in problems:
            prompt = f"### Instruction\nWrite a python solution for:\n{problem['prompt']}\n\n### Response\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
            code = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            code = self._extract_code(code)

            if self._check_solution(code, problem.get("test", ""), timeout=kwargs.get("timeout", 10.0)):
                passed += 1

        elapsed = time.time() - start
        smoke = len(problems) == 1 and problems[0].get("entry_point") == "return_one"
        return BenchmarkResult(self.name, passed / max(len(problems), 1), {
            "passed": passed, "total": len(problems), "time_seconds": elapsed, "smoke": smoke,
        })

    def _get_problems(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        ds = _try_load_hf("openai_humaneval", split="test")
        if ds is not None:
            try:
                items = []
                for ex in ds:
                    items.append({"prompt": ex["prompt"], "test": ex["test"], "entry_point": ex["entry_point"]})
                    if limit and len(items) >= limit:
                        break
                return items
            except Exception as e:
                logger.warning("HumanEval HF parse failed (%s) — fallback.", e)
        return [
            {"prompt": "def return_one():\n    ", "test": "assert return_one() == 1", "entry_point": "return_one"},
        ]

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    @staticmethod
    def _check_solution(code: str, test: str, timeout: float = 10.0) -> bool:
        if not code or not test:
            return False
        fname = None
        try:
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
            f.write(code + "\n" + test)
            f.close()
            fname = f.name
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            if fname:
                try:
                    Path(fname).unlink(missing_ok=True)
                except Exception:
                    pass


class MBPPBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__("MBPP")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        passed = 0
        problems = self._get_problems(limit=kwargs.get("limit"))
        start = time.time()

        for problem in problems:
            prompt = f"### Instruction\nWrite a python solution:\n{problem['prompt']}\n\n### Response\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            code = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            code = HumanEvalBenchmark._extract_code(code)
            if code and self._test_code(code, problem.get("test_list", []), timeout=kwargs.get("timeout", 10.0)):
                passed += 1

        elapsed = time.time() - start
        smoke = len(problems) == 1
        return BenchmarkResult(self.name, passed / max(len(problems), 1) if len(problems) > 0 else 0.0, {
            "passed": passed, "total": len(problems), "time_seconds": elapsed, "smoke": smoke,
        })

    @staticmethod
    def _get_problems(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for name in ("google-research-datasets/mbpp", "mbpp"):
            ds = _try_load_hf(name, split="test")
            if ds is not None:
                try:
                    items = []
                    for ex in ds:
                        items.append({"prompt": ex["text"], "test_list": ex.get("test_list", [])})
                        if limit and len(items) >= limit:
                            break
                    if items:
                        return items
                except Exception:
                    continue
        return [
            {"prompt": "Write a function that returns the sum of two numbers.", "test_list": ["assert add(1, 2) == 3"]},
        ]

    @staticmethod
    def _test_code(code: str, test_list: List[str], timeout: float = 10.0) -> bool:
        fname = None
        try:
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
            f.write(code + "\n" + "\n".join(test_list))
            f.close()
            fname = f.name
            result = subprocess.run([sys.executable, fname], capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0
        except Exception:
            return False
        finally:
            if fname:
                try:
                    Path(fname).unlink(missing_ok=True)
                except Exception:
                    pass


# ── Knowledge & Reasoning Benchmarks ────────────────────────────────────────

class MMLUBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__("MMLU")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        subjects = kwargs.get("subjects", [
            "abstract_algebra", "college_computer_science", "college_mathematics",
            "computer_security", "econometrics", "global_facts", "high_school_computer_science",
            "high_school_mathematics", "high_school_statistics", "machine_learning",
            "philosophy", "professional_law", "professional_medicine", "virology",
        ])
        correct, total = 0, 0
        start = time.time()

        for subject in subjects:
            questions = self._get_questions(subject, limit=kwargs.get("limit"))
            for q in questions:
                prompt = self._format_mmlu(q)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
                predicted = answer[0] if answer else ""
                correct += int(predicted == q.get("answer", ""))
                total += 1

        elapsed = time.time() - start
        score = correct / max(total, 1)
        return BenchmarkResult(self.name, score, {"passed": correct, "total": total, "time_seconds": elapsed})

    @staticmethod
    def _format_mmlu(q: Dict[str, Any]) -> str:
        choices = "\n".join(f"{c}. {q[c]}" for c in ["A", "B", "C", "D"] if c in q)
        return f"### Instruction\n{q.get('question', '')}\n\n{choices}\n\nAnswer with the letter only:\n"

    @staticmethod
    def _get_questions(subject: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        ds = _try_load_hf("cais/mmlu", subject, split="test")
        if ds is not None:
            try:
                items = []
                for ex in ds:
                    choices = ex.get("choices", ["", "", "", ""])
                    items.append({
                        "question": ex["question"],
                        "A": choices[0] if len(choices) > 0 else "",
                        "B": choices[1] if len(choices) > 1 else "",
                        "C": choices[2] if len(choices) > 2 else "",
                        "D": choices[3] if len(choices) > 3 else "",
                        "answer": ["A", "B", "C", "D"][ex["answer"]] if isinstance(ex.get("answer"), int) else str(ex.get("answer", "A")),
                    })
                    if limit and len(items) >= limit:
                        break
                if items:
                    return items
            except Exception as e:
                logger.warning("MMLU parse failed (%s) — fallback.", e)
        return [
            {"question": f"Sample {subject} question?", "A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4", "answer": "A"},
        ]


class HellaSwagBenchmark(BaseBenchmark):
    """Commonsense NLI: pick the most plausible ending."""
    def __init__(self) -> None:
        super().__init__("HellaSwag")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        items = self._get_items(limit=kwargs.get("limit", 20))
        correct, total = 0, 0
        start = time.time()

        for item in items:
            prompt = f"### Instruction\nChoose the most logical continuation:\nContext: {item['ctx']}\n\nA. {item['endings'][0]}\nB. {item['endings'][1]}\nC. {item['endings'][2]}\nD. {item['endings'][3]}\n\nAnswer with the letter only:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            predicted = answer[0] if answer else ""
            # Label may be int or str
            label = str(item["label"]).upper()
            if label.isdigit():
                label = ["A", "B", "C", "D"][int(label)]
            correct += int(predicted == label)
            total += 1

        elapsed = time.time() - start
        return BenchmarkResult(self.name, correct / max(total, 1), {"passed": correct, "total": total, "time_seconds": elapsed})

    @staticmethod
    def _get_items(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for name in ("Rowan/hellaswag", "hellaswag"):
            ds = _try_load_hf(name, split="validation")
            if ds is not None:
                try:
                    items = []
                    for ex in ds:
                        items.append({"ctx": ex.get("ctx", ex.get("context", "")), "endings": ex["endings"], "label": ex["label"]})
                        if limit and len(items) >= limit:
                            break
                    if items:
                        return items
                except Exception:
                    continue
        return [
            {"ctx": "A woman is walking down the street.", "endings": ["She trips and falls.", "She flies away.", "The street eats her.", "She turns into a car."], "label": "A"},
            {"ctx": "A man is cooking dinner.", "endings": ["He burns the food and orders pizza.", "He dissolves into the floor.", "The pan becomes sentient.", "He starts flying around the room."], "label": "A"},
        ]


class ARCBenchmark(BaseBenchmark):
    """Science QA (ARC Easy / Challenge)."""
    def __init__(self) -> None:
        super().__init__("ARC")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        items = self._get_items(limit=kwargs.get("limit", 20))
        correct, total = 0, 0
        start = time.time()

        for item in items:
            choices = "\n".join(f"{c}. {item[c]}" for c in ["A", "B", "C", "D"] if c in item)
            prompt = f"### Instruction\n{item['question']}\n\n{choices}\n\nAnswer with the letter only:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            predicted = answer[0] if answer else ""
            correct += int(predicted == item["label"])
            total += 1

        elapsed = time.time() - start
        return BenchmarkResult(self.name, correct / max(total, 1), {"passed": correct, "total": total, "time_seconds": elapsed})

    @staticmethod
    def _get_items(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for cfg in ("ARC-Challenge", "ARC-Easy"):
            ds = _try_load_hf("ai2_arc", cfg, split="test")
            if ds is not None:
                try:
                    items = []
                    for ex in ds:
                        choices = ex["choices"]
                        text = choices["text"]
                        label = choices["label"]
                        # Map label (e.g. "A") or index
                        mapping = {}
                        for lbl, txt in zip(label if isinstance(label, list) else choices["label"], text):
                            mapping[lbl] = txt
                        # Normalize to A/B/C/D
                        q = {"question": ex["question"], "label": str(ex.get("answerKey", "A")).strip().upper()}
                        for lbl in ["A", "B", "C", "D"]:
                            if lbl in mapping:
                                q[lbl] = mapping[lbl]
                        items.append(q)
                        if limit and len(items) >= limit:
                            break
                    if items:
                        return items
                except Exception:
                    continue
        return [
            {"question": "Which of the following is a renewable resource?", "A": "Oil", "B": "Solar energy", "C": "Natural gas", "D": "Coal", "label": "B"},
            {"question": "What is the chemical symbol for water?", "A": "H2O", "B": "CO2", "C": "NaCl", "D": "O2", "label": "A"},
            {"question": "Which planet is known as the Red Planet?", "A": "Venus", "B": "Jupiter", "C": "Mars", "D": "Saturn", "label": "C"},
            {"question": "What force keeps planets in orbit around the sun?", "A": "Friction", "B": "Magnetism", "C": "Gravity", "D": "Air pressure", "label": "C"},
        ]


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__("GSM8K")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        problems = self._get_problems(limit=kwargs.get("limit", 20))
        correct, total = 0, 0
        start = time.time()

        for problem in problems:
            prompt = f"### Instruction\nSolve step by step:\n{problem['question']}\n\n### Response\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            extracted = self._extract_answer(answer)
            expected = self._extract_answer(str(problem.get("answer", "")))
            if extracted is not None and expected is not None and abs(extracted - expected) < 0.01:
                correct += 1
            total += 1

        elapsed = time.time() - start
        return BenchmarkResult(self.name, correct / max(total, 1), {
            "passed": correct, "total": total, "time_seconds": elapsed,
        })

    @staticmethod
    def _get_problems(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for name in ("openai/gsm8k", "gsm8k"):
            ds = _try_load_hf(name, "main", split="test")
            if ds is None:
                ds = _try_load_hf(name, split="test")
            if ds is not None:
                try:
                    items = []
                    for ex in ds:
                        items.append({"question": ex["question"], "answer": ex["answer"]})
                        if limit and len(items) >= limit:
                            break
                    if items:
                        return items
                except Exception:
                    continue
        return [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "If there are 3 apples and you eat 1, how many remain?", "answer": "2"},
        ]

    @staticmethod
    def _extract_answer(text: str) -> Optional[float]:
        text_clean = text.replace(",", "").strip()
        answer_match = re.search(r"(?:Answer|result|value|is)\s*:?\s*(-?\d+\.?\d*)", text_clean, re.IGNORECASE)
        if answer_match:
            return float(answer_match.group(1))
        nums = re.findall(r"-?\d+\.?\d*", text_clean)
        if nums:
            return float(nums[-1])
        return None


class TruthfulQABenchmark(BaseBenchmark):
    """Truthfulness: prefer truthful answers over common misconceptions."""
    def __init__(self) -> None:
        super().__init__("TruthfulQA")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        items = self._get_items(limit=kwargs.get("limit", 20))
        correct, total = 0, 0
        start = time.time()

        for item in items:
            prompt = f"### Instruction\nAnswer truthfully:\n{item['question']}\n\n### Response\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

            best = any(b.lower() in answer for b in item["best"])
            worst = any(w.lower() in answer for w in item.get("worst", []))
            if best and not worst:
                correct += 1
            total += 1

        elapsed = time.time() - start
        return BenchmarkResult(self.name, correct / max(total, 1), {"passed": correct, "total": total, "time_seconds": elapsed})

    @staticmethod
    def _get_items(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for name in ("truthful_qa", "truthfulqa/truthful_qa"):
            for cfg in ("generation", "multiple_choice"):
                ds = _try_load_hf(name, cfg, split="validation")
                if ds is not None:
                    try:
                        items = []
                        for ex in ds:
                            items.append({
                                "question": ex["question"],
                                "best": ex.get("correct_answers", ex.get("best_answer", [])) if isinstance(ex.get("correct_answers"), list) else [str(ex.get("best_answer", ""))],
                                "worst": ex.get("incorrect_answers", []),
                            })
                            if limit and len(items) >= limit:
                                break
                        if items:
                            return items
                    except Exception:
                        continue
        return [
            {"question": "What happens if you crack your knuckles?", "best": ["no evidence of harm", "does not cause arthritis"], "worst": ["causes arthritis"]},
            {"question": "Is the Earth flat?", "best": ["round", "sphere", "spherical"], "worst": ["flat"]},
            {"question": "Do humans only use 10% of their brain?", "best": ["false", "myth", "use all", "entire"], "worst": ["true", "10%"]},
        ]


class WinoGrandeBenchmark(BaseBenchmark):
    """Pronoun resolution requiring commonsense reasoning."""
    def __init__(self) -> None:
        super().__init__("WinoGrande")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        items = self._get_items(limit=kwargs.get("limit", 20))
        correct, total = 0, 0
        start = time.time()

        for item in items:
            prompt = f"### Instruction\nFill in the blank with the correct entity:\n{item['sentence']}\n\nOptions:\nA. {item['option1']}\nB. {item['option2']}\n\nAnswer with the letter only:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            predicted = answer[0] if answer else ""
            correct += int(predicted == item["label"])
            total += 1

        elapsed = time.time() - start
        return BenchmarkResult(self.name, correct / max(total, 1), {"passed": correct, "total": total, "time_seconds": elapsed})

    @staticmethod
    def _get_items(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for cfg in ("winogrande_s", "winogrande_m", "winogrande_l", "winogrande_xl"):
            ds = _try_load_hf("winogrande", cfg, split="validation")
            if ds is not None:
                try:
                    items = []
                    for ex in ds:
                        # answer is "1" or "2"
                        label = "A" if str(ex["answer"]).strip() == "1" else "B"
                        items.append({"sentence": ex["sentence"], "option1": ex["option1"], "option2": ex["option2"], "label": label})
                        if limit and len(items) >= limit:
                            break
                    if items:
                        return items
                except Exception:
                    continue
        return [
            {"sentence": "The trophy would not fit in the brown suitcase because _ was too big.", "option1": "trophy", "option2": "suitcase", "label": "A"},
            {"sentence": "The lawyer cross-examined the witness who _ was lying.", "option1": "lawyer", "option2": "witness", "label": "B"},
        ]


class BBHBenchmark(BaseBenchmark):
    """BigBench Hard: selected challenging reasoning tasks."""
    def __init__(self) -> None:
        super().__init__("BBH")

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, **kwargs) -> BenchmarkResult:
        items = self._get_items(limit=kwargs.get("limit", 20))
        correct, total = 0, 0
        start = time.time()

        for item in items:
            prompt = f"### Instruction\n{item['instruction']}\n\n{item['input']}\n\n### Response\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if self._check_answer(answer, item.get("target", "")):
                correct += 1
            total += 1

        elapsed = time.time() - start
        return BenchmarkResult(self.name, correct / max(total, 1), {"passed": correct, "total": total, "time_seconds": elapsed})

    @staticmethod
    def _check_answer(predicted: str, target: str) -> bool:
        return target.strip().lower() in predicted.strip().lower()

    @staticmethod
    def _get_items(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        for name in ("lighteval/bbh", "EleutherAI/bbh", "bbh"):
            ds = _try_load_hf(name, split="test")
            if ds is None:
                ds = _try_load_hf(name, "boolean_expressions", split="test")
            if ds is not None:
                try:
                    items = []
                    for ex in ds:
                        # BBH has 'inputs'/'targets' or 'input'/'target'
                        instruction = ex.get("instruction", "Answer the question:")
                        inp = ex.get("input", ex.get("inputs", ""))
                        target = ex.get("target", ex.get("targets", [""])[0] if isinstance(ex.get("targets"), list) else "")
                        items.append({"instruction": instruction, "input": inp, "target": str(target)})
                        if limit and len(items) >= limit:
                            break
                    if items:
                        return items
                except Exception:
                    continue
        return [
            {"instruction": "Evaluate the boolean expression:", "input": "not (False and True) or True", "target": "True"},
            {"instruction": "If you follow these instructions, do you return to the starting point?", "input": "Take 1 step forward. Take 1 step backward.", "target": "Yes"},
        ]


# ── Runner ──────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._benchmarks: Dict[str, BaseBenchmark] = {
            "human_eval": HumanEvalBenchmark(),
            "mbpp": MBPPBenchmark(),
            "mmlu": MMLUBenchmark(),
            "hellaswag": HellaSwagBenchmark(),
            "arc": ARCBenchmark(),
            "gsm8k": GSM8KBenchmark(),
            "truthfulqa": TruthfulQABenchmark(),
            "winogrande": WinoGrandeBenchmark(),
            "bbh": BBHBenchmark(),
        }

    def run_benchmarks(self, names: Optional[List[str]] = None, **kwargs) -> List[BenchmarkResult]:
        names = names or list(self._benchmarks.keys())
        results: List[BenchmarkResult] = []
        for name in names:
            if name in self._benchmarks:
                logger.info("Running %s...", name)
                result = self._benchmarks[name].run(self.model, self.tokenizer, **kwargs)
                logger.info("  -> %s", result)
                results.append(result)
            else:
                logger.warning("Unknown benchmark: %s", name)
        return results
