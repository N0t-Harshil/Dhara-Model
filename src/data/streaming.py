from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import yaml
from datasets import load_dataset

from src.config.schema import DatasetEntryConfig

logger = logging.getLogger(__name__)


class MassiveDataCollector:
    def __init__(self, datasets_cfg: List[DatasetEntryConfig]) -> None:
        self.datasets = datasets_cfg
        self.target_libraries = [
            "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
            "pytorch", "torch", "tensorflow", "jax", "keras",
            "matplotlib", "seaborn", "plotly", "bokeh",
            "transformers", "tokenizers", "accelerate", "peft", "diffusers",
            "bitsandbytes", "deepspeed", "optimum", "vllm", "sglang",
            "outlines", "instructor", "pydantic_ai", "guidance",
            "langchain", "langgraph", "langserve", "llamaindex", "autogen",
            "crewai", "haystack", "semantic_kernel", "babyagi",
            "fastapi", "flask", "django", "uvicorn", "gunicorn", "docker",
            "kubernetes", "k8s", "boto3", "google-cloud-aiplatform",
            "chromadb", "pinecone", "qdrant", "weaviate", "milvus",
            "pymongo", "redis", "sqlalchemy", "psycopg2",
        ]

    def get_dataset_list(self) -> List[DatasetEntryConfig]:
        return self.datasets

    def stream_single_dataset(
        self,
        ds_info: DatasetEntryConfig,
        limit: Optional[int] = None,
        theme: str = "all",
        skip_samples: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        count = 0
        if limit is None:
            limit = ds_info.max_samples
        groups = self._get_groups()
        target_libs = groups.get(theme, self.target_libraries)

        state_key = f"_skip_{ds_info.path.replace('/','_').replace('-','_')}"
        if not hasattr(self, state_key):
            setattr(self, state_key, skip_samples)

        logger.info("Streaming %s samples from %s...", theme, ds_info.path)
        try:
            ds_kwargs = self._build_ds_kwargs(ds_info)
            while True:
                processed = 0
                ds = load_dataset(**ds_kwargs)
                initial_skip = getattr(self, state_key)

                for entry in ds:
                    processed += 1
                    sample = self._process_entry(entry, theme, target_libs)
                    if sample:
                        current_skip = getattr(self, state_key)
                        if current_skip > 0:
                            setattr(self, state_key, current_skip - 1)
                            continue
                        yield sample
                        count += 1
                        if limit and count >= limit:
                            return

                if count > 0:
                    break
                if processed == 0:
                    logger.warning("Dataset %s is empty.", ds_info.path)
                    break
                if getattr(self, state_key) == initial_skip:
                    logger.warning("Dataset %s yielded 0 valid samples. Breaking.", ds_info.path)
                    break
        except Exception as e:
            logger.error("Error streaming from %s: %s", ds_info.path, e)

        logger.info("Finished streaming %d samples from %s", count, ds_info.path)

    def stream_samples(
        self,
        limit: Optional[int] = None,
        theme: str = "all",
        stages: Optional[List[Tuple[str, int]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        if stages:
            yield from self._stream_curriculum(stages)
        else:
            yield from self._stream_for_theme(theme, limit)

    def _stream_curriculum(
        self, stages: List[Tuple[str, int]]
    ) -> Generator[Dict[str, Any], None, None]:
        stage_iter = iter(stages)
        try:
            curr_theme, curr_limit = next(stage_iter)
        except StopIteration:
            return

        curr_count = 0
        logger.info("=== CURRICULUM STAGE: %s (%d samples) ===", curr_theme.upper(), curr_limit)
        groups = self._get_groups()

        for ds_info in self.datasets:
            ds_kwargs = self._build_ds_kwargs(ds_info)
            try:
                ds = load_dataset(**ds_kwargs)
                ds_count = 0
                ds_limit = ds_info.max_samples
                for entry in ds:
                    target_libs = groups.get(curr_theme, self.target_libraries)
                    sample = self._process_entry(entry, curr_theme, target_libs)
                    if sample:
                        yield sample
                        curr_count += 1
                        ds_count += 1
                        if ds_limit and ds_count >= ds_limit:
                            break
                        if curr_count >= curr_limit:
                            try:
                                curr_theme, curr_limit = next(stage_iter)
                                curr_count = 0
                                logger.info("=== CURRICULUM STAGE: %s (%d samples) ===", curr_theme.upper(), curr_limit)
                            except StopIteration:
                                return
            except Exception as e:
                logger.error("Error in curriculum stream for %s: %s", ds_info.path, e)

    def _stream_for_theme(
        self, theme: str, limit: Optional[int]
    ) -> Generator[Dict[str, Any], None, None]:
        count = 0
        groups = self._get_groups()
        target_libs = groups.get(theme, self.target_libraries)

        for ds_info in self.datasets:
            logger.info("Streaming %s from %s...", theme, ds_info.path)
            ds_kwargs = self._build_ds_kwargs(ds_info)
            try:
                ds = load_dataset(**ds_kwargs)
                ds_count = 0
                ds_limit = ds_info.max_samples
                for entry in ds:
                    sample = self._process_entry(entry, theme, target_libs)
                    if sample:
                        yield sample
                        count += 1
                        ds_count += 1
                        if limit and count >= limit:
                            return
                        if ds_limit and ds_count >= ds_limit:
                            break
            except Exception as e:
                logger.error("Error streaming %s: %s", ds_info.path, e)

    def _build_ds_kwargs(self, ds_info: DatasetEntryConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "path": ds_info.path,
            "split": ds_info.split,
            "streaming": True,
        }
        if ds_info.data_dir:
            kwargs["data_dir"] = ds_info.data_dir
        if ds_info.name:
            kwargs["name"] = ds_info.name
        return kwargs

    def _get_groups(self) -> Dict[str, List[str]]:
        return {
            "core": [],
            "ai": ["numpy", "pandas", "pytorch", "torch", "tensorflow", "transformers", "scikit-learn", "matplotlib"],
            "agentic": ["langchain", "langgraph", "llamaindex", "autogen", "crewai", "pydantic_ai", "instructor"],
            "multi": ["rust", "golang", "cpp", "java", "typescript", "php", "ruby", "swift", "csharp", "kotlin", "sql", "shell", "dart", "scala", "lua", "zig", "haskell"],
            "all": self.target_libraries,
        }

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return "\n".join(str(v) for v in value if v is not None)
        return str(value).strip()

    def _process_entry(self, entry: dict, theme: str, target_libs: list) -> Optional[Dict[str, str]]:
        instruction, input_text, output = self._extract_fields(entry)
        content = "\n".join(p for p in (instruction, input_text, output) if p)
        lang = self._detect_entry_language(entry, content)

        content_lower = content.lower()
        if theme != "all":
            if theme == "core":
                if any(lib in content_lower for lib in self.target_libraries):
                    return None
            elif theme not in ("multi", "text"):
                if not any(lib in content_lower for lib in target_libs):
                    return None

        if not self._is_valid_sample(content, lang):
            return None

        return {
            "instruction": instruction or f"Complete this {lang} code:",
            "input": input_text,
            "output": output or content,
            "language": lang,
        }

    def _extract_fields(self, entry: dict) -> Tuple[str, str, str]:
        instruction = self._clean_text(entry.get("instruction", ""))
        input_text = self._clean_text(entry.get("input", ""))
        output = self._clean_text(entry.get("output", ""))

        # Tool-use / function calling: tool_def + instruction (check before standard path)
        if "tool_definition" in entry and instruction:
            tool = self._clean_text(entry["tool_definition"])
            resp = self._clean_text(entry.get("response", ""))
            return f"{tool}\n\n{instruction}", "", resp

        # Dolly format: context + instruction + response (check before standard path)
        if "context" in entry and entry.get("context") and "response" in entry:
            ctx = self._clean_text(entry["context"])
            inst = instruction if instruction else self._clean_text(entry.get("instruction", ""))
            resp = self._clean_text(entry["response"])
            return f"{ctx} {inst}".strip(), "", resp

        # Fast path: standard instruction/output already filled
        if instruction and output:
            return instruction, input_text, output

        # Alpaca/CodeAlpaca: instruction + response
        if instruction and "response" in entry:
            return instruction, "", self._clean_text(entry.get("response", ""))

        # Problem/solution (code contests, competitive programming)
        if "problem" in entry and "solution" in entry:
            return self._clean_text(entry["problem"]), "", self._clean_text(entry["solution"])

        # Query/answer (general QA)
        if "query" in entry and "answer" in entry:
            return self._clean_text(entry["query"]), "", self._clean_text(entry["answer"])

        # Question/answer (GSM8K, MATH, CoT datasets)
        if "question" in entry:
            if "answer" in entry:
                return self._clean_text(entry["question"]), "", self._clean_text(entry["answer"])
            if "rationale" in entry and "target" in entry:
                rationale = self._clean_text(entry["rationale"])
                target = self._clean_text(entry["target"])
                return self._clean_text(entry["question"]), "", f"{rationale} {target}"

        # FLAN format: inputs + targets
        if "inputs" in entry and "targets" in entry:
            return self._clean_text(entry["inputs"]), "", self._clean_text(entry["targets"])

        # Orca-style: question + response + system_prompt
        if "system_prompt" in entry and "response" in entry:
            sys_p = self._clean_text(entry["system_prompt"])
            q = instruction if instruction else self._clean_text(entry.get("question", ""))
            r = self._clean_text(entry["response"])
            return f"{sys_p}\n\n{q}" if q else sys_p, "", r

        # Chat/conversation formats (ShareGPT, OpenAssistant, UltraChat, etc.)
        if "messages" in entry or "chosen" in entry or "conversations" in entry:
            return self._extract_chat_fields(entry)

        # ShareGPT format with 'from'/'value' pairs
        if "from" in entry and "value" in entry:
            return self._extract_sharegpt_fields(entry)

        # Prompt + completion/response (UltraFeedback, Tulu, etc.)
        if "prompt" in entry:
            prompt_val = self._clean_text(entry["prompt"])
            completion = entry.get("completion") or entry.get("response", "")
            if completion:
                return prompt_val, "", self._clean_text(completion)
            # prompt might contain full conversation
            if not instruction:
                instruction = prompt_val

        # Code contests: description + solutions dict
        if "description" in entry and "solutions" in entry:
            return self._extract_code_contests(entry)

        # Function/docstring pairs (CodeSearchNet style)
        if "func_documentation_string" in entry and "func_code_string" in entry:
            return (
                self._clean_text(entry["func_documentation_string"]),
                "",
                self._clean_text(entry["func_code_string"]),
            )

        # Code + explanation (paired)
        if "code" in entry and "explanation" in entry:
            return self._clean_text(entry.get("explanation") or entry.get("description", "")), "", self._clean_text(entry["code"])

        # NLU datasets: premise/hypothesis or sentence1/sentence2
        if "sentence1" in entry and "sentence2" in entry:
            return self._clean_text(entry["sentence1"]), self._clean_text(entry["sentence2"]), self._clean_text(entry.get("label", ""))

        # SciQ: support + question + answer
        if "support" in entry and "answer" in entry:
            support = self._clean_text(entry["support"])
            q = self._clean_text(entry.get("question", instruction))
            a = self._clean_text(entry["answer"])
            return f"{support}\n\n{q}" if support else q, "", a

        # C4 / BookCorpus / PG19: just text content (no instruction)
        fallback = self._clean_text(entry.get("content") or entry.get("text") or entry.get("code") or "")
        if fallback and not instruction:
            return "", "", fallback

        # Last resort: if we have instruction but no output yet, try any field
        if instruction and not output:
            for field in ["response", "completion", "answer", "target", "result", "code", "text", "content"]:
                val = entry.get(field)
                if val:
                    return instruction, input_text, self._clean_text(val)

        return instruction, input_text, output or fallback

    def _extract_sharegpt_fields(self, entry: dict) -> Tuple[str, str, str]:
        """ShareGPT format: {'from': 'human', 'value': '...'} or {'from': 'gpt', 'value': '...'}."""
        role = str(entry.get("from", "")).lower()
        value = self._clean_text(entry.get("value", ""))
        if role in ("human", "user"):
            return value, "", ""
        if role in ("gpt", "assistant"):
            return "", "", value
        return "", "", value

    def _extract_chat_fields(self, entry: dict) -> Tuple[str, str, str]:
        messages = entry.get("messages") or entry.get("chosen") or entry.get("conversations") or []
        if not isinstance(messages, list):
            return self._clean_text(entry.get("prompt")), "", self._clean_text(entry.get("response") or entry.get("answer", ""))

        user_turns, assistant_turns = [], []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").lower()
            text = self._clean_text(msg.get("content") or msg.get("value") or msg.get("text"))
            if not text:
                continue
            if role in ("assistant", "gpt", "model", "bot"):
                assistant_turns.append(text)
            elif role in ("user", "human", "prompter", "customer"):
                user_turns.append(text)

        user_text = user_turns[-1] if user_turns else ""
        asst_text = assistant_turns[-1] if assistant_turns else ""
        return user_text, "", asst_text

    def _extract_code_contests(self, entry: dict) -> Tuple[str, str, str]:
        desc = self._clean_text(entry.get("description", ""))
        solutions = entry.get("solutions", {})
        if isinstance(solutions, dict):
            sol_list = solutions.get("solution", [])
        elif isinstance(solutions, list):
            sol_list = solutions
        else:
            return desc, "", ""
        if isinstance(sol_list, list) and len(sol_list) > 0:
            return desc, "", self._clean_text(sol_list[0])
        return desc, "", ""

    @staticmethod
    def _normalize_language(lang: str) -> str:
        aliases = {
            "c++": "cpp", "cc": "cpp", "cxx": "cpp", "py": "python",
            "python3": "python", "js": "javascript", "jsx": "javascript",
            "ts": "typescript", "tsx": "typescript", "go": "golang",
            "bash": "shell", "sh": "shell", "markdown": "text", "md": "text",
        }
        return aliases.get(lang.strip().lower(), lang.strip().lower())

    def _detect_entry_language(self, entry: dict, content: str) -> str:
        forced = entry.get("language") or entry.get("lang") or entry.get("programming_language") or entry.get("ext")
        if forced:
            return self._normalize_language(str(forced))
        return self._detect_language(content)

    @staticmethod
    def _detect_language(content: str) -> str:
        lower = content.lower()
        markers = {
            "python": ["def ", "import ", "if __name__", "elif ", "except ", "raise "],
            "rust": ["fn ", "let mut", "impl ", "match ", "pub "],
            "golang": ["func ", "package ", "chan ", "select {", "go "],
            "cpp": ["#include", "std::", "int main(", "public:", "virtual ", "::iterator"],
            "java": ["public class", "System.out.println", "@Override", "private static", "protected void"],
            "typescript": ["interface ", "type ", "as string", "readonly ", ": string"],
            "javascript": ["const ", "let ", "=>", "function ", "export default"],
            "csharp": ["using System", "namespace ", "Console.WriteLine"],
            "ruby": ["def ", "end", "require ", "module "],
            "swift": ["func ", "var ", "let ", "guard "],
            "kotlin": ["fun ", "val ", "var ", "data class"],
            "sql": ["SELECT ", "INSERT INTO", "CREATE TABLE"],
            "shell": ["#!/bin/bash", "#!/bin/sh", "if [[", "export "],
            "php": ["<?php", "$this->", "public function"],
            "perl": ["#!/usr/bin/perl", "use strict"],
            "scala": ["object ", "def main(", "import scala"],
            "lua": ["function ", "local "],
            "r": ["library(", "ggplot("],
            "haskell": ["::", "where", "data "],
            "dart": ["void main", "import 'package:"],
            "elixir": ["defmodule", "def do"],
            "julia": ["function ", "println("],
            "fsharp": ["let ", "module "],
        }
        scores = {}
        for lang, sigs in markers.items():
            count = sum(1 for s in sigs if s in lower)
            if count > 0:
                scores[lang] = count
        if scores:
            return max(scores, key=scores.get)
        nl_markers = ["the ", "is ", "are ", "was ", "were ", "have ", "has ", "do ", "does ", "an ", "this ", "that ", "with ", "for ", "not ", "but ", "can ", "all ", "its "]
        if sum(1 for m in nl_markers if m in lower) >= 2:
            return "text"
        return "python"

    @staticmethod
    def _is_valid_sample(content: str, language: str) -> bool:
        if not content or len(content) < 50 or len(content) > 250000:
            return False
        lowered = content.lower()
        bad = ["lorem ipsum", "todo: add code", "your code here", "coming soon", "placeholder", "under construction"]
        if any(m in lowered for m in bad):
            return False
        signals = {
            "python": ["def ", "import ", "class "],
            "rust": ["fn ", "let ", "use "],
            "golang": ["func ", "package "],
            "cpp": ["#include", "int ", "void "],
            "java": ["class ", "public ", "private ", "protected "],
            "typescript": ["interface ", "export "],
            "javascript": ["function", "const ", "let "],
            "csharp": ["class ", "void ", "int ", "string "],
            "ruby": ["def ", "end", "do "],
            "swift": ["func ", "var ", "let "],
            "kotlin": ["fun ", "val ", "var "],
            "php": ["<?php", "function "],
            "sql": ["select", "from", "where "],
            "shell": ["#!/", "echo ", "export "],
            "perl": ["my ", "sub ", "use "],
            "scala": ["def ", "val ", "object "],
            "lua": ["function ", "local "],
            "r": ["<-", "function(", "library"],
            "haskell": [" :: ", "->"],
            "dart": ["void ", "class ", "import "],
            "elixir": ["defmodule", "def "],
            "julia": ["function ", "println"],
            "fsharp": ["let ", "module "],
            "text": ["the ", "is ", "are ", "was "],
        }
        sigs = signals.get(language, [" "])
        return any(s in lowered for s in sigs)
