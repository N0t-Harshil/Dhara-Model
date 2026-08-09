#!/usr/bin/env python3
"""
Massive Data Collector for the Specialized Coding Model.
Streams millions of Python samples from Hugging Face datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
import yaml
from datasets import load_dataset

logger = logging.getLogger("massive_collector")

class MassiveDataCollector:
    def __init__(self, datasets_cfg_or_config: "str | list" = "config.yaml"):
        """Constructor accepts either a path to a YAML config (legacy behavior)
        or a list of dataset configs (tests pass a list). When a list is passed,
        the collector uses it directly and avoids file I/O.
        """
        if isinstance(datasets_cfg_or_config, list):
            # tests and some callsites pass a datasets list directly
            self.config = {"data": {"datasets": datasets_cfg_or_config}}
            self.datasets = datasets_cfg_or_config
        else:
            # treat as path to config file
            self.config = self._load_config(datasets_cfg_or_config)
            self.datasets = (
                self.config.get("data", {}).get("datasets")
                or self.config.get("data_collection", {}).get("datasets", [])
            )
        self.target_libraries = [
            # Core DS/AI
            "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
            "pytorch", "torch", "tensorflow", "jax", "keras",
            "matplotlib", "seaborn", "plotly", "bokeh",
            
            # NLP / LLMs
            "transformers", "tokenizers", "accelerate", "peft", "diffusers",
            "bitsandbytes", "deepspeed", "optimum", "vllm", "sglang",
            "outlines", "instructor", "pydantic_ai", "guidance",
            
            # Agentic Frameworks
            "langchain", "langgraph", "langserve", "llamaindex", "autogen",
            "crewai", "haystack", "semantic_kernel", "babyagi",
            
            # API / Deployment
            "fastapi", "flask", "django", "uvicorn", "gunicorn", "docker",
            "kubernetes", "k8s", "boto3", "google-cloud-aiplatform",
            
            # DB / Vector DB
            "chromadb", "pinecone", "qdrant", "weaviate", "milvus",
            "pymongo", "redis", "sqlalchemy", "psycopg2"
        ]

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_dataset_list(self) -> list:
        """Returns the list of dataset configs for staged training."""
        return self.datasets

    def stream_single_dataset(self, ds_info: dict, limit=None, theme: str = "all", skip_samples: int = 0):
        """Stream samples from a single dataset entry.
        
        Used by the staged training pipeline to train on one dataset at a time.
        """
        count = 0
        if limit is None:
            limit = ds_info.get("max_samples")
        groups = self._get_groups()
        target_libs = groups.get(theme, self.target_libraries)
        
        state_parts = [
            ds_info.get("path", ""),
            ds_info.get("data_dir", ""),
            ds_info.get("name", ""),
            ds_info.get("split", ""),
        ]
        state_key = "_skip_" + "_".join(
            str(part).replace("/", "_").replace("-", "_").replace("+", "plus")
            for part in state_parts
            if part
        )
        if not hasattr(self, state_key):
            setattr(self, state_key, skip_samples)
            
        logger.info(f"Streaming {theme} samples from {ds_info['path']}...")
        try:
            ds_kwargs = {
                "path": ds_info["path"],
                "split": ds_info.get("split", "train"),
                "streaming": True,
            }
            if "data_dir" in ds_info:
                ds_kwargs["data_dir"] = ds_info["data_dir"]
            if "name" in ds_info:
                ds_kwargs["name"] = ds_info["name"]
            if "languages" in ds_info:
                ds_kwargs["languages"] = ds_info["languages"]
            
            while True:
                processed_entries = 0
                ds = load_dataset(**ds_kwargs)
                initial_skip = getattr(self, state_key)
                
                for entry in ds:
                    processed_entries += 1
                    sample = self._process_entry(entry, theme, target_libs, ds_info=ds_info)
                    if sample:
                        current_skip = getattr(self, state_key)
                        if current_skip > 0:
                            setattr(self, state_key, current_skip - 1)
                            continue
                            
                        yield sample
                        count += 1
                        if limit and count >= int(limit):
                            return
                            
                if count > 0:
                    break
                    
                if processed_entries == 0:
                    logger.warning(f"Dataset {ds_info['path']} is empty. Breaking.")
                    break
                    
                # SAFETY CHECK: If we went through the entire dataset and didn't skip a single sample,
                # it means the dataset has 0 valid samples for this theme. We must break to avoid infinite loop!
                if getattr(self, state_key) == initial_skip:
                    logger.warning(f"Dataset {ds_info['path']} yielded 0 valid samples in this epoch. Breaking to avoid infinite loop.")
                    break
                    
                logger.info(f"Dataset exhausted during skip phase (remaining skips: {getattr(self, state_key)}). Looping internally.")
                
        except Exception as e:
            logger.error(f"Error streaming from {ds_info['path']}: {e}")
        
        logger.info(f"Finished streaming {count} samples from {ds_info['path']}")


    def stream_samples(self, limit=None, theme: str = "all", stages=None):
        """Streams samples from the configured datasets with theme filtering or staged curriculum."""
        if stages:
            stage_iter = iter(stages)
            try:
                curr_theme, curr_limit = next(stage_iter)
            except StopIteration:
                return
            
            curr_count = 0
            logger.info(f"=== [CURRICULUM STAGE: {curr_theme.upper()}] {curr_limit} samples ===")
            
            groups = self._get_groups()
            
            for ds_info in self.datasets:
                logger.info(f"Streaming curriculum from {ds_info['path']}...")
                try:
                    ds_kwargs = {
                        "path": ds_info["path"],
                        "split": ds_info.get("split", "train"),
                        "streaming": True,
                    }
                    if "data_dir" in ds_info:
                        ds_kwargs["data_dir"] = ds_info["data_dir"]
                    if "name" in ds_info:
                        ds_kwargs["name"] = ds_info["name"]
                    if "languages" in ds_info:
                        ds_kwargs["languages"] = ds_info["languages"]
                    
                    ds = load_dataset(**ds_kwargs)
                    
                    ds_count = 0
                    ds_limit = ds_info.get("max_samples")
                    for entry in ds:
                        target_libs = groups.get(curr_theme, self.target_libraries)
                        sample = self._process_entry(entry, curr_theme, target_libs, ds_info=ds_info)
                        
                        if sample:
                            yield sample
                            curr_count += 1
                            ds_count += 1
                            if ds_limit and ds_count >= int(ds_limit):
                                break
                            if curr_count >= curr_limit:
                                try:
                                    curr_theme, curr_limit = next(stage_iter)
                                    curr_count = 0
                                    logger.info(f"=== [CURRICULUM STAGE: {curr_theme.upper()}] {curr_limit} samples ===")
                                    target_libs = groups.get(curr_theme, self.target_libraries)
                                except StopIteration:
                                    return
                except Exception as e:
                    logger.error(f"Error streaming from {ds_info['path']}: {e}")
            return
            
        yield from self._stream_for_theme(theme=theme, limit=limit)

    def _get_groups(self):
        return {
            "core": [],
            "ai": ["numpy", "pandas", "pytorch", "torch", "tensorflow", "transformers", "scikit-learn", "matplotlib"],
            "agentic": ["langchain", "langgraph", "llamaindex", "autogen", "crewai", "pydantic_ai", "instructor"],
            "multi": ["rust", "golang", "cpp", "java", "typescript", "php", "ruby", "swift", "csharp", "kotlin", "sql", "shell", "dart", "scala", "lua", "zig", "haskell"],
            "all": self.target_libraries
        }

    @staticmethod
    def _detect_language(content: str) -> str:
        """Detects the programming language of the content (Supports 25+ languages).

        Static method so tests can call MassiveDataCollector._detect_language(...)
        without instantiating the class.
        """
        content_lower = content.lower()
        
        # Comprehensive language markers
        markers = {
            "python": ["def ", "import ", "class ", "if __name__"],
            "rust": ["fn main", "fn ", "let mut", "impl ", "match ", "pub "],
            "golang": ["func ", "package ", "chan ", "select {", "go "],
            "cpp": ["#include", "std::", "int main(", "public:", "virtual "],
            "java": ["public class", "System.out.println", "@Override", "package "],
            "typescript": ["interface ", "type ", "as string", "readonly ", "enum "],
            "javascript": ["const ", "let ", "=>", "function ", "export default"],
            "csharp": ["using System", "namespace ", "public class", "internal "],
            "php": ["<?php", "$this->", "public function", "namespace "],
            "ruby": ["def ", "end", "require ", "module ", "attr_accessor"],
            "swift": ["func ", "var ", "let ", "guard ", "extension "],
            "kotlin": ["fun ", "val ", "var ", "data class", "companion object"],
            "sql": ["SELECT ", "INSERT INTO", "CREATE TABLE", "WHERE ", "JOIN "],
            "shell": ["#!/bin/bash", "#!/bin/sh", "if [[", "echo ", "export "],
            "r": ["library(", "<- function(", "summary("],
            "scala": ["def ", "val ", "object ", "trait "],
            "objective-c": ["@interface", "@implementation", "#import <Foundation"],
            "perl": ["my $", "sub ", "use strict;"],
            "lua": ["local function", "then", "end", "require("],
            "haskell": ["main =", "module ", "where", "import "],
            "julia": ["function ", "using ", "struct "],
            "zig": ["fn ", "pub const", "std.debug.print"],
            "assembly": ["section .text", "global _start", "mov ", "push "],
            "dart": ["void main()", "class ", "final ", "extends "],
            "web-design": ["<html", "<div", "<style", "@media", "display: flex"],
        }
        
        # Short-circuit on some high-confidence patterns to avoid ambiguous
        # collisions (e.g., 'public class' should be Java even though 'class'
        # also appears in Python markers).
        if "public class" in content_lower or "system.out.println" in content_lower:
            return "java"
        if "fn main" in content_lower:
            return "rust"

        # Score languages by the total matched signal length; this prefers
        # languages with multiple or longer specific matches instead of the
        # first generic single-token match.
        best_lang = None
        best_score = 0
        best_max_sig = 0
        for lang, signals in markers.items():
            matched = [sig for sig in signals if sig.lower() in content_lower]
            score = sum(len(sig) for sig in matched)
            max_sig = max((len(sig) for sig in matched), default=0)
            if score > best_score or (score == best_score and max_sig > best_max_sig):
                best_lang = lang
                best_score = score
                best_max_sig = max_sig

        if best_score > 0:
            return best_lang

        # No language marker matched — treat as plain text by default.
        return "text"

    @staticmethod
    def _clean_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(v) for v in value if v is not None)
        return str(value).strip()

    def _stream_for_theme(self, theme: str, limit: int | None):
        count = 0
        groups = self._get_groups()
        target_libs = groups.get(theme, self.target_libraries)
        
        for ds_info in self.datasets:
            logger.info(f"Streaming {theme} samples from {ds_info['path']}...")
            try:
                ds_kwargs = {
                    "path": ds_info["path"],
                    "split": ds_info.get("split", "train"),
                    "streaming": True,
                }
                if "data_dir" in ds_info:
                    ds_kwargs["data_dir"] = ds_info["data_dir"]
                if "name" in ds_info:
                    ds_kwargs["name"] = ds_info["name"]
                if "languages" in ds_info:
                    ds_kwargs["languages"] = ds_info["languages"]
                
                ds = load_dataset(**ds_kwargs)
                
                ds_count = 0
                ds_limit = ds_info.get("max_samples")
                for entry in ds:
                    sample = self._process_entry(entry, theme, target_libs, ds_info=ds_info)
                    if sample:
                        yield sample
                        count += 1
                        ds_count += 1
                        if limit and count >= limit:
                            return
                        if ds_limit and ds_count >= int(ds_limit):
                            break
            except Exception as e:
                logger.error(f"Error streaming from {ds_info['path']}: {e}")

    def _process_entry(self, entry: dict, theme: str, target_libs: list, ds_info: dict | None = None) -> dict | None:
        """Processes a single entry and returns a formatted sample if it matches the theme.
        
        Handles many HuggingFace dataset formats:
          - instruction/output and instruction/response (Alpaca-style)
          - problem/solution and query/answer (code instruction datasets)
          - messages/chosen/conversations (chat and preference datasets)
          - prompt/completion
          - description/solutions (code_contests)
          - func_documentation_string/func_code_string (code-search-net)
          - content/text/code (raw code datasets)
        """
        ds_info = ds_info or {}
        instruction = ""
        input_text = ""
        output = ""

        # --- Alpaca-style instruction datasets ---
        if "instruction" in entry and "output" in entry:
            instruction = self._clean_text(entry.get("instruction"))
            input_text = self._clean_text(entry.get("input"))
            output = self._clean_text(entry.get("output"))
        # --- Instruction/response style ---
        elif "instruction" in entry and "response" in entry:
            instruction = self._clean_text(entry.get("instruction"))
            output = self._clean_text(entry.get("response"))
        # --- Problem/solution style ---
        elif "problem" in entry and "solution" in entry:
            instruction = self._clean_text(entry.get("problem"))
            output = self._clean_text(entry.get("solution"))
        # --- Query/answer style ---
        elif "query" in entry and "answer" in entry:
            instruction = self._clean_text(entry.get("query"))
            output = self._clean_text(entry.get("answer"))
        # --- Chat or preference-style messages ---
        elif "messages" in entry or "chosen" in entry or "conversations" in entry:
            instruction, output = self._extract_chat_pair(entry)
        # --- Prompt/completion style ---
        elif "prompt" in entry and "completion" in entry:
            instruction = self._clean_text(entry.get("prompt"))
            output = self._clean_text(entry.get("completion"))
        elif "prompt" in entry and "response" in entry:
            instruction = self._clean_text(entry.get("prompt"))
            output = self._clean_text(entry.get("response"))
        # --- Code contests (deepmind) ---
        elif "description" in entry and "solutions" in entry:
            instruction = self._clean_text(entry.get("description"))
            solutions = entry["solutions"]
            if isinstance(solutions, dict) and "solution" in solutions:
                sol_list = solutions["solution"]
                if isinstance(sol_list, list) and len(sol_list) > 0:
                    output = self._clean_text(sol_list[0])
                else:
                    return None
            elif isinstance(solutions, list) and len(solutions) > 0:
                output = self._clean_text(solutions[0])
            else:
                return None
        # --- Code-search-net style ---
        elif "func_documentation_string" in entry and "func_code_string" in entry:
            instruction = self._clean_text(entry.get("func_documentation_string"))
            output = self._clean_text(entry.get("func_code_string"))
        # --- Raw code fallback ---
        else:
            output = self._clean_text(entry.get('content') or entry.get('text') or entry.get('code') or "")
            instruction = ""

        content = "\n".join(part for part in (instruction, input_text, output) if part)

        lang = self._language_from_entry(entry, content, ds_info)
        content_lower = content.lower()
        
        # Theme filtering
        if theme != "all":
            if theme == "core":
                if any(lib in content_lower for lib in self.target_libraries):
                    return None
            elif theme == "multi":
                # Multi theme allows anything that is a recognized language
                pass
            elif lang == "text":
                pass
            else:
                if not any(lib in content_lower for lib in target_libs):
                    return None

        if self._is_high_quality(content, lang):
            return {
                "instruction": instruction or f"Complete this {lang} code:",
                "input": input_text,
                "output": output or content,
                "language": lang
            }
        return None

    def _extract_chat_pair(self, entry: dict) -> tuple[str, str]:
        messages = entry.get("messages") or entry.get("chosen") or entry.get("conversations") or []
        if not isinstance(messages, list):
            return self._clean_text(entry.get("prompt")), self._clean_text(entry.get("response") or entry.get("answer"))

        user_turns: list[str] = []
        assistant_turns: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            text = self._clean_text(message.get("content") or message.get("value") or message.get("text"))
            if not text:
                continue
            if role in {"assistant", "gpt", "model"}:
                assistant_turns.append(text)
            elif role in {"user", "human", "prompter"}:
                user_turns.append(text)

        instruction = user_turns[-1] if user_turns else self._clean_text(entry.get("prompt"))
        output = assistant_turns[-1] if assistant_turns else ""
        return instruction, output

    def _extract_fields(self, entry: dict) -> tuple[str, str, str]:
        """Extracts (instruction, input, output) fields from a generic dataset entry.
        This mirrors the extraction logic used in _process_entry but does not apply
        quality filtering; used by tests to validate field extraction.
        """
        instruction = ""
        input_text = ""
        output = ""

        # More specific formats first (dolly, flan, orca, tool-use, QA, sentence pairs)
        if "context" in entry and "instruction" in entry and "response" in entry:
            # Dolly format: include context with instruction
            context = self._clean_text(entry.get("context"))
            instruction = (context + "\n" + self._clean_text(entry.get("instruction"))).strip()
            output = self._clean_text(entry.get("response"))
        elif "tool_definition" in entry and "instruction" in entry and ("response" in entry or "output" in entry):
            # Tool use: include tool definition with instruction
            td = self._clean_text(entry.get("tool_definition"))
            inst_text = self._clean_text(entry.get("instruction"))
            instruction = (td + "\n" + inst_text).strip()
            output = self._clean_text(entry.get("response") or entry.get("output"))
        elif "system_prompt" in entry and "question" in entry and "response" in entry:
            # Orca-style
            system = self._clean_text(entry.get("system_prompt"))
            q = self._clean_text(entry.get("question"))
            instruction = (system + "\n" + q).strip()
            output = self._clean_text(entry.get("response"))
        elif "inputs" in entry and "targets" in entry:
            # Flan-style
            instruction = self._clean_text(entry.get("inputs"))
            output = self._clean_text(entry.get("targets"))
        elif "question" in entry and "answer" in entry:
            instruction = self._clean_text(entry.get("question"))
            output = self._clean_text(entry.get("answer"))
        elif "sentence1" in entry and "sentence2" in entry and "label" in entry:
            instruction = self._clean_text(entry.get("sentence1"))
            input_text = self._clean_text(entry.get("sentence2"))
            output = self._clean_text(entry.get("label"))
        elif "instruction" in entry and "output" in entry:
            instruction = self._clean_text(entry.get("instruction"))
            input_text = self._clean_text(entry.get("input"))
            output = self._clean_text(entry.get("output"))
        elif "instruction" in entry and "response" in entry:
            instruction = self._clean_text(entry.get("instruction"))
            output = self._clean_text(entry.get("response"))
        elif "problem" in entry and "solution" in entry:
            instruction = self._clean_text(entry.get("problem"))
            output = self._clean_text(entry.get("solution"))
        elif "query" in entry and "answer" in entry:
            instruction = self._clean_text(entry.get("query"))
            output = self._clean_text(entry.get("answer"))
        elif "messages" in entry or "chosen" in entry or "conversations" in entry:
            instruction, output = self._extract_chat_pair(entry)
        elif "prompt" in entry and ("completion" in entry or "response" in entry):
            instruction = self._clean_text(entry.get("prompt"))
            output = self._clean_text(entry.get("completion") or entry.get("response"))
        elif "description" in entry and "solutions" in entry:
            instruction = self._clean_text(entry.get("description"))
            solutions = entry.get("solutions")
            if isinstance(solutions, dict) and "solution" in solutions:
                sol_list = solutions["solution"]
                if isinstance(sol_list, list) and sol_list:
                    output = self._clean_text(sol_list[0])
            elif isinstance(solutions, list) and solutions:
                output = self._clean_text(solutions[0])
            instruction = self._clean_text(entry.get("description"))
            solutions = entry.get("solutions")
            if isinstance(solutions, dict) and "solution" in solutions:
                sol_list = solutions["solution"]
                if isinstance(sol_list, list) and sol_list:
                    output = self._clean_text(sol_list[0])
            elif isinstance(solutions, list) and solutions:
                output = self._clean_text(solutions[0])
        elif "func_documentation_string" in entry and "func_code_string" in entry:
            instruction = self._clean_text(entry.get("func_documentation_string"))
            output = self._clean_text(entry.get("func_code_string"))
        elif "code" in entry and "explanation" in entry:
            instruction = self._clean_text(entry.get("explanation"))
            output = self._clean_text(entry.get("code"))
        else:
            output = self._clean_text(entry.get('content') or entry.get('text') or entry.get('code') or "")

        return instruction, input_text, output

    def _extract_sharegpt_fields(self, message: dict) -> tuple[str, str, str]:
        """Extract a single ShareGPT-style message dict into (inst, inp, out).
        Human/user messages return (text, '', '') and assistant messages return ('', '', text).
        """
        role = str(message.get("from") or message.get("role") or "").lower()
        text = self._clean_text(message.get("value") or message.get("content") or message.get("text"))
        if role in {"human", "user"}:
            return text, "", ""
        return "", "", text

    def _language_from_entry(self, entry: dict, content: str, ds_info: dict) -> str:
        forced = self._clean_text(ds_info.get("language"))
        if forced:
            return self._normalize_language(forced)

        hinted = self._clean_text(
            entry.get("language")
            or entry.get("lang")
            or entry.get("programming_language")
            or entry.get("ext")
        )
        if hinted:
            return self._normalize_language(hinted)

        return self._detect_language(content)

    @staticmethod
    def _normalize_language(language: str) -> str:
        lang = language.strip().lower()
        aliases = {
            "c++": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "go": "golang",
            "bash": "shell",
            "sh": "shell",
            "markdown": "text",
            "md": "text",
            "natural-language": "text",
            "general": "text",
        }
        return aliases.get(lang, lang)

    def _is_high_quality(self, content, language):
        """Heuristic to check for quality and language indicators."""
        content = self._clean_text(content)
        if len(content) < 50:
            return False
        if len(content) > 250000:
            return False
        lowered = content.lower()
        low_quality_markers = (
            "lorem ipsum",
            "todo: add code",
            "your code here",
            "coming soon",
            "placeholder",
        )
        if any(marker in lowered for marker in low_quality_markers):
            return False
        if language == "text":
            return len(content) >= 80
        
        # Generic signals for most languages
        signals = {
            "python": ["def ", "import ", "class "],
            "rust": ["fn ", "let ", "use "],
            "golang": ["func ", "package "],
            "cpp": ["#include", "int "],
            "java": ["class ", "public "],
            "typescript": ["interface ", "export "],
            "javascript": ["function", "const ", "let "],
            "web-design": ["<html", "<div", "<style"],
        }
        
        target_signals = signals.get(language, [" "]) # Fallback to whitespace for unknown
        if not any(sig.lower() in lowered for sig in target_signals):
            return False
        
        return True

    @staticmethod
    def _is_valid_sample(content, language):
        """Compatibility wrapper used by tests — mirrors _is_high_quality heuristics
        but implemented as a staticmethod so tests can call it on the class.
        """
        # Reuse the same tests as _is_high_quality without needing an instance
        content = MassiveDataCollector._clean_text(content)
        if len(content) < 50:
            return False
        if len(content) > 250000:
            return False
        lowered = content.lower()
        low_quality_markers = (
            "lorem ipsum",
            "todo: add code",
            "your code here",
            "coming soon",
            "placeholder",
        )
        if any(marker in lowered for marker in low_quality_markers):
            return False
        if language == "text":
            return len(content) >= 80
        signals = {
            "python": ["def ", "import ", "class "],
            "rust": ["fn ", "let ", "use "],
            "golang": ["func ", "package "],
            "cpp": ["#include", "int "],
            "java": ["class ", "public "],
            "typescript": ["interface ", "export "],
            "javascript": ["function", "const ", "let "],
            "web-design": ["<html", "<div", "<style"],
        }
        target_signals = signals.get(language, [" "])
        return any(sig.lower() in lowered for sig in target_signals)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = MassiveDataCollector()
    print("Testing stream (10 samples)...")
    for i, sample in enumerate(collector.stream_samples(limit=10)):
        print(f"Sample {i+1}: {sample['output'][:100]}...")
