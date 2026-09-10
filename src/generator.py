#!/usr/bin/env python3
"""
Code generation interface for the Specialized Coding Model.

Provides higher-level wrappers around the base model's generation method
with batching, multi-language support, and code extraction utilities.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge_graph import GraphMemory

logger = logging.getLogger(__name__)


class CodeGenerator:
    """High-level code generator backed by a ``SpecializedCoderModel``.

    Args:
        model: A loaded ``SpecializedCoderModel`` instance.
    """

    # Languages the generator officially supports.
    SUPPORTED_LANGUAGES = (
        "python", "rust", "golang", "cpp", "java", "typescript", 
        "javascript", "csharp", "php", "ruby", "swift", "kotlin", 
        "sql", "shell", "r", "scala", "objective-c", "perl", 
        "lua", "haskell", "julia", "zig", "assembly", "dart", "web-design"
    )

    def __init__(self, model: Any) -> None:
        if not model.is_ready:
            raise RuntimeError("Model must be loaded before creating a CodeGenerator.")
        self.model = model
        self.memory: Optional["GraphMemory"] = None

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------

    def generate(
        self,
        problem: str,
        language: str,
        temperature: float = 0.8,
        max_new_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 50,
        agentic: bool = False,
    ) -> str:
        """Generate code for a single problem.

        Args:
            problem: Natural-language problem statement.
            language: ``python`` or ``javascript``.
            temperature: Sampling temperature (higher → more creative).
            max_new_tokens: Maximum number of tokens to produce.
            top_p: Nucleus-sampling probability mass.
            top_k: Top-k sampling.
            agentic: If True, wraps prompt in complex RAG thought process.

        Returns:
            Generated source code as a string.
        """
        language = self._normalise_language(language)

        if agentic:
            if self.memory is None:
                from src.knowledge_graph import GraphMemory
                self.memory = GraphMemory()
            # 1. Retrieve GraphRAG Context
            context = self.memory.retrieve_context(problem)
            
            # 2. Build Agentic Prompt
            final_problem = f"""You are an elite reasoning assistant. Before writing any code, you MUST think step-by-step inside <thought> tags. Analyze the problem, reference the provided Knowledge Graph context to ensure consistency with past architecture, and plan your solution. Then, write the final code.

[KNOWLEDGE GRAPH CONTEXT]
{context}

[PROBLEM]
{problem}"""
            logger.info("Executing Agentic Thinking Loop with Graph Context...")
        else:
            final_problem = problem

        raw = self.model.generate_code(
            problem=final_problem,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        code = self._extract_code(raw, language)
        
        if agentic:
            thought = self._extract_thought(raw)
            # 3. Store interaction in dynamic knowledge graph
            concepts = [language.capitalize(), "Code Generation"]
            self.memory.add_interaction(problem, code, concepts, [])
            return f"--- ACTIVE BRAIN (THOUGHT PROCESS) ---\n{thought}\n\n--- FINAL GENERATED CODE ---\n{code}"
        
        return code

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        problems: List[str],
        language: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """Generate code for multiple problems sequentially.

        Args:
            problems: List of problem descriptions.
            language: Target language for all problems.
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens per response.

        Returns:
            List of generated source code strings.
        """
        language = self._normalise_language(language)
        results: List[str] = []

        for i, problem in enumerate(problems):
            logger.info("Generating %d/%d: %s…", i + 1, len(problems), problem[:60])
            code = self.generate(
                problem=problem,
                language=language,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            results.append(code)

        return results

    # ------------------------------------------------------------------
    # Multi-language generation
    # ------------------------------------------------------------------

    def generate_all_languages(
        self,
        problem: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> Dict[str, str]:
        """Generate a solution in every supported language.

        Args:
            problem: Natural-language problem statement.
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens per response.

        Returns:
            Mapping of language → generated code.
        """
        results: Dict[str, str] = {}
        for lang in self.SUPPORTED_LANGUAGES:
            logger.info("Generating %s solution…", lang)
            results[lang] = self.generate(
                problem=problem,
                language=lang,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_language(language: str) -> str:
        """Normalise and validate the language string."""
        lang = language.strip().lower()
        aliases: Dict[str, str] = {
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "node": "javascript",
            "nodejs": "javascript",
            "ts": "typescript",
            "rs": "rust",
            "go": "golang",
            "rb": "ruby",
            "sh": "shell",
            "bash": "shell",
            "cs": "csharp",
            "cpp": "cpp",
            "c++": "cpp",
        }
        lang = aliases.get(lang, lang)
        if lang not in CodeGenerator.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Choose from: {', '.join(CodeGenerator.SUPPORTED_LANGUAGES)}"
            )
        return lang

    @staticmethod
    def _extract_code(raw: str, language: str) -> str:
        """Attempt to extract a clean code block from the model output.

        If the output contains a fenced code block (```), only the content
        inside the fence is returned.  Otherwise the full text is returned
        with minor cleanup.
        """
        # Try to extract fenced code block (language escaped for c++/objective-c)
        lang_esc = re.escape(language) if language else r"\w*"
        pattern = rf"```(?:{lang_esc})?\s*\n?(.*?)```"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: strip leading/trailing whitespace and stray markdown
        code = raw.strip()
        # Remove possible trailing instruction markers (anchored to newline)
        for marker in ("\n### Instruction", "\n### End"):
            idx = code.find(marker)
            if idx > 0:
                code = code[:idx].strip()

        return code

    @staticmethod
    def _extract_thought(raw: str) -> str:
        """Extract the chain-of-thought planning block."""
        pattern = r"<thought>(.*?)</thought>"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "No explicit thought block parsed (Model might need CoT fine-tuning)."
