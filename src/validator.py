#!/usr/bin/env python3
"""
Code validation and execution for the Specialized Coding Model.

Supports syntax checking and sandboxed execution of generated Python and
JavaScript code.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a code validation or execution attempt."""

    success: bool
    language: str
    output: str = ""
    error: str = ""
    return_code: int = 0


class CodeValidator:
    """Validate and execute generated code for Python and JavaScript.

    The validator runs code in a subprocess with configurable timeouts to
    guard against infinite loops and excessive resource use.
    """

    SUPPORTED_LANGUAGES = (
        "python", "rust", "golang", "cpp", "java", "typescript", 
        "javascript", "csharp", "php", "ruby", "swift", "kotlin", 
        "sql", "shell", "r", "scala", "objective-c", "perl", 
        "lua", "haskell", "julia", "zig", "assembly", "dart", "web-design"
    )

    # ------------------------------------------------------------------
    # Syntax checking
    # ------------------------------------------------------------------

    def check_syntax(self, code: str, language: str) -> ValidationResult:
        """Check whether *code* is syntactically valid.

        Args:
            code: Source code string.
            language: ``python`` or ``javascript``.

        Returns:
            A ``ValidationResult`` with ``success=True`` when the code parses.
        """
        language = language.strip().lower()

        if language == "python":
            return self._check_python_syntax(code)
        elif language == "javascript":
            return self._check_javascript_syntax(code)
        elif language in self.SUPPORTED_LANGUAGES:
            return self._check_heuristic_syntax(code, language)
        else:
            return ValidationResult(
                success=False,
                language=language,
                error=f"Unsupported language: {language}",
            )

    def _check_heuristic_syntax(self, code: str, language: str) -> ValidationResult:
        """Generic structure check for languages without a local compiler/parser."""
        if not code.strip():
            return ValidationResult(success=False, language=language, error="Empty code")
        
        # Check for basic balanced brackets
        stack = []
        brackets = {'(': ')', '[': ']', '{': '}'}
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return ValidationResult(success=False, language=language, error="Mismatched brackets")
        
        if stack:
            return ValidationResult(success=False, language=language, error="Unclosed brackets")
            
        return ValidationResult(success=True, language=language)

    def _check_python_syntax(self, code: str) -> ValidationResult:
        try:
            ast.parse(code)
            return ValidationResult(success=True, language="python")
        except SyntaxError as exc:
            return ValidationResult(
                success=False,
                language="python",
                error=f"SyntaxError at line {exc.lineno}: {exc.msg}",
            )

    def _check_javascript_syntax(self, code: str) -> ValidationResult:
        """Use Node.js ``--check`` flag for syntax validation."""
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True, text=True, timeout=10,
            )

            if result.returncode == 0:
                return ValidationResult(success=True, language="javascript")
            return ValidationResult(
                success=False,
                language="javascript",
                error=result.stderr.strip(),
            )
        except FileNotFoundError:
            return ValidationResult(
                success=False,
                language="javascript",
                error="Node.js is not installed or not on PATH.",
            )
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        language: str,
        timeout: int = 5,
    ) -> ValidationResult:
        """Execute *code* in a sandboxed subprocess.

        Args:
            code: Source code string.
            language: ``python`` or ``javascript``.
            timeout: Maximum execution time in seconds.

        Returns:
            ``ValidationResult`` containing stdout, stderr, and return code.
        """
        language = language.strip().lower()

        if language == "python":
            return self._execute_python(code, timeout)
        elif language == "javascript":
            return self._execute_javascript(code, timeout)
        elif language in self.SUPPORTED_LANGUAGES:
            return ValidationResult(
                success=False,
                language=language,
                error=f"Execution not yet implemented for {language} in this environment.",
            )
        else:
            return ValidationResult(
                success=False,
                language=language,
                error=f"Unsupported language: {language}",
            )

    def _execute_python(self, code: str, timeout: int) -> ValidationResult:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            return ValidationResult(
                success=result.returncode == 0,
                language="python",
                output=result.stdout.strip(),
                error=result.stderr.strip(),
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                success=False,
                language="python",
                error=f"Execution timed out after {timeout}s.",
            )
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

    def _execute_javascript(self, code: str, timeout: int) -> ValidationResult:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                ["node", tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )

            return ValidationResult(
                success=result.returncode == 0,
                language="javascript",
                output=result.stdout.strip(),
                error=result.stderr.strip(),
                return_code=result.returncode,
            )
        except FileNotFoundError:
            return ValidationResult(
                success=False,
                language="javascript",
                error="Node.js is not installed or not on PATH.",
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                success=False,
                language="javascript",
                error=f"Execution timed out after {timeout}s.",
            )
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def validate_output(
        self,
        code: str,
        expected: str,
        language: str,
        timeout: int = 5,
    ) -> ValidationResult:
        """Execute *code* and check that stdout matches *expected*.

        Args:
            code: Source code that prints its result.
            expected: Expected stdout content (stripped).
            language: ``python`` or ``javascript``.
            timeout: Execution timeout in seconds.

        Returns:
            ``ValidationResult`` — success is True only when output matches.
        """
        result = self.execute(code, language, timeout=timeout)
        if not result.success:
            return result

        actual = result.output.strip()
        expected = expected.strip()
        if actual == expected:
            return ValidationResult(
                success=True,
                language=language,
                output=actual,
            )
        return ValidationResult(
            success=False,
            language=language,
            output=actual,
            error=f"Output mismatch.\nExpected:\n{expected}\nGot:\n{actual}",
        )

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    def validate_batch(
        self,
        codes: list[str],
        language: str,
        timeout: int = 5,
    ) -> list[ValidationResult]:
        """Syntax-check and execute a batch of code snippets.

        Returns one ``ValidationResult`` per snippet.
        """
        results: list[ValidationResult] = []
        for i, code in enumerate(codes):
            logger.info("Validating snippet %d/%d (%s)", i + 1, len(codes), language)
            syntax = self.check_syntax(code, language)
            if not syntax.success:
                results.append(syntax)
                continue
            results.append(self.execute(code, language, timeout=timeout))
        return results
