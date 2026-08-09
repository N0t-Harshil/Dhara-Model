from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Optional

from src.data.ast_filter import extract_functions

logger = logging.getLogger(__name__)


def sample_functions(
    code: str,
    language: str,
    n: int = 5,
    min_body_lines: int = 3,
    max_body_lines: int = 500,
    seed: Optional[int] = None,
) -> List[str]:
    rng = random.Random(seed)
    funcs = extract_functions(code, language)
    filtered = [f for f in funcs if min_body_lines <= len(f.split('\n')) <= max_body_lines]
    if not filtered:
        return [code]
    if len(filtered) <= n:
        return filtered
    return rng.sample(filtered, n)


def sample_functions_or_fallback(
    code: str,
    language: str,
    config: object,
) -> str:
    if not config or not getattr(config, 'enabled', False):
        return code
    strategies = getattr(config, 'strategies', ['function', 'class'])
    min_len = getattr(config, 'min_body_lines', 3)
    max_len = getattr(config, 'max_body_lines', 500)
    limit = getattr(config, 'per_file_limit', 20)
    funcs = extract_functions(code, language)
    if not funcs:
        return code
    filtered = [f for f in funcs if min_len <= len(f.split('\n')) <= max_len]
    if not filtered:
        return code
    chosen = filtered[:limit]
    return '\n\n'.join(chosen)


def is_code_empty_or_trivial(code: str) -> bool:
    lines = [l for l in code.split('\n') if l.strip()]
    if len(lines) <= 2:
        return True
    non_comment = sum(1 for l in lines if not l.strip().startswith(('#', '//', '/*', '*', '--')))
    return non_comment <= 1
