from __future__ import annotations

import ast
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


IDENTIFIER_PATTERN = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]{1,}\b')
ONE_LINE_PATTERN = re.compile(r'.{500,}')


def _parse_python_ast(code: str) -> Optional[ast.Module]:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _python_extract_functions(code: str) -> List[str]:
    tree = _parse_python_ast(code)
    if tree is None:
        lines = code.split('\n')
        funcs: List[str] = []
        buf: List[str] = []
        in_func = False
        indent = 0
        for line in lines:
            if re.match(r'^\s*def\s+|^\s*async\s+def\s+|^\s*class\s+', line):
                if buf:
                    funcs.append('\n'.join(buf))
                buf = [line]
                in_func = True
                indent = len(line) - len(line.lstrip())
            elif in_func:
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    funcs.append('\n'.join(buf))
                    buf = []
                    in_func = False
                else:
                    buf.append(line)
        if buf:
            funcs.append('\n'.join(buf))
        return funcs
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', start + 1) or (start + 1)
            lines = code.split('\n')[start:end]
            funcs.append('\n'.join(lines))
    return funcs


def _generic_extract_functions(code: str, language: str) -> List[str]:
    patterns = {
        'python': [r'^\s*(?:async\s+)?def\s+\w+\s*\(', r'^\s*class\s+\w+'],
        'javascript': [r'^\s*(?:async\s+)?function\s+\w+\s*\(', r'^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function', r'^\s*class\s+\w+'],
        'typescript': [r'^\s*(?:async\s+)?function\s+\w+\s*\(', r'^\s*(?:const|let|var)\s+\w+\s*:\s*(?:string|number|boolean|any|void|never|unknown)', r'^\s*class\s+\w+'],
        'cpp': [r'^\s*(?:virtual\s+)?(?:void|int|bool|float|double|char|string|auto|size_t|uint\d+_t|int\d+_t)\s+\w+\s*\(', r'^\s*class\s+\w+'],
        'java': [r'^\s*(?:public|private|protected|static|final|\s)*(?:void|int|boolean|float|double|String|char|long|short|byte)\s+\w+\s*\(', r'^\s*(?:public|private|protected)?\s*class\s+\w+'],
        'rust': [r'^\s*(?:pub\s+)?(?:fn|unsafe\s+fn)\s+\w+', r'^\s*(?:pub\s+)?(?:struct|enum|trait|impl|mod)\s+\w+'],
        'go': [r'^\s*func\s+\w+', r'^\s*type\s+\w+\s+struct'],
        'csharp': [r'^\s*(?:public|private|protected|internal|static|virtual|override|\s)*(?:void|int|bool|float|double|string|char|long|short|byte|var)\s+\w+\s*\(', r'^\s*(?:public|private|protected|internal)?\s*class\s+\w+'],
        'sql': [r'^\s*CREATE\s+(?:TABLE|VIEW|INDEX|PROCEDURE|FUNCTION|TRIGGER)', r'^\s*SELECT\s+'],
        'shell': [r'^\s*function\s+\w+\s*\{', r'^\w+\(\)\s*\{'],
    }
    funcs: List[str] = []
    sigs = patterns.get(language, [])
    lines = code.split('\n')
    buf: List[str] = []
    in_block = False
    brace_depth = 0
    for line in lines:
        is_start = any(re.match(p, line) for p in sigs)
        if is_start and not in_block:
            if buf:
                funcs.append('\n'.join(buf))
            buf = [line]
            in_block = True
            brace_depth = line.count('{') - line.count('}')
        elif in_block:
            buf.append(line)
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and '}' in ''.join(buf[-3:]):
                funcs.append('\n'.join(buf))
                buf = []
                in_block = False
                brace_depth = 0
    if buf:
        funcs.append('\n'.join(buf))
    return funcs


def extract_functions(code: str, language: str) -> List[str]:
    if language == 'python':
        return _python_extract_functions(code)
    return _generic_extract_functions(code, language)


def ast_parseable(code: str, language: str) -> bool:
    if language == 'python':
        return _parse_python_ast(code) is not None
    return True


def count_identifiers(code: str) -> int:
    return len(IDENTIFIER_PATTERN.findall(code))


def has_one_enormous_line(code: str) -> bool:
    return bool(ONE_LINE_PATTERN.search(code))


def identifier_ratio(code: str) -> float:
    if not code:
        return 0.0
    identifiers = IDENTIFIER_PATTERN.findall(code)
    id_chars = sum(len(i) for i in identifiers)
    return id_chars / max(len(code), 1)


def executable_ratio(code: str, language: str) -> float:
    if not code:
        return 0.0
    comment_markers = {
        'python': ['#', '"""', "'''"],
        'javascript': ['//', '/*'],
        'typescript': ['//', '/*'],
        'cpp': ['//', '/*'],
        'java': ['//', '/*'],
        'rust': ['//', '/*'],
        'go': ['//', '/*'],
        'csharp': ['//', '/*'],
        'sql': ['--', '/*'],
        'shell': ['#'],
    }
    markers = comment_markers.get(language, ['#', '//'])
    lines = code.split('\n')
    executable = 0
    total = max(len(lines), 1)
    in_block_comment = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(m) for m in ['"""', "'''"]):
            in_block_comment = not in_block_comment
            continue
        if in_block_comment:
            continue
        if any(stripped.startswith(m) for m in markers if m not in ('"""', "'''")):
            continue
        executable += 1
    return executable / total


def is_autogenerated(text: str, autogen_patterns: Optional[List[str]] = None) -> bool:
    if autogen_patterns is None:
        autogen_patterns = [
            'auto-generated', 'autogenerated', 'do not edit',
            'this file is generated', 'this code was generated',
            'generated by the protocol buffer', 'generated by the swagger',
            'this file was automatically generated', 'source code is machine-generated',
        ]
    lower = text.lower()[:500]
    return any(p in lower for p in autogen_patterns)


def filter_code(
    code: str,
    language: str,
    config: Optional[object] = None,
) -> Tuple[bool, str]:
    if config is None:
        from src.config.schema import ASTFilterConfig
        config = ASTFilterConfig()

    if not code or len(code) < 30:
        return False, "too_short"

    if config.reject_autogen and is_autogenerated(code):
        return False, "autogenerated"

    if config.reject_one_liner and has_one_enormous_line(code):
        return False, "one_enormous_line"

    if config.reject_few_identifiers and identifier_ratio(code) < config.min_identifier_ratio:
        return False, "few_identifiers"

    if config.reject_no_parse:
        if config.code_filtering and not ast_parseable(code, language):
            return False, "no_parse"

    if config.min_executable_ratio > 0 and executable_ratio(code, language) < config.min_executable_ratio:
        return False, "low_executable_ratio"

    return True, "ok"


def _pool_filter_code(
    code: str,
    language: str,
    config: Optional[object] = None,
) -> Tuple[bool, str]:
    return filter_code(code, language, config)
