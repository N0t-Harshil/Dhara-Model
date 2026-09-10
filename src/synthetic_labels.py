"""
Synthetic Label Generator for Dhara Multi-Task Training.

Provides heuristic supervision targets for auxiliary heads:
- Intent Understanding (task type, difficulty, reasoning type, expected length)
- Tool Selection (router targets, argument extraction)
- Quality Assurance / Self-Evaluation (correctness, novelty, usefulness, uncertainty)
"""

import re
from typing import Any, Dict, List, Optional


def generate_intent_labels(text: str) -> Dict[str, Any]:
    """
    Generate synthetic intent classification labels from raw text/prompt.

    Returns
    -------
    dict
        Contains:
        - task_type: int (0: code, 1: math, 2: logic, 3: chat, 4: translation, 5: summarization, 6: qa, 7: general)
        - difficulty: int (0 to 4)
        - reasoning_type: int (0: direct, 1: chain_of_thought, 2: tool_assisted, 3: debate, 4: search, 5: code_exec, 6: continuous_ode, 7: recursive_qa)
        - expected_length: float (estimated response length in tokens)
    """
    text_lower = text.lower()
    
    # Task type heuristics (order matters: code before math etc.)
    if any(k in text_lower for k in ["def ", "class ", "import ", "function", "code", "python", "bug", "implement", "algorithm"]):
        task_type = 0 # Code
    elif any(k in text_lower for k in ["calculate", "solve", "equation", "math", "integral", "derivative", "theorem"]):
        task_type = 1 # Math
    elif any(k in text_lower for k in ["prove", "logic", "deduce", "reason", "if and only if", "implies"]):
        task_type = 2 # Logic
    elif any(k in text_lower for k in ["translate", "translation", "translating"]):
        task_type = 4
    elif any(k in text_lower for k in ["summarize", "summary", "summarise"]):
        task_type = 5
    elif "?" in text and len(text.split()) < 40:
        task_type = 6 # QA
    else:
        task_type = 7 # General

    # Difficulty heuristics based on length and keywords
    words = text.split()
    n = len(words)
    has_complex = any(k in text_lower for k in ["complex", "difficult", "advanced", "challenging", "hard"])
    if n > 100 or (n > 50 and has_complex):
        difficulty = 4
    elif n > 50 or has_complex:
        difficulty = 3
    elif n > 20:
        difficulty = 2
    elif n > 5:
        difficulty = 1
    else:
        difficulty = 0

    reasoning_type = 1 if difficulty >= 2 else 0
    # Rough token estimate: ~1.3 tokens per word, clamped
    expected_length = float(max(64, min(2048, int(n * 1.3 * 4))))

    return {
        "task_type": task_type,
        "difficulty": difficulty,
        "reasoning_type": reasoning_type,
        "expected_length": expected_length,
    }


def generate_tool_selection_labels(text: str, available_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate synthetic tool usage labels (which tools to call and confidence).

    Returns
    -------
    dict
        Contains:
        - tool_mask: list of float [calc, python, search, db]
        - primary_tool: int (0: calc, 1: python, 2: search, 3: db, -1: none)
    """
    text_lower = text.lower()
    tool_mask = [0.0, 0.0, 0.0, 0.0]
    primary_tool = -1

    # Calculator: require a numeric expression, not just any hyphen
    calc_pattern = re.compile(r"\d+\s*[\+\-\*\/\*\*\%]\s*\d+|(?:sin|cos|tan|log|sqrt|integral|derivative)\s*\(")
    if calc_pattern.search(text_lower) or any(k in text_lower for k in ["calculate numerically", "compute the value"]):
        tool_mask[0] = 1.0 # Calculator
        primary_tool = 0
    if "python" in text_lower or "exec" in text_lower or "run code" in text_lower or "execute" in text_lower:
        tool_mask[1] = 1.0 # Python Executor
        primary_tool = 1
    if any(k in text_lower for k in ["search", "find information", "who is", "what is", "lookup", "browse", "retrieve", "search for"]):
        tool_mask[2] = 1.0 # Search
        if primary_tool == -1:
            primary_tool = 2
    if any(k in text_lower for k in ["database", " sql ", "query", "select ", "table", "db "]):
        tool_mask[3] = 1.0 # DB
        if primary_tool == -1:
            primary_tool = 3

    # Respect available_tools filter if provided
    if available_tools is not None:
        name_to_idx = {"calc": 0, "calculator": 0, "python": 1, "search": 2, "db": 3, "database": 3}
        allowed = {name_to_idx.get(t.lower(), -1) for t in available_tools}
        for i in range(4):
            if i not in allowed:
                tool_mask[i] = 0.0
        if primary_tool not in allowed:
            # pick first allowed that is active, else none
            primary_tool = next((i for i, v in enumerate(tool_mask) if v > 0 and i in allowed), -1)

    return {
        "tool_mask": tool_mask,
        "primary_tool": primary_tool,
    }


def generate_qa_correctness_labels(
    question: str, answer: str, reference: Optional[str] = None
) -> Dict[str, float]:
    """
    Generate synthetic Quality Assurance / Self-Evaluation targets.

    Returns
    -------
    dict
        Contains:
        - usefulness: float [0, 1]
        - novelty: float [0, 1]
        - uncertainty: float [0, 1]
        - correctness: float [0, 1]
    """
    if not answer.strip():
        return {"usefulness": 0.0, "novelty": 0.0, "uncertainty": 1.0, "correctness": 0.0}

    ans_len = len(answer.split())
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    # Usefulness: length + lexical overlap with question (answers question)
    overlap = len(q_words & a_words) / max(len(q_words), 1)
    usefulness = min(1.0, 0.5 * min(1.0, ans_len / 30.0) + 0.5 * overlap)
    # Novelty: lexical diversity + length (heuristic, not learned)
    unique_ratio = len(a_words) / max(ans_len, 1)
    novelty = min(1.0, 0.6 * unique_ratio + 0.4 * min(1.0, ans_len / 50.0))
    uncertainty = 0.2 if ans_len > 10 else 0.8
    # Correctness: reference substring or high overlap fallback
    if reference and reference.strip().lower() in answer.lower():
        correctness = 1.0
    elif reference:
        ref_words = set(reference.lower().split())
        ref_overlap = len(ref_words & a_words) / max(len(ref_words), 1)
        correctness = 0.6 + 0.4 * ref_overlap
    else:
        correctness = 0.7

    return {
        "usefulness": float(usefulness),
        "novelty": float(novelty),
        "uncertainty": float(uncertainty),
        "correctness": float(correctness),
    }
