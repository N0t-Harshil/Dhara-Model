"""
Synthetic Label Generator for MethosV3 Multi-Task Training.

Provides stubs for generating synthetic supervision targets for auxiliary heads:
- Intent Understanding (task type, difficulty, reasoning type, expected length)
- Tool Selection (router targets, argument extraction)
- Quality Assurance / Self-Evaluation (correctness, novelty, usefulness, uncertainty)
"""

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
    
    # Task type heuristics
    if any(k in text_lower for k in ["def ", "class ", "import ", "function", "code", "python", "bug"]):
        task_type = 0 # Code
    elif any(k in text_lower for k in ["calculate", "solve", "equation", "math", "integral"]):
        task_type = 1 # Math
    elif any(k in text_lower for k in ["prove", "logic", "deduce", "reason"]):
        task_type = 2 # Logic
    else:
        task_type = 7 # General

    # Difficulty heuristics based on length and keywords
    words = text.split()
    if len(words) > 100 or "complex" in text_lower or "difficult" in text_lower:
        difficulty = 4
    elif len(words) > 50:
        difficulty = 3
    elif len(words) > 20:
        difficulty = 2
    elif len(words) > 5:
        difficulty = 1
    else:
        difficulty = 0

    reasoning_type = 1 if difficulty >= 2 else 0
    expected_length = float(max(64, len(words) * 4))

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

    if any(op in text for op in ["+", "-", "*", "/", "**", "sin", "cos"]):
        tool_mask[0] = 1.0 # Calculator
        primary_tool = 0
    if "python" in text_lower or "exec" in text_lower or "run code" in text_lower:
        tool_mask[1] = 1.0 # Python Executor
        primary_tool = 1
    if any(k in text_lower for k in ["search", "find information", "who is", "what is"]):
        tool_mask[2] = 1.0 # Search
        if primary_tool == -1:
            primary_tool = 2

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
    usefulness = min(1.0, ans_len / 20.0)
    novelty = 0.5
    uncertainty = 0.2 if ans_len > 10 else 0.8
    correctness = 1.0 if reference and reference.strip() in answer else 0.8

    return {
        "usefulness": usefulness,
        "novelty": novelty,
        "uncertainty": uncertainty,
        "correctness": correctness,
    }
