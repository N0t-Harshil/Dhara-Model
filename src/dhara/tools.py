from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SYMBOLIC_OPS: Dict[str, Callable] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "//": operator.floordiv, "%": operator.mod,
    "**": operator.pow,
}


def _extract_numbers(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", text)]


def _extract_expression(text: str) -> Optional[str]:
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if re.match(r"^[\d\s+\-*/().%**]+$", line):
            return line
    return None


def _safe_eval(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.Num, ast.UnaryOp, ast.BinOp,
                                     ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                                     ast.Mod, ast.Pow, ast.USub, ast.UAdd)):
                return None
        return float(eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


class SymbolicCalculator:
    def forward(self, x: torch.Tensor, hidden_text: Optional[str] = None) -> torch.Tensor:
        if hidden_text is None:
            return torch.zeros_like(x)
        expr = _extract_expression(hidden_text)
        if expr is not None:
            result = _safe_eval(expr)
            if result is not None:
                return torch.full_like(x, result / 100.0)
        nums = _extract_numbers(hidden_text)
        if len(nums) >= 2:
            return torch.full_like(x, sum(nums) / len(nums) / 100.0)
        return torch.zeros_like(x)


class SymbolicPythonExecutor:
    def forward(self, x: torch.Tensor, code_str: Optional[str] = None) -> torch.Tensor:
        if code_str is None:
            return torch.zeros_like(x)
        local_ns: Dict[str, Any] = {}
        try:
            exec(code_str, {"__builtins__": {}}, local_ns)
            result = local_ns.get("result", local_ns.get("output", None))
            if result is not None:
                if isinstance(result, (int, float)):
                    return torch.full_like(x, float(result) / 100.0)
                if isinstance(result, str):
                    return torch.full_like(x, sum(ord(c) for c in result[:100]) / 10000.0)
        except Exception:
            pass
        return torch.zeros_like(x)


class SymbolicSearch(nn.Module):
    def __init__(self, d_hidden: int, memory_size: int = 4096):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, d_hidden) * 0.02)
        self.query_proj = nn.Linear(d_hidden, d_hidden)
        self.key_proj = nn.Linear(d_hidden, d_hidden)
        self.value_proj = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        k = self.key_proj(self.memory).unsqueeze(0).expand(x.shape[0], -1, -1)
        v = self.value_proj(self.memory).unsqueeze(0).expand(x.shape[0], -1, -1)
        attn = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) * (x.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v).squeeze(1)


class SymbolicDatabase(nn.Module):
    def __init__(self, d_hidden: int, n_records: int = 2048):
        super().__init__()
        self.records = nn.Parameter(torch.randn(n_records, d_hidden) * 0.02)
        self.query_proj = nn.Linear(d_hidden, d_hidden)
        self.access_gate = nn.Linear(d_hidden, n_records)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        access_logits = self.access_gate(x)
        access_weights = F.softmax(access_logits, dim=-1)
        return torch.matmul(access_weights, self.records)


class SymbolicToolRouter(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.calculator = SymbolicCalculator()
        self.python_exec = SymbolicPythonExecutor()
        self.search = SymbolicSearch(d_hidden)
        self.database = SymbolicDatabase(d_hidden)
        self.router = nn.Linear(d_hidden, 4)
        self.fusion = nn.Linear(d_hidden * 4, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.hidden_text: Optional[str] = None
        self.code_str: Optional[str] = None

    def set_context(self, hidden_text: Optional[str] = None, code_str: Optional[str] = None) -> None:
        self.hidden_text = hidden_text
        self.code_str = code_str

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        route_logits = self.router(x)
        route_weights = F.softmax(route_logits, dim=-1)

        calc_out = self.calculator.forward(x, self.hidden_text)
        py_out = self.python_exec.forward(x, self.code_str)
        search_out = self.search.forward(x)
        db_out = self.database.forward(x)

        stacked = torch.stack([calc_out, py_out, search_out, db_out], dim=1)
        weighted = (stacked * route_weights.unsqueeze(-1)).sum(dim=1)
        fused = self.norm(weighted + x)

        return {
            "fused_tool_output": fused,
            "route_weights": route_weights,
            "calc_output": calc_out,
            "python_output": py_out,
            "search_output": search_out,
            "database_output": db_out,
        }


class InternalToolInterface(SymbolicToolRouter):
    pass
