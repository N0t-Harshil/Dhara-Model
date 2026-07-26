import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentUnderstanding(nn.Module):
    def __init__(self, d_model: int, n_task_types: int = 8, n_difficulty_levels: int = 5, n_reasoning_types: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.task_classifier = nn.Linear(d_model, n_task_types)
        self.difficulty_classifier = nn.Linear(d_model, n_difficulty_levels)
        self.length_predictor = nn.Linear(d_model, 1)
        self.reasoning_classifier = nn.Linear(d_model, n_reasoning_types)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict:
        pooled = x.mean(dim=1) if x.shape[1] > 0 else torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        h = self.norm(pooled)
        intent = {
            "task_type": self.task_classifier(h),
            "difficulty": self.difficulty_classifier(h),
            "expected_length": self.length_predictor(h).squeeze(-1),
            "reasoning_type": self.reasoning_classifier(h),
            "confidence": torch.sigmoid(self.confidence_head(h)).squeeze(-1),
        }
        return intent


class AdaptiveDifficultyRouter(nn.Module):
    def __init__(self, n_difficulty_levels: int = 5, min_steps: int = 2, max_steps: int = 32):
        super().__init__()
        self.n_levels = n_difficulty_levels
        self.min_steps = min_steps
        self.max_steps = max_steps

    def forward(self, difficulty_logits: torch.Tensor) -> torch.LongTensor:
        difficulty = torch.argmax(difficulty_logits, dim=-1)
        frac = difficulty.float() / max(self.n_levels - 1, 1)
        n_steps = self.min_steps + (frac * (self.max_steps - self.min_steps)).long()
        return n_steps
