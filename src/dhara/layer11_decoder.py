import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticDecoder(nn.Module):
    def __init__(self, d_hidden: int, n_semantic_concepts: int = 4096):
        super().__init__()
        self.concept_embeds = nn.Parameter(torch.randn(n_semantic_concepts, d_hidden) * 0.02)
        self.concept_proj = nn.Linear(d_hidden, d_hidden)
        self.concept_router = nn.Linear(d_hidden, n_semantic_concepts)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, h: torch.Tensor) -> dict:
        h = self.norm(h)
        route_logits = self.concept_router(h)
        route_weights = F.softmax(route_logits, dim=-1)
        concept_out = torch.matmul(route_weights, self.concept_embeds)
        concept_out = self.concept_proj(concept_out)
        return {
            "concept_logits": route_logits,
            "concept_weights": route_weights,
            "concept_out": concept_out,
        }


class LanguageDecoder(nn.Module):
    def __init__(self, d_hidden: int, n_language_groups: int = 8):
        super().__init__()
        self.language_embeds = nn.Embedding(n_language_groups, d_hidden)
        self.language_classifier = nn.Linear(d_hidden, n_language_groups)
        self.language_gate = nn.Linear(d_hidden, 1)

    def forward(self, h: torch.Tensor, language_hint: torch.Tensor = None) -> dict:
        lang_logits = self.language_classifier(h)
        lang_weights = F.softmax(lang_logits, dim=-1)
        gate = torch.sigmoid(self.language_gate(h))
        if language_hint is not None:
            lang_hint_onehot = F.one_hot(language_hint, num_classes=lang_logits.shape[-1]).float()
            # Learned-gated hint blend: the model can downweight the raw hint
            # instead of being forced to a fixed 0.5/0.5 mix.
            lang_weights = lang_weights * (1 - gate) + lang_hint_onehot * gate
        lang_ctx = torch.matmul(lang_weights, self.language_embeds.weight)
        return {
            "language_logits": lang_logits,
            "language_weights": lang_weights,
            "language_context": lang_ctx,
            "language_gate": gate,
        }


class TokenDecoder(nn.Module):
    def __init__(self, d_hidden: int, vocab_size: int, adaptive_top_k_min: int = 32, adaptive_top_k_max: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.adaptive_top_k_min = adaptive_top_k_min
        self.adaptive_top_k_max = adaptive_top_k_max
        self.hidden_to_vocab = nn.Linear(d_hidden, vocab_size, bias=False)
        self.difficulty_predictor = nn.Linear(d_hidden, 1)
        self.norm = nn.LayerNorm(d_hidden)
        self.sparsity_gate = nn.Linear(d_hidden, 1)

    def forward(self, h: torch.Tensor, language_bias: torch.Tensor = None) -> dict:
        h = self.norm(h)
        difficulty = torch.sigmoid(self.difficulty_predictor(h))
        top_k = self.adaptive_top_k_min + (difficulty * (self.adaptive_top_k_max - self.adaptive_top_k_min)).long()
        top_k = top_k.squeeze(-1).clamp(self.adaptive_top_k_min, self.adaptive_top_k_max)
        sparsity = torch.sigmoid(self.sparsity_gate(h))
        logits = self.hidden_to_vocab(h)
        # No learnable temperature inside the logits: a learned multiplier lets
        # the model "cheat" by shrinking the scale instead of fitting the
        # predictions (sampling temperature stays in generate()).
        if language_bias is not None:
            lang_bias_proj = torch.matmul(language_bias, self.hidden_to_vocab.weight.T)
            logits = logits + 0.1 * lang_bias_proj
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        batch, vocab = logits.shape
        k = min(top_k.max().item(), vocab)
        top_logits, top_indices = torch.topk(logits, k, dim=-1)
        return {
            "logits": logits,
            "top_logits": top_logits,
            "top_indices": top_indices,
            "top_k": top_k,
            "difficulty": difficulty.squeeze(-1),
            "sparsity": sparsity.squeeze(-1),
        }


class HierarchicalSparseDecoder(nn.Module):
    def __init__(self, d_hidden: int, vocab_size: int, d_model: int,
                 n_language_groups: int = 8, adaptive_top_k_min: int = 32,
                 adaptive_top_k_max: int = 2048, n_semantic_concepts: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.semantic = SemanticDecoder(d_hidden, n_semantic_concepts)
        self.language = LanguageDecoder(d_hidden, n_language_groups)
        self.token = TokenDecoder(d_hidden, vocab_size, adaptive_top_k_min, adaptive_top_k_max)
        self.fusion_gate = nn.Linear(d_hidden * 3, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, h: torch.Tensor, target_ids: torch.LongTensor = None,
                language_ids: torch.LongTensor = None) -> dict:
        h = self.norm(h)
        sem_out = self.semantic(h)
        lang_out = self.language(h, language_ids)
        fused = self.fusion_gate(torch.cat([h, sem_out["concept_out"], lang_out["language_context"]], dim=-1))
        tok_out = self.token(fused, lang_out["language_context"])
        return {
            "logits": tok_out["logits"],
            "top_logits": tok_out["top_logits"],
            "top_indices": tok_out["top_indices"],
            "top_k": tok_out["top_k"],
            "language_logits": lang_out["language_logits"],
            "difficulty": tok_out["difficulty"],
            "sparsity": tok_out["sparsity"],
            "concept_weights": sem_out["concept_weights"],
            "language_weights": lang_out["language_weights"],
            "semantic_out": sem_out["concept_out"],
            "language_context": lang_out["language_context"],
        }

    def compute_log_prob(self, h: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        out = self.forward(h)
        logits = out["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        safe_targets = targets.clamp(0, log_probs.shape[-1] - 1)
        return log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)

    def hierarchical_log_prob(self, h: torch.Tensor, targets: torch.LongTensor,
                               language_ids: torch.LongTensor = None) -> torch.Tensor:
        out = self.forward(h, language_ids=language_ids)
        logits = out["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        safe_targets = targets.clamp(0, log_probs.shape[-1] - 1)
        return log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)

    def hidden_to_vocab(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3:
            batch, seq_len, d = h.shape
            flat_h = h.reshape(-1, d)
            out = self.forward(flat_h)
            return out["logits"].view(batch, seq_len, -1)
        out = self.forward(h)
        return out["logits"]
