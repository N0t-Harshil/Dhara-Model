import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SemanticMetadataEmbedding(nn.Module):
    def __init__(self, d_model: int, n_token_categories: int = 8, n_languages: int = 16, n_doc_roles: int = 8):
        super().__init__()
        self.category_embed = nn.Embedding(n_token_categories, d_model // 4)
        self.language_embed = nn.Embedding(n_languages, d_model // 4)
        self.doc_role_embed = nn.Embedding(n_doc_roles, d_model // 4)
        self.out_proj = nn.Linear(d_model // 4 * 3, d_model)

    def forward(self, categories: torch.LongTensor, languages: torch.LongTensor, doc_roles: torch.LongTensor) -> torch.Tensor:
        cat = self.category_embed(categories)
        lang = self.language_embed(languages)
        role = self.doc_role_embed(doc_roles)
        meta = torch.cat([cat, lang, role], dim=-1)
        return self.out_proj(meta)


class IntelligentTokenizer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_token_categories: int = 8, n_languages: int = 16, n_doc_roles: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)
        self.semantic_meta = SemanticMetadataEmbedding(d_model, n_token_categories, n_languages, n_doc_roles)

    def forward(self, input_ids: torch.LongTensor, categories: torch.LongTensor = None,
                languages: torch.LongTensor = None, doc_roles: torch.LongTensor = None) -> torch.Tensor:
        tok = self.token_embedding(input_ids) * self.scale
        if categories is not None and languages is not None and doc_roles is not None:
            meta = self.semantic_meta(categories, languages, doc_roles)
            tok = tok + meta
        return tok
