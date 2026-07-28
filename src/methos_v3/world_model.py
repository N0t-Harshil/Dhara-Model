import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityExtractor(nn.Module):
    def __init__(self, d_model: int, max_entities: int = 64):
        super().__init__()
        self.max_entities = max_entities
        self.entity_embeds = nn.Parameter(torch.randn(max_entities, d_model) * 0.02)
        self.entity_classifier = nn.Linear(d_model, max_entities)
        self.entity_relation_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> dict:
        pooled = x.mean(dim=1)
        entity_logits = self.entity_classifier(pooled)
        entity_weights = F.softmax(entity_logits, dim=-1)
        entity_out = torch.matmul(entity_weights, self.entity_embeds)
        return {
            "entities": entity_out,
            "entity_logits": entity_logits,
            "entity_weights": entity_weights,
        }


class RelationNetwork(nn.Module):
    def __init__(self, d_model: int, n_relation_types: int = 16):
        super().__init__()
        self.n_relation_types = n_relation_types
        self.relation_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, n_relation_types),
        )
        self.relation_embeds = nn.Embedding(n_relation_types, d_model)

    def forward(self, entities: torch.Tensor) -> dict:
        batch, n_entities, d_model = entities.shape
        e_i = entities.unsqueeze(2).expand(-1, -1, n_entities, -1)
        e_j = entities.unsqueeze(1).expand(-1, n_entities, -1, -1)
        pair_feats = torch.cat([e_i, e_j], dim=-1)
        relation_logits = self.relation_scorer(pair_feats)
        relation_weights = F.softmax(relation_logits, dim=-1)
        relation_out = torch.einsum("bijk,kl->bijl", relation_weights, self.relation_embeds.weight)
        return {
            "relation_logits": relation_logits,
            "relation_weights": relation_weights,
            "relation_embeds": relation_out,
        }


class EventModel(nn.Module):
    def __init__(self, d_model: int, max_events: int = 32):
        super().__init__()
        self.max_events = max_events
        self.event_embeds = nn.Parameter(torch.randn(max_events, d_model) * 0.02)
        self.event_classifier = nn.Linear(d_model, max_events)
        self.temporal_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> dict:
        pooled = x.mean(dim=1)
        event_logits = self.event_classifier(pooled)
        event_weights = F.softmax(event_logits, dim=-1)
        event_out = torch.matmul(event_weights, self.event_embeds)
        temporal = self.temporal_proj(event_out)
        return {
            "events": event_out,
            "temporal_encoding": temporal,
            "event_logits": event_logits,
        }


class CauseEffectModel(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cause_proj = nn.Linear(d_model, d_model)
        self.effect_proj = nn.Linear(d_model, d_model)
        self.causal_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, events: torch.Tensor, relations: torch.Tensor) -> dict:
        cause = self.cause_proj(events)
        effect = self.effect_proj(events)
        causal_out, causal_attn = self.causal_attention(cause, effect, events)
        return {
            "causal_representation": causal_out,
            "causal_attention": causal_attn,
        }


class WorldModel(nn.Module):
    def __init__(self, d_model: int, max_entities: int = 64,
                 n_relation_types: int = 16, max_events: int = 32):
        super().__init__()
        self.entities = EntityExtractor(d_model, max_entities)
        self.relations = RelationNetwork(d_model, n_relation_types)
        self.events = EventModel(d_model, max_events)
        self.causality = CauseEffectModel(d_model)
        self.fusion = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict:
        entity_out = self.entities(x)
        entities = entity_out["entities"].unsqueeze(1)
        relation_out = self.relations(entities)
        event_out = self.events(x)
        causal_out = self.causality(event_out["events"].unsqueeze(1), relation_out["relation_embeds"].mean(dim=2))
        world_state = self.fusion(torch.cat([
            entity_out["entities"],
            relation_out["relation_embeds"].mean(dim=(1, 2)),
            event_out["events"],
            causal_out["causal_representation"].squeeze(1),
        ], dim=-1))
        world_state = self.norm(world_state)
        return {
            "world_state": world_state,
            "entities": entity_out["entities"],
            "entity_weights": entity_out["entity_weights"],
            "relations": relation_out["relation_embeds"],
            "relation_types": relation_out["relation_weights"].argmax(dim=-1),
            "events": event_out["events"],
            "temporal": event_out["temporal_encoding"],
            "causal": causal_out["causal_representation"],
        }
