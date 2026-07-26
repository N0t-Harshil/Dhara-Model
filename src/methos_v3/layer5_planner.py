import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgoalNode(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.dependency_gate = nn.Linear(d_model, 1)
        self.cost_predictor = nn.Linear(d_model, 1)
        self.depth_predictor = nn.Linear(d_model, 1)
        self.type_classifier = nn.Linear(d_model, 8)

    def forward(self, context: torch.Tensor) -> dict:
        h = context + self.embed.unsqueeze(0).unsqueeze(0)
        return {
            "dep_gate": torch.sigmoid(self.dependency_gate(h)).squeeze(-1),
            "cost": F.softplus(self.cost_predictor(h)).squeeze(-1),
            "depth": torch.sigmoid(self.depth_predictor(h)).squeeze(-1),
            "type_logits": self.type_classifier(h),
        }


class GoalGraphAttention(nn.Module):
    def __init__(self, d_model: int, max_subgoals: int = 64):
        super().__init__()
        self.max_subgoals = max_subgoals
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, goal_embeds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.query_proj(goal_embeds)
        k = self.key_proj(goal_embeds)
        v = self.value_proj(goal_embeds)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class HierarchicalPlanner(nn.Module):
    def __init__(self, d_model: int, max_subgoals: int = 64, n_planning_heads: int = 4,
                 max_depth: int = 4):
        super().__init__()
        self.max_subgoals = max_subgoals
        self.max_depth = max_depth
        self.goal_embed = nn.Parameter(torch.randn(max_subgoals, d_model) * 0.02)
        self.goal_gate = nn.Linear(d_model, 1)
        self.goal_bias = nn.Linear(d_model, d_model)
        self.graph_attn = GoalGraphAttention(d_model, max_subgoals)
        self.hierarchical_attn = GoalGraphAttention(d_model, max_subgoals)
        self.dependency_head = nn.Linear(d_model, max_subgoals * 2)
        self.order_head = nn.Linear(d_model, 1)
        self.cost_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.execution_head = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.depth_proj = nn.Linear(d_model, max_depth)
        self.subgoal_nodes = nn.ModuleList([SubgoalNode(d_model) for _ in range(max_subgoals)])

    def forward(self, context: torch.Tensor, task_type: torch.Tensor = None, max_subgoals: int = None) -> dict:
        batch = context.shape[0]
        device = context.device
        max_subgoals = max_subgoals or self.max_subgoals
        context_pooled = context.mean(dim=1)
        goal_embeds = self.goal_embed.unsqueeze(0).expand(batch, -1, -1)
        goal_bias = self.goal_bias(context_pooled).unsqueeze(1)
        goal_embeds = goal_embeds + goal_bias
        goal_embeds = self.norm(goal_embeds + self.proj(context_pooled).unsqueeze(1))
        refined = self.graph_attn(goal_embeds)
        depth_logits = self.depth_proj(context_pooled)
        depth_weights = F.softmax(depth_logits, dim=-1)
        goal_tree = []
        for d in range(self.max_depth):
            depth_scale = depth_weights[:, d:d+1].unsqueeze(-1)
            if d > 0:
                parent_mask = (torch.arange(self.max_subgoals, device=device) % (2 ** d) == 0).float()
                refined = refined * parent_mask.unsqueeze(0).unsqueeze(-1)
            tree_level = self.hierarchical_attn(refined)
            goal_tree.append(tree_level * depth_scale)
        hierarchical_goals = sum(goal_tree) / max(len(goal_tree), 1)
        hierarchical_goals = hierarchical_goals[:, :max_subgoals, :]
        dependencies = torch.sigmoid(self.dependency_head(hierarchical_goals))
        exec_graph = self.execution_head(hierarchical_goals)
        execution_order = torch.argsort(self.order_head(hierarchical_goals).squeeze(-1), dim=-1)
        cost_estimates = F.softplus(self.cost_head(hierarchical_goals)).squeeze(-1)
        total_cost = cost_estimates.sum(dim=-1)
        gate_scores = torch.sigmoid(self.goal_gate(hierarchical_goals).squeeze(-1))
        active_mask = gate_scores > 0.3

        subgoal_details = []
        for i in range(max_subgoals):
            sg = self.subgoal_nodes[i](hierarchical_goals[:, i:i+1, :])
            subgoal_details.append({
                "cost": sg["cost"],
                "depth": sg["depth"],
                "type": F.softmax(sg["type_logits"], dim=-1),
            })

        return {
            "goal_embeds": hierarchical_goals,
            "dependencies": dependencies,
            "execution_order": execution_order,
            "execution_graph": exec_graph,
            "cost_estimates": cost_estimates,
            "total_cost": total_cost,
            "gate_scores": gate_scores,
            "active_goals": active_mask,
            "goal_tree": goal_tree,
            "depth_weights": depth_weights,
            "subgoal_details": subgoal_details,
        }


class GlobalPlanner(HierarchicalPlanner):
    pass
