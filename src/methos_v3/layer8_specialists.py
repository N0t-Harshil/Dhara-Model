import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialistExpert(nn.Module):
    def __init__(self, d_hidden: int, name: str):
        super().__init__()
        self.name = name
        self.proposal = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 4),
            nn.GELU(),
            nn.Linear(d_hidden * 4, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.confidence = nn.Linear(d_hidden, 1)
        self.critique_encoder = nn.Linear(d_hidden * 2, d_hidden)
        self.repair_net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def forward(self, workspace: torch.Tensor) -> tuple:
        proposal = self.proposal(workspace)
        conf = torch.sigmoid(self.confidence(proposal))
        return proposal, conf

    def critique(self, workspace: torch.Tensor, other_proposal: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([workspace, other_proposal], dim=-1)
        return torch.sigmoid(self.critique_encoder(combined).mean(dim=-1, keepdim=True))

    def repair(self, own_proposal: torch.Tensor, critique_signal: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([own_proposal, critique_signal], dim=-1)
        return self.repair_net(combined)


class SpecialistCritic(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.critique = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 4),
            nn.GELU(),
            nn.Linear(d_hidden * 4, 1),
        )

    def forward(self, workspace: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([workspace, proposal], dim=-1)
        return torch.sigmoid(self.critique(combined))


class DebateSandbox(nn.Module):
    def __init__(self, d_hidden: int, expert_names: list = None, n_debate_rounds: int = 3):
        super().__init__()
        if expert_names is None:
            expert_names = ["programming", "math", "logic", "planning", "retrieval", "creative", "safety"]
        self.n_experts = len(expert_names)
        self.n_debate_rounds = n_debate_rounds
        self.experts = nn.ModuleList([SpecialistExpert(d_hidden, name) for name in expert_names])
        self.critic = SpecialistCritic(d_hidden)
        self.consensus = nn.Linear(d_hidden * self.n_experts, d_hidden)
        self.debate_fusion = nn.Linear(d_hidden * 2, d_hidden)
        self.cross_examination = nn.Linear(d_hidden * self.n_experts, d_hidden)
        self.repair_gate = nn.Linear(d_hidden, 1)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, workspace: torch.Tensor) -> dict:
        proposals = []
        confidences = []
        critique_scores = []

        for expert in self.experts:
            prop, conf = expert(workspace)
            proposals.append(prop.unsqueeze(1))
            confidences.append(conf)
        proposals_t = torch.cat(proposals, dim=1)
        conf_t = torch.cat(confidences, dim=-1)

        for round_idx in range(self.n_debate_rounds):
            new_proposals = []
            for i, expert in enumerate(self.experts):
                own_prop = proposals_t[:, i, :]
                others_avg = torch.stack(
                    [proposals_t[:, j, :] for j in range(self.n_experts) if j != i],
                    dim=1
                ).mean(dim=1)
                cross_input = self.cross_examination(
                    proposals_t.reshape(proposals_t.shape[0], -1)
                )
                repaired = expert.repair(own_prop, cross_input)
                repair_gate = torch.sigmoid(self.repair_gate(repaired))
                new_prop = (1 - repair_gate) * own_prop + repair_gate * repaired
                new_proposals.append(new_prop.unsqueeze(1))
            proposals_t = torch.cat(new_proposals, dim=1)
            new_confs = []
            for i, expert in enumerate(self.experts):
                _, new_conf = expert(proposals_t[:, i, :])
                new_confs.append(new_conf)
            conf_t = torch.cat(new_confs, dim=-1)

        for i, expert in enumerate(self.experts):
            score = self.critic(workspace, proposals_t[:, i, :])
            critique_scores.append(score)
        crit_t = torch.cat(critique_scores, dim=-1)

        scores = F.softmax(conf_t * crit_t, dim=-1).unsqueeze(-1)
        consensus_out = self.consensus(proposals_t.reshape(proposals_t.shape[0], -1))
        consensus_out = self.norm(consensus_out + workspace)
        best_idx = (conf_t * crit_t).argmax(dim=-1)
        best_proposal = proposals_t[torch.arange(proposals_t.shape[0]), best_idx]

        return {
            "consensus": consensus_out,
            "best_proposal": best_proposal,
            "proposals": proposals_t,
            "confidences": conf_t,
            "critiques": crit_t,
            "selected_idx": best_idx,
            "debate_rounds": self.n_debate_rounds,
        }


class SpecialistSandbox(DebateSandbox):
    pass
