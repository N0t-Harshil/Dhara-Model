import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class DomainDynamics(nn.Module):
    def __init__(self, d_state: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state + d_hidden, d_hidden * 2),
            nn.SiLU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )

    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, ctx], dim=-1))


class ODETick(nn.Module):
    def __init__(self, n_domains: int, d_state: int, d_hidden: int):
        super().__init__()
        self.domains = nn.ModuleList([DomainDynamics(d_state, d_hidden) for _ in range(n_domains)])
        self.domain_router = nn.Linear(d_state + d_hidden, n_domains)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        route_input = torch.cat([z, h], dim=-1)
        route_logits = self.domain_router(route_input)
        route_weights = F.softmax(route_logits, dim=-1)
        dz = torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        for i, dyn in enumerate(self.domains):
            dz = dz + route_weights[:, i:i+1] * dyn(z, h)
        return dz


class AdaptiveContinuousReasoning(nn.Module):
    def __init__(self, d_state: int, d_hidden: int, n_domains: int = 5, max_steps: int = 32):
        super().__init__()
        self.max_steps = max_steps
        self.ode_tick = ODETick(n_domains, d_state, d_hidden)
        self.scale = math.sqrt(d_hidden)

    def forward(self, h_compressed: torch.Tensor, z: torch.Tensor, n_steps: torch.LongTensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler-integrate the reasoning dynamics and return the reached
        endpoint state and the full trajectory.

        The endpoint (state reached at each sample's last active step) is the
        natural "after reasoning" representation; trailing updates freeze so
        early-stopped samples keep their endpoint. The recorded trajectory is
        returned for the smoothness aux loss (the previous design returned the
        trajectory MEAN, which flattened the path and made the smoothness loss,
        fed identical copies, permanently zero).
        """
        if n_steps is None:
            n_steps = torch.full((z.shape[0],), 8, device=z.device, dtype=torch.long)
        max_n = int(n_steps.max().item())
        z_out = z.clone()
        trajectories = []
        for step in range(max_n):
            active = step < n_steps
            dz = self.ode_tick(z_out, h_compressed)
            # Euler update with dt = 1/sqrt(d_hidden). Multiplying by
            # sqrt(d_hidden) (~100x for default config) blew up the state
            # magnitude multiplicatively at every step.
            new_z = z_out + dz / self.scale
            z_out = torch.where(active.unsqueeze(-1), new_z, z_out)
            trajectories.append(z_out.unsqueeze(1))
        if trajectories:
            z_traj = torch.cat(trajectories, dim=1)
        else:
            z_traj = z.unsqueeze(1)
        return z_out, z_traj
