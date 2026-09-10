from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EnergyFunction(nn.Module):
    """
    Energy function for the latent sandbox — measures coherence of a latent state.

    E(z) = ||z - g(z)||^2 + lambda_1 * ||grad_E||^2 + lambda_2 * R(z)

    where:
    - g(z) is a learned reconstruction/denoising autoencoder
    - ||z - g(z)||^2 measures how "natural" the state is (low energy = coherent)
    - ||grad_E||^2 is a smoothing regularizer
    - R(z) is a roughness penalty to discourage degenerate solutions

    Args:
        d_hidden: Hidden dimension from Layer 2
        d_latent: Internal latent dimension for energy computation
    """

    def __init__(self, d_hidden: int = 7168, d_latent: int = 1024):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_latent = d_latent

        # Encoder: z -> latent factors
        self.encoder = nn.Sequential(
            nn.Linear(d_hidden, d_latent * 2),
            nn.SiLU(),
            nn.Linear(d_latent * 2, d_latent),
        )

        # Decoder (denoiser): latent factors -> reconstructed z
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.SiLU(),
            nn.Linear(d_latent * 2, d_hidden),
        )

        # Regularization network for roughness penalty
        self.R_net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 4),
            nn.SiLU(),
            nn.Linear(d_hidden // 4, 1),
        )

    def forward(
        self,
        z: torch.Tensor,              # [batch, d_hidden]
        lambda_1: float = 0.1,
        lambda_2: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy and its gradient for a batch of latent states.

        Returns:
            energy: [batch] — scalar energy per sample
            grad: [batch, d_hidden] — gradient of energy w.r.t. z
        """
        # Reconstruction error: ||z - g(z)||^2
        z_latent = self.encoder(z)  # [batch, d_latent]
        z_recon = self.decoder(z_latent)  # [batch, d_hidden]
        recon_error = torch.sum((z - z_recon) ** 2, dim=-1)  # [batch]

        # Roughness penalty
        roughness = self.R_net(z).squeeze(-1)  # [batch]

        # Total energy
        energy = recon_error + lambda_2 * roughness
        # (gradient norm regularizer is computed via autograd during optimization)

        # Compute gradient of energy w.r.t. z
        # Only when grad is enabled AND z requires grad
        if z.requires_grad and torch.is_grad_enabled():
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=z,
                create_graph=True,
                retain_graph=True,
            )[0]  # [batch, d_hidden]
        else:
            grad = torch.zeros_like(z)

        return energy, grad


class LatentSandbox(nn.Module):
    """
    Layer 3: Latent-Space Sandbox — Parallel Vector Tree Reasoning

    Replaces auto-regressive token-by-token chain-of-thought with parallel
    simulation of K trajectories in latent space, evaluated by an energy
    function. The lowest-energy trajectory is selected as the reasoning path.

    Algorithm (similar to JEPA + MCTS energy optimization):
        1. Initialize K perturbed copies of the LTC output state
        2. For each of N simulation steps:
            a. Compute energy E(z_k) and gradient ∇E(z_k) for each trajectory
            b. Gradient descent update: z_k ← z_k - α * ∇E(z_k)
            c. Every select_every steps: select best, expand around it
        3. Return the state with minimum energy

    Args:
        d_hidden: Hidden dimension from Layer 2
        d_latent: Internal latent dimension for energy function
        n_trajectories: K — number of parallel simulation trajectories
        n_sim_steps: N — number of energy descent steps
        select_every: Steps between selection-and-expand rounds
        learning_rate: Alpha — gradient descent step size
        noise_scale: Epsilon — perturbation scale for trajectory initialization
    """

    def __init__(
        self,
        d_hidden: int = 7168,
        d_latent: int = 1024,
        n_trajectories: int = 8,
        n_sim_steps: int = 16,
        select_every: int = 4,
        learning_rate: float = 0.1,
        noise_scale: float = 0.05,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.n_trajectories = n_trajectories
        self.n_sim_steps = n_sim_steps
        self.select_every = select_every
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale

        # Energy function (learned)
        self.energy_fn = EnergyFunction(d_hidden=d_hidden, d_latent=d_latent)

        # Output projection
        self.out_proj = nn.Linear(d_hidden, d_hidden, bias=False)

        # Norm
        self.norm = nn.LayerNorm(d_hidden, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(
        self,
        z_ltc: torch.Tensor,        # [batch, d_hidden] — from Layer 2
        return_trajectories: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = z_ltc.shape[0]
        device = z_ltc.device
        dtype = z_ltc.dtype

        # Initialize K parallel trajectories by perturbing the LTC state
        # [batch, K, d_hidden]
        z_k = z_ltc.unsqueeze(1).expand(-1, self.n_trajectories, -1).clone()
        z_k = z_k + self.noise_scale * torch.randn_like(z_k)

        # Require grad for energy-based optimization
        z_k.requires_grad_(True)

        best_energies = []
        trajectory_log = [] if return_trajectories else None

        for step in range(self.n_sim_steps):
            # Ensure z_k requires grad for energy computation
            z_k = z_k.detach()
            z_k.requires_grad_(True)

            # Flatten batch*K for energy computation
            z_flat = z_k.view(batch * self.n_trajectories, self.d_hidden)
            # [batch*K, d_hidden]

            # Compute energy and gradient
            energy_flat, grad_flat = self.energy_fn(z_flat)
            # energy: [batch*K], grad: [batch*K, d_hidden]

            # Reshape back
            energy = energy_flat.view(batch, self.n_trajectories)  # [batch, K]
            grad = grad_flat.view(batch, self.n_trajectories, self.d_hidden)

            # Gradient descent on energy. Iterations 0..n-2 stay a memory-
            # bounded search (detached); the LAST step keeps the graph so the
            # learned energy_fn (and this module's other params) receive
            # end-to-end gradients.
            if step == self.n_sim_steps - 1:
                z_k = z_k - self.learning_rate * grad
                z_k = z_k + 0.01 * self.noise_scale * torch.randn_like(z_k)
            else:
                with torch.no_grad():
                    z_k = z_k - self.learning_rate * grad
                    z_k = z_k + 0.01 * self.noise_scale * torch.randn_like(z_k)

            # Track best energies
            best_k = energy.argmin(dim=1)
            best_energy = energy.gather(1, best_k.unsqueeze(1))
            best_energies.append(best_energy.mean().item())

            # Selection and expansion (MCTS-style)
            if (step + 1) % self.select_every == 0 and step < self.n_sim_steps - 1:
                with torch.no_grad():
                    best_idx = best_k
                    for b in range(batch):
                        z_best = z_k[b, best_idx[b]]
                        for k in range(self.n_trajectories):
                            noise = self.noise_scale * torch.randn(self.d_hidden, device=device, dtype=dtype)
                            z_k[b, k] = z_best + noise

            if trajectory_log is not None:
                trajectory_log.append(z_k.detach().clone())

        # Select the best trajectory for each batch item. Kept in-graph so
        # gradients flow from the output through energy_fn / out_proj / norm.
        z_flat_final = z_k.view(batch * self.n_trajectories, self.d_hidden)
        energy_final, _ = self.energy_fn(z_flat_final)
        energy_final = energy_final.view(batch, self.n_trajectories)
        best_k = energy_final.argmin(dim=1)  # [batch]

        # Gather best trajectory (indexing preserves the autograd graph)
        z_k3 = z_k.view(batch, self.n_trajectories, self.d_hidden)
        z_selected = torch.stack([z_k3[b, best_k[b]] for b in range(batch)])

        # Output projection
        output = self.out_proj(z_selected)  # [batch, d_hidden]
        output = self.norm(output)

        if return_trajectories:
            return output, torch.tensor(best_energies, device=device)

        return output, None


class LatentSandboxEfficient(LatentSandbox):
    """
    Memory-efficient version of LatentSandbox.

    Processes trajectories sequentially instead of in parallel to reduce
    peak memory usage. Useful when d_hidden * K * batch exceeds GPU memory.
    """

    def forward(
        self,
        z_ltc: torch.Tensor,
        return_trajectories: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = z_ltc.shape[0]
        device = z_ltc.device
        dtype = z_ltc.dtype

        # Initialize best state
        selected: List[torch.Tensor] = []

        for b in range(batch):
            z_single = z_ltc[b:b+1]  # [1, d_hidden]

            # Initialize K trajectories
            z_k = z_single.expand(self.n_trajectories, -1).clone()
            z_k = z_k + self.noise_scale * torch.randn_like(z_k)
            for step in range(self.n_sim_steps):
                z_k = z_k.detach()
                z_k.requires_grad_(True)

                energy, grad = self.energy_fn(z_k)

                if step == self.n_sim_steps - 1:
                    z_k = z_k - self.learning_rate * grad
                    z_k = z_k + 0.01 * self.noise_scale * torch.randn_like(z_k)
                else:
                    with torch.no_grad():
                        z_k = z_k - self.learning_rate * grad
                        z_k = z_k + 0.01 * self.noise_scale * torch.randn_like(z_k)

                if (step + 1) % self.select_every == 0 and step < self.n_sim_steps - 1:
                    with torch.no_grad():
                        best_k = energy.argmin()
                        z_best = z_k[best_k]
                        for k in range(self.n_trajectories):
                            z_k[k] = z_best + self.noise_scale * torch.randn(
                                self.d_hidden, device=device, dtype=dtype
                            )

            # Select best (kept in-graph so energy_fn / out_proj / norm train)
            energy, _ = self.energy_fn(z_k)
            selected.append(z_k[energy.argmin()])

        if selected:
            z_best_all = torch.stack(selected, dim=0)
        else:
            z_best_all = torch.zeros(batch, self.d_hidden, device=device, dtype=dtype)

        output = self.out_proj(z_best_all)
        output = self.norm(output)
        return output, None
