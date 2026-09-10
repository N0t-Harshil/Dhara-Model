from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nslt.layer3_sandbox import EnergyFunction

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    Node in the MCTS latent-space tree.

    Each node stores a latent state vector, visit statistics,
    and links to its parent and children in the search tree.

    Args:
        state: [d_hidden] — latent state at this node
        parent: Parent MCTSNode (None for root)
        action_id: Index of the action that led to this node
    """

    def __init__(
        self,
        state: torch.Tensor,
        parent: Optional["MCTSNode"] = None,
        action_id: int = -1,
    ):
        self.state = state.detach().clone()
        self.parent = parent
        self.action_id = action_id
        self.children: List["MCTSNode"] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.depth: int = parent.depth + 1 if parent else 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def mean_value(self) -> float:
        return self.total_value / max(self.visit_count, 1)

    def ucb_score(self, c_puct: float = 1.0) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.mean_value
        if self.parent is not None:
            exploration = c_puct * math.sqrt(
                math.log(self.parent.visit_count + 1) / (self.visit_count + 1e-8)
            )
        else:
            exploration = 0.0
        return exploitation + exploration

    def best_child(self, c_puct: float = 1.0) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb_score(c_puct))


class PolicyValueNetwork(nn.Module):
    """
    Policy-value network for MCTS guidance.

    Given a latent state and context, predicts:
    1. K candidate direction vectors for exploration (policy)
    2. Scalar value of the current state (value)

    Architecture:
        [state, context] -> MLP -> [K*d_hidden directions] + [scalar value]

    Args:
        d_hidden: Latent dimension
        n_directions: K — number of candidate directions to propose
    """

    def __init__(self, d_hidden: int = 7168, n_directions: int = 8):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_directions = n_directions

        self.net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )

        self.policy_head = nn.Linear(d_hidden, d_hidden * n_directions)
        self.value_head = nn.Linear(d_hidden, 1)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, d_hidden] — current latent state
            context: [batch, d_hidden] — global context from LTC

        Returns:
            directions: [batch, n_directions, d_hidden] — unit-normalized perturbations
            values: [batch] — predicted state values
        """
        combined = torch.cat([state, context], dim=-1)
        features = self.net(combined)

        directions = self.policy_head(features).view(-1, self.n_directions, self.d_hidden)
        directions = F.normalize(directions, dim=-1)

        values = self.value_head(features).squeeze(-1)
        return directions, values


class MCTSLatentSandbox(nn.Module):
    """
    Layer 3 (MCTS variant): Latent-Space Monte Carlo Tree Search.

    Replaces the energy-minimization sandbox with a proper MCTS that
    maintains a tree of latent states, uses UCB for selection, a learned
    policy for expansion, simulation via energy gradient descent, and
    value backup up the tree.

    Algorithm per forward pass:
        1. Initialize root node = LTC output state
        2. For N simulations:
            a. SELECT: Traverse tree from root using UCB until leaf
            b. EXPAND: Use policy network to generate K child directions
            c. SIMULATE: Run energy gradient descent from each new child
            d. BACKUP: Propagate terminal values up the selected path
        3. Return the state of the most-visited child of root

    Args:
        d_hidden: Hidden dimension
        d_latent: Internal latent dimension for energy function
        n_simulations: Number of MCTS simulations per forward pass
        n_directions: K — branching factor at each expansion
        n_sim_steps: N — number of energy descent steps per simulation rollout
        c_puct: Exploration constant for UCB
        learning_rate: Step size for energy gradient descent
        noise_scale: Perturbation scale for unused directions
        max_depth: Maximum tree depth
    """

    def __init__(
        self,
        d_hidden: int = 7168,
        d_latent: int = 1024,
        n_simulations: int = 16,
        n_directions: int = 8,
        n_sim_steps: int = 8,
        c_puct: float = 1.0,
        learning_rate: float = 0.1,
        noise_scale: float = 0.05,
        max_depth: int = 3,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.n_simulations = n_simulations
        self.n_directions = n_directions
        self.n_sim_steps = n_sim_steps
        self.c_puct = c_puct
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale
        self.max_depth = max_depth

        self.energy_fn = EnergyFunction(d_hidden=d_hidden, d_latent=d_latent)

        self.policy_value = PolicyValueNetwork(
            d_hidden=d_hidden, n_directions=n_directions
        )

        self.context_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.norm = nn.LayerNorm(d_hidden, eps=1e-6)

        self.n_trajectories = n_simulations

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        path = [node]
        while not node.is_leaf:
            node = node.best_child(self.c_puct)
            path.append(node)
        return node, path

    def _expand(
        self,
        leaf: MCTSNode,
        context: torch.Tensor,
        energy_grad: torch.Tensor,
    ) -> None:
        state = leaf.state.unsqueeze(0)
        context_b = context.unsqueeze(0) if context.dim() == 1 else context

        with torch.no_grad():
            directions, values = self.policy_value(state, context_b)

            for k in range(self.n_directions):
                # Gradient-based expansion: descend the energy surface at the
                # leaf, keeping a small policy-weighted exploration offset.
                perturbation = -self.learning_rate * energy_grad
                exploration = self.noise_scale * directions[0, k]
                noise = self.noise_scale * torch.randn(self.d_hidden, device=state.device)
                child_state = leaf.state + perturbation + exploration + noise
                child = MCTSNode(
                    state=child_state, parent=leaf, action_id=k
                )
                leaf.children.append(child)

    def _simulate(
        self, node: MCTSNode, context: torch.Tensor
    ) -> float:
        # Search-time rollout: memory-bounded energy gradient descent. The
        # returned scalar only feeds visit statistics / backup; the output
        # trajectory is replayed in-graph in _finalize_state so gradients
        # reach the learned modules.
        z = node.state.detach().clone()
        total_energy = 0.0
        for _ in range(self.n_sim_steps):
            z = z.detach().requires_grad_(True)
            energy, grad = self.energy_fn(z.unsqueeze(0))
            with torch.no_grad():
                z = z - self.learning_rate * grad.squeeze(0)
                total_energy += energy.detach().mean().item()
        return -total_energy

    def _finalize_state(
        self,
        best_child: MCTSNode,
        z_ltc_b: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Re-run the chosen trajectory in-graph so the module trains end to end."""
        dirs, _ = self.policy_value(z_ltc_b.unsqueeze(0), context.unsqueeze(0))
        k = best_child.action_id
        if 0 <= k < self.n_directions:
            _, g0 = self.energy_fn(z_ltc_b.unsqueeze(0))
            # Same update shape used during search-time expansion.
            z = z_ltc_b - self.learning_rate * g0[0]
            z = z + self.learning_rate * self.noise_scale * dirs[0, k]
        else:
            z = z_ltc_b.clone()
        for _ in range(self.n_sim_steps):
            _, g = self.energy_fn(z.unsqueeze(0))
            z = z - self.learning_rate * g[0]
        return z

    def _backup(self, path: List[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value

    def _gather_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        all_nodes = [root]
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children:
                all_nodes.append(child)
                stack.append(child)
        return all_nodes

    def forward(
        self,
        z_ltc: torch.Tensor,
        return_trajectories: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[float]]]:
        """
        Args:
            z_ltc: [batch, d_hidden] — from Layer 2
            return_trajectories: If True, return energy trace

        Returns:
            output: [batch, d_hidden] — MCTS-selected latent state
            energies: Optional energy trace list
        """
        batch = z_ltc.shape[0]
        device = z_ltc.device

        batch_outputs = []
        batch_energies = [] if return_trajectories else None

        for b in range(batch):
            context = self.context_proj(z_ltc[b])  # [d_hidden]
            root = MCTSNode(state=z_ltc[b])
            energy_trace = []

            for sim in range(self.n_simulations):
                leaf, path = self._select(root)

                if leaf.depth >= self.max_depth:
                    path_value = self._simulate(leaf, context)
                    energy_trace.append(-path_value)
                    self._backup(path, path_value)
                    continue

                # Real energy gradient at the leaf (search-time, detached).
                with torch.no_grad():
                    _, grad = self.energy_fn(leaf.state.unsqueeze(0))
                energy_grad = grad[0]

                self._expand(leaf, context, energy_grad)

                for child in leaf.children[-self.n_directions:]:
                    child_value = self._simulate(child, context)
                    child.visit_count = 1
                    child.total_value = child_value
                    energy_trace.append(-child_value)
                    self._backup([leaf, child], child_value)

            root_children = sorted(
                root.children, key=lambda c: c.visit_count, reverse=True
            )
            if not root_children:
                root_children = [root]
            best_child = root_children[0]

            # Differentiable finalization: replay the selected trajectory
            # in-graph so energy_fn / policy_value / out_proj / norm receive
            # end-to-end gradients.
            best_state = self._finalize_state(best_child, z_ltc[b], context)
            batch_outputs.append(best_state)
            if batch_energies is not None:
                batch_energies.append(energy_trace)

        output = torch.stack(batch_outputs, dim=0)
        output = self.out_proj(output)
        output = self.norm(output)

        return output, batch_energies


class MCTSLatentSandboxEfficient(MCTSLatentSandbox):
    """
    Memory-efficient MCTS variant.

    Reuses a single tree for all batch elements by processing the
    most-visited trajectory deterministically.
    """

    def forward(
        self,
        z_ltc: torch.Tensor,
        return_trajectories: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[float]]]:
        batch = z_ltc.shape[0]
        device = z_ltc.device

        context = self.context_proj(z_ltc)  # [batch, d_hidden]
        batch_outputs = []
        batch_energies = [] if return_trajectories else None

        for b in range(batch):
            cb = context[b]
            energy_trace = []

            best_z = z_ltc[b].detach().clone()
            best_score = -float("inf")
            best_k = -1

            for sim in range(self.n_simulations):
                with torch.no_grad():
                    directions, value = self.policy_value(z_ltc[b].unsqueeze(0), cb.unsqueeze(0))
                    _, grad = self.energy_fn(z_ltc[b].unsqueeze(0))
                energy_grad = grad[0]

                candidates = []
                candidate_scores = []
                for k in range(self.n_directions):
                    cand = (
                        best_z
                        - self.learning_rate * energy_grad
                        + self.learning_rate * self.noise_scale * directions[0, k]
                    )
                    candidates.append(cand)
                    sim_z = cand.detach().clone()
                    for _ in range(self.n_sim_steps):
                        sim_z = sim_z.detach().requires_grad_(True)
                        e, g = self.energy_fn(sim_z.unsqueeze(0))
                        with torch.no_grad():
                            sim_z = sim_z - self.learning_rate * g.squeeze(0)
                    with torch.no_grad():
                        e, _ = self.energy_fn(sim_z.unsqueeze(0))
                    score = -e.item()
                    candidate_scores.append(score)
                    energy_trace.append(e.item())

                best_idx = int(torch.tensor(candidate_scores).argmax())
                if candidate_scores[best_idx] > best_score:
                    best_score = candidate_scores[best_idx]
                    best_z = candidates[best_idx].detach()
                    best_k = best_idx

            # Differentiable finalization along the selected direction.
            dirs, _ = self.policy_value(z_ltc[b].unsqueeze(0), cb.unsqueeze(0))
            if best_k >= 0:
                _, g0 = self.energy_fn(z_ltc[b].unsqueeze(0))
                z = z_ltc[b] - self.learning_rate * g0[0]
                z = z + self.learning_rate * self.noise_scale * dirs[0, best_k]
            else:
                z = z_ltc[b].clone()
            for _ in range(self.n_sim_steps):
                _, g = self.energy_fn(z.unsqueeze(0))
                z = z - self.learning_rate * g[0]

            batch_outputs.append(z)
            if batch_energies is not None:
                batch_energies.append(energy_trace)

        output = torch.stack(batch_outputs, dim=0)
        output = self.out_proj(output)
        output = self.norm(output)

        return output, batch_energies
