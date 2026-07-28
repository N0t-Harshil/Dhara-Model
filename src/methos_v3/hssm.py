import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int, n_levels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_levels = n_levels
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dt_proj = nn.Linear(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(torch.abs(torch.randn(n_levels, d_state)) + 1e-4))
        self.B_proj = nn.ModuleList([nn.Linear(d_model, d_state) for _ in range(n_levels)])
        self.C_proj = nn.ModuleList([nn.Linear(d_model, d_state) for _ in range(n_levels)])
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple:
        batch, seq_len, _ = x.shape
        device = x.device
        residual = x
        x = self.norm(x)
        x_in, x_gate = self.in_proj(x).chunk(2, dim=-1)
        x_in = F.silu(x_in)
        dt = F.softplus(self.dt_proj(x_in))

        if state is None:
            h_init = torch.zeros(batch, self.n_levels, self.d_state, device=device, dtype=x.dtype)
        elif state.dim() == 2:
            h_init = state.unsqueeze(1).expand(-1, self.n_levels, -1).contiguous()
        elif state.shape[1] != self.n_levels:
            h_init = state[:, :1, :].expand(-1, self.n_levels, -1)[:, :self.n_levels, :].contiguous()
        else:
            h_init = state

        A_stacked = torch.stack([-F.softplus(self.A_log[lvl]) for lvl in range(self.n_levels)], dim=0)
        B_stacked = torch.stack([proj(x_in) for proj in self.B_proj], dim=0)
        C_stacked = torch.stack([proj(x_in) for proj in self.C_proj], dim=0)

        delta_exp = dt.unsqueeze(0)
        A_bar = torch.exp(delta_exp * A_stacked.unsqueeze(1).unsqueeze(1))
        B_bar = (A_bar - 1.0) / (A_stacked.unsqueeze(1).unsqueeze(1) + 1e-8)
        bb = B_bar * B_stacked

        h_t = h_init.transpose(0, 1).contiguous()
        h_lvl_out = torch.empty(self.n_levels, batch, seq_len, h_t.shape[-1], device=device, dtype=x.dtype)
        for t in range(seq_len):
            h_t = A_bar[:, :, t, :] * h_t + bb[:, :, t, :]
            h_lvl_out[:, :, t, :] = h_t

        h = h_t.transpose(0, 1)
        y = torch.einsum("lbtk,lbtk->lbt", C_stacked, h_lvl_out).unsqueeze(-1)
        y = y.sum(dim=0)
        y = y * F.silu(x_gate)
        y = self.out_proj(y)
        return residual + y, h


class HierarchicalSSMStack(nn.Module):
    def __init__(self, d_model: int, d_state: int, n_layers: int = 6, n_hssm_levels: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            HierarchicalSSM(d_model, d_state, n_hssm_levels) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple:
        states = []
        layer_state = state
        for layer in self.layers:
            x, h = layer(x, layer_state)
            if h is not None:
                states.append(h)
                layer_state = h.detach()
        if not states:
            return x, torch.zeros(x.shape[0], self.layers[0].n_levels, self.layers[0].d_state, device=x.device, dtype=x.dtype)
        final_state = torch.stack(states, dim=1).mean(dim=1)
        return x, final_state
