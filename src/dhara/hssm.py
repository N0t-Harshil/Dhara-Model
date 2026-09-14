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

        # Run the recurrent scan in fp32. bf16's small dynamic range and 3-bit
        # mantissa overflow to inf when the exp decay terms reach the clamp
        # floor (exp(+80) ~ 5.5e34 already near the bf16 limit and multiplied
        # again by bb before the cumsum), and 0*inf on the exp_pos factor then
        # poisons every downstream loss (memory reconstruction CE, intent, ...).
        state_dtype = h_init.dtype
        A_stacked = A_stacked.to(torch.float32)
        B_stacked = B_stacked.to(torch.float32)
        C_stacked = C_stacked.to(torch.float32)
        dt = dt.to(torch.float32)
        h_init = h_init.to(torch.float32)

        delta_exp = dt.unsqueeze(0)
        A_bar = torch.exp(delta_exp * A_stacked.unsqueeze(1).unsqueeze(1))
        B_bar = (A_bar - 1.0) / (A_stacked.unsqueeze(1).unsqueeze(1) + 1e-8)
        bb = B_bar * B_stacked

        log_A = delta_exp * A_stacked.unsqueeze(1).unsqueeze(1)
        log_prefix = torch.cumsum(log_A, dim=2)
        # Floor at -30 (exp(30) ~ 1.1e13, safe even in bf16): decay past
        # e^{-30} is numerically irrelevant and exp(-log_prefix) stays bounded.
        log_prefix = log_prefix.clamp(min=-30.0)
        exp_pos = torch.exp(log_prefix)
        exp_neg = torch.exp(-log_prefix)
        h_init_lvl = h_init.transpose(0, 1).contiguous().unsqueeze(2)
        scaled_bb = bb * exp_neg
        cumulative = torch.cumsum(scaled_bb, dim=2)
        h_lvl_out = exp_pos * (h_init_lvl + cumulative)

        h = h_lvl_out[:, :, -1, :].transpose(0, 1).to(state_dtype)
        y = torch.einsum("lbtk,lbtk->lbt", C_stacked, h_lvl_out).unsqueeze(-1)
        y = y.sum(dim=0).to(x_in.dtype)
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
                # Keep the graph connected across layers so the recurrent
                # state has a gradient path (detach() severed it).
                layer_state = h
        if not states:
            return x, torch.zeros(x.shape[0], self.layers[0].n_levels, self.layers[0].d_state, device=x.device, dtype=x.dtype)
        # Carry the deepest (top-level) layer's state onwards: it holds the
        # longest-horizon recurrence. Averaging over layers diluted the
        # high-level state with the per-token low-level states.
        final_state = states[-1]
        return x, final_state
