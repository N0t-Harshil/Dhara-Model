import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dhara.hssm import HierarchicalSSMStack


class ForgetGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, memory: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        age_norm = age / age.max().clamp(min=1)
        gate_input = torch.cat([memory, age_norm.unsqueeze(-1)], dim=-1)
        forget_fraction = self.gate(gate_input)
        return memory * (1 - forget_fraction)


class CompressionAE(nn.Module):
    def __init__(self, d_model: int, compressed_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, compressed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        compression_loss = F.mse_loss(decoded, x.detach(), reduction="mean")
        return decoded, decoded.detach() + (decoded - decoded.detach()), compression_loss


class PriorityScorer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )

    def forward(self, memories: torch.Tensor) -> torch.Tensor:
        return self.scorer(memories).squeeze(-1)


class MemoryRetriever(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = (d_model // n_heads) ** -0.5

    def forward(self, query: torch.Tensor, memory_bank: torch.Tensor, priority: torch.Tensor = None) -> torch.Tensor:
        q_batch = query.shape[0]
        _, mem_len, d_model = memory_bank.shape
        q = self.query_proj(query).view(q_batch, -1, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        k = self.key_proj(memory_bank.expand(q_batch, -1, -1)).view(q_batch, mem_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        v = self.value_proj(memory_bank.expand(q_batch, -1, -1)).view(q_batch, mem_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if priority is not None:
            attn = attn + priority.unsqueeze(1).unsqueeze(2)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(q_batch, -1, d_model)
        return self.out_proj(out)


class WorkingMemory(nn.Module):
    def __init__(self, d_model: int, capacity: int = 512):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.position = nn.Embedding(capacity, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple:
        batch, seq_len, _ = x.shape
        device = x.device
        if state is None or state.numel() == 0:
            state = torch.zeros(batch, self.capacity, self.d_model, device=device, dtype=x.dtype)
        head = x[:, -1:, :]
        state = torch.cat([head, state[:, :-1, :]], dim=1)
        pos = self.position.weight[:seq_len].unsqueeze(0)
        safe_len = min(seq_len, self.capacity)
        x_safe = x[:, :safe_len, :]
        state_safe = state[:, :safe_len, :]
        gate = torch.sigmoid(self.gate(torch.cat([x_safe, state_safe], dim=-1)))
        output = gate * x_safe + (1 - gate) * state_safe
        output = output + 0.01 * pos[:, :safe_len, :]
        if safe_len < seq_len:
            output = torch.cat([output, x[:, safe_len:, :]], dim=1)
        return output, state


class SemanticMemory(nn.Module):
    def __init__(self, d_model: int, n_concepts: int = 4096):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_concepts, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = self.key_proj(query)
        mem = self.memory.unsqueeze(0).expand(query.shape[0], -1, -1)
        k = self.key_proj(mem)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        v = self.value_proj(mem)
        out = torch.matmul(attn, v)
        return out


class LongContextMemory(nn.Module):
    def __init__(self, d_model: int, d_state: int, n_hssm_layers: int = 6, n_levels: int = 3):
        super().__init__()
        self.hssm = HierarchicalSSMStack(d_model, d_state, n_hssm_layers, n_levels)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        return self.hssm(x, state)


class EpisodicMemory(nn.Module):
    def __init__(self, d_model: int, max_episodes: int = 256):
        super().__init__()
        self.max_episodes = max_episodes
        self.register_buffer("episode_buffer", torch.zeros(1, max_episodes, d_model))
        self.register_buffer("episode_count", torch.zeros(1, dtype=torch.long))
        self.compress = nn.Linear(d_model * 2, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def store(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return None
        compressed = x.mean(dim=1).mean(dim=0, keepdim=True)
        return compressed

    def _apply_store(self, compressed: torch.Tensor) -> None:
        if compressed is None:
            return
        count = int(self.episode_count.item())
        if count < self.max_episodes:
            self.episode_buffer[0, count] = compressed
        else:
            self.episode_buffer[:, :-1, :] = self.episode_buffer[:, 1:, :].clone()
            self.episode_buffer[:, -1, :] = compressed
        self.episode_count += 1

    def retrieve(self, query: torch.Tensor, top_k: int = 16) -> torch.Tensor:
        count = min(int(self.episode_count.item()), self.max_episodes)
        if count == 0:
            return torch.zeros_like(query)
        mem = self.episode_buffer[:, :count, :].clone()
        q = self.query_proj(query)
        k = self.key_proj(mem)
        v = self.value_proj(mem)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.retrieve(query)


class MemoryManager(nn.Module):
    def __init__(self, d_model: int, d_state: int, n_hssm_layers: int = 6,
                 n_hssm_levels: int = 3, working_mem_capacity: int = 512,
                 n_semantic_concepts: int = 4096, max_episodes: int = 256):
        super().__init__()
        self.working = WorkingMemory(d_model, working_mem_capacity)
        self.semantic = SemanticMemory(d_model, n_semantic_concepts)
        self.long_context = LongContextMemory(d_model, d_state, n_hssm_layers, n_hssm_levels)
        self.episodic = EpisodicMemory(d_model, max_episodes)
        self.forget_gate = ForgetGate(d_model)
        self.compressor = CompressionAE(d_model, compressed_dim=max(64, d_model // 40))
        self.priority_scorer = PriorityScorer(d_model)
        self.retriever = MemoryRetriever(d_model)
        self.fusion = nn.Linear(d_model * 4, d_model)
        self.importance = nn.Linear(d_model, 1)
        self.register_buffer("mem_age", torch.zeros(1, max_episodes))
        self.register_buffer("mem_priority", torch.ones(1, max_episodes))

    def apply_updates(self, mem_state: dict) -> None:
        if mem_state is None:
            return
        pending = mem_state.get("pending", {})
        if pending.get("compressed") is not None:
            self.episodic._apply_store(pending["compressed"])
        if pending.get("decayed") is not None:
            self.episodic.episode_buffer.data.copy_(pending["decayed"])
        if pending.get("priority") is not None:
            self.mem_priority.data.copy_(pending["priority"])
        self.mem_age.data.add_(1)

    def forward(self, x: torch.Tensor, mem_state: dict = None) -> tuple:
        if mem_state is None:
            mem_state = {}
        wm_out, wm_state = self.working(x, mem_state.get("working"))
        sm_out = self.semantic(wm_out)
        lc_out, lc_state = self.long_context(wm_out, mem_state.get("long_context"))
        compressed = self.episodic.store(lc_out)
        ep_out = self.episodic.retrieve(lc_out)

        importance = torch.sigmoid(self.importance(lc_out.mean(dim=1)))
        ep_buf = self.episodic.episode_buffer.clone()
        m_age = self.mem_age.clone()
        decayed = self.forget_gate(ep_buf, m_age)
        _, _, compression_loss = self.compressor(lc_out.mean(dim=1))
        priority = self.priority_scorer(decayed)
        retrieved = self.retriever(lc_out.mean(dim=1, keepdim=True), decayed, priority)
        fused = self.fusion(torch.cat([wm_out, sm_out, lc_out, ep_out], dim=-1))
        pending = {
            "compressed": compressed,
            "decayed": decayed.detach(),
            "priority": priority.detach(),
        }
        new_state = {
            "working": wm_state.detach(),
            "long_context": lc_state.detach(),
            "pending": pending,
        }
        meta = {
            "importance": importance,
            "compression_loss": compression_loss,
            "priority": priority,
            "retrieved": retrieved,
        }
        return fused, new_state, meta


class HierarchicalMemoryEngine(MemoryManager):
    pass
