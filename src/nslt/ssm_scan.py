from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass

_HAS_CUDA = torch.cuda.is_available()


# ── Reference Implementations (CPU / GPU-fallback) ────────────────────────


def selective_scan_vectorized(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    d_state: int,
    dt_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, d_inner = x.shape

    expand_weight = torch.eye(d_state, dt_rank, device=x.device, dtype=x.dtype)
    delta_exp = F.linear(delta, expand_weight)
    B_exp = F.linear(B, expand_weight)
    C_exp = F.linear(C, expand_weight)

    A_neg = A
    A_bar = torch.exp(delta_exp * A_neg.unsqueeze(0).unsqueeze(0))
    B_bar = (A_bar - 1.0) / (A_neg + 1e-10) * B_exp

    log_A_bar = delta_exp * A_neg.unsqueeze(0).unsqueeze(0)
    log_prefix = torch.cumsum(log_A_bar, dim=1)

    b = B_bar * x[:, :, :d_state]

    scaled = b * torch.exp(-log_prefix)
    cumulative = torch.cumsum(scaled, dim=1)
    h = cumulative * torch.exp(log_prefix)

    h_final = h[:, -1, :]

    y = torch.sum(C_exp * h, dim=-1, keepdim=True)

    outputs = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
    outputs[:, :, :1] = y
    if d_inner > d_state:
        outputs[:, :, d_state:] = x[:, :, d_state:]

    return outputs, h_final


def selective_scan_sequential(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_rank: int,
    return_h_seq: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[0]

    expand_weight = torch.eye(d_state, dt_rank, device=x.device, dtype=x.dtype)
    delta_exp = F.linear(delta, expand_weight)
    B_exp = F.linear(B, expand_weight)
    C_exp = F.linear(C, expand_weight)

    A_neg = A
    A_bar = torch.exp(delta_exp * A_neg.unsqueeze(0).unsqueeze(0))
    B_bar = (A_bar - 1.0) / (A_neg + 1e-10) * B_exp

    h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    outputs = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
    h_seq = []

    for t in range(seq_len):
        h = A_bar[:, t, :] * h + B_bar[:, t, :] * x[:, t, :d_state]
        h_seq.append(h.unsqueeze(1))
        y = torch.sum(C_exp[:, t, :] * h, dim=-1, keepdim=True)
        outputs[:, t, :y.shape[-1]] = y

    if d_inner > d_state:
        outputs[:, :, d_state:] = x[:, :, d_state:]

    if return_h_seq:
        return outputs, h, torch.cat(h_seq, dim=1)
    return outputs, h


# ── TorchScript CPU Helper ────────────────────────────────────────────────


@torch.jit.script
def _scan_cpu_loop(
    A_bar: torch.Tensor,
    B_contrib: torch.Tensor,
    C_exp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, d_state = A_bar.shape
    h = torch.zeros(batch, d_state, device=A_bar.device, dtype=A_bar.dtype)
    y = torch.zeros(batch, seq_len, device=A_bar.device, dtype=A_bar.dtype)

    for t in range(seq_len):
        h = A_bar[:, t, :] * h + B_contrib[:, t, :]
        y[:, t] = torch.sum(C_exp[:, t, :] * h, dim=-1)

    return y, h


def selective_scan_jit(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[0]

    expand_weight = torch.eye(d_state, dt_rank, device=x.device, dtype=x.dtype)
    delta_exp = F.linear(delta, expand_weight)
    B_exp = F.linear(B, expand_weight)
    C_exp = F.linear(C, expand_weight)

    A_neg = A
    A_bar = torch.exp(delta_exp * A_neg.unsqueeze(0).unsqueeze(0))
    B_bar = (A_bar - 1.0) / (A_neg + 1e-10)
    B_contrib = B_bar * B_exp * x[:, :, :d_state]

    y, h = _scan_cpu_loop(A_bar, B_contrib, C_exp)

    outputs = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
    outputs[:, :, 0] = y
    if d_inner > d_state:
        outputs[:, :, d_state:] = x[:, :, d_state:]

    return outputs, h


# ── Triton SSM Scan Kernel ────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _ssm_scan_kernel(
        a_ptr, b_ptr, c_ptr,
        y_ptr, h_ptr,
        seq_len, d_state,
        stride_a_s, stride_a_d,
        stride_b_s, stride_b_d,
        stride_c_s, stride_c_d,
        stride_y_s,
        stride_h_d,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # This kernel processes 1 state dimension across the full sequence.
        # pid encodes both batch and state dimension.
        # Stride-based indexing enables handling [batch, seq_len, d_state] tensors.

        offs = tl.arange(0, BLOCK_SIZE)

        # Load A_bar, B_contrib for this (batch, state) slice
        a = tl.load(a_ptr + offs * stride_a_s, mask=offs < seq_len, other=1.0)
        b = tl.load(b_ptr + offs * stride_b_s, mask=offs < seq_len, other=0.0)

        # Blelloch up-sweep (reduction to last element)
        step = 1
        while step < BLOCK_SIZE:
            cond = (offs + step) < seq_len
            a_i = tl.load(a_ptr + offs * stride_a_s)
            b_i = tl.load(b_ptr + offs * stride_b_s)
            a_j = tl.load(a_ptr + (offs + step) * stride_a_s)
            b_j = tl.load(b_ptr + (offs + step) * stride_b_s)
            a_new = a_j * a_i
            b_new = a_j * b_i + b_j
            tl.store(a_ptr + (offs + step) * stride_a_s, tl.where(cond, a_new, a_j))
            tl.store(b_ptr + (offs + step) * stride_b_s, tl.where(cond, b_new, b_j))
            step *= 2

        # Down-sweep (build scan)
        tl.store(b_ptr + 0 * stride_b_s, 0.0)
        tl.store(a_ptr + 0 * stride_a_s, 1.0)

        step = BLOCK_SIZE // 2
        while step > 0:
            cond = (offs + step) < seq_len
            a_i = tl.load(a_ptr + offs * stride_a_s)
            b_i = tl.load(b_ptr + offs * stride_b_s)
            a_j = tl.load(a_ptr + (offs + step) * stride_a_s)
            b_j = tl.load(b_ptr + (offs + step) * stride_b_s)
            a_new = a_j * a_i
            b_new = a_j * b_i + b_j
            tl.store(a_ptr + offs * stride_a_s, tl.where(cond, a_new, a_i))
            tl.store(b_ptr + offs * stride_b_s, tl.where(cond, b_new, b_i))
            tl.store(a_ptr + (offs + step) * stride_a_s, tl.where(cond, a_i, a_j))
            tl.store(b_ptr + (offs + step) * stride_b_s, tl.where(cond, b_i, b_j))
            step //= 2

        # h[t] = b[t] (after scan, b holds the inclusive result)
        h_vals = tl.load(b_ptr + offs * stride_b_s, mask=offs < seq_len, other=0.0)

        # Load C and compute y[t] = C[t] * h[t]
        c_vals = tl.load(c_ptr + offs * stride_c_s, mask=offs < seq_len, other=0.0)
        y_vals = c_vals * h_vals
        tl.store(y_ptr + offs * stride_y_s, y_vals, mask=offs < seq_len)

        # Store final state
        if seq_len > 0:
            h_final = tl.load(b_ptr + (seq_len - 1) * stride_b_s)
            tl.store(h_ptr + pid * stride_h_d, h_final)


def selective_scan_triton(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_rank: int,
    block_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[0]

    expand_weight = torch.eye(d_state, dt_rank, device=x.device, dtype=x.dtype)
    delta_exp = F.linear(delta, expand_weight)
    B_exp = F.linear(B, expand_weight)
    C_exp = F.linear(C, expand_weight)

    A_neg = A
    A_bar = torch.exp(delta_exp * A_neg.unsqueeze(0).unsqueeze(0))
    B_bar = (A_bar - 1.0) / (A_neg + 1e-10)
    B_contrib = B_bar * B_exp * x[:, :, :d_state]

    if not _HAS_CUDA or not _HAS_TRITON:
        A_bar_raw = A_bar
        h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(batch, seq_len, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = A_bar_raw[:, t, :] * h + B_contrib[:, t, :]
            y[:, t] = torch.sum(C_exp[:, t, :] * h, dim=-1)
        y_out = y.unsqueeze(-1)
        h_out = h
    else:
        y_out = torch.zeros(batch, seq_len, 1, device=x.device, dtype=x.dtype)
        h_out = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)

        num_programs = batch * d_state
        block_size = min(block_size, triton.next_power_of_2(seq_len))

        def grid(meta):
            return (num_programs,)

        _ssm_scan_kernel[grid](
            A_bar, B_contrib, C_exp,
            y_out, h_out,
            seq_len, d_state,
            A_bar.stride(1), A_bar.stride(2),
            B_contrib.stride(1), B_contrib.stride(2),
            C_exp.stride(1), C_exp.stride(2),
            y_out.stride(1),
            h_out.stride(1),
            BLOCK_SIZE=block_size,
        )

    outputs = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
    outputs[:, :, :1] = y_out
    if d_inner > d_state:
        outputs[:, :, d_state:] = x[:, :, d_state:]

    return outputs, h_out


# ── Custom Autograd Function ──────────────────────────────────────────────


class SSMScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, A, B, C, dt_rank, use_triton):
        ctx.dt_rank = dt_rank

        if use_triton and _HAS_TRITON and x.is_cuda:
            y, h = selective_scan_triton(x, delta, A, B, C, dt_rank)
            h_final_h = h
            h_seq = None
        else:
            y, h, h_seq = selective_scan_sequential(x, delta, A, B, C, dt_rank, return_h_seq=True)
            h_final_h = h
        ctx.h_seq = h_seq
        ctx.save_for_backward(x, delta, A, B, C, h_final_h, torch.tensor(use_triton))
        return y, h

    @staticmethod
    def backward(ctx, grad_y, grad_h):
        x, delta, A, B, C, h, use_triton_tensor = ctx.saved_tensors
        use_triton = use_triton_tensor.item()
        dt_rank = ctx.dt_rank

        batch, seq_len, d_inner = x.shape
        d_state = A.shape[0]

        expand_weight = torch.eye(d_state, dt_rank, device=x.device, dtype=x.dtype)
        delta_exp = F.linear(delta, expand_weight)

        A_neg = A
        A_bar = torch.exp(delta_exp * A_neg.unsqueeze(0).unsqueeze(0))
        B_bar = (A_bar - 1.0) / (A_neg + 1e-10)

        flat_grad = torch.zeros(batch, seq_len, d_state, device=x.device, dtype=x.dtype)
        flat_grad[:, :, 0] = grad_y[:, :, 0]

        # Recompute h_seq for correct per-timestep gradient computation
        h_seq = ctx.h_seq
        if h_seq is None:
            h_replay = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
            h_seq_list = []
            for t in range(seq_len):
                h_replay = A_bar[:, t, :] * h_replay + B_bar[:, t, :] * x[:, t, :d_state]
                h_seq_list.append(h_replay.unsqueeze(1))
            h_seq = torch.cat(h_seq_list, dim=1)

        grad_C = torch.zeros(batch, seq_len, dt_rank, device=x.device, dtype=x.dtype)
        C_proj = expand_weight.T

        dh = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
        for t in reversed(range(seq_len)):
            dh = dh + grad_h if t == seq_len - 1 else dh
            C_t = F.linear(C[:, t, :], expand_weight)
            dh_prev = A_bar[:, t, :] * dh + C_t * flat_grad[:, t, :]
            grad_C[:, t, :] = F.linear(h_seq[:, t, :] * flat_grad[:, t, :], C_proj)
            dh = dh_prev

        grad_x = torch.zeros_like(x)
        grad_x[:, :, :d_state] = B_bar * flat_grad
        if d_inner > d_state:
            grad_x[:, :, d_state:] = grad_y[:, :, d_state:]

        grad_delta = torch.zeros_like(delta)
        dt_proj = expand_weight.T
        h_replay = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            dh_delta = A_neg * A_bar[:, t, :] * (
                flat_grad[:, t, :] * h_replay
                + flat_grad[:, t, :] * x[:, t, :d_state] * B_bar[:, t, :]
            )
            grad_delta[:, t, :] = F.linear(dh_delta, dt_proj)
            h_replay = A_bar[:, t, :] * h_replay + B_bar[:, t, :] * x[:, t, :d_state]

        grad_A = torch.zeros_like(A)
        h_replay2 = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            grad_A = grad_A + torch.sum(
                delta_exp[:, t, :] * A_bar[:, t, :] * (
                    flat_grad[:, t, :] * h_replay2
                    + flat_grad[:, t, :] * x[:, t, :d_state] * B_bar[:, t, :]
                ),
                dim=0,
            )
            h_replay2 = A_bar[:, t, :] * h_replay2 + B_bar[:, t, :] * x[:, t, :d_state]
        grad_A = grad_A * -torch.exp(A)

        grad_B = torch.zeros_like(B)
        for t in range(seq_len):
            grad_B[:, t, :] = F.linear(
                flat_grad[:, t, :] * x[:, t, :d_state] * B_bar[:, t, :],
                dt_proj,
            )

        return grad_x, grad_delta, grad_A, grad_B, grad_C, None, None


# ── Unified Dispatch ──────────────────────────────────────────────────────


_SCAN_MODE = "auto"


def set_scan_mode(mode: str):
    global _SCAN_MODE
    assert mode in ("auto", "sequential", "vectorized", "triton", "cuda", "jit")
    _SCAN_MODE = mode


def get_scan_mode() -> str:
    return _SCAN_MODE


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_rank: int,
    mode: Optional[str] = None,
    use_autograd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode is None:
        mode = _SCAN_MODE

    if use_autograd:
        use_triton = mode in ("triton", "cuda") or (mode == "auto" and x.is_cuda and _HAS_TRITON)
        return SSMScanFunction.apply(x, delta, A, B, C, dt_rank, use_triton)

    if mode == "sequential":
        return selective_scan_sequential(x, delta, A, B, C, dt_rank)
    if mode == "jit":
        return selective_scan_jit(x, delta, A, B, C, dt_rank)
    if mode == "vectorized":
        return selective_scan_vectorized(x, delta, A, B, C, A.shape[0], dt_rank)
    if mode in ("triton", "cuda"):
        return selective_scan_triton(x, delta, A, B, C, dt_rank)
    if mode == "auto":
        if x.is_cuda and _HAS_TRITON:
            return selective_scan_triton(x, delta, A, B, C, dt_rank)
        if x.is_cuda:
            return selective_scan_vectorized(x, delta, A, B, C, A.shape[0], dt_rank)
        return selective_scan_sequential(x, delta, A, B, C, dt_rank)

    raise ValueError(f"Unknown scan mode: {mode}")
