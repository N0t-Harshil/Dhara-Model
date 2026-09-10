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


def _fold_drive(x: torch.Tensor, d_state: int) -> torch.Tensor:
    """Fold all ``d_inner`` input channels into a ``d_state``-sized drive.

    Previously only the first ``d_state`` channels drove the recurrent state;
    channels ``d_state:`` were raw passthrough (or dead zeros), so most of the
    block's input never reached the state. Strided folding gives every input
    channel a gradient path into the state while keeping the O(1)-memory
    ``[batch, d_state]`` state.
    """
    batch, seq_len, d_inner = x.shape
    drive = x[:, :, :d_state].clone()
    if d_inner > d_state:
        tail = x[:, :, d_state:]
        groups = (tail.shape[-1] + d_state - 1) // d_state
        padded = F.pad(tail, (0, groups * d_state - tail.shape[-1]))
        drive = drive + padded.reshape(batch, seq_len, groups, d_state).sum(dim=2)
    return drive


def _expand_output(y: torch.Tensor, x: torch.Tensor, d_inner: int) -> torch.Tensor:
    """Combine the per-timestep scan output ``[B, T, 1]`` with the input.

    Every output channel carries the input (residual) PLUS the broadcast
    scan signal — previously only channel 0 carried the scan result, the
    middle channels were dead zeros, and the tail was raw passthrough.
    """
    return x + y.expand(*y.shape[:-1], d_inner)


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
    # NOTE: B_exp is applied to the drive below (matching the other backends).
    B_bar = (A_bar - 1.0) / (A_neg + 1e-10)

    log_A_bar = delta_exp * A_neg.unsqueeze(0).unsqueeze(0)
    log_prefix = torch.cumsum(log_A_bar, dim=1)

    b = B_bar * B_exp * _fold_drive(x, d_state)

    scaled = b * torch.exp(-log_prefix)
    cumulative = torch.cumsum(scaled, dim=1)
    h = cumulative * torch.exp(log_prefix)

    h_final = h[:, -1, :]

    y = torch.sum(C_exp * h, dim=-1, keepdim=True)

    return _expand_output(y, x, d_inner), h_final


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
    # NOTE: B_exp is applied to the drive below (matching the other backends);
    # do NOT fold it into B_bar here as well.
    B_bar = (A_bar - 1.0) / (A_neg + 1e-10)

    h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    outputs = torch.zeros(batch, seq_len, 1, device=x.device, dtype=x.dtype)
    h_seq = []
    drive = _fold_drive(x, d_state)

    for t in range(seq_len):
        h = A_bar[:, t, :] * h + B_bar[:, t, :] * B_exp[:, t, :] * drive[:, t, :]
        h_seq.append(h.unsqueeze(1))
        y = torch.sum(C_exp[:, t, :] * h, dim=-1, keepdim=True)
        outputs[:, t, :] = y

    if return_h_seq:
        return _expand_output(outputs, x, d_inner), h, torch.cat(h_seq, dim=1)
    return _expand_output(outputs, x, d_inner), h


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
    B_contrib = B_bar * _fold_drive(x, d_state)

    y, h = _scan_cpu_loop(A_bar, B_contrib, C_exp)

    return _expand_output(y.unsqueeze(-1), x, d_inner), h


# ── Triton SSM Scan Kernel ────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _ssm_scan_kernel(
        a_ptr, b_ptr, c_ptr,
        y_ptr, h_ptr,
        seq_len, d_state,
        stride_a_b, stride_a_s, stride_a_d,
        stride_b_b, stride_b_s, stride_b_d,
        stride_c_b, stride_c_s, stride_c_d,
        stride_y_b, stride_y_s,
        stride_h_b, stride_h_d,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # This kernel processes 1 state dimension across the full sequence.
        # pid encodes both batch and state dimension.
        # Stride-based indexing enables handling [batch, seq_len, d_state] tensors.
        b_id = pid // d_state
        s_id = pid % d_state
        a_base = b_id * stride_a_b + s_id * stride_a_d
        b_base = b_id * stride_b_b + s_id * stride_b_d
        c_base = b_id * stride_c_b + s_id * stride_c_d
        y_base = b_id * stride_y_b
        h_off = b_id * stride_h_b + s_id * stride_h_d

        offs = tl.arange(0, BLOCK_SIZE)

        # Each affine element (a, b) represents h_out = a * h_in + b. An
        # inclusive scan composes elements [0..t] so that b[t] = h[t].
        #
        # Hillis-Steele scan: at iteration with distance `step`, element t
        # absorbs element (t - step):  (a, b)_t <- (a_t * a_{t-step}, a_t * b_{t-step} + b_t).
        # Every load/store is masked to the [0, seq_len) slice; without masks
        # the trailing BLOCK_SIZE positions hit out-of-bounds memory belonging
        # to other (batch, state) slices and corrupt them.
        step = 1
        while step < BLOCK_SIZE:
            t = offs + step
            m = t < seq_len
            a_i = tl.load(a_ptr + a_base + offs * stride_a_s, mask=offs < seq_len, other=1.0)
            b_i = tl.load(b_ptr + b_base + offs * stride_b_s, mask=offs < seq_len, other=0.0)
            a_j = tl.load(a_ptr + a_base + t * stride_a_s, mask=m, other=1.0)
            b_j = tl.load(b_ptr + b_base + t * stride_b_s, mask=m, other=0.0)
            a_new = a_j * a_i
            b_new = a_j * b_i + b_j
            tl.store(a_ptr + a_base + t * stride_a_s, a_new, mask=m)
            tl.store(b_ptr + b_base + t * stride_b_s, b_new, mask=m)
            step *= 2

        # h[t] = b[t] after the inclusive scan
        h_vals = tl.load(b_ptr + b_base + offs * stride_b_s, mask=offs < seq_len, other=0.0)

        # Load C and compute y[t] = C[t] * h[t]
        c_vals = tl.load(c_ptr + c_base + offs * stride_c_s, mask=offs < seq_len, other=0.0)
        y_vals = c_vals * h_vals
        tl.store(y_ptr + y_base + offs * stride_y_s, y_vals, mask=offs < seq_len)

        # Store final state
        h_final = tl.load(b_ptr + b_base + (seq_len - 1) * stride_b_s, mask=seq_len > 0, other=0.0)
        tl.store(h_ptr + h_off, h_final)


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
    B_contrib = B_bar * _fold_drive(x, d_state)

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

        if seq_len > 0:
            num_programs = batch * d_state
            block_size = triton.next_power_of_2(seq_len)

            def grid(meta):
                return (num_programs,)

            _ssm_scan_kernel[grid](
                A_bar, B_contrib, C_exp,
                y_out, h_out,
                seq_len, d_state,
                A_bar.stride(0), A_bar.stride(1), A_bar.stride(2),
                B_contrib.stride(0), B_contrib.stride(1), B_contrib.stride(2),
                C_exp.stride(0), C_exp.stride(1), C_exp.stride(2),
                y_out.stride(0), y_out.stride(1),
                h_out.stride(0), h_out.stride(1),
                BLOCK_SIZE=block_size,
            )

    return _expand_output(y_out, x, d_inner), h_out


# ── Custom Autograd Function ──────────────────────────────────────────────


class SSMScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, A, B, C, dt_rank, use_triton):
        ctx.dt_rank = dt_rank

        if use_triton and _HAS_TRITON and x.is_cuda:
            y, h = selective_scan_triton(x, delta, A, B, C, dt_rank)
        else:
            y, h, h_seq = selective_scan_sequential(x, delta, A, B, C, dt_rank, return_h_seq=True)
        ctx.save_for_backward(x, delta, A, B, C)
        return y, h

    @staticmethod
    def backward(ctx, grad_y, grad_h):
        x, delta, A, B, C = ctx.saved_tensors
        dt_rank = ctx.dt_rank

        batch, seq_len, d_inner = x.shape
        d_state = A.shape[0]

        if grad_y is None:
            grad_y = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        if grad_h is None:
            grad_h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)

        eps = 1e-10
        with torch.no_grad():
            expand_weight = torch.eye(d_state, dt_rank, device=x.device, dtype=x.dtype)
            delta_exp = F.linear(delta, expand_weight)  # [B, T, d_state]
            A_neg = A
            A_bar = torch.exp(delta_exp * A_neg.unsqueeze(0).unsqueeze(0))
            denom = A_neg.unsqueeze(0).unsqueeze(0) + eps
            B_bar = (A_bar - 1.0) / denom
            B_exp = F.linear(B, expand_weight)  # [B, T, d_state]
            C_exp = F.linear(C, expand_weight)  # [B, T, d_state]

            # Forward folds all channels into the drive (see _fold_drive) and
            # broadcasts the scalar scan output to every channel (see
            # _expand_output) — the adjoint mirrors both operations.
            x_ssm = _fold_drive(x.detach(), d_state)

            # Replay hidden states h[t] (fresh tensors, no autograd graph, so the
            # explicit gradients below are not double-counted).
            h_seq_list = []
            h_replay = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
            for t in range(seq_len):
                h_replay = A_bar[:, t, :] * h_replay + B_bar[:, t, :] * B_exp[:, t, :] * x_ssm[:, t, :]
                h_seq_list.append(h_replay.clone())
            h_seq = torch.stack(h_seq_list, dim=1)

            # y[t] is broadcast to every channel: sum incoming grads first.
            g = grad_y.sum(dim=-1)  # [B, T]

            gA_bar = torch.zeros_like(A_bar)  # dL/dA_bar[t, d]
            gB = torch.zeros_like(A_bar)      # dL/dB_contrib[t, d]
            gC_exp = torch.zeros_like(A_bar)  # dL/dC_exp[t, d]
            dh = grad_h.clone()               # dL/dh[t] accumulated from the right
            for t in reversed(range(seq_len)):
                dh = dh + C_exp[:, t, :] * g[:, t].unsqueeze(-1)
                gC_exp[:, t, :] = h_seq[:, t, :] * g[:, t].unsqueeze(-1)
                gB[:, t, :] = dh
                if t > 0:
                    gA_bar[:, t, :] = dh * h_seq[:, t - 1, :]
                dh = A_bar[:, t, :] * dh  # propagate to h[t-1]

            # B_contrib = B_bar * B_exp * drive
            gB_bar = gB * B_exp * x_ssm
            gB_exp = gB * B_bar * x_ssm
            gx_drive = gB * B_bar * B_exp

            # Un-fold the drive gradient back to the original channels
            # (mirror of _fold_drive) and add the residual path
            # (output = x + broadcast(y) ⇒ dL/dx includes grad_y directly).
            grad_x = grad_y.clone()
            grad_x[:, :, :d_state] = grad_x[:, :, :d_state] + gx_drive
            if d_inner > d_state:
                tail = x[:, :, d_state:]
                groups = (tail.shape[-1] + d_state - 1) // d_state
                padded = F.pad(tail, (0, groups * d_state - tail.shape[-1]))
                # Each group's gradient is gx_drive (sum over group members).
                grad_tail = gx_drive.unsqueeze(-1).expand(
                    batch, seq_len, d_state, groups).reshape(batch, seq_len, groups * d_state)
                grad_x[:, :, d_state:] = grad_x[:, :, d_state:] + grad_tail[:, :, : tail.shape[-1]]

            # delta_exp[t, d] = delta[t, r] * eye[d, r]  ->  adjoint picks r == d
            # dA_bar/ddelta_exp = A_bar * A_neg
            # dB_bar/ddelta_exp = (1 / denom) * A_bar * A_neg
            gDelta_exp = gA_bar * A_bar * A_neg.unsqueeze(0).unsqueeze(0)
            gDelta_exp = gDelta_exp + gB_bar * (A_bar * A_neg.unsqueeze(0).unsqueeze(0) / denom)
            grad_delta = torch.zeros_like(delta)
            k = min(dt_rank, d_state)
            grad_delta[:, :, :k] = gDelta_exp[:, :, :k]

            # A (shared across batch/time):
            # dA_bar/dA = A_bar * delta_exp
            # dB_bar/dA = (A_bar * delta_exp * denom - (A_bar - 1)) / denom^2
            dA_bar_dA = A_bar * delta_exp
            dB_bar_dA = (A_bar * delta_exp * denom - (A_bar - 1.0)) / (denom * denom)
            grad_A = (gA_bar * dA_bar_dA + gB_bar * dB_bar_dA).sum(dim=(0, 1))

            # B_exp[t, d] = B[t, r] * eye[d, r]  ->  adjoint picks r == d
            grad_B = torch.zeros_like(B)
            grad_B[:, :, :k] = gB_exp[:, :, :k]

            # C_exp[t, d] = C[t, r] * eye[d, r]  ->  adjoint picks r == d
            grad_C = torch.zeros_like(B)
            grad_C[:, :, :k] = gC_exp[:, :, :k]

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
        # The Triton kernel mutates its inputs in place and has no autograd
        # graph, so it can only be used when no gradient is required.
        _needs_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (x, delta, A, B, C)
        )
        if _needs_grad:
            use_triton = False
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
        _needs_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (x, delta, A, B, C)
        )
        if x.is_cuda and _HAS_TRITON and not _needs_grad:
            return selective_scan_triton(x, delta, A, B, C, dt_rank)
        if x.is_cuda:
            return selective_scan_vectorized(x, delta, A, B, C, A.shape[0], dt_rank)
        return selective_scan_sequential(x, delta, A, B, C, dt_rank)

    raise ValueError(f"Unknown scan mode: {mode}")
