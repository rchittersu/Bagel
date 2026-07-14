# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Fused FP8 (e4m3) quantization *prologue* kernels for the TP BAGEL LLM.

``torch._scaled_mm`` already fuses the GEMM epilogue (scale + fp32 accumulate ->
bf16), so the only quant work left unfused is the per-token activation quant. These
Triton kernels fuse that quant with the op that *produces* the activation, so the
activation is read from DRAM once instead of several times:

  - ``rmsnorm_fp8_quant(x, weight, eps)`` : RMSNorm + per-token quant. Feeds q/k/v
    (producer ``input_layernorm``) and gate/up (producer ``post_attention_layernorm``).
  - ``silumul_fp8_quant(gate, up)``       : ``silu(gate) * up`` + per-token quant.
    Feeds ``down_proj``.
  - ``quant_fp8_only(x)``                  : per-token quant of an already-produced
    activation (used on the MoT path where the norm was applied per-expert upstream).

Each returns ``(x_fp8 [M, K] e4m3, act_scale [M, 1] fp32)`` ready for
``fp8_w8a8_linear(..., x_fp8=x_fp8, act_scale=act_scale)``. A pure-PyTorch reference
with identical math backs every kernel; the dispatchers below default to the
reference (``USE_TRITON = False``) until the kernels are validated on the target
H100 (see ``tools/fp8_kernel_test.py``), then flip the flag.
"""

from typing import Tuple

import torch

from modeling.quant_utils import (
    FP8_DTYPE, FP8_MAX, INT8_DTYPE, INT8_MAX,
    quantize_fp8_rowwise, quantize_int8_rowwise,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton always present on the GPU box
    _HAS_TRITON = False

# Validated on the target H100 via tools/fp8_kernel_test.py (kernels match the
# PyTorch reference within e4m3 tolerance; fused silumul prologue ~9.5x faster than
# the naive multi-pass). Enabled so the fused kernels are used.
USE_TRITON = True

_EPS_SCALE = 1e-12  # guards a degenerate all-zero row (scale -> 1.0)


# --------------------------------------------------------------------------- #
# PyTorch reference implementations (correctness oracle + default path)
# --------------------------------------------------------------------------- #
def rmsnorm_fp8_quant_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(x) then per-token e4m3 quant, all in float32. Matches the kernel math."""
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps) * weight.to(torch.float32)
    return quantize_fp8_rowwise(normed)


def silumul_fp8_quant_ref(
    gate: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``silu(gate) * up`` then per-token e4m3 quant, all in float32."""
    gf = gate.to(torch.float32)
    act = (gf * torch.sigmoid(gf)) * up.to(torch.float32)
    return quantize_fp8_rowwise(act)


def quant_fp8_only_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8_rowwise(x)


def silumul_int8_quant_ref(
    gate: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``silu(gate) * up`` then per-token int8 quant, all in float32."""
    gf = gate.to(torch.float32)
    act = (gf * torch.sigmoid(gf)) * up.to(torch.float32)
    return quantize_int8_rowwise(act)


def quant_int8_only_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_int8_rowwise(x)


def dequant_int32_ref(acc: torch.Tensor, act_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """Reference int32 -> bf16 dequant: ``acc * act_scale[M,1] * w_scale[1,N]``."""
    out = acc.to(torch.float32) * act_scale.reshape(-1, 1) * w_scale.reshape(1, -1)
    return out.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_fp8_quant_kernel(
        x_ptr, w_ptr, out_ptr, scale_ptr,
        stride_xm, stride_om,
        N, eps, fp8_max,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per token-row; the whole row fits in one block (BLOCK >= N).
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        normed = x * rstd * w

        amax = tl.max(tl.abs(normed), axis=0)
        scale = amax / fp8_max
        scale = tl.where(scale > 0.0, scale, 1.0)

        y = normed / scale
        y = tl.minimum(tl.maximum(y, -fp8_max), fp8_max)
        tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _silumul_fp8_quant_kernel(
        g_ptr, u_ptr, out_ptr, scale_ptr,
        stride_gm, stride_um, stride_om,
        N, fp8_max,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per token-row; long rows (N=intermediate) are tiled, two passes.
        row = tl.program_id(0)
        n_tiles = tl.cdiv(N, BLOCK_SIZE)

        # Pass 1: row amax of silu(gate) * up.
        amax = 0.0
        for t in range(0, n_tiles):
            cols = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            g = tl.load(g_ptr + row * stride_gm + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + row * stride_um + cols, mask=mask, other=0.0).to(tl.float32)
            act = (g * tl.sigmoid(g)) * u
            amax = tl.maximum(amax, tl.max(tl.abs(act), axis=0))

        scale = amax / fp8_max
        scale = tl.where(scale > 0.0, scale, 1.0)

        # Pass 2: requantize and store.
        for t in range(0, n_tiles):
            cols = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            g = tl.load(g_ptr + row * stride_gm + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + row * stride_um + cols, mask=mask, other=0.0).to(tl.float32)
            act = (g * tl.sigmoid(g)) * u
            y = act / scale
            y = tl.minimum(tl.maximum(y, -fp8_max), fp8_max)
            tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _quant_fp8_only_kernel(
        x_ptr, out_ptr, scale_ptr,
        stride_xm, stride_om,
        N, fp8_max,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        scale = amax / fp8_max
        scale = tl.where(scale > 0.0, scale, 1.0)
        y = x / scale
        y = tl.minimum(tl.maximum(y, -fp8_max), fp8_max)
        tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row, scale)

    def _empty_out(m, n, device):
        return (
            torch.empty((m, n), dtype=FP8_DTYPE, device=device),
            torch.empty((m, 1), dtype=torch.float32, device=device),
        )

    def rmsnorm_fp8_quant_triton(x, weight, eps):
        x = x.contiguous()
        m, n = x.shape
        out, scale = _empty_out(m, n, x.device)
        block = triton.next_power_of_2(n)
        num_warps = min(32, max(4, block // 256))
        _rmsnorm_fp8_quant_kernel[(m,)](
            x, weight, out, scale,
            x.stride(0), out.stride(0),
            n, eps, FP8_MAX,
            BLOCK_SIZE=block, num_warps=num_warps,
        )
        return out, scale

    def silumul_fp8_quant_triton(gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        m, n = gate.shape
        out, scale = _empty_out(m, n, gate.device)
        block = min(2048, triton.next_power_of_2(n))
        _silumul_fp8_quant_kernel[(m,)](
            gate, up, out, scale,
            gate.stride(0), up.stride(0), out.stride(0),
            n, FP8_MAX,
            BLOCK_SIZE=block, num_warps=8,
        )
        return out, scale

    def quant_fp8_only_triton(x):
        x = x.contiguous()
        m, n = x.shape
        out, scale = _empty_out(m, n, x.device)
        block = triton.next_power_of_2(n)
        num_warps = min(32, max(4, block // 256))
        _quant_fp8_only_kernel[(m,)](
            x, out, scale,
            x.stride(0), out.stride(0),
            n, FP8_MAX,
            BLOCK_SIZE=block, num_warps=num_warps,
        )
        return out, scale

    # ----- int8 variants (round-half-away; int cast truncates toward zero) ----- #
    def _empty_int8(m, n, device):
        return (
            torch.empty((m, n), dtype=INT8_DTYPE, device=device),
            torch.empty((m, 1), dtype=torch.float32, device=device),
        )

    @triton.jit
    def _quant_int8_only_kernel(
        x_ptr, out_ptr, scale_ptr,
        stride_xm, stride_om,
        N, qmax,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        scale = amax / qmax
        scale = tl.where(scale > 0.0, scale, 1.0)
        y = x / scale
        y = y + tl.where(y >= 0, 0.5, -0.5)
        y = tl.minimum(tl.maximum(y, -qmax), qmax)
        tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _silumul_int8_quant_kernel(
        g_ptr, u_ptr, out_ptr, scale_ptr,
        stride_gm, stride_um, stride_om,
        N, qmax,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        n_tiles = tl.cdiv(N, BLOCK_SIZE)
        amax = 0.0
        for t in range(0, n_tiles):
            cols = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            g = tl.load(g_ptr + row * stride_gm + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + row * stride_um + cols, mask=mask, other=0.0).to(tl.float32)
            act = (g * tl.sigmoid(g)) * u
            amax = tl.maximum(amax, tl.max(tl.abs(act), axis=0))
        scale = amax / qmax
        scale = tl.where(scale > 0.0, scale, 1.0)
        for t in range(0, n_tiles):
            cols = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            g = tl.load(g_ptr + row * stride_gm + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + row * stride_um + cols, mask=mask, other=0.0).to(tl.float32)
            act = (g * tl.sigmoid(g)) * u
            y = act / scale
            y = y + tl.where(y >= 0, 0.5, -0.5)
            y = tl.minimum(tl.maximum(y, -qmax), qmax)
            tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _dequant_int32_kernel(
        acc_ptr, as_ptr, ws_ptr, out_ptr,
        N, stride_am, stride_om,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Fused int32 -> bf16 dequant: out[m,n] = acc[m,n] * act_scale[m] * w_scale[n].
        row = tl.program_id(0)
        a_scale = tl.load(as_ptr + row)
        n_tiles = tl.cdiv(N, BLOCK_SIZE)
        for t in range(0, n_tiles):
            cols = t * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            acc = tl.load(acc_ptr + row * stride_am + cols, mask=mask, other=0).to(tl.float32)
            ws = tl.load(ws_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out = acc * a_scale * ws
            tl.store(out_ptr + row * stride_om + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

    def quant_int8_only_triton(x):
        x = x.contiguous()
        m, n = x.shape
        out, scale = _empty_int8(m, n, x.device)
        block = triton.next_power_of_2(n)
        num_warps = min(32, max(4, block // 256))
        _quant_int8_only_kernel[(m,)](
            x, out, scale, x.stride(0), out.stride(0), n, INT8_MAX,
            BLOCK_SIZE=block, num_warps=num_warps,
        )
        return out, scale

    def silumul_int8_quant_triton(gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        m, n = gate.shape
        out, scale = _empty_int8(m, n, gate.device)
        block = min(2048, triton.next_power_of_2(n))
        _silumul_int8_quant_kernel[(m,)](
            gate, up, out, scale,
            gate.stride(0), up.stride(0), out.stride(0), n, INT8_MAX,
            BLOCK_SIZE=block, num_warps=8,
        )
        return out, scale

    def dequant_int32_triton(acc, act_scale, w_scale):
        acc = acc.contiguous()
        m, n = acc.shape
        out = torch.empty((m, n), dtype=torch.bfloat16, device=acc.device)
        block = min(2048, triton.next_power_of_2(n))
        _dequant_int32_kernel[(m,)](
            acc, act_scale, w_scale, out, n, acc.stride(0), out.stride(0),
            BLOCK_SIZE=block, num_warps=8,
        )
        return out


# --------------------------------------------------------------------------- #
# Dispatchers (reference by default; Triton once validated on-box)
# --------------------------------------------------------------------------- #
def _use_triton(x: torch.Tensor) -> bool:
    return USE_TRITON and _HAS_TRITON and x.is_cuda


def rmsnorm_fp8_quant(x, weight, eps):
    if _use_triton(x):
        return rmsnorm_fp8_quant_triton(x, weight, eps)
    return rmsnorm_fp8_quant_ref(x, weight, eps)


def silumul_fp8_quant(gate, up):
    if _use_triton(gate):
        return silumul_fp8_quant_triton(gate, up)
    return silumul_fp8_quant_ref(gate, up)


def quant_fp8_only(x):
    if _use_triton(x):
        return quant_fp8_only_triton(x)
    return quant_fp8_only_ref(x)


def quant_int8_only(x):
    if _use_triton(x):
        return quant_int8_only_triton(x)
    return quant_int8_only_ref(x)


def silumul_int8_quant(gate, up):
    if _use_triton(gate):
        return silumul_int8_quant_triton(gate, up)
    return silumul_int8_quant_ref(gate, up)


def dequant_int32(acc, act_scale, w_scale):
    if _use_triton(acc):
        return dequant_int32_triton(acc, act_scale, w_scale)
    return dequant_int32_ref(acc, act_scale, w_scale)


# Scheme routers used by the linears / MLP so a single call site serves fp8 and int8.
# Marked torch.compiler.disable: these launch raw Triton kernels Dynamo can't trace, so
# they run eager (opaque) and the surrounding transformer still compiles.
@torch.compiler.disable
def quant_only(x, scheme):
    return quant_int8_only(x) if scheme == "int8" else quant_fp8_only(x)


@torch.compiler.disable
def silumul_quant(gate, up, scheme):
    return silumul_int8_quant(gate, up) if scheme == "int8" else silumul_fp8_quant(gate, up)
