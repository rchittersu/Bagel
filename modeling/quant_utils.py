# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
FP8 (e4m3) W8A8 quantization helpers for the tensor-parallel BAGEL LLM.

Backend: ``torch._scaled_mm`` on Hopper (SM90). The scaling / dequant runs in a
hardware-fused epilogue (fp32 accumulate -> bf16 out), so the only quant work
left unfused is the per-token activation quant *prologue* (see
``modeling/fp8_triton.py`` for the fused-prologue kernels).

Numerics (rowwise scaling):
  - Weight     : per-output-channel e4m3 + ``weight_scale[1, N]`` (quantized once at load)
  - Activation : per-token (per-row)   e4m3 + ``act_scale[M, 1]``  (quantized each forward)
  - GEMM       : ``out = scale_a * (x_fp8 @ w_fp8.T) * scale_b`` accumulated in fp32

Both shard kinds reconstruct the same float result the ``F.linear`` path
produces, up to e4m3 rounding:
  - RowParallel  : ``_scaled_mm`` returns the bf16 partial already dequantized, then
                   the existing ``tp_all_reduce`` sums true partial sums (exact).
  - ColumnParallel: the replicated input is quantized identically on every rank, so
                   the sharded output stays consistent. No all-reduce.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# e4m3 finite range. Both activations and weights use e4m3 for forward inference
# (more mantissa than e5m2; e5m2 is for gradients and is not used here).
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

# Below this many tokens the GEMM is not compute-bound, ``_scaled_mm`` gives no
# speedup, and quantizing the activations only costs accuracy -> fall back to
# bf16. Notably covers the 1-token text-decode step in ``generate_text``.
FP8_MIN_TOKENS = 32

# ``_scaled_mm`` requires the contraction dim K to be a multiple of 16.
FP8_K_MULTIPLE = 16

# Use FP8 accumulation in the MMA (the fast Hopper path) instead of fp32 accumulate.
# This is the main GEMM-speed lever for _scaled_mm; off by default in torch, which
# leaves FP8 roughly tied with bf16 cuBLAS. Slight precision cost, fine for inference.
FP8_FAST_ACCUM = True

_EPS = 1e-12


def _scaled_mm_available() -> bool:
    return hasattr(torch, "_scaled_mm")


def can_use_scaled_mm(num_tokens: int, k: int) -> bool:
    """Preconditions for the fast ``_scaled_mm`` path (else we fall back to bf16)."""
    return (
        _scaled_mm_available()
        and num_tokens >= FP8_MIN_TOKENS
        and k % FP8_K_MULTIPLE == 0
    )


# --------------------------------------------------------------------------- #
# Quantization primitives
# --------------------------------------------------------------------------- #
def quantize_weight_rowwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel (row) symmetric e4m3 quant of a ``[N, K]`` weight.

    Returns ``(w_fp8 [N, K] e4m3, w_scale [1, N] fp32)``. The scale is shaped
    ``[1, N]`` so it can be passed straight to ``_scaled_mm`` as ``scale_b`` for
    the ``[K, N]`` operand ``w_fp8.t()``.
    """
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp_(min=_EPS)  # [N, 1]
    scale = amax / FP8_MAX                                     # [N, 1]
    w_fp8 = (w / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return w_fp8, scale.reshape(1, -1).contiguous()


def quantize_fp8_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token (per-row) symmetric e4m3 quant of a 2-D ``[M, K]`` activation.

    Returns ``(x_fp8 [M, K] e4m3, act_scale [M, 1] fp32)``. This is the reference
    prologue; ``modeling/fp8_triton.py`` provides fused (norm/act + quant) variants
    that are diff-tested against this.
    """
    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_(min=_EPS)  # [M, 1]
    scale = amax / FP8_MAX                                       # [M, 1]
    x_fp8 = (xf / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


def dequantize_weight(
    w_fp8: torch.Tensor, w_scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Reconstruct an approximate float ``[N, K]`` weight from its e4m3 + scale.

    Used only for the bf16 fallback path (small-M / odd-K). ``w_scale`` is ``[1, N]``.
    """
    return w_fp8.to(dtype) * w_scale.reshape(-1, 1).to(dtype)


# --------------------------------------------------------------------------- #
# FP8 linear
# --------------------------------------------------------------------------- #
def scaled_mm_fp8(
    x_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Low-level rowwise FP8 GEMM on 2-D operands -> bf16 ``[M, N]``.

    ``x_fp8 [M, K]`` e4m3 row-major, ``act_scale [M, 1]``; ``w_fp8 [N, K]`` e4m3,
    ``w_scale [1, N]``. ``_scaled_mm`` wants mat2 column-major, which is exactly
    ``w_fp8.t()``. Scaling + fp32 accumulate happen in the fused epilogue.
    """
    out = torch._scaled_mm(
        x_fp8.contiguous(),
        w_fp8.t(),
        scale_a=act_scale.reshape(-1, 1).to(torch.float32),
        scale_b=w_scale.reshape(1, -1).to(torch.float32),
        out_dtype=torch.bfloat16,
        use_fast_accum=FP8_FAST_ACCUM,
    )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def fp8_w8a8_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    fallback_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 W8A8 matmul: ``y = (x_fp8 @ w_fp8.T) * scales (+ bias)`` via ``_scaled_mm``.

    ``x`` may be N-D; only the last dim (``K``) is contracted. Quantizes ``x`` with
    the reference per-token prologue. When the ``_scaled_mm`` preconditions are not
    met (small ``M`` / odd ``K``), falls back to bf16 ``F.linear`` using
    ``fallback_weight`` (a cached dequantized weight) if provided.
    """
    out_features = w_fp8.shape[0]
    in_features = w_fp8.shape[1]
    lead_shape = x.shape[:-1]
    num_tokens = x.numel() // in_features if x.numel() else 0

    if not can_use_scaled_mm(num_tokens, in_features):
        w = fallback_weight if fallback_weight is not None else dequantize_weight(w_fp8, w_scale)
        return F.linear(x, w.to(x.dtype), bias)

    # Use the fused Triton per-token quant (falls back to the PyTorch reference
    # when Triton is unavailable). The naive reference quant is multi-pass and, run
    # per projection per layer, eats the GEMM savings -- so route it through the kernel.
    from modeling.fp8_triton import quant_fp8_only
    x_fp8, act_scale = quant_fp8_only(x.reshape(-1, in_features))
    out = scaled_mm_fp8(x_fp8, act_scale, w_fp8, w_scale, bias)
    return out.reshape(*lead_shape, out_features)


# --------------------------------------------------------------------------- #
# Selection pass
# --------------------------------------------------------------------------- #
# Only quantize the large matmuls inside the LLM decoder stack.
_LLM_LAYER_PREFIX = "language_model.model.layers."
_QKV_PREFIXES = ("q_proj", "k_proj", "v_proj")
_DOWN_PREFIX = "down_proj"


def quantize_model_fp8(
    model: torch.nn.Module,
    include_qkv: bool = False,
    skip_down: bool = False,
    min_numel: int = 0,
    verbose: bool = False,
) -> int:
    """Convert selected TP linears in the LLM decoder stack to FP8 in place.

    Target = the matmuls that actually win in FP8: MLP ``gate/up/down_proj`` and
    attention ``o_proj`` (plus every ``_moe_gen`` twin), for both the und and gen
    experts. ``q/k/v_proj`` are off by default -- at TP>=4 their shards are too
    small to be compute-bound and FP8 is a measured net loss there (set
    ``include_qkv=True`` only at low TP). ``lm_head``, embeddings, ViT, VAE and
    connectors stay bf16.

    Must be called *after* the checkpoint is loaded (real weight values) and
    before ``eval()``. Returns the number of layers quantized.

    Knobs (exposed as CLI flags upstream):
      - ``include_qkv`` : also quantize q/k/v (off if they regress accuracy).
      - ``skip_down``   : keep the outlier-prone ``down_proj`` in bf16.
      - ``min_numel``   : skip linears smaller than this ``in*out`` (tiny k/v).
    """
    # Imported here to avoid a circular import (tp_utils imports nothing from us).
    from modeling.tp_utils import ColumnParallelLinear, RowParallelLinear

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            continue
        if _LLM_LAYER_PREFIX not in name:
            continue

        leaf = name.rsplit(".", 1)[-1]
        if not include_qkv and any(leaf.startswith(p) for p in _QKV_PREFIXES):
            continue
        if skip_down and leaf.startswith(_DOWN_PREFIX):
            continue
        if min_numel and module.weight.numel() < min_numel:
            continue

        module._fp8_name = name  # label for the in-run profiler (fp8_profile)
        module.quantize_()
        count += 1
        if verbose:
            print(f"[fp8] quantized {name}", flush=True)

    return count
