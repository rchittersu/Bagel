# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
W8A8 quantization helpers for the tensor-parallel BAGEL LLM. Two selectable schemes:

  - "fp8"  : e4m3 via ``torch._scaled_mm``. Scaling/dequant runs in a hardware-fused
             epilogue (fp32 accumulate -> bf16), so only the activation-quant prologue
             is unfused.
  - "int8" : symmetric via ``torch._int_mm`` (int32 out). No fused epilogue, so the
             dequant ``int32 * act_scale * w_scale`` is folded into one Triton pass
             (``dequant_int32``) to keep it a fair fight with fp8 on Hopper.

The fused-prologue / dequant kernels live in ``modeling/fp8_triton.py``.

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

# INT8 symmetric range. Alternative W8A8 scheme via torch._int_mm; unlike _scaled_mm
# it returns int32 and does NOT fuse the dequant, so the int8 path pairs the GEMM with
# a fused Triton dequant epilogue (modeling/fp8_triton.dequant_int32).
INT8_DTYPE = torch.int8
INT8_MAX = 127.0

# Per-scheme quant range / storage dtype.
_QUANT_MAX = {"fp8": FP8_MAX, "int8": INT8_MAX}
_QUANT_DTYPE = {"fp8": FP8_DTYPE, "int8": INT8_DTYPE}

# Below this many tokens the GEMM is not compute-bound, _scaled_mm gives no speedup,
# and quantizing the activations only costs accuracy -> fall back to bf16. Calibrated
# from the in-run profiler (modeling/fp8_profile): M=315 measured a net loss, M>=3136
# a clear win, so the crossover sits between -- 2048 stays safely on the win side.
# Tunable at runtime via set_quant_min_tokens() (--quant-min-tokens).
QUANT_MIN_TOKENS = 2048

# ``_scaled_mm`` requires the contraction dim K to be a multiple of 16.
FP8_K_MULTIPLE = 16

# Use FP8 accumulation in the MMA (the fast Hopper path) instead of fp32 accumulate.
# This is the main GEMM-speed lever for _scaled_mm; off by default in torch, which
# leaves FP8 roughly tied with bf16 cuBLAS. Slight precision cost, fine for inference.
FP8_FAST_ACCUM = True

_EPS = 1e-12


def _scaled_mm_available() -> bool:
    return hasattr(torch, "_scaled_mm")


def _int_mm_available() -> bool:
    return hasattr(torch, "_int_mm")


def set_quant_min_tokens(n: int) -> None:
    """Override the token-count gate below which linears fall back to bf16."""
    global QUANT_MIN_TOKENS
    QUANT_MIN_TOKENS = n


def can_use_scaled_mm(num_tokens: int, k: int) -> bool:
    """Preconditions for the fast ``_scaled_mm`` path (else we fall back to bf16)."""
    return (
        _scaled_mm_available()
        and num_tokens >= QUANT_MIN_TOKENS
        and k % FP8_K_MULTIPLE == 0
    )


def can_use_int_mm(num_tokens: int, k: int) -> bool:
    """Preconditions for the ``torch._int_mm`` path (else fall back to bf16).

    Same token / K-alignment gate as FP8; ``_int_mm`` also wants K a multiple of 16.
    """
    return (
        _int_mm_available()
        and num_tokens >= QUANT_MIN_TOKENS
        and k % FP8_K_MULTIPLE == 0
    )


def can_quant(scheme: str, num_tokens: int, k: int) -> bool:
    """Scheme-aware precondition check (fp8 -> _scaled_mm, int8 -> _int_mm)."""
    return can_use_int_mm(num_tokens, k) if scheme == "int8" else can_use_scaled_mm(num_tokens, k)


# --------------------------------------------------------------------------- #
# Quantization primitives
# --------------------------------------------------------------------------- #
def _quantize_rowwise(x: torch.Tensor, dim: int, scheme: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quant along ``dim`` for the given scheme. Returns (q, scale)."""
    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=dim, keepdim=True).clamp_(min=_EPS)
    qmax = _QUANT_MAX[scheme]
    scale = amax / qmax
    q = xf / scale
    if scheme == "int8":
        q = q.round_().clamp_(-qmax, qmax).to(INT8_DTYPE)
    else:
        q = q.clamp_(-qmax, qmax).to(FP8_DTYPE)
    return q, scale


def quantize_weight_rowwise(w: torch.Tensor, scheme: str = "fp8") -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel (row) symmetric quant of a ``[N, K]`` weight.

    Returns ``(w_q [N, K], w_scale [1, N] fp32)``. The scale is shaped ``[1, N]`` so it
    can be passed straight to ``_scaled_mm`` (scale_b) / the int8 dequant epilogue.
    """
    w_q, scale = _quantize_rowwise(w, dim=1, scheme=scheme)  # scale [N, 1]
    return w_q, scale.reshape(1, -1).contiguous()


def quantize_fp8_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token e4m3 quant of a 2-D ``[M, K]`` activation -> (x_fp8, act_scale[M,1])."""
    return _quantize_rowwise(x, dim=-1, scheme="fp8")


def quantize_int8_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token int8 quant of a 2-D ``[M, K]`` activation -> (x_int8, act_scale[M,1])."""
    return _quantize_rowwise(x, dim=-1, scheme="int8")


def dequantize_weight(
    w_q: torch.Tensor, w_scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Reconstruct an approximate float ``[N, K]`` weight from its quant + scale.

    Works for both schemes (e4m3 or int8 cast to ``dtype`` then scaled). Used only for
    the bf16 fallback path (small-M / odd-K). ``w_scale`` is ``[1, N]``.
    """
    return w_q.to(dtype) * w_scale.reshape(-1, 1).to(dtype)


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


def int8_mm_dequant(
    x_int8: torch.Tensor,
    act_scale: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Low-level rowwise INT8 GEMM on 2-D operands -> bf16 ``[M, N]``.

    ``torch._int_mm`` returns int32 ``[M, N]``; unlike ``_scaled_mm`` it has no fused
    epilogue, so the dequant ``out = int32 * act_scale[M,1] * w_scale[1,N]`` is folded
    into a single Triton pass (``dequant_int32``) rather than several bf16 ops.
    """
    acc = torch._int_mm(x_int8.contiguous(), w_int8.t())  # int32 [M, N]
    from modeling.fp8_triton import dequant_int32
    out = dequant_int32(acc, act_scale, w_scale)          # bf16 [M, N]
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def mm_dequant(
    x_q: torch.Tensor,
    act_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    scheme: str,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scheme-aware low-level GEMM on pre-quantized 2-D operands -> bf16 ``[M, N]``."""
    if scheme == "int8":
        return int8_mm_dequant(x_q, act_scale, w_q, w_scale, bias)
    return scaled_mm_fp8(x_q, act_scale, w_q, w_scale, bias)


def quant_w8a8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    scheme: str = "fp8",
    bias: Optional[torch.Tensor] = None,
    fallback_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A8 matmul (fp8 via ``_scaled_mm`` or int8 via ``_int_mm`` + fused dequant).

    ``x`` may be N-D; only the last dim (``K``) is contracted. Quantizes ``x`` per-token
    via the fused Triton prologue. When preconditions aren't met (small ``M`` / odd ``K``)
    falls back to bf16 ``F.linear`` using ``fallback_weight`` if provided.
    """
    out_features = w_q.shape[0]
    in_features = w_q.shape[1]
    lead_shape = x.shape[:-1]
    num_tokens = x.numel() // in_features if x.numel() else 0

    if not can_quant(scheme, num_tokens, in_features):
        w = fallback_weight if fallback_weight is not None else dequantize_weight(w_q, w_scale)
        return F.linear(x, w.to(x.dtype), bias)

    from modeling.fp8_triton import quant_only
    x_q, act_scale = quant_only(x.reshape(-1, in_features), scheme)
    out = mm_dequant(x_q, act_scale, w_q, w_scale, scheme, bias)
    return out.reshape(*lead_shape, out_features)


def fp8_w8a8_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    fallback_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Back-compat thin wrapper: FP8 W8A8 linear."""
    return quant_w8a8_linear(x, w_fp8, w_scale, "fp8", bias, fallback_weight)


# --------------------------------------------------------------------------- #
# Selection pass
# --------------------------------------------------------------------------- #
# Only quantize the large matmuls inside the LLM decoder stack.
_LLM_LAYER_PREFIX = "language_model.model.layers."
_QKV_PREFIXES = ("q_proj", "k_proj", "v_proj")
_DOWN_PREFIX = "down_proj"


def quantize_model(
    model: torch.nn.Module,
    scheme: str = "fp8",
    include_qkv: bool = False,
    skip_down: bool = False,
    min_numel: int = 0,
    verbose: bool = False,
) -> int:
    """Convert selected TP linears in the LLM decoder stack to W8A8 (``scheme``) in place.

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
        module.quantize_(scheme)
        count += 1
        if verbose:
            print(f"[{scheme}] quantized {name}", flush=True)

    return count


def quantize_model_fp8(model, include_qkv=False, skip_down=False, min_numel=0, verbose=False) -> int:
    """Back-compat wrapper: quantize to FP8."""
    return quantize_model(model, "fp8", include_qkv, skip_down, min_numel, verbose)
