# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel (Megatron-style) primitives for BAGEL inference.

The model is a custom Qwen2 "Mixture-of-Transformers" stack with hand-written
packed attention + a NaiveCache, so HF / vLLM auto-TP cannot be used. This
module provides the minimal building blocks to shard the LLM decoder layers
across `WORLD_SIZE` GPUs and run them SPMD under `torchrun`.

Sharding scheme (per decoder layer):
  - q/k/v_proj (+ _moe_gen)  : column-parallel (split heads across ranks)
  - o_proj      (+ _moe_gen) : row-parallel  -> all-reduce
  - gate/up_proj(+ _moe_gen) : column-parallel
  - down_proj   (+ _moe_gen) : row-parallel  -> all-reduce
Everything else (layernorms, RoPE, embeddings, lm_head, ViT, VAE) is replicated.

When WORLD_SIZE == 1 (no torchrun / single GPU) every primitive degrades to a
plain nn.Linear with no communication, so the single-GPU code path is byte-for-
byte unchanged.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modeling import fp8_profile


# --------------------------------------------------------------------------- #
# Process-group state
# --------------------------------------------------------------------------- #
_TP_INITIALIZED = False
_TP_WORLD_SIZE = 1
_TP_RANK = 0
_TP_LOCAL_RANK = 0


def init_tensor_parallel() -> int:
    """Initialise the default NCCL process group from torchrun env vars.

    Returns the local rank (== cuda device index for this process). Safe to call
    when launched without torchrun: stays single-process (world size 1).
    """
    global _TP_INITIALIZED, _TP_WORLD_SIZE, _TP_RANK, _TP_LOCAL_RANK

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        _TP_INITIALIZED = True
        return 0

    _TP_RANK = int(os.environ.get("RANK", "0"))
    _TP_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    _TP_WORLD_SIZE = world_size

    torch.cuda.set_device(_TP_LOCAL_RANK)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    _TP_INITIALIZED = True
    return _TP_LOCAL_RANK


def get_tp_world_size() -> int:
    return _TP_WORLD_SIZE


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_local_rank() -> int:
    return _TP_LOCAL_RANK


def is_tp_enabled() -> bool:
    return _TP_WORLD_SIZE > 1


def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum-reduce a tensor across all TP ranks in place (no-op for world size 1)."""
    if _TP_WORLD_SIZE > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def tp_barrier() -> None:
    if _TP_WORLD_SIZE > 1:
        dist.barrier()


# --------------------------------------------------------------------------- #
# Parallel linear layers
# --------------------------------------------------------------------------- #
def _quantize_linear_(linear: nn.Module, scheme: str = "fp8") -> None:
    """Convert a parallel linear's float ``weight`` Parameter to W8A8 buffers
    (``weight_q`` + ``weight_scale``) for ``scheme`` (fp8|int8), freeing the float weight.

    Shared by Column/RowParallelLinear.quantize_(). Must run after the checkpoint
    is loaded (real values) and before eval. Idempotent.
    """
    if getattr(linear, "quant_scheme", None) is not None:
        return
    from modeling.quant_utils import quantize_weight_rowwise

    device = linear.weight.device
    w_q, w_scale = quantize_weight_rowwise(linear.weight.data, scheme)
    # Drop the float Parameter (register_parameter(None) is the idiomatic free); the
    # original tensor is then unreferenced and released.
    linear.weight = None
    linear.register_buffer("weight_q", w_q.to(device))
    linear.register_buffer("weight_scale", w_scale.to(device))
    linear._fallback_weight = None
    linear.quant_scheme = scheme


class ColumnParallelLinear(nn.Module):
    """Linear with the output dimension sharded across TP ranks.

    Holds a local weight of shape [out_features // world_size, in_features]. The
    activation stays sharded on the output dim (we never gather here -- the
    following RowParallelLinear consumes the sharded activation directly).
    `out_features` must be divisible by the TP world size.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        ws = get_tp_world_size()
        assert out_features % ws == 0, (
            f"ColumnParallelLinear out_features={out_features} not divisible by TP world size {ws}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_local = out_features // ws

        self.weight = nn.Parameter(torch.empty(self.out_features_local, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_local)) if bias else None

        # W8A8 state (opt-in via quantize_()): quant_scheme is None | "fp8" | "int8".
        self.quant_scheme = None
        self._fallback_weight = None  # lazily-built bf16 weight for the small-M fallback

    def quantize_(self, scheme: str = "fp8") -> None:
        """Replace the loaded float weight with W8A8 buffers (fp8|int8), freeing the float weight."""
        _quantize_linear_(self, scheme)

    def _fallback(self) -> torch.Tensor:
        if self._fallback_weight is None:
            from modeling.quant_utils import dequantize_weight
            self._fallback_weight = dequantize_weight(self.weight_q, self.weight_scale)
        return self._fallback_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_scheme is None:
            return F.linear(x, self.weight, self.bias)
        from modeling.quant_utils import quant_w8a8_linear, can_quant
        # Only profile when the quantized path actually runs (token gate met); below the
        # gate the call falls back to bf16, which isn't a datapoint and clutters the report.
        if fp8_profile.is_enabled():
            m = x.numel() // self.in_features
            if can_quant(self.quant_scheme, m, self.in_features):
                return fp8_profile.run(
                    getattr(self, "_fp8_name", "col_proj"), m,
                    lambda: quant_w8a8_linear(x, self.weight_q, self.weight_scale, self.quant_scheme,
                                              self.bias, fallback_weight=self._fallback()),
                    lambda: F.linear(x, self._fallback(), self.bias),
                )
        return quant_w8a8_linear(x, self.weight_q, self.weight_scale, self.quant_scheme,
                                 self.bias, fallback_weight=self._fallback())

    def forward_prequantized(self, x_q: torch.Tensor, act_scale: torch.Tensor,
                             lead_shape=None) -> torch.Tensor:
        """W8A8 forward consuming an externally fused-quant'd 2-D input.

        Lets several column-parallel projections that share one input (q/k/v, or
        gate/up) reuse a single fused-prologue quant. Bias is added inline, exactly
        as the float path. Caller guarantees the GEMM preconditions.
        """
        from modeling.quant_utils import mm_dequant
        out = mm_dequant(x_q, act_scale, self.weight_q, self.weight_scale, self.quant_scheme, self.bias)
        if lead_shape is not None:
            out = out.reshape(*lead_shape, self.out_features_local)
        return out


class RowParallelLinear(nn.Module):
    """Linear with the input dimension sharded across TP ranks.

    Holds a local weight of shape [out_features, in_features // world_size]. The
    incoming activation is assumed already sharded on the input dim; each rank
    computes a partial output and we all-reduce to reconstruct the full result.
    Bias (if any) is added once, after the all-reduce. `in_features` must be
    divisible by the TP world size.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        ws = get_tp_world_size()
        assert in_features % ws == 0, (
            f"RowParallelLinear in_features={in_features} not divisible by TP world size {ws}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_local = in_features // ws

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_local))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # W8A8 state (opt-in via quantize_()): quant_scheme is None | "fp8" | "int8".
        self.quant_scheme = None
        self._fallback_weight = None  # lazily-built bf16 weight for the small-M fallback

    def quantize_(self, scheme: str = "fp8") -> None:
        """Replace the loaded float weight with W8A8 buffers (fp8|int8), freeing the float weight."""
        _quantize_linear_(self, scheme)

    def _fallback(self) -> torch.Tensor:
        if self._fallback_weight is None:
            from modeling.quant_utils import dequantize_weight
            self._fallback_weight = dequantize_weight(self.weight_q, self.weight_scale)
        return self._fallback_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_scheme is not None:
            from modeling.quant_utils import quant_w8a8_linear, can_quant
            # Bias is added AFTER the all-reduce (each rank holds a partial sum), so
            # the GEMM itself must not add it. Time the GEMM only; the all-reduce is
            # identical for quant and bf16 and would just cancel in the comparison.
            # Only profile when the quantized path actually runs (token gate met).
            out = None
            if fp8_profile.is_enabled():
                m = x.numel() // self.in_features_local
                if can_quant(self.quant_scheme, m, self.in_features_local):
                    out = fp8_profile.run(
                        getattr(self, "_fp8_name", "row_proj"), m,
                        lambda: quant_w8a8_linear(x, self.weight_q, self.weight_scale, self.quant_scheme,
                                                  bias=None, fallback_weight=self._fallback()),
                        lambda: F.linear(x, self._fallback()),
                    )
            if out is None:
                out = quant_w8a8_linear(x, self.weight_q, self.weight_scale, self.quant_scheme,
                                        bias=None, fallback_weight=self._fallback())
        else:
            out = F.linear(x, self.weight)
        out = tp_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_prequantized(self, x_q: torch.Tensor, act_scale: torch.Tensor,
                             lead_shape=None) -> torch.Tensor:
        """W8A8 forward consuming an externally fused-quant'd 2-D input (e.g. the
        ``silu(gate)*up`` -> quant prologue feeding ``down_proj``). Keeps the
        all-reduce-then-bias ordering of the float path. Caller guarantees the
        GEMM preconditions.
        """
        from modeling.quant_utils import mm_dequant
        scheme = self.quant_scheme
        if not fp8_profile.is_enabled():
            out = mm_dequant(x_q, act_scale, self.weight_q, self.weight_scale, scheme, bias=None)
        else:
            m = act_scale.numel()  # act_scale is [M, 1]
            out = fp8_profile.run(
                getattr(self, "_fp8_name", "row_proj"), m,
                lambda: mm_dequant(x_q, act_scale, self.weight_q, self.weight_scale, scheme, bias=None),
                # Representative bf16 GEMM on the same shapes (values irrelevant for timing).
                lambda: F.linear(x_q.to(torch.bfloat16), self._fallback()),
            )
        if lead_shape is not None:
            out = out.reshape(*lead_shape, self.out_features)
        out = tp_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------------------------------------------------------------- #
# Sharded checkpoint loader
# --------------------------------------------------------------------------- #
# Projection names sharded on the OUTPUT dim (rows / dim 0).
_COLUMN_PROJ = (
    "q_proj", "k_proj", "v_proj",
    "q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen",
    "gate_proj", "up_proj",
)
# Projection names sharded on the INPUT dim (cols / dim 1).
_ROW_PROJ = (
    "o_proj", "o_proj_moe_gen",
    "down_proj",
)
# Only shard inside the LLM decoder stack; ViT/VAE/connectors stay replicated.
_LLM_LAYER_PREFIX = "language_model.model.layers."


def _shard_kind(name: str) -> Optional[str]:
    """Return 'col', 'row', or None for a checkpoint parameter name."""
    if _LLM_LAYER_PREFIX not in name:
        return None
    for proj in _COLUMN_PROJ:
        if f".{proj}." in name:
            return "col"
    for proj in _ROW_PROJ:
        if f".{proj}." in name:
            return "row"
    return None


def load_sharded_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load `checkpoint_path` (safetensors) into `model`, slicing TP-sharded
    tensors to this rank. Replicated tensors are loaded whole. Parameters/buffers
    not present in the checkpoint (e.g. rotary inv_freq) are left as initialised.

    The model is expected to already be materialised on the target device with
    TP-local parameter shapes (i.e. built while the TP group is active).
    """
    from safetensors import safe_open

    ws = get_tp_world_size()
    rank = get_tp_rank()
    state = model.state_dict()

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        ckpt_keys = set(f.keys())
        for name, param in state.items():
            if name not in ckpt_keys:
                continue

            kind = _shard_kind(name) if ws > 1 else None
            if kind is None:
                tensor = f.get_tensor(name)
            else:
                sl = f.get_slice(name)
                shape = sl.get_shape()
                if kind == "col":
                    local = shape[0] // ws
                    tensor = sl[rank * local:(rank + 1) * local]
                else:  # row
                    if len(shape) == 1:
                        # 1-D row-parallel tensors don't occur here, but be safe.
                        tensor = sl[:]
                    else:
                        local = shape[1] // ws
                        tensor = sl[:, rank * local:(rank + 1) * local]

            with torch.no_grad():
                param.copy_(tensor.to(dtype=param.dtype, device=param.device))
