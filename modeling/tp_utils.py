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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
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
