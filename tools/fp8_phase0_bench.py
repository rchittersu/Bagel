# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Phase 0 de-risk microbenchmark: FP8 (_scaled_mm) linear vs bf16 F.linear at BAGEL's
real LLM matmul shapes. Run on the target H100 BEFORE relying on the full rollout.

    python tools/fp8_phase0_bench.py --world-size 4 --tokens 4096

Reports, per layer (q/k/v/o/gate/up/down), the realized per-GEMM speedup `s` and the
share spent in the (unfused) activation-quant prologue. Plug the median `s` and the
profiled GEMM wall-time fraction `f` into  speedup = 1 / ((1-f) + f/s)  for the
end-to-end projection. Decision rule from the plan: proceed only if s >~ 1.3.
"""

import argparse

import torch

from modeling.quant_utils import (
    quantize_weight_rowwise, scaled_mm_fp8, int8_mm_dequant, can_quant, FP8_FAST_ACCUM,
)
from modeling.fp8_triton import quant_only

# (name, in_features, out_features) at world_size 1. Column-parallel layers shard
# out_features; row-parallel (o_proj, down_proj) shard in_features.
HIDDEN = 3584
INTERMEDIATE = 18944
HEADS, HEAD_DIM, KV_HEADS = 28, 128, 4
LAYERS = [
    ("q_proj",    HIDDEN, HEADS * HEAD_DIM,    "col"),
    ("k_proj",    HIDDEN, KV_HEADS * HEAD_DIM, "col"),
    ("v_proj",    HIDDEN, KV_HEADS * HEAD_DIM, "col"),
    ("o_proj",    HEADS * HEAD_DIM, HIDDEN,    "row"),
    ("gate_proj", HIDDEN, INTERMEDIATE,        "col"),
    ("up_proj",   HIDDEN, INTERMEDIATE,        "col"),
    ("down_proj", INTERMEDIATE, HIDDEN,        "row"),
]


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _gemm_only(scheme, x_q, act_scale, w_q, w_scale):
    if scheme == "int8":
        return int8_mm_dequant(x_q, act_scale, w_q, w_scale)
    return scaled_mm_fp8(x_q, act_scale, w_q, w_scale)


def bench_layer(name, in_f, out_f, kind, world_size, tokens, device, scheme):
    # Apply TP sharding to the contracted/produced dim.
    if kind == "col":
        out_f = out_f // world_size
    else:
        in_f = in_f // world_size

    x = torch.randn(tokens, in_f, device=device, dtype=torch.bfloat16)
    w = torch.randn(out_f, in_f, device=device, dtype=torch.bfloat16) * (in_f ** -0.5)

    bf16 = _time(lambda: torch.nn.functional.linear(x, w))

    if not can_quant(scheme, tokens, in_f):
        print(f"{name:10s} in={in_f:6d} out={out_f:6d}  "
              f"bf16={bf16:.3f}ms   {scheme} skipped (preconditions: K%16 / min-tokens)")
        return

    w_q, w_scale = quantize_weight_rowwise(w, scheme)
    # Use the shipped Triton per-token quant (what the model runs).
    quant = _time(lambda: quant_only(x, scheme))
    full = _time(lambda: _gemm_only(scheme, *quant_only(x, scheme), w_q, w_scale))
    x_q0, as0 = quant_only(x, scheme)
    gemm_only = _time(lambda: _gemm_only(scheme, x_q0, as0, w_q, w_scale))

    s = bf16 / full
    print(f"{name:10s} in={in_f:6d} out={out_f:6d}  "
          f"bf16={bf16:.3f}ms  {scheme}={full:.3f}ms  (gemm={gemm_only:.3f} quant={quant:.3f})  "
          f"s={s:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=1, help="simulate TP sharding of the dims")
    ap.add_argument("--tokens", type=int, default=4096, help="M (latent tokens per step)")
    ap.add_argument("--scheme", choices=["fp8", "int8"], default="fp8")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "run on the H100 box"
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} sm={cap[0]}{cap[1]} torch={torch.__version__}")
    print(f"scheme={args.scheme} M(tokens)={args.tokens} world_size={args.world_size} fast_accum={FP8_FAST_ACCUM}\n")

    for name, in_f, out_f, kind in LAYERS:
        bench_layer(name, in_f, out_f, kind, args.world_size, args.tokens, device, args.scheme)


if __name__ == "__main__":
    main()
