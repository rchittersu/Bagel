# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Correctness + speed test for the fused FP8 prologue kernels and the FP8 linear.

    python tools/fp8_kernel_test.py

Validates, on the target H100:
  1. Each Triton prologue kernel matches its PyTorch reference (dequantized result
     within e4m3 tolerance). Required before flipping fp8_triton.USE_TRITON = True.
  2. fp8_w8a8_linear reconstructs F.linear (bf16) with high cosine similarity.
  3. The fused kernels are faster than the naive multi-pass prologue.

Exit code is non-zero if any correctness check fails.
"""

import sys

import torch
import torch.nn.functional as F

from modeling.quant_utils import (
    FP8_MAX, quantize_weight_rowwise, quantize_fp8_rowwise, quant_w8a8_linear,
)
import modeling.fp8_triton as fp8t


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _dequant(x_fp8, scale):
    return x_fp8.float() * scale.float()


def check_kernel(name, triton_fn, ref_fn, *inputs, rel_tol=0.06):
    if not fp8t._HAS_TRITON:
        print(f"[skip] {name}: triton not importable")
        return True
    xq_t, s_t = triton_fn(*inputs)
    xq_r, s_r = ref_fn(*inputs)
    dq_t, dq_r = _dequant(xq_t, s_t), _dequant(xq_r, s_r)
    denom = dq_r.abs().amax().clamp(min=1e-6)
    max_rel = (dq_t - dq_r).abs().amax().item() / denom.item()
    scale_match = torch.allclose(s_t.float(), s_r.float(), rtol=1e-3, atol=1e-6)
    ok = max_rel < rel_tol and scale_match
    print(f"[{'ok' if ok else 'FAIL'}] {name}: max_rel={max_rel:.4f} scale_match={scale_match}")
    return ok


def main():
    assert torch.cuda.is_available(), "run on the H100 box"
    device = torch.device("cuda")
    torch.manual_seed(0)
    ok = True

    M, H, I = 4096, 3584, 18944
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    w_norm = torch.randn(H, device=device, dtype=torch.bfloat16)
    gate = torch.randn(M, I, device=device, dtype=torch.bfloat16)
    up = torch.randn(M, I, device=device, dtype=torch.bfloat16)

    # 1. Triton kernels vs reference.
    ok &= check_kernel("rmsnorm_fp8_quant", fp8t.rmsnorm_fp8_quant_triton,
                       fp8t.rmsnorm_fp8_quant_ref, x, w_norm, 1e-6)
    ok &= check_kernel("silumul_fp8_quant", fp8t.silumul_fp8_quant_triton,
                       fp8t.silumul_fp8_quant_ref, gate, up)
    ok &= check_kernel("quant_fp8_only", fp8t.quant_fp8_only_triton,
                       fp8t.quant_fp8_only_ref, x)

    # 1b. INT8 Triton kernels vs reference.
    ok &= check_kernel("quant_int8_only", fp8t.quant_int8_only_triton,
                       fp8t.quant_int8_only_ref, x)
    ok &= check_kernel("silumul_int8_quant", fp8t.silumul_int8_quant_triton,
                       fp8t.silumul_int8_quant_ref, gate, up)
    if fp8t._HAS_TRITON:
        acc = torch.randint(-(1 << 20), 1 << 20, (M, H), device=device, dtype=torch.int32)
        a_s = torch.rand(M, 1, device=device) * 1e-3
        w_s = torch.rand(1, H, device=device) * 1e-3
        d_t = fp8t.dequant_int32_triton(acc, a_s, w_s)
        d_r = fp8t.dequant_int32_ref(acc, a_s, w_s)
        deq_ok = torch.allclose(d_t.float(), d_r.float(), rtol=1e-3, atol=1e-4)
        ok &= deq_ok
        print(f"[{'ok' if deq_ok else 'FAIL'}] dequant_int32 vs ref: "
              f"max_abs={(d_t.float() - d_r.float()).abs().amax().item():.2e}")

    # 2. FP8 / INT8 linear vs bf16 F.linear.
    w = torch.randn(H, H, device=device, dtype=torch.bfloat16) * (H ** -0.5)
    y_ref = F.linear(x, w)
    for scheme in ("fp8", "int8"):
        w_q, w_scale = quantize_weight_rowwise(w, scheme)
        y_q = quant_w8a8_linear(x, w_q, w_scale, scheme)
        cos = _cos(y_ref, y_q)
        lin_ok = cos > 0.99
        ok &= lin_ok
        print(f"[{'ok' if lin_ok else 'FAIL'}] {scheme}_w8a8_linear vs F.linear: cos={cos:.5f}")

    # 3. Fused vs naive prologue speed (informational).
    if fp8t._HAS_TRITON:
        def naive():
            gf = gate.float()
            act = (gf * torch.sigmoid(gf)) * up.float()
            return quantize_fp8_rowwise(act)
        def _t(fn, it=50):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            for _ in range(it):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / it
        print(f"[time] silumul prologue: fused={_t(lambda: fp8t.silumul_fp8_quant_triton(gate, up)):.3f}ms "
              f"naive={_t(naive):.3f}ms")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
