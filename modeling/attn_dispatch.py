# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Attention backend dispatch for BAGEL's packed/varlen attention.

Selects the backend from the ``BAGEL_ATTN_BACKEND`` environment variable:
  - ``sage``                 -> SageAttention (quantized)
  - ``flash3``               -> FlashAttention-3 (Hopper-native; install separately as
                                ``flash_attn_interface``, the flash-attention hopper/ build)
  - anything else / unset    -> the existing FlashAttention-2 path (default)

All call sites (LLM ``qwen2_navit`` + ViT ``siglip_navit``) go through
``attn_varlen_func`` with the FlashAttention varlen convention, so swapping
backends is a single env var with no code change. q/k/v are packed
``[total_tokens, n_heads, head_dim]``; GQA is allowed (k/v may have fewer heads).

The Sage path below is a best-effort against the common ``sageattn_varlen`` API;
adjust ``_sage_varlen`` to match the exact signature of your installed
``sageattention`` build (arg order, GQA support, tensor layout).
"""

import os

from flash_attn import flash_attn_varlen_func

from modeling import attn_profile

# Resolved once at import; override at runtime with set_attn_backend().
_BACKEND = os.environ.get("BAGEL_ATTN_BACKEND", "flash").lower()

# Expand grouped KV heads to match q before sage (only needed if sage lacks native GQA).
# Off by default: the expand is a 7x copy of the frozen context KV every call.
_SAGE_EXPAND_GQA = os.environ.get("BAGEL_SAGE_EXPAND_GQA", "0") == "1"


def get_attn_backend() -> str:
    """Return the active backend name: "sage" or "flash"."""
    return "sage" if _BACKEND == "sage" else "flash"


def set_attn_backend(name: str) -> None:
    """Override the backend at runtime (else BAGEL_ATTN_BACKEND decides)."""
    global _BACKEND
    _BACKEND = (name or "flash").lower()


def _flash_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    return flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )


def _flash3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    # FlashAttention-3 (Hopper). Installed separately as ``flash_attn_interface`` (the
    # flash-attention repo's hopper/ build). FP16/BF16 is exact; FP8 is a separate opt-in.
    from flash_attn_interface import flash_attn_varlen_func as fa3_varlen

    out = fa3_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal=causal,
    )
    # FA3 returns (out, softmax_lse) on some versions; take the output tensor.
    return out[0] if isinstance(out, tuple) else out


def _sage_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    # Imported lazily so the default (flash) path has no sageattention dependency.
    from sageattention import sageattn_varlen

    # GQA: FlashAttention does grouped KV natively. Only expand k/v to match q if the
    # installed sage build can't (BAGEL_SAGE_EXPAND_GQA=1). The expand is a 7x KV copy
    # of the (frozen) context every call, so leave it OFF when sage handles GQA.
    hq, hkv = q.shape[-2], k.shape[-2]
    if _SAGE_EXPAND_GQA and hq != hkv:
        rep = hq // hkv
        k = k.repeat_interleave(rep, dim=-2)
        v = v.repeat_interleave(rep, dim=-2)

    # Packed varlen layout is [total_tokens, n_heads, head_dim] (same as flash).
    # NOTE: conform arg order / names to your installed sageattn_varlen signature.
    return sageattn_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        is_causal=causal,
    )


def attn_varlen_func(
    q, k, v, *,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=False,
):
    """Unified packed/varlen attention -> ``[total_tokens, n_heads, head_dim]``.

    Dispatches to Sage or FlashAttention per ``BAGEL_ATTN_BACKEND``. q/k/v packed
    ``[total_tokens, n_heads, head_dim]``; GQA allowed.
    """
    args = (q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal)
    backends = {
        "flash": lambda: _flash_varlen(*args),
        "flash3": lambda: _flash3_varlen(*args),
        "sage": lambda: _sage_varlen(*args),
    }
    primary_name = _BACKEND if _BACKEND in backends else "flash"

    if not attn_profile.is_enabled():
        return backends[primary_name]()

    # A/B: time the active backend and a baseline on the same inputs; return the active
    # result. Baseline = FA2-flash (the reference), or sage when flash is active.
    # Key by (q_len, kv_len) -- attention is q->context cross-attn, kv_len >> q_len.
    q_len, kv_len = int(q.shape[0]), int(k.shape[0])
    other_name = "flash" if primary_name != "flash" else "sage"
    return attn_profile.run(
        q_len, kv_len,
        primary_name, backends[primary_name],
        other_name, backends[other_name],
    )
