# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Attention backend dispatch for BAGEL's packed/varlen attention.

Selects the backend from the ``BAGEL_ATTN_BACKEND`` environment variable:
  - ``sage``                 -> SageAttention (quantized)
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

# Resolved once at import; override at runtime with set_attn_backend().
_BACKEND = os.environ.get("BAGEL_ATTN_BACKEND", "flash").lower()


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


def _sage_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    # Imported lazily so the default (flash) path has no sageattention dependency.
    from sageattention import sageattn_varlen

    # GQA: FlashAttention does grouped KV natively, but some SageAttention builds
    # require equal q / kv heads. Expand k/v to match q when they differ. If your
    # sage build handles GQA natively, drop this block.
    hq, hkv = q.shape[-2], k.shape[-2]
    if hq != hkv:
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
    if _BACKEND == "sage":
        return _sage_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal)
    return _flash_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal)
