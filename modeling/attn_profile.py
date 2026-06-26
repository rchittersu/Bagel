# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Opt-in in-run A/B timing for the attention backends (flash vs sage).

When enabled, ``attn_dispatch.attn_varlen_func`` times BOTH backends on the same
inputs per call (CUDA events) and returns the active backend's result. Disabled ->
a single bool check and the normal dispatch.

Keyed by (q_len, kv_len): this is NOT self-attention -- in the denoise loop the
latent query attends to the frozen context KV, so kv_len >> q_len and varies per
call. Showing both reveals whether cost tracks q_len*kv_len (flash) or is dominated
by K/V quantization that scales with kv_len (sage).

If the non-active backend raises (e.g. sageattention not installed), A/B silently
degrades to timing only the active backend.
"""

from collections import defaultdict

import torch

_ENABLED = False
_AB = True
_ab_broken = False
_events = []  # (q_len, kv_len, backend_name, start_event, end_event)


def enable(ab: bool = True) -> None:
    global _ENABLED, _AB
    _ENABLED = True
    _AB = ab


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def reset() -> None:
    _events.clear()


def is_enabled() -> bool:
    return _ENABLED


def _time(q_len, kv_len, name, fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    _events.append((q_len, kv_len, name, start, end))
    return out


def run(q_len, kv_len, primary_name, primary_fn, other_name, other_fn):
    """Time the primary backend (result returned) and, in A/B mode, the other one."""
    global _ab_broken
    out = _time(q_len, kv_len, primary_name, primary_fn)
    if _AB and not _ab_broken and other_fn is not None:
        try:
            _time(q_len, kv_len, other_name, other_fn)
        except Exception as ex:  # secondary backend unavailable -> stop A/B, keep primary
            _ab_broken = True
            if _events and _events[-1][2] == other_name:
                _events.pop()
            print(f"[attn-profile] A/B disabled: {other_name} backend failed ({ex})", flush=True)
    return out


def report(reset_after: bool = True) -> None:
    if not _events:
        print("[attn-profile] no events recorded", flush=True)
        return
    torch.cuda.synchronize()

    agg = defaultdict(lambda: {"calls": 0, "ms": 0.0})
    for q_len, kv_len, name, s, e in _events:
        a = agg[(q_len, kv_len, name)]
        a["calls"] += 1
        a["ms"] += s.elapsed_time(e)

    print("\n[attn-profile] attention time by (q_len, kv_len) -- totals over the run:", flush=True)
    print(f"{'q_len':>7} {'kv_len':>7} {'kv/q':>5} {'backend':<8} {'calls':>6} {'total ms':>10} {'avg ms':>9}",
          flush=True)
    per_shape = defaultdict(dict)
    for (q_len, kv_len, name) in sorted(agg.keys()):
        a = agg[(q_len, kv_len, name)]
        ratio = kv_len / max(q_len, 1)
        print(f"{q_len:>7} {kv_len:>7} {ratio:>5.0f} {name:<8} {a['calls']:>6} "
              f"{a['ms']:>10.2f} {a['ms'] / max(a['calls'], 1):>9.4f}", flush=True)
        per_shape[(q_len, kv_len)][name] = a["ms"]

    print("-" * 64, flush=True)
    for (q_len, kv_len) in sorted(per_shape):
        d = per_shape[(q_len, kv_len)]
        if "flash" in d and "sage" in d and d["sage"] > 0:
            sp = d["flash"] / d["sage"]
            print(f"q={q_len:>6} kv={kv_len:>6}  sage vs flash: {sp:.2f}x  "
                  f"({'sage faster' if sp > 1 else 'sage slower'})", flush=True)

    tot = defaultdict(float)
    for (q_len, kv_len, name), a in agg.items():
        tot[name] += a["ms"]
    print("-" * 64, flush=True)
    for name in sorted(tot):
        print(f"  total {name}: {tot[name]:.1f} ms", flush=True)

    if reset_after:
        reset()
