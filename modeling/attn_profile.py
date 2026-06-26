# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Opt-in in-run A/B timing for the attention backends (flash vs sage).

When enabled, ``attn_dispatch.attn_varlen_func`` times BOTH backends on the same
inputs per call (CUDA events, keyed by query-token count) and returns the active
backend's result. Disabled -> a single bool check and the normal dispatch. One run
then prints flash vs sage per token bucket with the speedup, so the real token
distribution (prefill / CFG / denoise) drives the comparison.

If the non-active backend raises (e.g. sageattention not installed), A/B silently
degrades to timing only the active backend.
"""

from collections import defaultdict

import torch

_ENABLED = False
_AB = True
_ab_broken = False  # set if the secondary backend errors once
_events = []  # (m, backend_name, start_event, end_event)


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


def _time(m: int, name: str, fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    _events.append((m, name, start, end))
    return out


def run(m, primary_name, primary_fn, other_name, other_fn):
    """Time the primary backend (result returned) and, in A/B mode, the other one."""
    global _ab_broken
    out = _time(m, primary_name, primary_fn)
    if _AB and not _ab_broken and other_fn is not None:
        try:
            _time(m, other_name, other_fn)
        except Exception as ex:  # secondary backend unavailable -> stop A/B, keep primary
            _ab_broken = True
            _events.pop() if _events and _events[-1][1] == other_name else None
            print(f"[attn-profile] A/B disabled: {other_name} backend failed ({ex})", flush=True)
    return out


def report(reset_after: bool = True) -> None:
    if not _events:
        print("[attn-profile] no events recorded", flush=True)
        return
    torch.cuda.synchronize()

    agg = defaultdict(lambda: {"calls": 0, "ms": 0.0})
    for m, name, s, e in _events:
        a = agg[(m, name)]
        a["calls"] += 1
        a["ms"] += s.elapsed_time(e)

    print("\n[attn-profile] attention time grouped by tokens (totals over the run):", flush=True)
    print(f"{'tokens':>8} {'backend':<8} {'calls':>6} {'total ms':>10} {'avg ms':>9}", flush=True)
    per_m = defaultdict(dict)
    for (m, name) in sorted(agg.keys()):
        a = agg[(m, name)]
        print(f"{m:>8} {name:<8} {a['calls']:>6} {a['ms']:>10.2f} {a['ms'] / max(a['calls'], 1):>9.4f}",
              flush=True)
        per_m[m][name] = a["ms"]

    print("-" * 55, flush=True)
    for m in sorted(per_m):
        d = per_m[m]
        if "flash" in d and "sage" in d and d["sage"] > 0:
            sp = d["flash"] / d["sage"]
            print(f"{m:>8} sage vs flash: {sp:.2f}x  "
                  f"({'sage faster' if sp > 1 else 'sage slower'})", flush=True)

    tot = defaultdict(float)
    for (m, name), a in agg.items():
        tot[name] += a["ms"]
    print("-" * 55, flush=True)
    for name in sorted(tot):
        print(f"  total {name}: {tot[name]:.1f} ms", flush=True)

    if reset_after:
        reset()
