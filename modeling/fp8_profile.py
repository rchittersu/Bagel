# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Opt-in in-run timing for the FP8 quant linears.

Wraps each FP8 projection's forward: when enabled it times the FP8 path (and, in
A/B mode, the bf16 equivalent on the same shapes) with CUDA events, keyed by layer
and token count M. When disabled it is a single bool check -- the linear runs its
normal path with no wrapping. Aggregating by M shows, for the *real* workload, which
layers actually win in FP8 and at which token counts -- the per-call truth the
microbenchmark (tools/fp8_phase0_bench.py) only approximates at a fixed M.

Usage (see infer_tp.py --fp8-profile):
    from modeling import fp8_profile
    fp8_profile.enable(ab=True)
    ... run generation ...
    fp8_profile.report()
"""

import re
from collections import defaultdict

import torch

_ENABLED = False
_AB = False
# (short_name, M, kind, start_event, end_event); elapsed read once at report() time
# to avoid a per-call cuda sync.
_events = []


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


def ab_enabled() -> bool:
    return _AB


def _short(name: str) -> str:
    # Collapse the layer index so all 28 decoder layers aggregate per projection type
    # (keeps the expert distinction, e.g. mlp vs mlp_moe_gen).
    return re.sub(r"^.*layers\.\d+\.", "", name) or name


def _time(name: str, m: int, kind: str, fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    _events.append((_short(name), m, kind, start, end))
    return out


def run(name: str, m: int, fp8_fn, bf16_fn):
    """Time fp8_fn (always) and bf16_fn (A/B mode); return the FP8 result.

    Call site only builds the lambdas when is_enabled(), so the off path is free.
    """
    out = _time(name, m, "quant", fp8_fn)
    if _AB and bf16_fn is not None:
        _time(name, m, "bf16", bf16_fn)
    return out


def report(reset_after: bool = True) -> None:
    """Print per-(tokens, layer) aggregated quant (fp8|int8) vs bf16 time, sorted by tokens.

    Scheme-agnostic: the 'quant ms' column holds whichever scheme actually ran.
    """
    if not _events:
        print("[quant-profile] no events recorded", flush=True)
        return
    torch.cuda.synchronize()  # one sync: all events are now complete

    agg = defaultdict(lambda: {"calls": 0, "quant": 0.0, "bf16": 0.0})
    for name, m, kind, s, e in _events:
        a = agg[(m, name)]
        a[kind] += s.elapsed_time(e)
        if kind == "quant":
            a["calls"] += 1

    print("\n[quant-profile] per-layer time grouped by tokens (totals over the run):", flush=True)
    print(f"{'tokens':>8} {'layer':<26} {'calls':>6} {'quant ms':>9} {'bf16 ms':>9} {'s':>6}", flush=True)
    bucket_tot = defaultdict(lambda: {"quant": 0.0, "bf16": 0.0})
    for (m, name) in sorted(agg.keys()):
        a = agg[(m, name)]
        s = (a["bf16"] / a["quant"]) if (a["bf16"] and a["quant"]) else float("nan")
        print(f"{m:>8} {name:<26} {a['calls']:>6} {a['quant']:>9.2f} {a['bf16']:>9.2f} {s:>6.2f}",
              flush=True)
        bucket_tot[m]["quant"] += a["quant"]
        bucket_tot[m]["bf16"] += a["bf16"]

    print("-" * 70, flush=True)
    for m in sorted(bucket_tot.keys()):
        b = bucket_tot[m]
        s = (b["bf16"] / b["quant"]) if (b["bf16"] and b["quant"]) else float("nan")
        verdict = "WIN" if s > 1 else "loss"
        print(f"{m:>8} {'TOTAL @ this token count':<26} {'':>6} "
              f"{b['quant']:>9.2f} {b['bf16']:>9.2f} {s:>6.2f}  {verdict}", flush=True)

    if reset_after:
        reset()
