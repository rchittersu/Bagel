# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel edit-dataset inference for BAGEL.

Runs an image-editing dataset through the tensor-parallel model. Unlike the
data-parallel eval scripts (eval/gen/gen_images_mp_imgedit.py), here ALL ranks
collaborate on ONE edit at a time (SPMD lockstep), so each edit is faster but the
dataset is walked sequentially.

Lockstep contract (important): every rank builds the SAME dataset, iterates it in
the SAME order, and runs every sample together so the per-layer all-reduces stay
in sync. Do NOT use a DistributedSampler / shard the dataset -- that desyncs the
collectives and will hang. Only rank 0 writes outputs.

Launch (one process per GPU):

    torchrun --nproc_per_node=4 infer_tp_edit.py \
        --model_path models/BAGEL-7B-MoT \
        --output_dir edits_out \
        --cfg_img_scale 2.0 --cfg_text_scale 4.0

Plug your dataset into `build_dataset()` below. It is assumed to be a map-style
dataset (supports len() and dataset[i]) whose items yield (text, image), where
`text` is the edit instruction and `image` is a PIL.Image or a path to one.
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

from data.data_utils import pil_img2rgb
from infer_tp import load_inferencer, set_seed
from modeling.tp_utils import (
    init_tensor_parallel, get_tp_rank, get_tp_world_size, tp_barrier,
)


# --------------------------------------------------------------------------- #
# Plug your dataset in here.
# --------------------------------------------------------------------------- #
def build_dataset(args):
    """Return a map-style dataset where dataset[i] -> (text, image).

    Replace the body with the construction of YOUR dataset class. It must be
    built identically on every rank (same args, same order) -- no random shuffle
    unless seeded identically. Example:

        from your_module import EditDataset
        return EditDataset(args.data_root, split="test")
    """
    raise NotImplementedError(
        "Edit build_dataset() to construct and return your edit dataset "
        "(map-style, dataset[i] -> (text, image))."
    )


def normalize_item(item):
    """Coerce a dataset item into (instruction_text, PIL.Image).

    Accepts (text, image) or (image, text); `image` may be a PIL.Image or a path.
    """
    a, b = item
    # Figure out which element is the image.
    if isinstance(a, Image.Image) or (isinstance(a, str) and os.path.exists(a)):
        image, text = a, b
    else:
        text, image = a, b
    if isinstance(image, str):
        image = Image.open(image)
    image = pil_img2rgb(image)
    return text, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
    parser.add_argument("--output_dir", type=str, default="edits_out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1, help="only process the first N items (debug)")
    parser.add_argument("--resume", action="store_true", help="skip items whose output already exists")
    parser.add_argument("--think", action="store_true", help="enable chain-of-thought editing")
    # Edit CFG defaults mirror eval/gen/gen_images_mp_imgedit.py.
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--fp8", action="store_true",
                        help="run the large LLM matmuls in FP8 W8A8 (e4m3) on Hopper")
    parser.add_argument("--fp8-include-qkv", dest="fp8_include_qkv", action="store_true",
                        help="also quantize attention q/k/v (only helps at low TP; "
                             "measured a net loss at TP>=4, so off by default)")
    parser.add_argument("--fp8-skip-down", action="store_true",
                        help="keep the outlier-prone MLP down_proj in bf16 when --fp8 is set")
    parser.add_argument("--fp8-min-tokens", type=int, default=None,
                        help="token-count gate below which FP8 linears fall back to bf16 "
                             "(default 2048, calibrated from the in-run profiler)")
    args = parser.parse_args()

    local_rank = init_tensor_parallel()
    rank = get_tp_rank()
    world_size = get_tp_world_size()
    device = torch.device(f"cuda:{local_rank}")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[TP-edit] world_size={world_size}, building model on each rank ...")
    inferencer = load_inferencer(
        args.model_path, device, torch.bfloat16,
        fp8=args.fp8, fp8_include_qkv=args.fp8_include_qkv, fp8_skip_down=args.fp8_skip_down,
        fp8_min_tokens=args.fp8_min_tokens,
    )

    dataset = build_dataset(args)
    n = len(dataset)
    if args.limit >= 0:
        n = min(n, args.limit)
    log(f"[TP-edit] dataset has {len(dataset)} items, processing {n}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        manifest = open(os.path.join(args.output_dir, "manifest.jsonl"), "a")

    tp_barrier()
    t0 = time.time()
    done = 0

    for idx in range(n):
        out_path = os.path.join(args.output_dir, f"{idx:06d}.png")
        # Deterministic skip: identical on every rank -> stays in lockstep.
        if args.resume and os.path.exists(out_path):
            continue

        # Re-seed per sample BEFORE the fetch so any randomness in __getitem__
        # (augmentation, etc.) AND the diffusion noise are reproducible and
        # identical across ranks. Divergent inputs across ranks would desync the
        # tensor-parallel all-reduces.
        set_seed(args.seed + idx)
        text, image = normalize_item(dataset[idx])

        output = inferencer(
            image=image,
            text=text,
            think=args.think,
            num_timesteps=args.num_timesteps,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
            timestep_shift=args.timestep_shift,
            understanding_output=False,
        )

        if rank == 0:
            output["image"].save(out_path)
            record = {"index": idx, "prompt": text, "output": os.path.basename(out_path)}
            if output.get("text"):
                record["think"] = output["text"]
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

        done += 1
        if done % 10 == 0:
            log(f"[TP-edit] {done}/{n} done ({(time.time() - t0) / done:.2f}s/item)")

    tp_barrier()
    if rank == 0:
        manifest.close()
        print(f"[TP-edit] finished {done} items in {time.time() - t0:.1f}s -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
