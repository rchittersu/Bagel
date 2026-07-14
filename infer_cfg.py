# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
CFG-parallel inference for BAGEL: one classifier-free-guidance branch per process.

Instead of tensor-parallel sharding each matmul (2 all-reduces/layer), each rank runs
ONE full, unsharded CFG branch on its own KV cache, and the branches sync only once per
denoise step (an all_gather of v_t). This removes the per-layer collectives, makes every
matmul full-size (so FP8 wins more and torch.compile fuses the whole transformer), and
fits since a 7B model is ~7GB (FP8) / 14GB (bf16) per GPU.

Branch <-> rank mapping (must match): rank 0 = main (cond), 1 = cfg_text, 2 = cfg_img.
World size = number of active branches (3 for image editing with text+img guidance; 2 for
text-only CFG, with --cfg_img_scale 1). Tensor parallelism is OFF.

    torchrun --nproc_per_node=3 infer_cfg.py \
        --model_path models/BAGEL-7B-MoT \
        --image test_images/women.jpg \
        --prompt "make the background a snowy mountain at sunset" \
        --output out_cfg.png
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from PIL import Image

from data.data_utils import pil_img2rgb
from infer_tp import load_inferencer, set_seed


def init_cfg_parallel():
    """Init the NCCL group for the CFG branches. Does NOT touch tensor-parallel state, so
    each rank builds a full (unsharded) model."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
    parser.add_argument("--image", type=str, default=None, help="input image for editing (omit for t2i)")
    parser.add_argument("--prompt", type=str, default="a teapot shaped like a cat, studio photo")
    parser.add_argument("--output", type=str, default="out_cfg.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=1024, help="t2i output size when --image is omitted")
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--no-vit", action="store_true", help="skip ViT encode of the input image")
    # Quantization (applies to the full-size linears -- FP8/INT8 win more here than under TP).
    parser.add_argument("--quant", choices=["fp8", "int8"], default=None)
    parser.add_argument("--fp8", action="store_true", help="shorthand for --quant fp8")
    parser.add_argument("--int8", action="store_true", help="shorthand for --quant int8")
    parser.add_argument("--quant-include-qkv", dest="quant_include_qkv", action="store_true")
    parser.add_argument("--quant-skip-down", dest="quant_skip_down", action="store_true")
    parser.add_argument("--quant-min-tokens", dest="quant_min_tokens", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=None,
                        help="untimed warmup gens before measuring (default 3 with --benchmark)")
    args = parser.parse_args()

    rank, local_rank, world_size = init_cfg_parallel()
    device = torch.device(f"cuda:{local_rank}")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    if world_size not in (2, 3):
        raise SystemExit(f"CFG-parallel expects world size 2 (t2i) or 3 (edit), got {world_size}")

    log(f"[CFG] world_size={world_size} (rank {rank} = branch {rank}); TP off, full model per rank")

    quant = args.quant or ("fp8" if args.fp8 else "int8" if args.int8 else None)
    inferencer = load_inferencer(
        args.model_path, device, torch.bfloat16,
        quant=quant, include_qkv=args.quant_include_qkv, skip_down=args.quant_skip_down,
        min_tokens=args.quant_min_tokens,
    )

    image = pil_img2rgb(Image.open(args.image)) if args.image else None

    def generate():
        # Identical seed on every rank so the (broadcast) init noise and sampling match.
        set_seed(args.seed)
        return inferencer.gen_image_cfg_parallel(
            branch=rank, cfg_group=None, num_branches=world_size,
            image=image, text=args.prompt,
            image_shape=None if image is not None else (args.image_size, args.image_size),
            decode=(rank == 0), use_vit=not args.no_vit,
            cfg_text_scale=args.cfg_text_scale, cfg_img_scale=args.cfg_img_scale,
            cfg_interval=(0.4, 1.0), cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
            num_timesteps=args.num_timesteps, timestep_shift=args.timestep_shift,
        )

    n_warmup = args.warmup if args.warmup is not None else (3 if args.benchmark else 0)
    for i in range(n_warmup):
        log(f"[CFG] warmup {i + 1}/{n_warmup} ...")
        generate()
    if n_warmup:
        torch.cuda.synchronize()
        dist.barrier()

    log(f"[CFG] generating: {args.prompt!r}")
    if args.benchmark:
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()

    out = generate()

    if args.benchmark:
        torch.cuda.synchronize()
        dist.barrier()
        log(f"[CFG] generation took {time.time() - t0:.2f}s "
            f"({args.num_timesteps} steps, world_size={world_size})")

    if rank == 0:
        out.save(args.output)
        print(f"[CFG] saved -> {args.output}", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
