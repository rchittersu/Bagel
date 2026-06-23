# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel inference entrypoint for BAGEL.

Launch with torchrun, one process per GPU:

    torchrun --nproc_per_node=4 infer_tp.py \
        --model_path models/BAGEL-7B-MoT \
        --prompt "a teapot shaped like a cat, studio photo" \
        --output out.png

All ranks run the same code (SPMD). The 28 LLM decoder layers are tensor-
sharded across the GPUs (see modeling/tp_utils.py); ViT / VAE / embeddings
are replicated. The Bagel model is constructed with TP-local parameter shapes
and the checkpoint is sliced per-rank on load, so no rank ever holds the full
weights. The reconstructed activations are identical on every rank, so it is
safe for rank 0 to be the one that saves the output image.

Falls back to a normal single-GPU run when launched without torchrun
(WORLD_SIZE unset / == 1).
"""

import argparse
import contextlib
import os
import random
import time

import numpy as np
import torch

from accelerate import init_empty_weights  # noqa: F401  (kept for parity / debugging)

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.tp_utils import (
    init_tensor_parallel, get_tp_rank, get_tp_world_size, load_sharded_checkpoint,
    tp_barrier,
)
from modeling.qwen2 import Qwen2Tokenizer


def set_seed(seed):
    """Seed every rank identically so replicated (noise / sampling) computations match."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_model(model_path, device, dtype):
    """Construct the Bagel model with TP-local shapes directly on `device`.

    We build on the real device (not meta) so that non-checkpoint buffers such as
    the rotary embedding inv_freq are materialised correctly. Parallel linears are
    already local-sized, so per-rank memory is ~1/world_size of the LLM weights.
    """
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=False)
    finally:
        torch.set_default_dtype(prev_default_dtype)

    return model, vae_model, config


def load_inferencer(model_path, device, dtype=torch.bfloat16):
    """Build the TP-sharded Bagel model on `device`, load weights, and wrap it in
    an InterleaveInferencer. Reused by both the single-prompt entrypoint and the
    dataset edit driver. `init_tensor_parallel()` must have been called first.
    """
    model, vae_model, config = build_model(model_path, device, dtype)

    # Slice + load the checkpoint into this rank's local shards.
    load_sharded_checkpoint(model, os.path.join(model_path, "ema.safetensors"))
    model = model.eval()

    # Keep the VAE in its loaded precision; the inferencer autocasts to bf16 for compute.
    vae_model = vae_model.to(device).eval()

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
    parser.add_argument("--prompt", type=str, default="a teapot shaped like a cat, studio photo")
    parser.add_argument("--output", type=str, default="out_tp.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true", help="time the generation and report on rank 0")
    args = parser.parse_args()

    local_rank = init_tensor_parallel()
    rank = get_tp_rank()
    world_size = get_tp_world_size()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[TP] world_size={world_size}, building model on each rank ...")

    inferencer = load_inferencer(args.model_path, device, dtype)

    # Identical seed on every rank -> identical replicated noise / sampling.
    set_seed(args.seed)
    tp_barrier()

    log(f"[TP] generating: {args.prompt!r}")
    if args.benchmark:
        torch.cuda.synchronize()
        tp_barrier()
        t0 = time.time()

    output = inferencer(
        text=args.prompt,
        think=False,
        image_shapes=(args.image_size, args.image_size),
        num_timesteps=args.num_timesteps,
        cfg_text_scale=args.cfg_text_scale,
        understanding_output=False,
    )

    if args.benchmark:
        torch.cuda.synchronize()
        tp_barrier()
        if rank == 0:
            print(f"[TP] generation took {time.time() - t0:.2f}s "
                  f"({args.num_timesteps} steps, world_size={world_size})", flush=True)

    if rank == 0:
        image = output["image"]
        image.save(args.output)
        print(f"[TP] saved -> {args.output}", flush=True)

    tp_barrier()


if __name__ == "__main__":
    main()
