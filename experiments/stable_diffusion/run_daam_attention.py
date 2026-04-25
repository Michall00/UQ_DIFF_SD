"""
experiments/stable_diffusion/run_daam_attention.py
--------------------------------------------------
DAAM cross-attention heat maps for attention-weighted uncertainty aggregation.

This script intentionally runs outside the main Stable Diffusion UQ environment:
the upstream DAAM package pins older diffusers/transformers versions. Use the
Makefile target, which supplies a DAAM-compatible dependency set via `uv run`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sd_uq.aggregation import attention_weighted_scores, normalize_attention_map
from sd_uq.plotting import save_heatmap_png


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="a human hand with five fingers")
    p.add_argument("--words", type=str, default="hand,fingers")
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_offset", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    p.add_argument(
        "--uncertainty_npz",
        type=str,
        default="",
        help="Optional laplace_results.npz to aggregate with DAAM maps.",
    )
    p.add_argument(
        "--uncertainty_keys",
        type=str,
        default="var_maps",
        help=(
            "Comma-separated uncertainty map arrays inside --uncertainty_npz, "
            "for example var_maps,bayesdiff_var_maps."
        ),
    )
    p.add_argument("--out_dir", type=str, default="assets/stable_diffusion/daam")
    p.add_argument("--save_images", action="store_true")
    return p.parse_args()


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def parse_words(raw: str) -> list[str]:
    words = [word.strip() for word in raw.split(",") if word.strip()]
    if not words:
        raise ValueError("Could not infer DAAM words from prompt. Pass --words explicitly.")
    return words


def parse_uncertainty_keys(raw: str) -> list[str]:
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if not keys:
        raise ValueError("--uncertainty_keys must contain at least one key")
    return keys


def daam_prefix_for_key(key: str) -> str:
    if key == "var_maps":
        return "var"
    if key.endswith("_maps"):
        return key[:-5]
    return key


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    words = parse_words(args.words)
    uncertainty_keys = parse_uncertainty_keys(args.uncertainty_keys)

    try:
        from daam import trace
        from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
    except ImportError as exc:
        raise SystemExit(
            "DAAM dependencies are missing. Run through the Makefile target, e.g. "
            "`make sd-daam-ddim`, which uses a DAAM-compatible uv environment."
        ) from exc

    dtype = resolve_torch_dtype(args.torch_dtype, args.device)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    scheduler_cls = DDIMScheduler if args.scheduler == "ddim" else DDPMScheduler
    pipe.scheduler = scheduler_cls.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = pipe.to(args.device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    uncertainty = None
    if args.uncertainty_npz:
        uncertainty = np.load(args.uncertainty_npz)
        for key in uncertainty_keys:
            if key not in uncertainty:
                raise KeyError(
                    f"{key!r} not found in {args.uncertainty_npz}. "
                    f"Available keys: {list(uncertainty.keys())}"
                )
        args.n_samples = min(
            args.n_samples,
            min(int(uncertainty[key].shape[0]) for key in uncertainty_keys),
        )

    daam_maps = []
    weighted = {
        key: {"mean": [], "sum": []}
        for key in uncertainty_keys
    }
    seeds = []

    for i in range(args.n_samples):
        seed = args.seed + args.seed_offset + i
        seeds.append(seed)
        generator = torch.Generator(device=args.device).manual_seed(seed)

        print(f"DAAM sample {i + 1}/{args.n_samples} seed={seed} words={words}")
        with torch.no_grad():
            with trace(pipe) as tc:
                out = pipe(
                    args.prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                )
                try:
                    global_heat_map = tc.compute_global_heat_map(args.prompt, normalize=True)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "DAAM did not capture any cross-attention maps. This can happen "
                        "with tiny/testing pipelines or unsupported diffusers model "
                        "internals. Use a standard SD v1.x checkpoint such as "
                        "CompVis/stable-diffusion-v1-4 or runwayml/stable-diffusion-v1-5."
                    ) from exc

        word_maps = []
        for word in words:
            try:
                word_maps.append(global_heat_map.compute_word_heat_map(word).value)
            except ValueError as exc:
                print(f"  Skipping DAAM word {word!r}: {exc}")
        if not word_maps:
            raise RuntimeError(f"No DAAM maps could be computed for words: {words}")

        attn_map = normalize_attention_map(torch.stack(word_maps).mean(dim=0))
        daam_maps.append(attn_map)

        if args.save_images:
            out.images[0].save(Path(args.out_dir) / f"sample_{i:03d}.png")
            save_heatmap_png(attn_map, str(Path(args.out_dir) / f"daam_{i:03d}.png"))

        if uncertainty is not None:
            for key in uncertainty_keys:
                mean_i, sum_i = attention_weighted_scores(uncertainty[key][i], attn_map)
                weighted[key]["mean"].append(mean_i)
                weighted[key]["sum"].append(sum_i)

    result = {
        "daam_maps": np.stack(daam_maps).astype(np.float32),
        "daam_words": np.array(words),
        "prompt": np.array(args.prompt),
        "seeds": np.array(seeds, dtype=np.int64),
    }
    if uncertainty is not None:
        result["uncertainty_keys"] = np.array(uncertainty_keys)
        for key in uncertainty_keys:
            prefix = daam_prefix_for_key(key)
            result[f"daam_{prefix}_mean"] = np.array(weighted[key]["mean"], dtype=np.float32)
            result[f"daam_{prefix}_sum"] = np.array(weighted[key]["sum"], dtype=np.float32)
            plain_mean_key = f"{prefix}_mean"
            if plain_mean_key in uncertainty:
                result[f"plain_{prefix}_mean"] = uncertainty[plain_mean_key][: args.n_samples].astype(np.float32)

    out_path = Path(args.out_dir) / "daam_results.npz"
    np.savez_compressed(out_path, **result)
    print(f"Saved DAAM results to {out_path}")


if __name__ == "__main__":
    main()
