"""
Run Stable Diffusion UQ with FLARE + BayesDiff maps, then aggregate them with DAAM.

This is an orchestration script: the DAAM package needs a separate dependency
set, so the second stage is launched through `uv run --with daam==0.2.0 ...`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_z0", type=int, default=4)
    p.add_argument("--n_lap_pairs", type=int, default=100)
    p.add_argument("--laplace_mode", choices=["last_layer", "subnet"], default="last_layer")
    p.add_argument("--subnet_n_params", type=int, default=50_000)
    p.add_argument("--subnet_max_tensors", type=int, default=12)
    p.add_argument("--subnet_mc_samples", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    p.add_argument("--plot_max_samples", type=int, default=16)
    p.add_argument("--out_dir", type=str, default="assets/stable_diffusion/daam_bayesdiff")
    p.add_argument("--daam_python", type=str, default="3.11")
    p.add_argument(
        "--daam_with",
        nargs="*",
        default=["daam==0.2.0", "huggingface-hub==0.17.3"],
        help="Additional uv --with packages for the DAAM stage.",
    )
    p.add_argument("--save_daam_images", action="store_true")
    p.add_argument("--skip_uq", action="store_true", help="Reuse existing laplace_results.npz.")
    p.add_argument("--skip_daam", action="store_true", help="Only run the UQ stage.")
    return p.parse_args()


def run(cmd: list[str]):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    uq_dir = out_dir / "uq"
    daam_dir = out_dir / "daam"
    uq_dir.mkdir(parents=True, exist_ok=True)
    daam_dir.mkdir(parents=True, exist_ok=True)

    sd_script = Path(__file__).with_name("run_sd_laplace.py")
    daam_script = Path(__file__).with_name("run_daam_attention.py")
    uncertainty_npz = uq_dir / "laplace_results.npz"

    if not args.skip_uq:
        uq_cmd = [
            sys.executable,
            str(sd_script),
            "--model_id",
            args.model_id,
            "--prompt",
            args.prompt,
            "--device",
            args.device,
            "--torch_dtype",
            args.torch_dtype,
            "--scheduler",
            args.scheduler,
            "--laplace_mode",
            args.laplace_mode,
            "--uq_method",
            "both",
            "--steps",
            str(args.steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--n_z0",
            str(args.n_z0),
            "--n_lap_pairs",
            str(args.n_lap_pairs),
            "--n_samples",
            str(args.n_samples),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--plot_max_samples",
            str(args.plot_max_samples),
            "--seed",
            str(args.seed),
            "--out_dir",
            str(uq_dir),
        ]
        if args.laplace_mode == "subnet":
            uq_cmd.extend(
                [
                    "--subnet_n_params",
                    str(args.subnet_n_params),
                    "--subnet_max_tensors",
                    str(args.subnet_max_tensors),
                    "--subnet_mc_samples",
                    str(args.subnet_mc_samples),
                ]
            )
        run(uq_cmd)

    if args.skip_daam:
        return

    daam_cmd = ["uv", "run", "--python", args.daam_python]
    for package in args.daam_with:
        daam_cmd.extend(["--with", package])
    daam_cmd.extend(
        [
            "python",
            str(daam_script),
            "--model_id",
            args.model_id,
            "--prompt",
            args.prompt,
            "--words",
            args.words,
            "--device",
            args.device,
            "--torch_dtype",
            args.torch_dtype,
            "--scheduler",
            args.scheduler,
            "--steps",
            str(args.steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--n_samples",
            str(args.n_samples),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--seed",
            str(args.seed),
            "--uncertainty_npz",
            str(uncertainty_npz),
            "--uncertainty_keys",
            "var_maps,bayesdiff_var_maps",
            "--out_dir",
            str(daam_dir),
        ]
    )
    if args.save_daam_images:
        daam_cmd.append("--save_images")
    run(daam_cmd)

    print(f"UQ results: {uncertainty_npz}")
    print(f"DAAM results: {daam_dir / 'daam_results.npz'}")


if __name__ == "__main__":
    main()
