"""
Run Stable Diffusion UQ experiments for a JSONL/TXT prompt batch.

Each prompt is written to a separate directory under `--out_root`, with one
subdirectory per Laplace mode. The script also writes `results.txt`, which can
be passed to evaluate_tifa_uq.py via `--results_file`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, help="JSONL, CSV, or TXT prompt file.")
    p.add_argument("--out_root", default="assets/stable_diffusion/recap_probe")
    p.add_argument("--methods", default="last_layer,subnet")
    p.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--pipeline", choices=["auto", "sd", "sdxl"], default="auto")
    p.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_z0", type=int, default=4)
    p.add_argument("--n_lap_pairs", type=int, default=100)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--subnet_n_params", type=int, default=50_000)
    p.add_argument("--subnet_max_tensors", type=int, default=12)
    p.add_argument("--subnet_mc_samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def slugify(value: str, max_len: int = 48) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "_":
            out.append("_")
    slug = "".join(out).strip("_")
    return slug[:max_len].strip("_") or "prompt"


def read_prompts(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    records = []
    if suffix == ".jsonl":
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = str(data.get("prompt") or data.get("recaption") or data.get("caption"))
            records.append({"id": str(data.get("id", f"prompt_{i:04d}")), "prompt": prompt})
    elif suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for i, data in enumerate(reader):
                prompt = data.get("prompt") or data.get("recaption") or data.get("caption")
                if prompt:
                    records.append({"id": str(data.get("id", f"prompt_{i:04d}")), "prompt": prompt})
    else:
        for i, line in enumerate(path.read_text().splitlines()):
            prompt = line.strip()
            if prompt:
                records.append({"id": f"prompt_{i:04d}", "prompt": prompt})
    if not records:
        raise RuntimeError(f"No prompts found in {path}")
    return records


def main():
    args = get_args()
    prompts = read_prompts(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    script = Path(__file__).with_name("run_sd_laplace.py")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    result_paths = []

    for prompt_idx, record in enumerate(prompts):
        prompt = record["prompt"]
        prompt_id = f"{prompt_idx:03d}_{slugify(record['id'])}"
        for method in methods:
            if method not in {"last_layer", "subnet"}:
                raise ValueError(f"Unsupported method: {method}")
            out_dir = out_root / prompt_id / method
            result_path = out_dir / "laplace_results.npz"
            result_paths.append(result_path)
            if args.skip_existing and result_path.exists():
                print(f"Skipping existing {result_path}")
                continue

            cmd = [
                sys.executable,
                str(script),
                "--model_id",
                args.model_id,
                "--pipeline",
                args.pipeline,
                "--prompt",
                prompt,
                "--device",
                args.device,
                "--torch_dtype",
                args.torch_dtype,
                "--scheduler",
                args.scheduler,
                "--laplace_mode",
                method,
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
                "--seed",
                str(args.seed + prompt_idx * 1000),
                "--out_dir",
                str(out_dir),
            ]
            if method == "subnet":
                cmd.extend(
                    [
                        "--subnet_n_params",
                        str(args.subnet_n_params),
                        "--subnet_max_tensors",
                        str(args.subnet_max_tensors),
                        "--subnet_mc_samples",
                        str(args.subnet_mc_samples),
                    ]
                )

            print(f"Running {prompt_id} {method}: {prompt}")
            subprocess.run(cmd, check=True)

    results_file = out_root / "results.txt"
    existing = [path for path in result_paths if path.exists()]
    results_file.write_text("\n".join(str(path) for path in existing) + "\n")
    print(f"Saved {len(existing)} result paths to {results_file}")


if __name__ == "__main__":
    main()
