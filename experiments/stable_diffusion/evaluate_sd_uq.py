"""
Evaluate Stable Diffusion FLARE-style uncertainty results.

The script reads one or more laplace_results.npz files produced by
run_sd_laplace.py, computes CLIPScore for saved images, and reports whether
higher uncertainty scores identify lower-quality / lower-alignment samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more laplace_results.npz files.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="assets/stable_diffusion/eval",
        help="Directory for summary CSV/JSON/Markdown outputs.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt. By default the prompt stored in each npz is used.",
    )
    p.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model used for CLIPScore.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--filter_fracs",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated fractions of most-uncertain samples to reject.",
    )
    p.add_argument(
        "--skip_clip",
        action="store_true",
        help="Only compute uncertainty summaries, no CLIPScore.",
    )
    return p.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def parse_fracs(raw: str) -> list[float]:
    fracs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0 <= value < 1:
            raise ValueError(f"Filter fraction must be in [0, 1): {value}")
        fracs.append(value)
    return fracs


def npz_prompt(data, fallback: str | None) -> str:
    if fallback is not None:
        return fallback
    if "prompt" not in data:
        raise ValueError("No prompt found in npz. Pass --prompt.")
    value = data["prompt"]
    if value.shape == ():
        return str(value.item())
    return str(value)


def to_pil_images(images: np.ndarray) -> list[Image.Image]:
    pil_images = []
    for image in images:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
        pil_images.append(Image.fromarray(arr))
    return pil_images


def load_clip(model_id: str, device: str, dtype: torch.dtype):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for CLIPScore. Run `make sd-sync` or pass --skip_clip."
        ) from exc

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def clip_scores(
    images: np.ndarray,
    prompt: str,
    model,
    processor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    pil_images = to_pil_images(images)
    scores = []
    for start in range(0, len(pil_images), batch_size):
        batch_images = pil_images[start:start + batch_size]
        prompts = [prompt] * len(batch_images)
        inputs = processor(
            text=prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            model_dtype = next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.float()
        text_embeds = outputs.text_embeds.float()
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        batch_scores = (image_embeds * text_embeds).sum(dim=-1)
        scores.extend(batch_scores.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    if kind == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def gini(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    out = []
    for x in flat:
        x = np.sort(np.asarray(x, dtype=np.float64))
        total = x.sum()
        if total <= 0:
            out.append(float("nan"))
            continue
        n = len(x)
        weights = 2 * np.arange(1, n + 1) - n - 1
        out.append(float(weights.dot(x) / (n * total)))
    return np.asarray(out, dtype=np.float32)


def entropy01(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    probs = flat / (flat.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    return (ent / math.log(flat.shape[1])).astype(np.float32)


def uncertainty_metrics(data) -> dict[str, np.ndarray]:
    var_maps = np.asarray(data["var_maps"], dtype=np.float32)
    flat = var_maps.reshape(var_maps.shape[0], -1)
    metrics = {
        "var_mean": np.asarray(data["var_mean"], dtype=np.float32)
        if "var_mean" in data else flat.mean(axis=1),
        "var_sum": np.asarray(data["var_sum"], dtype=np.float32)
        if "var_sum" in data else flat.sum(axis=1),
        "var_p95": np.asarray(data["var_p95"], dtype=np.float32)
        if "var_p95" in data else np.percentile(flat, 95, axis=1).astype(np.float32),
        "var_max": flat.max(axis=1).astype(np.float32),
        "var_gini": gini(var_maps),
        "var_entropy": entropy01(var_maps),
        "var_peak_mean": (flat.max(axis=1) / (flat.mean(axis=1) + 1e-12)).astype(np.float32),
    }
    if "var_attn_mean" in data:
        value = np.asarray(data["var_attn_mean"], dtype=np.float32)
        if np.isfinite(value).any():
            metrics["var_attn_mean"] = value
    if "var_attn_sum" in data:
        value = np.asarray(data["var_attn_sum"], dtype=np.float32)
        if np.isfinite(value).any():
            metrics["var_attn_sum"] = value
    return metrics


def method_name(path: str) -> str:
    parent = Path(path).parent.name
    return parent or Path(path).stem


def summarize_method(
    method: str,
    result_path: str,
    prompt: str,
    metrics: dict[str, np.ndarray],
    quality: np.ndarray | None,
    filter_fracs: list[float],
) -> list[dict[str, object]]:
    rows = []
    n = len(next(iter(metrics.values())))
    for metric_name, scores in metrics.items():
        row = {
            "method": method,
            "result_path": result_path,
            "prompt": prompt,
            "n": n,
            "uncertainty_metric": metric_name,
            "uq_mean": float(np.nanmean(scores)),
            "uq_std": float(np.nanstd(scores)),
            "uq_min": float(np.nanmin(scores)),
            "uq_max": float(np.nanmax(scores)),
        }
        if quality is not None:
            row["clip_mean_all"] = float(np.nanmean(quality))
            row["pearson_uq_clip"] = corr(scores, quality, "pearson")
            row["spearman_uq_clip"] = corr(scores, quality, "spearman")
            order = np.argsort(scores)
            row["best_uq_clip_mean_20pct"] = float(np.nanmean(quality[order[:max(1, int(0.2 * n))]]))
            row["worst_uq_clip_mean_20pct"] = float(np.nanmean(quality[order[-max(1, int(0.2 * n)):]]))
            for frac in filter_fracs:
                keep_n = max(1, int(round((1.0 - frac) * n)))
                kept = order[:keep_n]
                rejected = order[keep_n:]
                prefix = f"reject_top_{int(frac * 100)}pct"
                row[f"{prefix}_clip_mean_kept"] = float(np.nanmean(quality[kept]))
                row[f"{prefix}_clip_delta"] = float(np.nanmean(quality[kept]) - np.nanmean(quality))
                row[f"{prefix}_n_rejected"] = int(len(rejected))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]):
    primary = [r for r in rows if r["uncertainty_metric"] in {"var_mean", "var_p95", "var_attn_mean"}]
    if not primary:
        primary = rows
    columns = [
        "method",
        "uncertainty_metric",
        "n",
        "uq_mean",
        "clip_mean_all",
        "pearson_uq_clip",
        "spearman_uq_clip",
        "reject_top_20pct_clip_mean_kept",
        "reject_top_20pct_clip_delta",
    ]
    available = [c for c in columns if any(c in r for r in primary)]
    lines = ["# Stable Diffusion UQ Evaluation", ""]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("| " + " | ".join(["---"] * len(available)) + " |")
    for row in primary:
        values = []
        for col in available:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Negative `pearson_uq_clip` / `spearman_uq_clip` means higher uncertainty is associated with lower CLIPScore.")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filter_fracs = parse_fracs(args.filter_fracs)

    clip = None
    if not args.skip_clip:
        dtype = resolve_dtype(args.torch_dtype, args.device)
        clip = load_clip(args.clip_model, args.device, dtype)

    all_summary_rows = []
    all_sample_rows = []

    for result_path in args.results:
        data = np.load(result_path, allow_pickle=True)
        method = method_name(result_path)
        prompt = npz_prompt(data, args.prompt)
        images = np.asarray(data["images"], dtype=np.float32)
        metrics = uncertainty_metrics(data)

        quality = None
        if clip is not None:
            model, processor = clip
            quality = clip_scores(images, prompt, model, processor, args.device, args.batch_size)

        all_summary_rows.extend(
            summarize_method(method, result_path, prompt, metrics, quality, filter_fracs)
        )

        n = images.shape[0]
        for i in range(n):
            row = {"method": method, "result_path": result_path, "sample_idx": i}
            if quality is not None:
                row["clip_score"] = float(quality[i])
            for name, values in metrics.items():
                row[name] = float(values[i])
            all_sample_rows.append(row)

    write_csv(out_dir / "summary.csv", all_summary_rows)
    write_csv(out_dir / "per_sample.csv", all_sample_rows)
    write_markdown(out_dir / "summary.md", all_summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(all_summary_rows, indent=2) + "\n")

    print(f"Saved {out_dir / 'summary.csv'}")
    print(f"Saved {out_dir / 'per_sample.csv'}")
    print(f"Saved {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
