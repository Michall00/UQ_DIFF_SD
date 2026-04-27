"""
Evaluate Stable Diffusion uncertainty scores with PUNC-like prompt uncertainty.

The script captions generated images with an OpenAI vision model, compares each
caption with the original prompt, and reports whether the saved UQ metrics track
semantic prompt-space uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from evaluate_sd_uq import (
    corr,
    method_name,
    npz_prompt,
    parse_fracs,
    to_pil_images,
    uncertainty_metrics,
    write_csv,
)
from uqdiff.punc_like.captioner import (
    DEFAULT_OPENAI_VISION_MODEL,
    OpenAICaptioner,
    image_cache_key,
)
from uqdiff.punc_like.scorer import EmbeddingPuncScorer, TokenPuncScorer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="One or more laplace_results.npz files produced by run_sd_laplace.py.",
    )
    p.add_argument(
        "--results_file",
        type=str,
        default="",
        help="Text file with one laplace_results.npz path per line.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="assets/stable_diffusion/punc_eval",
        help="Directory for summary/per-sample outputs.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt. By default each npz prompt is used.",
    )
    p.add_argument(
        "--caption_cache",
        type=str,
        default="",
        help="JSON cache for OpenAI captions. Defaults to <out_dir>/punc_captions.json.",
    )
    p.add_argument(
        "--require_cached_captions",
        action="store_true",
        help="Fail if an image caption is missing from --caption_cache.",
    )
    p.add_argument(
        "--refresh_captions",
        action="store_true",
        help="Regenerate captions even when cached captions exist.",
    )
    p.add_argument(
        "--openai_model",
        type=str,
        default=DEFAULT_OPENAI_VISION_MODEL,
        help="OpenAI vision-capable model used for captioning.",
    )
    p.add_argument(
        "--image_detail",
        choices=["low", "auto", "high"],
        default="low",
        help="OpenAI image input detail level.",
    )
    p.add_argument(
        "--caption_max_side",
        type=int,
        default=768,
        help="Resize longest image side before sending to OpenAI. 0 disables resizing.",
    )
    p.add_argument(
        "--similarity",
        choices=["token", "embedding"],
        default="token",
        help="Use ROUGE-like token overlap or BERTScore-like embedding overlap.",
    )
    p.add_argument(
        "--embedding_checkpoint",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Text encoder for --similarity embedding.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit samples per results file. 0 evaluates all samples.",
    )
    p.add_argument(
        "--filter_fracs",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated fractions of most-uncertain samples to reject.",
    )
    return p.parse_args()


def load_caption_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "captions" in payload:
        payload = payload["captions"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported caption cache format: {path}")
    return {str(key): dict(value) for key, value in payload.items()}


def save_caption_cache(path: Path, cache: dict[str, dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "captions": cache}, indent=2, ensure_ascii=False) + "\n"
    )


def resolve_result_paths(args) -> list[str]:
    paths = list(args.results)
    if args.results_file:
        results_file = Path(args.results_file)
        paths.extend(
            line.strip()
            for line in results_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not paths:
        raise ValueError("Pass --results or --results_file.")
    return paths


def truncate_metrics(metrics: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(values[:n]) for name, values in metrics.items()}


def build_scorer(args):
    if args.similarity == "embedding":
        return EmbeddingPuncScorer(
            checkpoint=args.embedding_checkpoint,
            device=args.device,
        )
    return TokenPuncScorer()


def summarize_method(
    method: str,
    result_path: str,
    prompt: str,
    metrics: dict[str, np.ndarray],
    punc_rows: list[dict[str, Any]],
    filter_fracs: list[float],
) -> list[dict[str, object]]:
    rows = []
    targets = {
        "punc_total": np.asarray([r["punc_total"] for r in punc_rows], dtype=np.float32),
        "punc_epistemic": np.asarray([r["punc_epistemic"] for r in punc_rows], dtype=np.float32),
        "punc_aleatoric": np.asarray([r["punc_aleatoric"] for r in punc_rows], dtype=np.float32),
        "caption_f1": np.asarray([r["caption_f1"] for r in punc_rows], dtype=np.float32),
    }
    n = len(punc_rows)
    for metric_name, scores in metrics.items():
        scores = np.asarray(scores[:n], dtype=np.float32)
        row = {
            "method": method,
            "result_path": result_path,
            "prompt": prompt,
            "n": n,
            "uncertainty_metric": metric_name,
            "uq_mean": float(np.nanmean(scores)),
            "uq_std": float(np.nanstd(scores)),
            "punc_total_mean_all": float(np.nanmean(targets["punc_total"])),
            "punc_epistemic_mean_all": float(np.nanmean(targets["punc_epistemic"])),
            "punc_aleatoric_mean_all": float(np.nanmean(targets["punc_aleatoric"])),
            "caption_f1_mean_all": float(np.nanmean(targets["caption_f1"])),
        }
        for target_name, values in targets.items():
            row[f"pearson_uq_{target_name}"] = corr(scores, values, "pearson")
            row[f"spearman_uq_{target_name}"] = corr(scores, values, "spearman")

        order = np.argsort(scores)
        for frac in filter_fracs:
            keep_n = max(1, int(round((1.0 - frac) * n)))
            kept = order[:keep_n]
            rejected = order[keep_n:]
            prefix = f"reject_top_{int(frac * 100)}pct"
            kept_punc = float(np.nanmean(targets["punc_total"][kept]))
            row[f"{prefix}_punc_total_mean_kept"] = kept_punc
            row[f"{prefix}_punc_total_delta"] = kept_punc - float(np.nanmean(targets["punc_total"]))
            row[f"{prefix}_n_rejected"] = int(len(rejected))
        rows.append(row)
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]):
    primary_names = {
        "var_mean",
        "var_p95",
        "var_attn_mean",
        "bayesdiff_var_mean",
        "bayesdiff_var_p95",
        "daam_var_mean",
    }
    primary = [r for r in rows if r["uncertainty_metric"] in primary_names] or rows
    columns = [
        "method",
        "uncertainty_metric",
        "n",
        "punc_total_mean_all",
        "punc_epistemic_mean_all",
        "punc_aleatoric_mean_all",
        "pearson_uq_punc_total",
        "spearman_uq_punc_total",
        "reject_top_20pct_punc_total_mean_kept",
        "reject_top_20pct_punc_total_delta",
    ]
    available = [c for c in columns if any(c in r for r in primary)]
    lines = ["# Stable Diffusion UQ x PUNC-like Evaluation", ""]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("| " + " | ".join(["---"] * len(available)) + " |")
    for row in primary:
        values = []
        for col in available:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6g}" if math.isfinite(value) else "nan"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append(
        "Positive `pearson_uq_punc_total` / `spearman_uq_punc_total` means higher "
        "UQ is associated with higher prompt-space uncertainty. Negative "
        "`reject_top_*_punc_total_delta` means rejection by UQ lowered PUNC uncertainty."
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.caption_cache) if args.caption_cache else out_dir / "punc_captions.json"
    caption_cache = load_caption_cache(cache_path)
    filter_fracs = parse_fracs(args.filter_fracs)
    scorer = build_scorer(args)
    captioner = None

    all_summary_rows = []
    all_sample_rows = []

    for result_path in resolve_result_paths(args):
        data = np.load(result_path, allow_pickle=True)
        method = method_name(result_path)
        prompt = npz_prompt(data, args.prompt)
        images = to_pil_images(np.asarray(data["images"], dtype=np.float32))
        if args.max_samples:
            images = images[: args.max_samples]
        metrics = truncate_metrics(uncertainty_metrics(data), len(images))

        punc_rows = []
        for sample_idx, image in enumerate(tqdm(images, desc=f"PUNC {method}")):
            key = image_cache_key(
                image,
                prompt=prompt,
                model=args.openai_model,
                detail=args.image_detail,
                max_side=args.caption_max_side,
            )
            cached = caption_cache.get(key)
            if cached is not None and not args.refresh_captions:
                caption = str(cached["caption"])
            else:
                if args.require_cached_captions:
                    raise KeyError(f"No cached PUNC caption for sample {sample_idx}: {key}")
                if captioner is None:
                    captioner = OpenAICaptioner(
                        model=args.openai_model,
                        detail=args.image_detail,
                        max_side=args.caption_max_side,
                    )
                caption = captioner.caption(image, prompt=prompt)
                caption_cache[key] = {
                    "caption": caption,
                    "prompt": prompt,
                    "model": args.openai_model,
                    "detail": args.image_detail,
                    "max_side": args.caption_max_side,
                }
                save_caption_cache(cache_path, caption_cache)

            punc = scorer.score(prompt, caption)
            row = {
                "method": method,
                "result_path": result_path,
                "prompt": prompt,
                "sample_idx": sample_idx,
                "caption": caption,
                "caption_cache_key": key,
                "openai_model": args.openai_model,
                "similarity": args.similarity,
                **punc,
            }
            for metric_name, values in metrics.items():
                row[metric_name] = float(values[sample_idx])
            punc_rows.append(row)
            all_sample_rows.append(row)

        all_summary_rows.extend(
            summarize_method(method, result_path, prompt, metrics, punc_rows, filter_fracs)
        )

    write_csv(out_dir / "summary.csv", all_summary_rows)
    write_csv(out_dir / "per_sample.csv", all_sample_rows)
    write_markdown(out_dir / "summary.md", all_summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(all_summary_rows, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"Saved {out_dir / 'summary.csv'}")
    print(f"Saved {out_dir / 'per_sample.csv'}")
    print(f"Saved {out_dir / 'summary.md'}")
    print(f"Saved {cache_path}")


if __name__ == "__main__":
    main()
