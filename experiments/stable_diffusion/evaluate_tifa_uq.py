"""
Evaluate Stable Diffusion uncertainty scores against TIFA-like faithfulness.

This script bridges the Stable Diffusion UQ outputs from run_sd_laplace.py with
the local uqdiff.tifa_like evaluator. It writes sample-level and question-level
CSVs so uncertainty scores can be compared against semantic prompt-image
failures.
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
from uqdiff.tifa_like.heuristic_question_gen import generate_heuristic_questions
from uqdiff.tifa_like.question_gen import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOGETHER_MODEL,
    generate_questions,
    generate_questions_together,
)
from uqdiff.tifa_like.schemas import TifaQuestion
from uqdiff.tifa_like.scorer import score_image
from uqdiff.tifa_like.vqa import DEFAULT_SBERT_CHECKPOINT, DEFAULT_VQA_CHECKPOINT, TifaLikeVQAModel


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more laplace_results.npz files produced by run_sd_laplace.py.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="assets/stable_diffusion/tifa_eval",
        help="Directory for summary/per-sample/per-question outputs.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt. By default each npz prompt is used.",
    )
    p.add_argument(
        "--question_cache",
        type=str,
        default="",
        help=(
            "JSON cache for generated TIFA questions. Defaults to "
            "<out_dir>/tifa_questions.json."
        ),
    )
    p.add_argument(
        "--require_cached_questions",
        action="store_true",
        help="Fail if a prompt is missing from --question_cache instead of calling OpenAI.",
    )
    p.add_argument(
        "--refresh_questions",
        action="store_true",
        help="Regenerate questions even when cached questions exist.",
    )
    p.add_argument(
        "--question_source",
        choices=["openai", "together", "heuristic"],
        default="openai",
        help="Use OpenAI, Together AI, or an offline heuristic question generator.",
    )
    p.add_argument(
        "--openai_model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model for question generation.",
    )
    p.add_argument(
        "--together_model",
        type=str,
        default=DEFAULT_TOGETHER_MODEL,
        help="Together AI model for question generation.",
    )
    p.add_argument(
        "--prompt_path",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help="TIFA-like question generation prompt YAML.",
    )
    p.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Limit questions per prompt. 0 keeps all generated/cached questions.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit samples per results file. 0 evaluates all samples.",
    )
    p.add_argument(
        "--vqa_checkpoint",
        type=str,
        default=DEFAULT_VQA_CHECKPOINT,
        help="Hugging Face BLIP VQA checkpoint.",
    )
    p.add_argument(
        "--sbert_checkpoint",
        type=str,
        default=DEFAULT_SBERT_CHECKPOINT,
        help="Hugging Face sentence embedding checkpoint for answer-choice matching.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--filter_fracs",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated fractions of most-uncertain samples to reject.",
    )
    return p.parse_args()


def model_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def normalize_question_dict(question: Any) -> dict[str, Any]:
    data = model_dump(question)
    return {
        "element": str(data["element"]),
        "question": str(data["question"]),
        "choices": [str(choice) for choice in data["choices"]],
        "answer": str(data["answer"]),
        "element_type": str(data["element_type"]),
    }


def to_question(question: Any) -> TifaQuestion:
    return TifaQuestion(**normalize_question_dict(question))


def load_question_cache(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return {"__default__": [normalize_question_dict(q) for q in payload]}
    if isinstance(payload, dict) and "prompts" in payload:
        payload = payload["prompts"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported question cache format: {path}")

    cache = {}
    for prompt, questions in payload.items():
        if not isinstance(questions, list):
            raise ValueError(f"Questions for prompt {prompt!r} must be a list")
        cache[str(prompt)] = [normalize_question_dict(q) for q in questions]
    return cache


def save_question_cache(path: Path, cache: dict[str, list[dict[str, Any]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "prompts": cache}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def resolve_questions(
    prompt: str,
    args,
    cache: dict[str, list[dict[str, Any]]],
    cache_path: Path,
) -> list[TifaQuestion]:
    if not args.refresh_questions:
        cached = cache.get(prompt) or cache.get("__default__")
        if cached is not None:
            return [to_question(q) for q in cached[: args.max_questions or None]]

    if args.require_cached_questions:
        raise KeyError(
            f"No cached TIFA questions for prompt {prompt!r} in {cache_path}"
        )

    if args.question_source == "heuristic":
        questions = generate_heuristic_questions(
            prompt,
            max_questions=args.max_questions or 12,
        )
    elif args.question_source == "together":
        prompt_path = Path(args.prompt_path) if args.prompt_path else None
        questions = generate_questions_together(
            prompt,
            model=args.together_model,
            prompt_path=prompt_path,
        )
    else:
        prompt_path = Path(args.prompt_path) if args.prompt_path else None
        questions = generate_questions(prompt, model=args.openai_model, prompt_path=prompt_path)
    normalized = [normalize_question_dict(q) for q in questions]
    if args.max_questions:
        normalized = normalized[: args.max_questions]
    cache[prompt] = normalized
    save_question_cache(cache_path, cache)
    return [to_question(q) for q in normalized]


def summarize_method(
    method: str,
    result_path: str,
    prompt: str,
    metrics: dict[str, np.ndarray],
    tifa_scores: np.ndarray,
    n_questions: int,
    filter_fracs: list[float],
) -> list[dict[str, object]]:
    rows = []
    n = len(tifa_scores)
    for metric_name, scores in metrics.items():
        scores = np.asarray(scores[:n], dtype=np.float32)
        row = {
            "method": method,
            "result_path": result_path,
            "prompt": prompt,
            "n": n,
            "n_questions": n_questions,
            "uncertainty_metric": metric_name,
            "uq_mean": float(np.nanmean(scores)),
            "uq_std": float(np.nanstd(scores)),
            "uq_min": float(np.nanmin(scores)),
            "uq_max": float(np.nanmax(scores)),
            "tifa_mean_all": float(np.nanmean(tifa_scores)),
            "pearson_uq_tifa": corr(scores, tifa_scores, "pearson"),
            "spearman_uq_tifa": corr(scores, tifa_scores, "spearman"),
        }

        order = np.argsort(scores)
        window = max(1, int(0.2 * n))
        row["best_uq_tifa_mean_20pct"] = float(np.nanmean(tifa_scores[order[:window]]))
        row["worst_uq_tifa_mean_20pct"] = float(np.nanmean(tifa_scores[order[-window:]]))
        for frac in filter_fracs:
            keep_n = max(1, int(round((1.0 - frac) * n)))
            kept = order[:keep_n]
            rejected = order[keep_n:]
            prefix = f"reject_top_{int(frac * 100)}pct"
            row[f"{prefix}_tifa_mean_kept"] = float(np.nanmean(tifa_scores[kept]))
            row[f"{prefix}_tifa_delta"] = float(np.nanmean(tifa_scores[kept]) - np.nanmean(tifa_scores))
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
    primary = [r for r in rows if r["uncertainty_metric"] in primary_names]
    if not primary:
        primary = rows

    columns = [
        "method",
        "uncertainty_metric",
        "n",
        "n_questions",
        "tifa_mean_all",
        "pearson_uq_tifa",
        "spearman_uq_tifa",
        "reject_top_20pct_tifa_mean_kept",
        "reject_top_20pct_tifa_delta",
    ]
    available = [c for c in columns if any(c in r for r in primary)]
    lines = ["# Stable Diffusion UQ x TIFA Evaluation", ""]
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
        "Negative `pearson_uq_tifa` / `spearman_uq_tifa` means higher uncertainty "
        "is associated with lower TIFA faithfulness."
    )
    path.write_text("\n".join(lines) + "\n")


def truncate_metrics(metrics: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(values[:n]) for name, values in metrics.items()}


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.question_cache) if args.question_cache else out_dir / "tifa_questions.json"
    filter_fracs = parse_fracs(args.filter_fracs)

    question_cache = load_question_cache(cache_path)
    vqa_model = TifaLikeVQAModel(
        vqa_checkpoint=args.vqa_checkpoint,
        sbert_checkpoint=args.sbert_checkpoint,
        device=args.device,
    )

    all_summary_rows = []
    all_sample_rows = []
    all_question_rows = []

    for result_path in args.results:
        data = np.load(result_path, allow_pickle=True)
        method = method_name(result_path)
        prompt = npz_prompt(data, args.prompt)
        questions = resolve_questions(prompt, args, question_cache, cache_path)

        images = to_pil_images(np.asarray(data["images"], dtype=np.float32))
        if args.max_samples:
            images = images[: args.max_samples]
        n = len(images)
        metrics = truncate_metrics(uncertainty_metrics(data), n)

        tifa_scores = []
        for sample_idx, image in enumerate(tqdm(images, desc=f"TIFA {method}")):
            tifa_result = score_image(vqa_model, questions, image)
            tifa_scores.append(tifa_result.score)
            sample_row = {
                "method": method,
                "result_path": result_path,
                "prompt": prompt,
                "sample_idx": sample_idx,
                "tifa_score": tifa_result.score,
            }
            for metric_name, values in metrics.items():
                sample_row[metric_name] = float(values[sample_idx])
            all_sample_rows.append(sample_row)

            for question_idx, question_result in enumerate(tifa_result.question_results):
                row = question_result.model_dump()
                row["choices"] = "|".join(row["choices"])
                all_question_rows.append(
                    {
                        "method": method,
                        "result_path": result_path,
                        "prompt": prompt,
                        "sample_idx": sample_idx,
                        "question_idx": question_idx,
                        **row,
                    }
                )

        tifa_scores_np = np.asarray(tifa_scores, dtype=np.float32)
        all_summary_rows.extend(
            summarize_method(
                method,
                result_path,
                prompt,
                metrics,
                tifa_scores_np,
                len(questions),
                filter_fracs,
            )
        )

    write_csv(out_dir / "summary.csv", all_summary_rows)
    write_csv(out_dir / "per_sample.csv", all_sample_rows)
    write_csv(out_dir / "per_question.csv", all_question_rows)
    write_markdown(out_dir / "summary.md", all_summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(all_summary_rows, indent=2) + "\n")

    print(f"Saved {out_dir / 'summary.csv'}")
    print(f"Saved {out_dir / 'per_sample.csv'}")
    print(f"Saved {out_dir / 'per_question.csv'}")
    print(f"Saved {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
