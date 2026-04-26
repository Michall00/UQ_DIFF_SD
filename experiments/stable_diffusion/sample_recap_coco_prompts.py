"""
Sample probe prompts from UCSC-VLAA/Recap-COCO-30K.

The script prefers prompts with visual attributes that are useful for TIFA-like
faithfulness checks: people, counts, colors, spatial relations, and interactions.
It writes JSONL records with `id` and `prompt` fields.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

COUNT_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "several",
    "many",
}
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
HUMAN_WORDS = {
    "boy",
    "child",
    "children",
    "girl",
    "man",
    "men",
    "person",
    "people",
    "woman",
    "women",
}
SPATIAL_WORDS = {
    "above",
    "behind",
    "beside",
    "between",
    "front",
    "inside",
    "near",
    "next",
    "on",
    "under",
}
ACTION_WORDS = {
    "carrying",
    "eating",
    "holding",
    "playing",
    "riding",
    "sitting",
    "standing",
    "wearing",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="UCSC-VLAA/Recap-COCO-30K")
    p.add_argument("--split", default="train")
    p.add_argument("--caption_column", default="recaption")
    p.add_argument("--out", default="assets/stable_diffusion/recap_coco_prompts.jsonl")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_words", type=int, default=8)
    p.add_argument("--max_words", type=int, default=35)
    p.add_argument(
        "--top_pool",
        type=int,
        default=1000,
        help="Sample from the highest-scoring prompt pool instead of the full dataset.",
    )
    return p.parse_args()


def words(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]*|\d+", text)]


def prompt_score(prompt: str) -> int:
    token_set = set(words(prompt))
    score = 0
    score += 2 * int(bool(token_set & HUMAN_WORDS))
    score += 2 * int(bool(token_set & COUNT_WORDS) or any(token.isdigit() for token in token_set))
    score += int(bool(token_set & COLOR_WORDS))
    score += int(bool(token_set & SPATIAL_WORDS))
    score += int(bool(token_set & ACTION_WORDS))
    score += int("," in prompt)
    return score


def normalize_prompt(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split())


def main():
    args = get_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install Hugging Face datasets, e.g. "
            "`uv run --with datasets python experiments/stable_diffusion/sample_recap_coco_prompts.py`."
        ) from exc

    dataset = load_dataset(args.dataset, split=args.split)
    if args.caption_column not in dataset.column_names:
        raise KeyError(
            f"Column {args.caption_column!r} not found. Available columns: {dataset.column_names}"
        )

    candidates = []
    seen = set()
    for row_idx, row in enumerate(dataset):
        prompt = normalize_prompt(row[args.caption_column])
        token_count = len(words(prompt))
        if not args.min_words <= token_count <= args.max_words:
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "row_idx": row_idx,
                "prompt": prompt,
                "score": prompt_score(prompt),
            }
        )

    if not candidates:
        raise RuntimeError("No prompts matched the filters.")

    candidates.sort(key=lambda item: (-item["score"], item["row_idx"]))
    pool = candidates[: min(len(candidates), args.top_pool)]
    rng = random.Random(args.seed)
    selected = rng.sample(pool, k=min(args.n, len(pool)))
    selected.sort(key=lambda item: item["row_idx"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, item in enumerate(selected):
            record = {
                "id": f"recap_{i:04d}",
                "prompt": item["prompt"],
                "source": args.dataset,
                "row_idx": item["row_idx"],
                "score": item["score"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(selected)} prompts to {out_path}")


if __name__ == "__main__":
    main()
