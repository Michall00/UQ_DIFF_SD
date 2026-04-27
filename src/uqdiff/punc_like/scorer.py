from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "this",
    "to",
    "with",
}


def _normalize_token(token: str) -> str:
    token = token.lower().strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z0-9']+", text):
        token = _normalize_token(raw)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return set(tokens)


def punc_from_precision_recall(precision: float, recall: float) -> dict[str, float]:
    precision = float(max(0.0, min(1.0, precision)))
    recall = float(max(0.0, min(1.0, recall)))
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "caption_precision": precision,
        "caption_recall": recall,
        "caption_f1": f1,
        "punc_aleatoric": 1.0 - precision,
        "punc_epistemic": 1.0 - recall,
        "punc_total": 1.0 - f1,
    }


@dataclass
class TokenPuncScorer:
    """ROUGE-like concept overlap approximation for PUNC."""

    def score(self, prompt: str, caption: str) -> dict[str, float]:
        prompt_tokens = content_tokens(prompt)
        caption_tokens = content_tokens(caption)
        overlap = prompt_tokens & caption_tokens
        precision = len(overlap) / len(caption_tokens) if caption_tokens else 0.0
        recall = len(overlap) / len(prompt_tokens) if prompt_tokens else 0.0
        scores = punc_from_precision_recall(precision, recall)
        scores["prompt_token_count"] = float(len(prompt_tokens))
        scores["caption_token_count"] = float(len(caption_tokens))
        scores["overlap_token_count"] = float(len(overlap))
        return scores


class EmbeddingPuncScorer:
    """BERTScore-like precision/recall using a Hugging Face text encoder."""

    def __init__(
        self,
        checkpoint: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | torch.device | None = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for embedding PUNC scoring."
            ) from exc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _embeddings(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True,
        ).to(self.device)
        output = self.model(**{k: v for k, v in encoded.items() if k != "special_tokens_mask"})
        embeddings = F.normalize(output.last_hidden_state[0].float(), p=2, dim=-1)
        mask = encoded["attention_mask"][0].bool()
        if "special_tokens_mask" in encoded:
            mask = mask & ~encoded["special_tokens_mask"][0].bool()
        embeddings = embeddings[mask]
        if embeddings.numel() == 0:
            return torch.empty(0, output.last_hidden_state.shape[-1], device=self.device)
        return embeddings

    def score(self, prompt: str, caption: str) -> dict[str, float]:
        prompt_emb = self._embeddings(prompt)
        caption_emb = self._embeddings(caption)
        if prompt_emb.numel() == 0 or caption_emb.numel() == 0:
            return punc_from_precision_recall(0.0, 0.0)

        sim = (caption_emb @ prompt_emb.T).clamp_min(0.0)
        precision = float(sim.max(dim=1).values.mean().item())
        recall = float(sim.max(dim=0).values.mean().item())
        return punc_from_precision_recall(precision, recall)
