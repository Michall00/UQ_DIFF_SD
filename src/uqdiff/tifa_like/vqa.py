from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BlipForQuestionAnswering

DEFAULT_VQA_CHECKPOINT = "Salesforce/blip-vqa-capfilt-large"
DEFAULT_SBERT_CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"


class AnswerResult(TypedDict):
    """Multiple-choice answer payload returned by the VQA wrapper."""

    free_form_answer: str
    multiple_choice_answer: str


class BLIPModel:
    """BLIP VQA model wrapper."""

    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = BlipForQuestionAnswering.from_pretrained(checkpoint)
        self.model.to(self.device)  # type: ignore[invalid-argument-type]
        self.model.eval()

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=50)
        decoded_answers = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return str(decoded_answers[0])


class SBERTMatcher:
    """Sentence embedding matcher for mapping free-form VQA answers to choices."""

    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, sentences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(
            attention_mask.sum(dim=1),
            min=1e-9,
        )
        return F.normalize(pooled, p=2, dim=1).cpu()

    def match(self, answer: str, choices: list[str]) -> str:
        answer_emb = self._embed([answer])
        choices_emb = self._embed(choices)
        best_idx = int(torch.argmax(choices_emb @ answer_emb.T).item())
        return choices[best_idx]


class TifaLikeVQAModel:
    """Combined VQA + multiple-choice matching model."""

    def __init__(
        self,
        vqa_checkpoint: str = DEFAULT_VQA_CHECKPOINT,
        sbert_checkpoint: str = DEFAULT_SBERT_CHECKPOINT,
        device: str | torch.device | None = None,
    ):
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vqa = BLIPModel(checkpoint=vqa_checkpoint, device=resolved_device)
        self.matcher = SBERTMatcher(checkpoint=sbert_checkpoint, device=resolved_device)

    def answer_multiple_choice(
        self,
        image: str | Path | Image.Image,
        question: str,
        choices: list[str],
    ) -> AnswerResult:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        free_form = self.vqa.answer(image, question)
        multiple_choice = free_form if free_form in choices else self.matcher.match(free_form, choices)
        return {
            "free_form_answer": free_form,
            "multiple_choice_answer": multiple_choice,
        }
