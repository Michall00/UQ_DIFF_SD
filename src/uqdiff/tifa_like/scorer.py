from __future__ import annotations

from pathlib import Path
from statistics import mean

from PIL import Image

from .schemas import QuestionResult, TifaQuestion, TifaResult
from .vqa import TifaLikeVQAModel


def score_image(
    vqa_model: TifaLikeVQAModel,
    questions: list[TifaQuestion],
    image: str | Path | Image.Image,
) -> TifaResult:
    """Compute a TIFA-like faithfulness score for one image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    results: list[QuestionResult] = []
    for question in questions:
        vqa_answer = vqa_model.answer_multiple_choice(
            image,
            question.question,
            question.choices,
        )
        score = int(vqa_answer["multiple_choice_answer"] == question.answer)
        results.append(
            QuestionResult(
                element=question.element,
                question=question.question,
                choices=question.choices,
                expected_answer=question.answer,
                element_type=question.element_type,
                free_form_answer=vqa_answer["free_form_answer"],
                multiple_choice_answer=vqa_answer["multiple_choice_answer"],
                score=score,
            )
        )

    tifa_score = mean(result.score for result in results) if results else 0.0
    return TifaResult(score=tifa_score, question_results=results)
