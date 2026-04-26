from __future__ import annotations

from pathlib import Path

import yaml
from openai import OpenAI

from .schemas import TifaQuestion, TifaQuestionSet

DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompt.yaml")


def _load_prompt(path: Path) -> dict[str, str]:
    with path.open() as f:
        return yaml.safe_load(f)


def generate_questions(
    caption: str,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    prompt_path: Path | None = None,
) -> list[TifaQuestion]:
    """Generate TIFA-like verification questions for a text-to-image prompt.

    The OpenAI client reads `OPENAI_API_KEY` from the environment by default.
    """
    model = model or DEFAULT_OPENAI_MODEL
    prompt_path = prompt_path or DEFAULT_PROMPT_PATH
    client = client or OpenAI()
    prompt = _load_prompt(prompt_path)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user_template"].format(caption=caption)},
        ],
        response_format=TifaQuestionSet,
    )

    result = completion.choices[0].message.parsed
    if result is None:
        return []

    for question in result.questions:
        if question.element_type in ("human", "animal"):
            question.element_type = "animal/human"

    return result.questions
