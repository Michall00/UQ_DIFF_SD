from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from openai import OpenAI

from .schemas import TifaQuestion, TifaQuestionSet

DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompt.yaml")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def _load_prompt(path: Path) -> dict[str, str]:
    with path.open() as f:
        return yaml.safe_load(f)


def _default_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        return OpenAI()

    api_key = api_key.strip()
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "OPENAI_API_KEY contains non-ASCII characters. Reset the environment "
            "variable to the raw API key only, without terminal output or quotes."
        ) from exc

    return OpenAI(api_key=api_key)


def _env_ascii(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set.")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{name} contains non-ASCII characters. Reset it to the raw API key only."
        ) from exc
    return value


def _default_together_client() -> OpenAI:
    return OpenAI(
        api_key=_env_ascii("TOGETHER_API_KEY"),
        base_url=TOGETHER_BASE_URL,
    )


def _normalize_questions(questions: list[TifaQuestion]) -> list[TifaQuestion]:
    for question in questions:
        if question.element_type in ("human", "animal"):
            question.element_type = "animal/human"
    return questions


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
    client = client or _default_openai_client()
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
    return _normalize_questions(result.questions)


def generate_questions_together(
    caption: str,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    prompt_path: Path | None = None,
) -> list[TifaQuestion]:
    """Generate TIFA-like questions with Together AI's OpenAI-compatible API."""
    model = model or DEFAULT_TOGETHER_MODEL
    prompt_path = prompt_path or DEFAULT_PROMPT_PATH
    client = client or _default_together_client()
    prompt = _load_prompt(prompt_path)
    schema = TifaQuestionSet.model_json_schema()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    prompt["system"]
                    + "\n\nRespond only in JSON following this schema:\n"
                    + json.dumps(schema)
                ),
            },
            {"role": "user", "content": prompt["user_template"].format(caption=caption)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tifa_question_set",
                "schema": schema,
            },
        },
    )

    content = completion.choices[0].message.content
    if not content:
        return []
    result = TifaQuestionSet.model_validate_json(content)
    return _normalize_questions(result.questions)
