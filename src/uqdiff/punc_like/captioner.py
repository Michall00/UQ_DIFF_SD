from __future__ import annotations

import base64
import hashlib
import io
import os
from dataclasses import dataclass

from PIL import Image

DEFAULT_OPENAI_VISION_MODEL = "gpt-4.1-mini"

DEFAULT_CAPTION_INSTRUCTION = """
Describe the visible content of the image as a concise factual caption.
Mention the main objects, attributes, counts, colors, relations, actions, and
text if visible. Do not judge image quality. Return only the caption.
""".strip()


def validate_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "OPENAI_API_KEY contains a non-ASCII character. Re-export it as plain text."
        ) from exc


def image_to_jpeg_bytes(image: Image.Image, max_side: int = 768, quality: int = 90) -> bytes:
    image = image.convert("RGB")
    if max_side > 0 and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def image_data_url(image: Image.Image, max_side: int = 768, quality: int = 90) -> str:
    payload = base64.b64encode(image_to_jpeg_bytes(image, max_side, quality)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def image_cache_key(
    image: Image.Image,
    prompt: str,
    model: str,
    detail: str,
    max_side: int,
    quality: int = 90,
) -> str:
    digest = hashlib.sha1()
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(detail.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(max_side).encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(image_to_jpeg_bytes(image, max_side, quality))
    return digest.hexdigest()


@dataclass
class OpenAICaptioner:
    model: str = DEFAULT_OPENAI_VISION_MODEL
    detail: str = "low"
    max_side: int = 768
    quality: int = 90
    instruction: str = DEFAULT_CAPTION_INSTRUCTION

    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        validate_openai_api_key()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is required for PUNC captioning. Run `uv sync --extra tifa`."
            ) from exc

        prompt_hint = ""
        if prompt:
            prompt_hint = (
                "\nOriginal generation prompt, for context only. Describe the image, "
                "not what the prompt asks for:\n"
                f"{prompt}"
            )

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{self.instruction}{prompt_hint}",
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url(
                                image,
                                max_side=self.max_side,
                                quality=self.quality,
                            ),
                            "detail": self.detail,
                        },
                    ],
                }
            ],
            max_output_tokens=140,
        )
        caption = response.output_text.strip()
        if not caption:
            raise RuntimeError("OpenAI returned an empty caption.")
        return caption
