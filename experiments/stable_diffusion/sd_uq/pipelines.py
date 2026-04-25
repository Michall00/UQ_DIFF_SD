"""Pipeline loading and conditioning helpers for SD/SDXL UQ experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


@dataclass
class DiffusionConditioning:
    text_emb: torch.Tensor
    uncond_emb: torch.Tensor
    cond_kwargs: dict[str, torch.Tensor] | None = None
    uncond_kwargs: dict[str, torch.Tensor] | None = None

    def cond_forward_kwargs(self) -> dict[str, dict[str, torch.Tensor]]:
        if not self.cond_kwargs:
            return {}
        return {"added_cond_kwargs": self.cond_kwargs}

    def cfg_forward_kwargs(self) -> dict[str, dict[str, torch.Tensor]]:
        if not self.cond_kwargs:
            return {}
        if not self.uncond_kwargs:
            raise ValueError("CFG conditioning requires unconditional added kwargs.")
        return {
            "added_cond_kwargs": {
                key: torch.cat([self.uncond_kwargs[key], value], dim=0)
                for key, value in self.cond_kwargs.items()
            }
        }


def cfg_embeddings(cond: DiffusionConditioning) -> torch.Tensor:
    return torch.cat([cond.uncond_emb, cond.text_emb])


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def infer_pipeline_family(model_id: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lower = model_id.lower()
    if "sdxl" in lower or "xl" in lower:
        return "sdxl"
    return "sd"


def load_pipeline(model_id: str, family: str, torch_dtype: torch.dtype):
    pipeline_cls = StableDiffusionXLPipeline if family == "sdxl" else StableDiffusionPipeline
    return pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype)


def encode_conditioning(pipe, family: str, prompt: str, height: int, width: int, device: str):
    if family == "sdxl":
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        projection_dim = pipe.text_encoder_2.config.projection_dim
        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        ).to(device)
        return DiffusionConditioning(
            text_emb=prompt_embeds.detach(),
            uncond_emb=negative_prompt_embeds.detach(),
            cond_kwargs={
                "text_embeds": pooled_prompt_embeds.detach(),
                "time_ids": add_time_ids,
            },
            uncond_kwargs={
                "text_embeds": negative_pooled_prompt_embeds.detach(),
                "time_ids": add_time_ids,
            },
        ), None, None

    tok = pipe.tokenizer
    text_input = tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0].detach()

    uncond_input = tok(
        "",
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_emb = pipe.text_encoder(uncond_input.input_ids.to(device))[0].detach()

    return DiffusionConditioning(text_emb=text_emb, uncond_emb=uncond_emb), tok, text_input

