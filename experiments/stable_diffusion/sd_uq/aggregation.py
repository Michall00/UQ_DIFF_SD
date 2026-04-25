"""Uncertainty and attention-map aggregation helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def latent_gray(var_map: np.ndarray) -> np.ndarray:
    """Return a 2D uncertainty map from a latent map with optional channels."""
    var = np.asarray(var_map, dtype=np.float32)
    if var.ndim == 3:
        return var.mean(axis=0).astype(np.float32)
    if var.ndim == 2:
        return var.astype(np.float32)
    raise ValueError(f"Expected 2D or 3D uncertainty map, got shape {var.shape}")


def uncertainty_stats(var_map: np.ndarray) -> dict[str, float]:
    var = np.asarray(var_map, dtype=np.float32)
    return {
        "mean": float(var.mean()),
        "sum": float(var.sum()),
        "p95": float(np.percentile(var, 95)),
    }


def normalize_attention_map(attention_map: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(attention_map, torch.Tensor):
        attn = attention_map.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attention_map, dtype=np.float32)
    attn = np.clip(attn, 0, None).astype(np.float32)
    total = float(attn.sum())
    if total > 0:
        attn = attn / total
    return attn.astype(np.float32)


def attention_weighted_scores(
    var_map: np.ndarray,
    attention_map: np.ndarray,
) -> tuple[float, float]:
    var_gray = latent_gray(var_map)
    attn = normalize_attention_map(attention_map)
    if attn.shape != var_gray.shape:
        attn_t = torch.from_numpy(attn).view(1, 1, *attn.shape).float()
        attn = F.interpolate(
            attn_t,
            size=var_gray.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        attn = normalize_attention_map(attn)

    if float(attn.sum()) <= 0:
        attn = np.full_like(var_gray, 1.0 / var_gray.size)

    weighted_mean = float((var_gray * attn).sum())
    weighted_sum = float(weighted_mean * var_gray.size)
    return weighted_mean, weighted_sum

