"""Plotting helpers for Stable Diffusion uncertainty experiments."""

from __future__ import annotations

import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from sd_uq.aggregation import latent_gray


def _as_rgb_float(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {arr.shape}.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size and finite.max() > 1.0:
        arr = arr / 255.0
    return np.nan_to_num(np.clip(arr, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


def save_heatmap_png(attn_map: np.ndarray, path: str):
    plt.figure(figsize=(5, 5))
    plt.imshow(attn_map, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_results(images, var_maps, prompt, out_dir, max_samples=16, filename="sd_laplace_flare.png"):
    if max_samples <= 0:
        print("Skipping plot (--plot_max_samples <= 0).")
        return
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 14))
    if n == 1:
        axes = axes[:, None]

    fig.suptitle(f'Laplace-FLARE UQ — "{prompt}"', fontsize=13, y=0.98)

    for i in range(n):
        image = _as_rgb_float(images[i])
        var_gray = latent_gray(var_maps[i])

        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis("off")

        im = axes[1, i].imshow(var_gray, cmap="hot", interpolation="bilinear")
        axes[1, i].set_title("UQ map (latent)")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        std_up = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(var_gray).unsqueeze(0).unsqueeze(0).float(),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        )
        std_norm = (std_up / (std_up.max() + 1e-10)).astype(np.float32)
        overlay = image.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = std_norm
        overlay = np.clip(0.6 * overlay + 0.4 * red, 0, 1).astype(np.float32)
        axes[2, i].imshow(overlay)
        axes[2, i].set_title("Image + uncertainty overlay")
        axes[2, i].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")
