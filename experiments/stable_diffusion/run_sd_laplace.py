"""
experiments/stable_diffusion/run_sd_laplace.py
-----------------------------------------------
Laplace + FLARE uncertainty quantification for Stable Diffusion.

Supports two Laplace modes:
  - last_layer: diagonal LLLA on UNet's conv_out (~11.5K params)
  - subnet: random network-wide diagonal subnetwork Laplace with MC gamma2

Then accumulates per-step epistemic variance through reverse diffusion via
FLARE transport:

    u_proj = Σ_t (Π_{s>t} a_s)² · b_t² · γ²_t

where γ²_t = diag(J_t Σ J_t^T) is the Laplace predictive variance at step t,
computed efficiently for Conv2d via unfolded patches.

Usage:
    uv run python experiments/stable_diffusion/run_sd_laplace.py \\
        --prompt "a photo of a cat sitting on a chair" \\
        --n_samples 2 --device mps

    uv run python experiments/stable_diffusion/run_sd_laplace.py \\
        --prompt "a human hand with five fingers" \\
        --n_samples 2 --steps 30 --device mps
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="a photo of a cat sitting on a chair")
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--n_samples", type=int, default=2,
                   help="Images to generate with uncertainty")
    p.add_argument("--n_z0", type=int, default=2,
                   help="Reference latents for Laplace dataset")
    p.add_argument("--n_lap_pairs", type=int, default=50,
                   help="Regression pairs for Laplace fitting")
    p.add_argument("--steps", type=int, default=30,
                   help="Reverse diffusion inference steps")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Reverse sampler used for image generation and FLARE transport.",
    )
    p.add_argument(
        "--laplace_mode",
        type=str,
        default="last_layer",
        choices=["last_layer", "subnet"],
        help="Use conv_out LLLA or a random network-wide UNet subnetwork.",
    )
    p.add_argument(
        "--subnet_n_params",
        type=int,
        default=50_000,
        help="Number of selected scalar parameters for --laplace_mode subnet.",
    )
    p.add_argument(
        "--subnet_max_tensors",
        type=int,
        default=12,
        help="Maximum parameter tensors selected across UNet blocks for subnet Laplace.",
    )
    p.add_argument(
        "--subnet_mc_samples",
        type=int,
        default=2,
        help="Monte Carlo weight perturbations per diffusion step for subnet gamma2.",
    )
    p.add_argument(
        "--attention_aggregation",
        type=str,
        default="none",
        choices=["none", "cross"],
        help="Optional attention-aware uncertainty aggregation.",
    )
    p.add_argument(
        "--attention_token_indices",
        type=str,
        default="",
        help="Comma-separated prompt token indices for cross-attention weighting. Empty means all non-special prompt tokens.",
    )
    p.add_argument(
        "--save_attention_maps",
        action="store_true",
        help="Store per-sample cross-attention maps in laplace_results.npz.",
    )
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. Use auto for float16 on CUDA and float32 elsewhere.",
    )
    p.add_argument("--out_dir", type=str, default="assets/stable_diffusion/laplace")
    return p.parse_args()


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def parse_token_indices(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_attention_token_weights(tokenizer, text_input, token_indices: list[int]):
    input_ids = text_input.input_ids[0]
    attn_mask = getattr(text_input, "attention_mask", None)
    if attn_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid = attn_mask[0].bool()

    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    if token_indices:
        for idx in token_indices:
            if 0 <= idx < weights.numel():
                weights[idx] = 1.0
    else:
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        for i, token_id in enumerate(input_ids.tolist()):
            if valid[i] and token_id not in special_ids:
                weights[i] = 1.0
        if weights.sum() == 0:
            weights[valid] = 1.0

    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    used_tokens = [tokens[i] for i, value in enumerate(weights.tolist()) if value > 0]
    return weights, tokens, used_tokens


class CrossAttentionStore:
    def __init__(self):
        self.enabled = False
        self.latent_hw: tuple[int, int] | None = None
        self.token_weights: torch.Tensor | None = None
        self._step_maps: list[torch.Tensor] = []
        self._sample_maps: list[torch.Tensor] = []

    def start_step(self, latent_hw: tuple[int, int], token_weights: torch.Tensor):
        self.enabled = True
        self.latent_hw = latent_hw
        self.token_weights = token_weights
        self._step_maps = []

    def finish_step(self):
        self.enabled = False
        if not self._step_maps:
            return
        step_map = torch.stack(self._step_maps).mean(dim=0)
        self._sample_maps.append(step_map.cpu())
        self._step_maps = []

    def reset_sample(self):
        self.enabled = False
        self._step_maps = []
        self._sample_maps = []

    def sample_map(self) -> torch.Tensor | None:
        if not self._sample_maps:
            return None
        attn_map = torch.stack(self._sample_maps).mean(dim=0)
        attn_map = attn_map.clamp_min(0)
        total = attn_map.sum()
        if total > 0:
            attn_map = attn_map / total
        return attn_map

    def add(self, attn_probs: torch.Tensor):
        if not self.enabled or self.latent_hw is None or self.token_weights is None:
            return
        if attn_probs.ndim != 4:
            return

        batch, _, query_len, key_len = attn_probs.shape
        if key_len != self.token_weights.numel():
            return

        side = int(query_len ** 0.5)
        if side * side != query_len:
            return

        cond_idx = 1 if batch > 1 else 0
        token_weights = self.token_weights.to(
            device=attn_probs.device, dtype=torch.float32
        )
        if token_weights.sum() <= 0:
            return

        selected = (
            attn_probs[cond_idx].float()
            * token_weights.view(1, 1, key_len)
        ).sum(dim=-1)
        spatial = selected.mean(dim=0).view(1, 1, side, side)
        spatial = F.interpolate(
            spatial,
            size=self.latent_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        self._step_maps.append(spatial.detach().cpu())


class CaptureCrossAttnProcessor:
    def __init__(self, store: CrossAttentionStore):
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        scale = getattr(attn, "scale", head_dim ** -0.5)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = attention_scores.float().softmax(dim=-1).to(query.dtype)
        if is_cross:
            self.store.add(attention_probs)

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_cross_attention_capture(unet, store: CrossAttentionStore):
    processors = {}
    for name in unet.attn_processors.keys():
        if "attn2" in name:
            processors[name] = CaptureCrossAttnProcessor(store)
        else:
            processors[name] = AttnProcessor2_0()
    unet.set_attn_processor(processors)


def attention_weighted_scores(var_map: np.ndarray, attention_map: np.ndarray):
    var_gray = np.asarray(var_map, dtype=np.float32).mean(axis=0)
    attn = np.asarray(attention_map, dtype=np.float32)
    if attn.shape != var_gray.shape:
        attn_t = torch.from_numpy(attn).view(1, 1, *attn.shape).float()
        attn = F.interpolate(
            attn_t,
            size=var_gray.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
    attn = np.clip(attn, 0, None)
    attn_sum = float(attn.sum())
    if attn_sum <= 0:
        attn = np.full_like(var_gray, 1.0 / var_gray.size)
    else:
        attn = attn / attn_sum
    weighted_mean = float((var_gray * attn).sum())
    weighted_sum = float(weighted_mean * var_gray.size)
    return weighted_mean, weighted_sum


def gamma2_conv2d(
    features: torch.Tensor,
    posterior_var: torch.Tensor,
    conv: nn.Conv2d,
) -> torch.Tensor:
    """
    Per-output epistemic variance for a Conv2d last layer with diagonal
    Laplace posterior.

    features      : (B, C_in, H, W) — input to conv_out captured via hook
    posterior_var  : (P,) diagonal posterior variance vector
    conv           : the Conv2d layer

    Returns: (B, C_out, H, W) gamma2 map, clamped >= 0
    """
    features = features.float()
    var = posterior_var.to(features.device, torch.float32)

    C_out, C_in, kH, kW = conv.weight.shape
    patches = F.unfold(features, (kH, kW), padding=conv.padding)
    B, CkK, L = patches.shape

    n_w = C_out * CkK
    W_var = var[:n_w].view(C_out, CkK)
    b_var = var[n_w:n_w + C_out] if conv.bias is not None else 0.0

    gamma2 = torch.einsum("bpl,cp->bcl", patches.pow(2), W_var)
    if conv.bias is not None:
        gamma2 = gamma2 + b_var.view(1, C_out, 1)

    H_out, W_out = features.shape[2], features.shape[3]
    return gamma2.view(B, C_out, H_out, W_out).clamp_min(0)


def ddim_transport_coeffs(
    abar: torch.Tensor,
    timesteps: list[torch.Tensor],
    i: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_int = timesteps[i].item()
    t_next = timesteps[i + 1].item() if i < len(timesteps) - 1 else 0

    ab_t = abar[t_int]
    ab_next = abar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

    a = torch.sqrt(ab_next / ab_t.clamp_min(1e-12))
    b = (
        torch.sqrt(1 - ab_next)
        - torch.sqrt(ab_next * (1 - ab_t) / ab_t.clamp_min(1e-12))
    )
    return a, b


def ddpm_transport_coeffs(
    scheduler: DDPMScheduler,
    t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear DDPM transport coefficients for epsilon prediction.

    DDPM adds sampler noise in scheduler.step(). FLARE uses only the epistemic
    term transported through eps_theta, so the extra sampler variance is not
    accumulated here.
    """
    alpha_prod_t = scheduler.alphas_cumprod[t]
    prev_t = scheduler.previous_timestep(t)
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t]
        if prev_t >= 0
        else scheduler.one.to(alpha_prod_t.device)
    )

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_x0_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t.clamp_min(1e-12)
    current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t.clamp_min(1e-12)

    a = pred_x0_coeff / alpha_prod_t.sqrt().clamp_min(1e-12) + current_sample_coeff
    b = -pred_x0_coeff * beta_prod_t.sqrt() / alpha_prod_t.sqrt().clamp_min(1e-12)
    return a, b


def _subnet_group_name(param_name: str) -> str:
    parts = param_name.split(".")
    if len(parts) >= 2 and parts[0] in {"down_blocks", "up_blocks"}:
        return ".".join(parts[:2])
    if parts[0] == "mid_block":
        return "mid_block"
    return parts[0]


class RandomSubnetDiagLaplace:
    """
    Diagonal Laplace over a random network-wide UNet subnetwork.

    Fitting uses an empirical diagonal Fisher on selected scalar parameters.
    Prediction uses Monte Carlo weight perturbations from the diagonal posterior
    to estimate diag(J Sigma J^T) in latent space. This avoids per-pixel
    Jacobians for Stable Diffusion while still moving beyond conv_out-only LLLA.
    """

    def __init__(
        self,
        unet: nn.Module,
        n_params: int,
        max_tensors: int,
        seed: int,
        prior_prec: float = 1.0,
    ):
        self.unet = unet
        self.n_params_target = n_params
        self.max_tensors = max_tensors
        self.seed = seed
        self.prior_prec = prior_prec
        self.selected = self._select_params()
        self.n_data = 0

    def _candidate_params(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
        for name, param in self.unet.named_parameters():
            if not param.requires_grad or not torch.is_floating_point(param):
                continue
            if param.ndim < 2:
                continue
            if name.startswith("conv_out."):
                continue
            if not (
                name.startswith("conv_in.")
                or name.startswith("time_embedding.")
                or name.startswith("down_blocks.")
                or name.startswith("mid_block.")
                or name.startswith("up_blocks.")
            ):
                continue
            groups.setdefault(_subnet_group_name(name), []).append((name, param))
        return groups

    def _select_params(self) -> list[dict[str, object]]:
        rng = np.random.default_rng(self.seed)
        groups = self._candidate_params()
        group_names = list(groups)
        rng.shuffle(group_names)
        for group in group_names:
            rng.shuffle(groups[group])

        chosen: list[tuple[str, nn.Parameter]] = []
        while len(chosen) < self.max_tensors and any(groups.values()):
            for group in group_names:
                if groups[group]:
                    chosen.append(groups[group].pop())
                    if len(chosen) >= self.max_tensors:
                        break

        if not chosen:
            raise RuntimeError("No eligible UNet tensors found for subnet Laplace.")

        per_tensor = int(np.ceil(self.n_params_target / len(chosen)))
        selected: list[dict[str, object]] = []
        remaining = self.n_params_target

        for name, param in chosen:
            if remaining <= 0:
                break
            n_select = min(param.numel(), per_tensor, remaining)
            if n_select <= 0:
                continue
            idx_np = rng.choice(param.numel(), size=n_select, replace=False)
            idx_np.sort()
            selected.append(
                {
                    "name": name,
                    "param": param,
                    "idx": torch.from_numpy(idx_np).long(),
                    "H": torch.zeros(n_select, dtype=torch.float32),
                    "posterior_var": torch.empty(n_select, dtype=torch.float32),
                }
            )
            remaining -= n_select

        if not selected:
            raise RuntimeError("Empty random subnetwork after parameter selection.")
        return selected

    @property
    def n_params(self) -> int:
        return sum(int(item["idx"].numel()) for item in self.selected)

    def describe(self):
        print(f"Random subnet tensors: {len(self.selected)}")
        print(f"  Selected scalar params: {self.n_params:,}")
        for item in self.selected:
            print(f"    {item['name']}: {item['idx'].numel():,}")

    def _set_fit_requires_grad(self) -> list[bool]:
        original = [p.requires_grad for p in self.unet.parameters()]
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for item in self.selected:
            param = item["param"]
            param.requires_grad_(True)
        return original

    def _restore_requires_grad(self, original: list[bool]):
        for param, requires_grad in zip(self.unet.parameters(), original):
            param.requires_grad_(requires_grad)

    def fit_from_unet(self, unet, z0_set, abar, text_emb, T, n_pairs, device):
        n_z0 = z0_set.shape[0]
        latent_dtype = next(unet.parameters()).dtype
        original_requires_grad = self._set_fit_requires_grad()
        unet.train(False)

        try:
            for _ in tqdm(range(n_pairs), desc="  Subnet Fisher"):
                unet.zero_grad(set_to_none=True)

                idx = torch.randint(0, n_z0, (1,)).item()
                z0 = z0_set[idx:idx + 1]
                t = torch.randint(0, T, (1,), device=device)
                abar_t = abar[t].view(1, 1, 1, 1)
                eps = torch.randn_like(z0)
                z_t = (
                    torch.sqrt(abar_t) * z0.float()
                    + torch.sqrt(1 - abar_t) * eps.float()
                ).to(dtype=latent_dtype)

                emb = text_emb.expand(1, -1, -1)
                pred = unet(z_t, t, encoder_hidden_states=emb).sample
                loss = 0.5 * (pred.float() - eps.float()).pow(2).mean()
                loss.backward()

                for item in self.selected:
                    param = item["param"]
                    grad = param.grad
                    if grad is None:
                        continue
                    sel_idx = item["idx"].to(grad.device)
                    h = grad.detach().flatten()[sel_idx].float().pow(2)
                    item["H"] = item["H"].to(h.device) + h

                self.n_data += 1
        finally:
            self._restore_requires_grad(original_requires_grad)
            unet.zero_grad(set_to_none=True)

        if self.n_data > 0:
            for item in self.selected:
                item["H"] = item["H"] / self.n_data

    def optimize_prior(self):
        theta2_sum = torch.tensor(0.0)
        for item in self.selected:
            param = item["param"]
            sel_idx = item["idx"].to(param.device)
            theta = param.detach().flatten()[sel_idx].float()
            theta2_sum = theta2_sum.to(theta.device) + theta.pow(2).sum()
        self.prior_prec = float(self.n_params / (theta2_sum + 1e-8))
        self._update_posterior_variance()

    def _update_posterior_variance(self):
        for item in self.selected:
            prec = item["H"].float() + self.prior_prec
            item["posterior_var"] = 1.0 / prec.clamp_min(1e-12)

    def perturb_(self, generator: torch.Generator) -> list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]]:
        perturbations = []
        for item in self.selected:
            param = item["param"]
            device = param.device
            flat = param.data.view(-1)
            sel_idx = item["idx"].to(device)
            var = item["posterior_var"].to(device=device, dtype=torch.float32)
            noise = torch.randn(
                var.shape, device=device, dtype=torch.float32, generator=generator
            ) * var.sqrt()
            noise_param = noise.to(dtype=param.dtype)
            flat[sel_idx] += noise_param
            perturbations.append((param, sel_idx, noise_param))
        return perturbations

    def restore_(self, perturbations: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]]):
        for param, sel_idx, noise in reversed(perturbations):
            param.data.view(-1)[sel_idx] -= noise


class ManualDiagLaplace:
    """
    Diagonal GGN-Laplace for a single Conv2d layer.

    Computes H_diag ≈ (1/N) Σ_i J_i^T J_i  element-wise (diagonal GGN).
    For Conv2d, J is structured: each output pixel depends on a local patch.

    posterior_precision = H_diag + prior_prec * I
    posterior_variance  = 1 / posterior_precision
    """

    def __init__(self, conv: nn.Conv2d, prior_prec: float = 1.0):
        self.conv = conv
        self.prior_prec = prior_prec
        n_params = sum(p.numel() for p in conv.parameters())
        self.H_diag = torch.zeros(n_params)
        self.n_data = 0

    @torch.no_grad()
    def accumulate(self, features: torch.Tensor, residuals: torch.Tensor):
        """
        Accumulate diagonal GGN from one batch.

        features  : (B, C_in, H, W) — input to conv_out
        residuals : (B, C_out, H, W) — (prediction - target) at conv_out output
        """
        conv = self.conv
        features = features.float()
        C_out, C_in, kH, kW = conv.weight.shape

        patches = F.unfold(features, (kH, kW), padding=conv.padding)
        B, CkK, L = patches.shape

        p2 = patches.pow(2).sum(dim=(0, 2))
        H_w = p2.unsqueeze(0).expand(C_out, -1).reshape(-1)

        if conv.bias is not None:
            H_b = torch.full((C_out,), float(B * L), device=features.device)
            H_diag = torch.cat([H_w, H_b])
        else:
            H_diag = H_w

        self.H_diag = self.H_diag.to(H_diag.device) + H_diag
        self.n_data += B * L

    def fit_from_unet(self, unet, feat_cap, z0_set, abar, text_emb, T, n_pairs, device):
        """
        Build diagonal Hessian by running forward diffusion + UNet forward.
        """
        n_z0 = z0_set.shape[0]
        for i in tqdm(range(n_pairs), desc="  Laplace Hessian"):
            idx = torch.randint(0, n_z0, (1,)).item()
            z0 = z0_set[idx:idx+1]
            t = torch.randint(0, T, (1,), device=device)
            abar_t = abar[t].view(1, 1, 1, 1)
            eps = torch.randn_like(z0)
            z_t = (
                torch.sqrt(abar_t) * z0.float()
                + torch.sqrt(1 - abar_t) * eps.float()
            ).to(dtype=z0.dtype)

            emb = text_emb.expand(1, -1, -1)
            pred = unet(z_t, t, encoder_hidden_states=emb).sample

            feats = feat_cap.features
            residual = pred - eps
            self.accumulate(feats, residual)

        if self.n_data > 0:
            self.H_diag = self.H_diag / self.n_data

    def optimize_prior(self):
        """Simple empirical Bayes: prior_prec = P / sum(θ²)."""
        theta = torch.cat([p.detach().reshape(-1) for p in self.conv.parameters()])
        self.prior_prec = float(theta.numel() / (theta.pow(2).sum() + 1e-8))

    @property
    def posterior_variance(self) -> torch.Tensor:
        prec = self.H_diag + self.prior_prec
        return (1.0 / prec.clamp_min(1e-12))


class FeatureCapture:
    def __init__(self, module: nn.Module):
        self.features: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.features = inp[0].detach()

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def generate_z0(unet, scheduler, text_emb, uncond_emb, latent_shape,
                n_z0, steps, guidance_scale, device, base_seed):
    z0_list = []
    latent_dtype = next(unet.parameters()).dtype
    for i in range(n_z0):
        scheduler.set_timesteps(steps)
        gen = torch.Generator(device=device).manual_seed(base_seed + i)
        z = torch.randn(
            1, *latent_shape, device=device, dtype=latent_dtype, generator=gen
        )

        for t in tqdm(scheduler.timesteps, desc=f"  z0 [{i+1}/{n_z0}]", leave=False):
            z_in = torch.cat([z, z])
            emb = torch.cat([uncond_emb, text_emb])
            pred = unet(z_in, t, encoder_hidden_states=emb).sample
            eu, ec = pred.chunk(2)
            eps = eu + guidance_scale * (ec - eu)
            z = scheduler.step(eps, t, z, generator=gen).prev_sample.to(dtype=latent_dtype)

        z0_list.append(z)
    return torch.cat(z0_list)


@torch.no_grad()
def gamma2_subnet_mc(
    unet,
    laplace: RandomSubnetDiagLaplace,
    z: torch.Tensor,
    t: torch.Tensor,
    text_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float,
    base_eps: torch.Tensor,
    n_mc: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if n_mc <= 0:
        raise ValueError("--subnet_mc_samples must be > 0 for subnet Laplace.")

    z_in = torch.cat([z, z])
    emb = torch.cat([uncond_emb, text_emb])
    gamma2 = torch.zeros_like(base_eps, dtype=torch.float32)

    for _ in range(n_mc):
        perturbations = laplace.perturb_(generator)
        try:
            pred = unet(z_in, t, encoder_hidden_states=emb).sample
            eu, ec = pred.chunk(2)
            eps = eu + guidance_scale * (ec - eu)
            gamma2 = gamma2 + (eps.float() - base_eps.float()).pow(2)
        finally:
            laplace.restore_(perturbations)

    return (gamma2 / n_mc).clamp_min(0)


@torch.no_grad()
def sample_with_flare(
    unet, laplace, scheduler, text_emb, uncond_emb, abar,
    latent_shape, steps, guidance_scale, device, seed,
    feat_cap, scheduler_name, laplace_mode, subnet_mc_samples,
    attention_store=None, attention_token_weights=None,
):
    """DDIM sampling with FLARE epistemic uncertainty transport.

    At each DDIM step:
    1. CFG forward pass (hook captures features for uncond+cond)
    2. Compute per-pixel gamma2_t from conditional features via diagonal Laplace
    3. Compute DDIM transport factors a_t, b_t from alpha_bar schedule
    4. Accumulate: Var_proj += cum_a2 * b_t^2 * gamma2_t; cum_a2 *= a_t^2
    5. Execute DDIM step

    Returns (z0, Var_proj) where Var_proj is the accumulated epistemic variance.
    """
    scheduler.set_timesteps(steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    laplace_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    latent_dtype = next(unet.parameters()).dtype
    z = torch.randn(
        1, *latent_shape, device=device, dtype=latent_dtype, generator=gen
    )

    Var_proj = torch.zeros_like(z, dtype=torch.float32)
    cum_a2 = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float32)

    timesteps = list(scheduler.timesteps)
    if attention_store is not None:
        attention_store.reset_sample()

    for i, t in enumerate(tqdm(timesteps, desc="  FLARE sample", leave=False)):
        t_int = t.item()

        z_in = torch.cat([z, z])
        emb = torch.cat([uncond_emb, text_emb])
        if attention_store is not None and attention_token_weights is not None:
            attention_store.start_step(latent_shape[1:], attention_token_weights)
        pred = unet(z_in, t, encoder_hidden_states=emb).sample
        if attention_store is not None:
            attention_store.finish_step()
        eu, ec = pred.chunk(2)
        eps = eu + guidance_scale * (ec - eu)

        if laplace_mode == "last_layer":
            cond_feats = feat_cap.features[1:2]
            gamma2_t = gamma2_conv2d(cond_feats, laplace.posterior_variance, unet.conv_out)
        elif laplace_mode == "subnet":
            gamma2_t = gamma2_subnet_mc(
                unet, laplace, z, t, text_emb, uncond_emb,
                guidance_scale, eps, subnet_mc_samples, laplace_gen,
            )
        else:
            raise ValueError(f"Unsupported Laplace mode: {laplace_mode}")

        if scheduler_name == "ddim":
            a, b = ddim_transport_coeffs(abar, timesteps, i, device)
        elif scheduler_name == "ddpm":
            a, b = ddpm_transport_coeffs(scheduler, t_int)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        Var_proj = Var_proj + cum_a2 * (b ** 2) * gamma2_t
        cum_a2 = cum_a2 * (a ** 2)

        z = scheduler.step(eps, t, z, generator=gen).prev_sample.to(dtype=latent_dtype)

    attention_map = None
    if attention_store is not None:
        attention_map = attention_store.sample_map()
    return z, Var_proj, attention_map


def decode_latent(vae, z, device):
    z = z / vae.config.scaling_factor
    img = vae.decode(z.to(device)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return img.detach().float().cpu().permute(0, 2, 3, 1).numpy()[0]


def plot_results(images, var_maps, prompt, out_dir):
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 14))
    if n == 1:
        axes = axes[:, None]

    fig.suptitle(f'Laplace-FLARE UQ — "{prompt}"', fontsize=13, y=0.98)

    for i in range(n):
        image = np.asarray(images[i], dtype=np.float32)
        var_map = np.asarray(var_maps[i], dtype=np.float32)

        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis("off")

        var_gray = var_map.mean(axis=0).astype(np.float32)
        im = axes[1, i].imshow(var_gray, cmap="hot", interpolation="bilinear")
        axes[1, i].set_title(f"γ² FLARE (latent)")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        std_up = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(var_gray).unsqueeze(0).unsqueeze(0).float(),
                size=images[i].shape[:2], mode="bilinear", align_corners=False,
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
    path = os.path.join(out_dir, "sd_laplace_flare.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def main():
    """Run the full Laplace-FLARE pipeline for Stable Diffusion.

    Steps:
        1. Load SD pipeline (UNet, VAE, DDIM scheduler)
        2. Encode prompt + unconditional embedding for CFG
        3. Setup latent shape, feature capture hook, alpha_bar schedule
        4. Generate reference z0 latents via DDIM
        5. Fit diagonal Laplace on conv_out via forward diffusion pairs
        6. Sample images with FLARE uncertainty transport
        7. Save results (.npz) and plot
    """
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_id}...")
    torch_dtype = resolve_torch_dtype(args.torch_dtype, device)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    unet = pipe.unet
    vae = pipe.vae
    unet.eval()
    vae.eval()
    pipe.text_encoder.eval()
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    scheduler_cls = DDIMScheduler if args.scheduler == "ddim" else DDPMScheduler
    scheduler = scheduler_cls.from_pretrained(args.model_id, subfolder="scheduler")

    tok = pipe.tokenizer
    text_input = tok(
        args.prompt, padding="max_length",
        max_length=tok.model_max_length, truncation=True,
        return_attention_mask=True, return_tensors="pt",
    )
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0].detach()

    uncond_input = tok(
        "", padding="max_length",
        max_length=tok.model_max_length, truncation=True,
        return_attention_mask=True, return_tensors="pt",
    )
    with torch.no_grad():
        uncond_emb = pipe.text_encoder(uncond_input.input_ids.to(device))[0].detach()

    latent_shape = (unet.config.in_channels, args.height // 8, args.width // 8)
    feat_cap = FeatureCapture(unet.conv_out)
    abar = scheduler.alphas_cumprod.to(device)

    n_conv = sum(p.numel() for p in unet.conv_out.parameters())
    print(f"UNet conv_out: {unet.conv_out}")
    print(f"  Trainable params for LLLA: {n_conv}")

    print(
        f"Generating {args.n_z0} reference latents via "
        f"{args.scheduler.upper()} ({args.steps} steps)..."
    )
    t0 = time.time()
    z0_set = generate_z0(
        unet, scheduler, text_emb, uncond_emb, latent_shape,
        args.n_z0, args.steps, args.guidance_scale, device, args.seed,
    )
    print(f"  Done in {time.time()-t0:.0f}s. z0_set shape: {z0_set.shape}")

    if args.laplace_mode == "last_layer":
        print(f"Fitting manual diagonal Laplace on conv_out ({n_conv} params)...")
    else:
        print(
            "Fitting random subnet diagonal Laplace "
            f"({args.subnet_n_params:,} target params, "
            f"{args.subnet_max_tensors} tensors max)..."
        )
    t0 = time.time()
    if args.laplace_mode == "last_layer":
        laplace = ManualDiagLaplace(unet.conv_out)
        laplace.fit_from_unet(
            unet, feat_cap, z0_set, abar, text_emb,
            T=1000, n_pairs=args.n_lap_pairs, device=device,
        )
    else:
        laplace = RandomSubnetDiagLaplace(
            unet,
            n_params=args.subnet_n_params,
            max_tensors=args.subnet_max_tensors,
            seed=args.seed,
        )
        laplace.describe()
        laplace.fit_from_unet(
            unet, z0_set, abar, text_emb,
            T=1000, n_pairs=args.n_lap_pairs, device=device,
        )
    laplace.optimize_prior()
    print(f"  Hessian + prior done in {time.time()-t0:.0f}s")
    print(f"  prior_prec = {laplace.prior_prec:.2e}")

    attention_store = None
    attention_token_weights = None
    attention_tokens = []
    attention_all_tokens = []
    if args.attention_aggregation == "cross":
        token_indices = parse_token_indices(args.attention_token_indices)
        attention_token_weights, attention_all_tokens, attention_tokens = build_attention_token_weights(
            tok, text_input, token_indices
        )
        attention_store = CrossAttentionStore()
        install_cross_attention_capture(unet, attention_store)
        print("Cross-attention aggregation enabled.")
        print(f"  Attention tokens: {attention_tokens}")

    images = []
    var_maps = []
    attention_maps = []
    var_mean = []
    var_sum = []
    var_p95 = []
    var_attn_mean = []
    var_attn_sum = []
    for i in range(args.n_samples):
        seed = args.seed + 100 + i
        print(f"Sampling image {i+1}/{args.n_samples} (seed={seed})...")
        t0 = time.time()
        z0, var_proj, attention_map = sample_with_flare(
            unet, laplace, scheduler, text_emb, uncond_emb, abar,
            latent_shape, args.steps, args.guidance_scale,
            device, seed, feat_cap, args.scheduler,
            args.laplace_mode, args.subnet_mc_samples,
            attention_store, attention_token_weights,
        )
        print(f"  Done in {time.time()-t0:.0f}s")

        img = decode_latent(vae, z0, device)
        var_np = var_proj.squeeze(0).detach().cpu().numpy()
        images.append(img)
        var_maps.append(var_np)
        var_mean.append(float(var_np.mean()))
        var_sum.append(float(var_np.sum()))
        var_p95.append(float(np.percentile(var_np, 95)))
        if attention_map is not None:
            attn_np = attention_map.numpy().astype(np.float32)
            attention_maps.append(attn_np)
            attn_mean, attn_sum = attention_weighted_scores(var_np, attn_np)
            var_attn_mean.append(attn_mean)
            var_attn_sum.append(attn_sum)
        else:
            var_attn_mean.append(np.nan)
            var_attn_sum.append(np.nan)

    results = {
        "images": np.stack(images),
        "var_maps": np.stack(var_maps),
        "var_mean": np.array(var_mean, dtype=np.float32),
        "var_sum": np.array(var_sum, dtype=np.float32),
        "var_p95": np.array(var_p95, dtype=np.float32),
        "var_attn_mean": np.array(var_attn_mean, dtype=np.float32),
        "var_attn_sum": np.array(var_attn_sum, dtype=np.float32),
        "attention_tokens": np.array(attention_tokens),
        "attention_all_tokens": np.array(attention_all_tokens),
        "prompt": np.array(args.prompt),
    }
    if args.save_attention_maps and attention_maps:
        results["attention_maps"] = np.stack(attention_maps).astype(np.float32)

    np.savez_compressed(
        os.path.join(args.out_dir, "laplace_results.npz"),
        **results,
    )
    print(f"Saved results to {args.out_dir}/laplace_results.npz")

    plot_results(images, var_maps, args.prompt, args.out_dir)
    feat_cap.remove()
    print("Done!")


if __name__ == "__main__":
    main()
