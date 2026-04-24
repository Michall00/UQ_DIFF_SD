"""
experiments/stable_diffusion/run_sd_laplace.py
-----------------------------------------------
Last-Layer Laplace + FLARE uncertainty quantification for Stable Diffusion.

Fits a diagonal LLLA on UNet's conv_out (~11.5K params), then accumulates
per-step epistemic variance through DDIM reverse diffusion via FLARE transport:

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
                   help="DDIM inference steps")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Reverse sampler used for image generation and FLARE transport.",
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
    var = posterior_var.to(features.device, features.dtype)

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
            z_t = torch.sqrt(abar_t) * z0 + torch.sqrt(1 - abar_t) * eps

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
    for i in range(n_z0):
        scheduler.set_timesteps(steps)
        gen = torch.Generator(device=device).manual_seed(base_seed + i)
        z = torch.randn(1, *latent_shape, device=device, generator=gen)

        for t in tqdm(scheduler.timesteps, desc=f"  z0 [{i+1}/{n_z0}]", leave=False):
            z_in = torch.cat([z, z])
            emb = torch.cat([uncond_emb, text_emb])
            pred = unet(z_in, t, encoder_hidden_states=emb).sample
            eu, ec = pred.chunk(2)
            eps = eu + guidance_scale * (ec - eu)
            z = scheduler.step(eps, t, z).prev_sample

        z0_list.append(z)
    return torch.cat(z0_list)


@torch.no_grad()
def sample_with_flare(
    unet, laplace: ManualDiagLaplace, scheduler, text_emb, uncond_emb, abar,
    latent_shape, steps, guidance_scale, device, seed,
    feat_cap, scheduler_name,
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
    z = torch.randn(1, *latent_shape, device=device, generator=gen)

    Var_proj = torch.zeros_like(z)
    cum_a2 = torch.ones(1, 1, 1, 1, device=device)

    timesteps = list(scheduler.timesteps)

    for i, t in enumerate(tqdm(timesteps, desc="  FLARE sample", leave=False)):
        t_int = t.item()

        z_in = torch.cat([z, z])
        emb = torch.cat([uncond_emb, text_emb])
        pred = unet(z_in, t, encoder_hidden_states=emb).sample
        eu, ec = pred.chunk(2)
        eps = eu + guidance_scale * (ec - eu)

        cond_feats = feat_cap.features[1:2]
        gamma2_t = gamma2_conv2d(cond_feats, laplace.posterior_variance, unet.conv_out)

        if scheduler_name == "ddim":
            a, b = ddim_transport_coeffs(abar, timesteps, i, device)
        elif scheduler_name == "ddpm":
            a, b = ddpm_transport_coeffs(scheduler, t_int)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        Var_proj = Var_proj + cum_a2 * (b ** 2) * gamma2_t
        cum_a2 = cum_a2 * (a ** 2)

        z = scheduler.step(eps, t, z, generator=gen).prev_sample

    return z, Var_proj


def decode_latent(vae, z, device):
    z = z / vae.config.scaling_factor
    img = vae.decode(z.to(device)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]


def plot_results(images, var_maps, prompt, out_dir):
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 14))
    if n == 1:
        axes = axes[:, None]

    fig.suptitle(f'Laplace-FLARE UQ — "{prompt}"', fontsize=13, y=0.98)

    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis("off")

        var_gray = var_maps[i].mean(axis=0)
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
        std_norm = std_up / (std_up.max() + 1e-10)
        overlay = images[i].copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = std_norm
        overlay = np.clip(0.6 * overlay + 0.4 * red, 0, 1)
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
    scheduler_cls = DDIMScheduler if args.scheduler == "ddim" else DDPMScheduler
    scheduler = scheduler_cls.from_pretrained(args.model_id, subfolder="scheduler")

    tok = pipe.tokenizer
    text_input = tok(
        args.prompt, padding="max_length",
        max_length=tok.model_max_length, truncation=True, return_tensors="pt",
    )
    text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = tok(
        "", padding="max_length",
        max_length=tok.model_max_length, truncation=True, return_tensors="pt",
    )
    uncond_emb = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    latent_shape = (unet.config.in_channels, args.height // 8, args.width // 8)
    feat_cap = FeatureCapture(unet.conv_out)
    abar = scheduler.alphas_cumprod.to(device)

    n_conv = sum(p.numel() for p in unet.conv_out.parameters())
    print(f"UNet conv_out: {unet.conv_out}")
    print(f"  Trainable params for LLLA: {n_conv}")

    print(f"Generating {args.n_z0} reference latents via DDIM ({args.steps} steps)...")
    t0 = time.time()
    z0_set = generate_z0(
        unet, scheduler, text_emb, uncond_emb, latent_shape,
        args.n_z0, args.steps, args.guidance_scale, device, args.seed,
    )
    print(f"  Done in {time.time()-t0:.0f}s. z0_set shape: {z0_set.shape}")

    print(f"Fitting manual diagonal Laplace on conv_out ({n_conv} params)...")
    t0 = time.time()
    laplace = ManualDiagLaplace(unet.conv_out)
    laplace.fit_from_unet(
        unet, feat_cap, z0_set, abar, text_emb,
        T=1000, n_pairs=args.n_lap_pairs, device=device,
    )
    laplace.optimize_prior()
    print(f"  Hessian + prior done in {time.time()-t0:.0f}s")
    print(f"  prior_prec = {laplace.prior_prec:.2e}")

    images = []
    var_maps = []
    var_mean = []
    var_sum = []
    var_p95 = []
    for i in range(args.n_samples):
        seed = args.seed + 100 + i
        print(f"Sampling image {i+1}/{args.n_samples} (seed={seed})...")
        t0 = time.time()
        z0, var_proj = sample_with_flare(
            unet, laplace, scheduler, text_emb, uncond_emb, abar,
            latent_shape, args.steps, args.guidance_scale,
            device, seed, feat_cap, args.scheduler,
        )
        print(f"  Done in {time.time()-t0:.0f}s")

        img = decode_latent(vae, z0, device)
        var_np = var_proj.squeeze(0).detach().cpu().numpy()
        images.append(img)
        var_maps.append(var_np)
        var_mean.append(float(var_np.mean()))
        var_sum.append(float(var_np.sum()))
        var_p95.append(float(np.percentile(var_np, 95)))

    np.savez_compressed(
        os.path.join(args.out_dir, "laplace_results.npz"),
        images=np.stack(images),
        var_maps=np.stack(var_maps),
        var_mean=np.array(var_mean, dtype=np.float32),
        var_sum=np.array(var_sum, dtype=np.float32),
        var_p95=np.array(var_p95, dtype=np.float32),
        prompt=np.array(args.prompt),
    )
    print(f"Saved results to {args.out_dir}/laplace_results.npz")

    plot_results(images, var_maps, args.prompt, args.out_dir)
    feat_cap.remove()
    print("Done!")


if __name__ == "__main__":
    main()
