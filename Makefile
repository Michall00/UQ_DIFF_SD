.PHONY: help sd-sync sd-smoke sd-last-ddim sd-last-ddpm sd-subnet-smoke sd-subnet-ddim sd-subnet-ddpm sd-show

MODEL_ID ?= CompVis/stable-diffusion-v1-4
TINY_MODEL_ID ?= hf-internal-testing/tiny-stable-diffusion-pipe
PROMPT ?= a human hand with five fingers
DEVICE ?= cuda
TORCH_DTYPE ?= auto
GUIDANCE_SCALE ?= 7.5
SEED ?= 42

STEPS_DDIM ?= 30
STEPS_DDPM ?= 100
HEIGHT ?= 512
WIDTH ?= 512
N_Z0 ?= 4
N_LAP_PAIRS ?= 100
N_SAMPLES ?= 8

SUBNET_N_PARAMS ?= 50000
SUBNET_MAX_TENSORS ?= 12
SUBNET_MC_SAMPLES ?= 2

SD_SCRIPT := experiments/stable_diffusion/run_sd_laplace.py

help:
	@echo "Stable Diffusion UQ targets:"
	@echo "  make sd-sync          install SD dependencies via uv"
	@echo "  make sd-smoke         tiny CPU last-layer smoke test"
	@echo "  make sd-last-ddim     SD v1.4 conv_out LLLA + DDIM"
	@echo "  make sd-last-ddpm     SD v1.4 conv_out LLLA + DDPM"
	@echo "  make sd-subnet-smoke  SD v1.4 small subnet sanity run"
	@echo "  make sd-subnet-ddim   SD v1.4 random subnet FLARE + DDIM"
	@echo "  make sd-subnet-ddpm   SD v1.4 random subnet FLARE + DDPM"
	@echo "  make sd-show OUT_DIR=...  print uncertainty ranking from results"
	@echo ""
	@echo "Common overrides:"
	@echo "  make sd-subnet-ddim PROMPT=\"a red cube on a blue sphere\" N_SAMPLES=16"
	@echo "  make sd-subnet-ddim SUBNET_N_PARAMS=100000 SUBNET_MC_SAMPLES=4"
	@echo "  make sd-subnet-ddpm STEPS_DDPM=1000"

sd-sync:
	uv sync --extra stable-diffusion --extra dev

sd-smoke:
	uv run python $(SD_SCRIPT) \
		--model_id "$(TINY_MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device cpu \
		--torch_dtype float32 \
		--scheduler ddim \
		--laplace_mode last_layer \
		--steps 1 \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 1 \
		--n_lap_pairs 1 \
		--n_samples 1 \
		--height 64 \
		--width 64 \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/smoke_tiny_last_layer

sd-last-ddim:
	uv run python $(SD_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--laplace_mode last_layer \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/last_layer_ddim

sd-last-ddpm:
	uv run python $(SD_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddpm \
		--laplace_mode last_layer \
		--steps $(STEPS_DDPM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/last_layer_ddpm

sd-subnet-smoke:
	uv run python $(SD_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--laplace_mode subnet \
		--steps 20 \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 2 \
		--n_lap_pairs 50 \
		--n_samples 2 \
		--subnet_n_params 10000 \
		--subnet_max_tensors 6 \
		--subnet_mc_samples 1 \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/subnet_smoke_ddim

sd-subnet-ddim:
	uv run python $(SD_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--laplace_mode subnet \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--subnet_n_params $(SUBNET_N_PARAMS) \
		--subnet_max_tensors $(SUBNET_MAX_TENSORS) \
		--subnet_mc_samples $(SUBNET_MC_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/subnet_ddim

sd-subnet-ddpm:
	uv run python $(SD_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddpm \
		--laplace_mode subnet \
		--steps $(STEPS_DDPM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--subnet_n_params $(SUBNET_N_PARAMS) \
		--subnet_max_tensors $(SUBNET_MAX_TENSORS) \
		--subnet_mc_samples $(SUBNET_MC_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/subnet_ddpm

sd-show:
	uv run python -c "import numpy as np; p='$(OUT_DIR)/laplace_results.npz'; d=np.load(p); order=np.argsort(d['var_mean']); print('file:', p); print('least uncertain:', order[:10], d['var_mean'][order[:10]]); print('most uncertain:', order[-10:][::-1], d['var_mean'][order[-10:][::-1]])"
