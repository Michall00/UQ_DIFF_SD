.PHONY: help all all-daam sd-sync sd-smoke sd-last-ddim sd-last-ddpm sd-subnet-smoke sd-subnet-ddim sd-subnet-ddpm sd-eval-ddim sd-tifa-eval sd-punc-eval sd-filtering-run sd-filtering-clip sd-filtering-ranking sd-filtering-tifa sd-filtering-punc sd-recap-prompts sd-recap-run sd-recap-tifa sd-recap-punc sd-benchmark-ddim sdxl-last-ddim sdxl-subnet-ddim sdxl-eval-ddim sdxl-benchmark-ddim sd-daam-ddim sd-daam-subnet-ddim sd-daam-bayesdiff-ddim sd-daam-bayesdiff-ddpm sd-daam-bayesdiff-subnet-ddim sd-daam-bayesdiff-subnet-ddpm sd-daam-show sd-show

MODEL_ID ?= CompVis/stable-diffusion-v1-4
SDXL_MODEL_ID ?= stabilityai/stable-diffusion-xl-base-1.0
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
SDXL_HEIGHT ?= 1024
SDXL_WIDTH ?= 1024
N_Z0 ?= 4
N_LAP_PAIRS ?= 100
N_SAMPLES ?= 8
PLOT_MAX_SAMPLES ?= 16

SUBNET_N_PARAMS ?= 50000
SUBNET_MAX_TENSORS ?= 12
SUBNET_MC_SAMPLES ?= 2
ATTENTION_AGGREGATION ?= none
ATTENTION_TOKEN_INDICES ?=
SAVE_ATTENTION_MAPS ?=
DAAM_WORDS ?=
DAAM_PYTHON ?= 3.11
DAAM_UNCERTAINTY_NPZ ?= assets/stable_diffusion/subnet_ddim/laplace_results.npz
DAAM_UNCERTAINTY_KEYS ?= var_maps
DAAM_BAYESDIFF_OUT_DIR ?= assets/stable_diffusion/daam_bayesdiff
DAAM_SCORE_KEY ?= daam_var_mean
DAAM_WITH := --python $(DAAM_PYTHON) --with daam==0.2.0 --with huggingface-hub==0.17.3 --with matplotlib
DAAM_SAVE_IMAGES ?= --save_images
DAAM_BAYESDIFF_SAVE_IMAGES ?= --save_daam_images
EVAL_NPZS ?= assets/stable_diffusion/last_layer_ddim/laplace_results.npz assets/stable_diffusion/subnet_ddim/laplace_results.npz
EVAL_OUT_DIR ?= assets/stable_diffusion/eval_ddim
SDXL_EVAL_NPZS ?= assets/stable_diffusion/sdxl_last_layer_ddim/laplace_results.npz assets/stable_diffusion/sdxl_subnet_ddim/laplace_results.npz
SDXL_EVAL_OUT_DIR ?= assets/stable_diffusion/eval_sdxl_ddim
EVAL_CLIP_MODEL ?= openai/clip-vit-base-patch32
EVAL_BATCH_SIZE ?= 16
EVAL_FILTER_FRACS ?= 0.1,0.2,0.3
TIFA_EVAL_RESULTS ?= $(EVAL_NPZS)
TIFA_OUT_DIR ?= assets/stable_diffusion/tifa_eval
TIFA_QUESTION_CACHE ?= $(TIFA_OUT_DIR)/tifa_questions.json
TIFA_OPENAI_MODEL ?= gpt-4.1
TIFA_TOGETHER_MODEL ?= meta-llama/Llama-3.3-70B-Instruct-Turbo
TIFA_QUESTION_SOURCE ?= openai
TIFA_MAX_SAMPLES ?= 0
TIFA_MAX_QUESTIONS ?= 0
TIFA_REQUIRE_CACHED ?=
PUNC_EVAL_RESULTS ?= $(EVAL_NPZS)
PUNC_OUT_DIR ?= assets/stable_diffusion/punc_eval
PUNC_CAPTION_CACHE ?= $(PUNC_OUT_DIR)/punc_captions.json
PUNC_OPENAI_MODEL ?= gpt-4.1-mini
PUNC_IMAGE_DETAIL ?= low
PUNC_MAX_SAMPLES ?= 0
PUNC_SIMILARITY ?= token
PUNC_REQUIRE_CACHED ?=
RECAP_PROMPTS ?= assets/stable_diffusion/recap_coco_prompts.jsonl
RECAP_OUT_ROOT ?= assets/stable_diffusion/recap_probe
RECAP_N_PROMPTS ?= 30
RECAP_METHODS ?= last_layer,subnet
RECAP_N_SAMPLES ?= 8
FILTER_PROMPTS ?= experiments/stable_diffusion/filtering_prompts.txt
FILTER_OUT_ROOT ?= assets/stable_diffusion/filtering_ranking
FILTER_METHODS ?= last_layer,subnet
FILTER_N_SAMPLES ?= 100
FILTER_LIMIT ?= 0
FILTER_SKIP_EXISTING ?= --skip_existing

SD_SCRIPT := experiments/stable_diffusion/run_sd_laplace.py
DAAM_SCRIPT := experiments/stable_diffusion/run_daam_attention.py
DAAM_BAYESDIFF_SCRIPT := experiments/stable_diffusion/run_daam_bayesdiff_pipeline.py
EVAL_SCRIPT := experiments/stable_diffusion/evaluate_sd_uq.py
TIFA_EVAL_SCRIPT := experiments/stable_diffusion/evaluate_tifa_uq.py
PUNC_EVAL_SCRIPT := experiments/stable_diffusion/evaluate_punc_uq.py
RECAP_PROMPT_SCRIPT := experiments/stable_diffusion/sample_recap_coco_prompts.py
PROMPT_BATCH_SCRIPT := experiments/stable_diffusion/run_prompt_batch.py

help:
	@echo "Stable Diffusion UQ targets:"
	@echo "  make all              run all non-smoke SD + DAAM/BayesDiff experiments"
	@echo "  make all-daam         run all DAAM/BayesDiff experiments"
	@echo "  make sd-sync          install SD dependencies via uv"
	@echo "  make sd-smoke         tiny CPU last-layer smoke test"
	@echo "  make sd-last-ddim     SD v1.4 conv_out LLLA + DDIM"
	@echo "  make sd-last-ddpm     SD v1.4 conv_out LLLA + DDPM"
	@echo "  make sd-subnet-smoke  SD v1.4 small subnet sanity run"
	@echo "  make sd-subnet-ddim   SD v1.4 random subnet FLARE + DDIM"
	@echo "  make sd-subnet-ddpm   SD v1.4 random subnet FLARE + DDPM"
	@echo "  make sd-eval-ddim     evaluate last_layer_ddim and subnet_ddim with CLIPScore"
	@echo "  make sd-tifa-eval     evaluate UQ rankings with local TIFA-like VQA scoring"
	@echo "  make sd-punc-eval     evaluate UQ rankings with OpenAI PUNC-like caption scoring"
	@echo "  make sd-recap-prompts sample Recap-COCO prompts for a prompt batch"
	@echo "  make sd-recap-run     run last_layer/subnet UQ for sampled Recap-COCO prompts"
	@echo "  make sd-recap-tifa    run TIFA-like eval over the Recap-COCO batch"
	@echo "  make sd-recap-punc    run PUNC-like eval over the Recap-COCO batch"
	@echo "  make sd-filtering-ranking  run per-prompt filtering/ranking with CLIPScore"
	@echo "  make sd-filtering-tifa     run TIFA-like eval for filtering/ranking outputs"
	@echo "  make sd-filtering-punc     run PUNC-like eval for filtering/ranking outputs"
	@echo "  make sd-benchmark-ddim  run last/subnet DDIM, then evaluate"
	@echo "  make sdxl-last-ddim   SDXL base conv_out LLLA + DDIM"
	@echo "  make sdxl-subnet-ddim SDXL base random subnet FLARE + DDIM"
	@echo "  make sdxl-benchmark-ddim  run SDXL last/subnet DDIM, then evaluate"
	@echo "  make sd-daam-ddim     DAAM maps + weighted scores for last_layer_ddim"
	@echo "  make sd-daam-subnet-ddim  DAAM maps + weighted scores for subnet_ddim"
	@echo "  make sd-daam-bayesdiff-ddim  run FLARE+BayesDiff maps, then DAAM weighting"
	@echo "  make sd-daam-bayesdiff-ddpm  same with DDPM t4 sampler variance"
	@echo "  make sd-daam-bayesdiff-subnet-ddim  same, using random subnet gamma2"
	@echo "  make sd-daam-bayesdiff-subnet-ddpm  subnet version with DDPM"
	@echo "  make sd-daam-show OUT_DIR=...  print DAAM-weighted ranking"
	@echo "  make sd-show OUT_DIR=...  print uncertainty ranking from results"
	@echo ""
	@echo "Common overrides:"
	@echo "  make sd-subnet-ddim PROMPT=\"a red cube on a blue sphere\" N_SAMPLES=16"
	@echo "  make sd-subnet-ddim SUBNET_N_PARAMS=100000 SUBNET_MC_SAMPLES=4"
	@echo "  make sd-subnet-ddpm STEPS_DDPM=1000"
	@echo "  make sd-subnet-ddim ATTENTION_AGGREGATION=cross SAVE_ATTENTION_MAPS=--save_attention_maps"
	@echo "  make sd-benchmark-ddim N_SAMPLES=100"
	@echo "  make sd-filtering-ranking FILTER_N_SAMPLES=100 FILTER_LIMIT=10"
	@echo "  make sdxl-benchmark-ddim PROMPT=\"a soccer match in a packed stadium\" N_SAMPLES=50"
	@echo "  make sd-daam-subnet-ddim DAAM_WORDS=hand,fingers  # empty DAAM_WORDS auto-infers prompt words"
	@echo "  make sd-daam-bayesdiff-ddim DAAM_WORDS=player,ball N_SAMPLES=16"
	@echo "  make all-daam DAAM_SAVE_IMAGES= DAAM_BAYESDIFF_SAVE_IMAGES=  # skip DAAM PNGs"

all: sd-last-ddim sd-last-ddpm sd-subnet-ddim sd-subnet-ddpm all-daam

all-daam: sd-daam-bayesdiff-ddim sd-daam-bayesdiff-ddpm sd-daam-bayesdiff-subnet-ddim sd-daam-bayesdiff-subnet-ddpm

sd-benchmark-ddim: sd-last-ddim sd-subnet-ddim sd-eval-ddim

sdxl-benchmark-ddim: sdxl-last-ddim sdxl-subnet-ddim sdxl-eval-ddim

sd-sync:
	uv sync --extra stable-diffusion --extra tifa --extra dev

sd-smoke:
	uv run python $(SD_SCRIPT) \
		--model_id "$(TINY_MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device cpu \
		--torch_dtype float32 \
		--scheduler ddim \
		--laplace_mode last_layer \
		--attention_aggregation none \
		--steps 1 \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 1 \
		--n_lap_pairs 1 \
		--n_samples 1 \
		--height 64 \
		--width 64 \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
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
		--attention_aggregation $(ATTENTION_AGGREGATION) \
		--attention_token_indices "$(ATTENTION_TOKEN_INDICES)" \
		$(SAVE_ATTENTION_MAPS) \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
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
		--attention_aggregation $(ATTENTION_AGGREGATION) \
		--attention_token_indices "$(ATTENTION_TOKEN_INDICES)" \
		$(SAVE_ATTENTION_MAPS) \
		--steps $(STEPS_DDPM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
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
		--attention_aggregation $(ATTENTION_AGGREGATION) \
		--attention_token_indices "$(ATTENTION_TOKEN_INDICES)" \
		$(SAVE_ATTENTION_MAPS) \
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
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
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
		--attention_aggregation $(ATTENTION_AGGREGATION) \
		--attention_token_indices "$(ATTENTION_TOKEN_INDICES)" \
		$(SAVE_ATTENTION_MAPS) \
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
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
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
		--attention_aggregation $(ATTENTION_AGGREGATION) \
		--attention_token_indices "$(ATTENTION_TOKEN_INDICES)" \
		$(SAVE_ATTENTION_MAPS) \
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
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/subnet_ddpm

sd-eval-ddim:
	uv run python $(EVAL_SCRIPT) \
		--results $(EVAL_NPZS) \
		--out_dir "$(EVAL_OUT_DIR)" \
		--clip_model "$(EVAL_CLIP_MODEL)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--batch_size $(EVAL_BATCH_SIZE) \
		--filter_fracs "$(EVAL_FILTER_FRACS)"

sd-tifa-eval:
	uv run --extra tifa python $(TIFA_EVAL_SCRIPT) \
		--results $(TIFA_EVAL_RESULTS) \
		--out_dir "$(TIFA_OUT_DIR)" \
		--question_cache "$(TIFA_QUESTION_CACHE)" \
		--question_source "$(TIFA_QUESTION_SOURCE)" \
		--openai_model "$(TIFA_OPENAI_MODEL)" \
		--together_model "$(TIFA_TOGETHER_MODEL)" \
		--device $(DEVICE) \
		--max_samples $(TIFA_MAX_SAMPLES) \
		--max_questions $(TIFA_MAX_QUESTIONS) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		$(TIFA_REQUIRE_CACHED)

sd-punc-eval:
	uv run --extra tifa python $(PUNC_EVAL_SCRIPT) \
		--results $(PUNC_EVAL_RESULTS) \
		--out_dir "$(PUNC_OUT_DIR)" \
		--caption_cache "$(PUNC_CAPTION_CACHE)" \
		--openai_model "$(PUNC_OPENAI_MODEL)" \
		--image_detail "$(PUNC_IMAGE_DETAIL)" \
		--max_samples $(PUNC_MAX_SAMPLES) \
		--similarity "$(PUNC_SIMILARITY)" \
		--device $(DEVICE) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		$(PUNC_REQUIRE_CACHED)

sd-filtering-run:
	uv run --extra stable-diffusion python $(PROMPT_BATCH_SCRIPT) \
		--prompts "$(FILTER_PROMPTS)" \
		--out_root "$(FILTER_OUT_ROOT)" \
		--methods "$(FILTER_METHODS)" \
		--model_id "$(MODEL_ID)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_samples $(FILTER_N_SAMPLES) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--subnet_n_params $(SUBNET_N_PARAMS) \
		--subnet_max_tensors $(SUBNET_MAX_TENSORS) \
		--subnet_mc_samples $(SUBNET_MC_SAMPLES) \
		--limit $(FILTER_LIMIT) \
		$(FILTER_SKIP_EXISTING)

sd-filtering-clip:
	uv run --extra stable-diffusion python $(EVAL_SCRIPT) \
		--results_file "$(FILTER_OUT_ROOT)/results.txt" \
		--out_dir "$(FILTER_OUT_ROOT)/clip_eval" \
		--clip_model "$(EVAL_CLIP_MODEL)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--batch_size $(EVAL_BATCH_SIZE) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		--include_random_baseline

sd-filtering-ranking:
	$(MAKE) sd-filtering-run
	$(MAKE) sd-filtering-clip

sd-filtering-tifa:
	uv run --extra tifa python $(TIFA_EVAL_SCRIPT) \
		--results_file "$(FILTER_OUT_ROOT)/results.txt" \
		--out_dir "$(FILTER_OUT_ROOT)/tifa_eval" \
		--question_cache "$(FILTER_OUT_ROOT)/tifa_eval/tifa_questions.json" \
		--question_source "$(TIFA_QUESTION_SOURCE)" \
		--openai_model "$(TIFA_OPENAI_MODEL)" \
		--together_model "$(TIFA_TOGETHER_MODEL)" \
		--device $(DEVICE) \
		--max_samples $(TIFA_MAX_SAMPLES) \
		--max_questions $(TIFA_MAX_QUESTIONS) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		$(TIFA_REQUIRE_CACHED)

sd-filtering-punc:
	uv run --extra tifa python $(PUNC_EVAL_SCRIPT) \
		--results_file "$(FILTER_OUT_ROOT)/results.txt" \
		--out_dir "$(FILTER_OUT_ROOT)/punc_eval" \
		--caption_cache "$(FILTER_OUT_ROOT)/punc_eval/punc_captions.json" \
		--openai_model "$(PUNC_OPENAI_MODEL)" \
		--image_detail "$(PUNC_IMAGE_DETAIL)" \
		--max_samples $(PUNC_MAX_SAMPLES) \
		--similarity "$(PUNC_SIMILARITY)" \
		--device $(DEVICE) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		$(PUNC_REQUIRE_CACHED)

sd-recap-prompts:
	uv run --extra recap python $(RECAP_PROMPT_SCRIPT) \
		--n $(RECAP_N_PROMPTS) \
		--out "$(RECAP_PROMPTS)"

sd-recap-run:
	uv run --extra stable-diffusion python $(PROMPT_BATCH_SCRIPT) \
		--prompts "$(RECAP_PROMPTS)" \
		--out_root "$(RECAP_OUT_ROOT)" \
		--methods "$(RECAP_METHODS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_samples $(RECAP_N_SAMPLES) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--subnet_n_params $(SUBNET_N_PARAMS) \
		--subnet_max_tensors $(SUBNET_MAX_TENSORS) \
		--subnet_mc_samples $(SUBNET_MC_SAMPLES) \
		--skip_existing

sd-recap-tifa:
	uv run --extra tifa python $(TIFA_EVAL_SCRIPT) \
		--results_file "$(RECAP_OUT_ROOT)/results.txt" \
		--out_dir "$(RECAP_OUT_ROOT)/tifa_eval" \
		--question_cache "$(RECAP_OUT_ROOT)/tifa_eval/tifa_questions.json" \
		--question_source "$(TIFA_QUESTION_SOURCE)" \
		--openai_model "$(TIFA_OPENAI_MODEL)" \
		--together_model "$(TIFA_TOGETHER_MODEL)" \
		--device $(DEVICE) \
		--max_samples $(TIFA_MAX_SAMPLES) \
		--max_questions $(TIFA_MAX_QUESTIONS) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		$(TIFA_REQUIRE_CACHED)

sd-recap-punc:
	uv run --extra tifa python $(PUNC_EVAL_SCRIPT) \
		--results_file "$(RECAP_OUT_ROOT)/results.txt" \
		--out_dir "$(RECAP_OUT_ROOT)/punc_eval" \
		--caption_cache "$(RECAP_OUT_ROOT)/punc_eval/punc_captions.json" \
		--openai_model "$(PUNC_OPENAI_MODEL)" \
		--image_detail "$(PUNC_IMAGE_DETAIL)" \
		--max_samples $(PUNC_MAX_SAMPLES) \
		--similarity "$(PUNC_SIMILARITY)" \
		--device $(DEVICE) \
		--filter_fracs "$(EVAL_FILTER_FRACS)" \
		$(PUNC_REQUIRE_CACHED)

sdxl-last-ddim:
	uv run python $(SD_SCRIPT) \
		--pipeline sdxl \
		--model_id "$(SDXL_MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--laplace_mode last_layer \
		--attention_aggregation none \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(SDXL_HEIGHT) \
		--width $(SDXL_WIDTH) \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/sdxl_last_layer_ddim

sdxl-subnet-ddim:
	uv run python $(SD_SCRIPT) \
		--pipeline sdxl \
		--model_id "$(SDXL_MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--laplace_mode subnet \
		--attention_aggregation none \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--subnet_n_params $(SUBNET_N_PARAMS) \
		--subnet_max_tensors $(SUBNET_MAX_TENSORS) \
		--subnet_mc_samples $(SUBNET_MC_SAMPLES) \
		--height $(SDXL_HEIGHT) \
		--width $(SDXL_WIDTH) \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--out_dir assets/stable_diffusion/sdxl_subnet_ddim

sdxl-eval-ddim:
	uv run python $(EVAL_SCRIPT) \
		--results $(SDXL_EVAL_NPZS) \
		--out_dir "$(SDXL_EVAL_OUT_DIR)" \
		--clip_model "$(EVAL_CLIP_MODEL)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--batch_size $(EVAL_BATCH_SIZE) \
		--filter_fracs "$(EVAL_FILTER_FRACS)"

sd-daam-ddim:
	uv run $(DAAM_WITH) python $(DAAM_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--words "$(DAAM_WORDS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--uncertainty_npz assets/stable_diffusion/last_layer_ddim/laplace_results.npz \
		--uncertainty_keys "$(DAAM_UNCERTAINTY_KEYS)" \
		--out_dir assets/stable_diffusion/daam_last_layer_ddim \
		$(DAAM_SAVE_IMAGES)

sd-daam-subnet-ddim:
	uv run $(DAAM_WITH) python $(DAAM_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--words "$(DAAM_WORDS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--seed $(SEED) \
		--uncertainty_npz "$(DAAM_UNCERTAINTY_NPZ)" \
		--uncertainty_keys "$(DAAM_UNCERTAINTY_KEYS)" \
		--out_dir assets/stable_diffusion/daam_subnet_ddim \
		$(DAAM_SAVE_IMAGES)

sd-daam-bayesdiff-ddim:
	uv run python $(DAAM_BAYESDIFF_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--words "$(DAAM_WORDS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
		--steps $(STEPS_DDIM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--laplace_mode last_layer \
		--daam_python $(DAAM_PYTHON) \
		--out_dir "$(DAAM_BAYESDIFF_OUT_DIR)/last_layer_ddim" \
		$(DAAM_BAYESDIFF_SAVE_IMAGES)

sd-daam-bayesdiff-ddpm:
	uv run python $(DAAM_BAYESDIFF_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--words "$(DAAM_WORDS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddpm \
		--steps $(STEPS_DDPM) \
		--guidance_scale $(GUIDANCE_SCALE) \
		--n_z0 $(N_Z0) \
		--n_lap_pairs $(N_LAP_PAIRS) \
		--n_samples $(N_SAMPLES) \
		--height $(HEIGHT) \
		--width $(WIDTH) \
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--laplace_mode last_layer \
		--daam_python $(DAAM_PYTHON) \
		--out_dir "$(DAAM_BAYESDIFF_OUT_DIR)/last_layer_ddpm" \
		$(DAAM_BAYESDIFF_SAVE_IMAGES)

sd-daam-bayesdiff-subnet-ddim:
	uv run python $(DAAM_BAYESDIFF_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--words "$(DAAM_WORDS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddim \
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
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--laplace_mode subnet \
		--daam_python $(DAAM_PYTHON) \
		--out_dir "$(DAAM_BAYESDIFF_OUT_DIR)/subnet_ddim" \
		$(DAAM_BAYESDIFF_SAVE_IMAGES)

sd-daam-bayesdiff-subnet-ddpm:
	uv run python $(DAAM_BAYESDIFF_SCRIPT) \
		--model_id "$(MODEL_ID)" \
		--prompt "$(PROMPT)" \
		--words "$(DAAM_WORDS)" \
		--device $(DEVICE) \
		--torch_dtype $(TORCH_DTYPE) \
		--scheduler ddpm \
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
		--plot_max_samples $(PLOT_MAX_SAMPLES) \
		--seed $(SEED) \
		--laplace_mode subnet \
		--daam_python $(DAAM_PYTHON) \
		--out_dir "$(DAAM_BAYESDIFF_OUT_DIR)/subnet_ddpm" \
		$(DAAM_BAYESDIFF_SAVE_IMAGES)

sd-show:
	uv run python -c "import numpy as np; p='$(OUT_DIR)/laplace_results.npz'; d=np.load(p); order=np.argsort(d['var_mean']); print('file:', p); print('least uncertain:', order[:10], d['var_mean'][order[:10]]); print('most uncertain:', order[-10:][::-1], d['var_mean'][order[-10:][::-1]])"

sd-daam-show:
	uv run python -c "import numpy as np; p='$(OUT_DIR)/daam_results.npz'; key='$(DAAM_SCORE_KEY)'; d=np.load(p); order=np.argsort(d[key]); print('file:', p); print('key:', key); print('words:', d['daam_words']); print('least uncertain:', order[:10], d[key][order[:10]]); print('most uncertain:', order[-10:][::-1], d[key][order[-10:][::-1]])"
