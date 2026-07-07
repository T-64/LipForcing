#!/bin/bash
# =============================================================================
# Stage 1 — Diffusion Forcing (shift=5) for the 14B causal student
# =============================================================================
#
# LoRA on the transformer blocks + selective full fine-tune on the audio path
# and patch embedding, FSDP, bf16/fp32 mixed precision, on the t769 2-step
# schedule:
#
#   sample_t_cfg.t_list:   [0.999, 0.937, 0.833, 0.624, 0.0] -> [0.999, 0.769, 0.0]
#   student_sample_steps:  4 -> 2
#
# The student is trained only at the noise levels used at Stage-2 inference,
# with a regime that prioritizes audio-path adaptation while constraining the
# bulk of the network to a low-rank update.
#
# Usage:
#   bash scripts/train_stage1_df.sh
#
#   # Smoke test (50 iters):
#   MAX_ITER=50 SAVE_EVERY=50 bash scripts/train_stage1_df.sh
#
# Resume:
#   RESUME=True bash scripts/train_stage1_df.sh
# =============================================================================

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

export CONFIG_PATH="lipforcing/configs/experiments/OmniAvatar/config_df.py"

# Launch / batch settings (effective batch = BATCH_SIZE * NGPU * GRAD_ACCUM).
NGPU="${NGPU:-4}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_ITER="${MAX_ITER:-5000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
RESUME="${RESUME:-False}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU * GRAD_ACCUM))

# Environment.
export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-./OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# wandb mode: online when credentials are configured, offline otherwise (local
# logs only, no login prompt). Override explicitly with WANDB_MODE=disabled etc.
WANDB_MODE="${WANDB_MODE:-$([ -n "${WANDB_API_KEY}" ] && echo online || echo offline)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIPFORCING_OUTPUT_ROOT="${LIPFORCING_OUTPUT_ROOT:-./outputs}"
export NEG_TEXT_EMB_PATH="${NEG_TEXT_EMB_PATH:-/path/to/neg_text_emb.pt}"

# Student init checkpoint: the pretrained 14B V2V adapter the causal student is
# initialized from (loaded by the config into CausalOmniAvatarWan's
# omniavatar_ckpt_path). This release is 14B-only.
export OMNIAVATAR_STUDENT_CKPT_14B="${OMNIAVATAR_STUDENT_CKPT_14B:-/path/to/omniavatar_14b_init.pt}"
if [[ ! -f "${OMNIAVATAR_STUDENT_CKPT_14B}" ]]; then
    echo "ERROR: OMNIAVATAR_STUDENT_CKPT_14B does not exist: ${OMNIAVATAR_STUDENT_CKPT_14B}" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-df_14b_lora_t769}"
LOG_GROUP="${LOG_GROUP:-stage1_df}"

# Additional user overrides, appended last (so they win on conflict).
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

echo "============================================="
echo "  Stage 1 — DF (shift=5), 14B LoRA + unfreeze + t769"
echo "============================================="
echo "  Config:           ${CONFIG_PATH}"
echo "  GPUs:             ${NGPU}"
echo "  Per-GPU batch:    ${BATCH_SIZE}"
echo "  Grad accum:       ${GRAD_ACCUM}"
echo "  Effective batch:  ${EFFECTIVE_BATCH}  (= ${BATCH_SIZE} x ${NGPU} x ${GRAD_ACCUM})"
echo "  Max iter:         ${MAX_ITER}"
echo "  Save every:       ${SAVE_EVERY}"
echo "  Student ckpt:     ${OMNIAVATAR_STUDENT_CKPT_14B}"
echo "  Run name:         ${RUN_NAME}"
echo "  Log group:        ${LOG_GROUP}"
echo "  Output root:      ${LIPFORCING_OUTPUT_ROOT}"
echo "  Schedule:         t_list=[0.999, 0.769, 0.0], student_sample_steps=2"
echo "  Resume:           ${RESUME}"
echo "============================================="
echo ""

torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=${CONFIG_PATH} \
    - dataloader_train.batch_size=${BATCH_SIZE} \
    trainer.ddp=False \
    trainer.fsdp=True \
    trainer.grad_accum_rounds=${GRAD_ACCUM} \
    trainer.max_iter=${MAX_ITER} \
    trainer.save_ckpt_iter=${SAVE_EVERY} \
    trainer.resume=${RESUME} \
    log_config.group="${LOG_GROUP}" \
    log_config.name="${RUN_NAME}" \
    log_config.project="LipForcing" \
    log_config.wandb_entity="${WANDB_ENTITY:-}" \
    log_config.wandb_mode="${WANDB_MODE}" \
    ${EXTRA_OVERRIDES}
