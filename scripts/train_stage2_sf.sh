#!/bin/bash
# =============================================================================
# Stage 2 — Self-Forcing (Re-DMD, beta=2 + TAEW) for the 14B causal student
# =============================================================================
#
# Distills the Stage-1 DF student into a few-step causal student with a
# SyncNet-v2 sync-C reward (Re-DMD, beta=2, TAEW decoder) on the t769 2-step
# schedule. The critic (fake_score) LR is matched to the 2e-6 student LR;
# both are set in the config.
#
# Effective batch 16 = BS=1/GPU x 4 GPUs x grad_accum=4.
#
# Usage:
#   bash scripts/train_stage2_sf.sh
#
# Resume:
#   RESUME=True bash scripts/train_stage2_sf.sh
# =============================================================================

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

export CONFIG_PATH="lipforcing/configs/experiments/OmniAvatar/config_sf.py"

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-600}"
SAVE_EVERY="${SAVE_EVERY:-100}"
RESUME="${RESUME:-False}"

# Environment.
export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-./OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# wandb mode: online when credentials are configured, offline otherwise (local
# logs only, no login prompt). Override explicitly with WANDB_MODE=disabled etc.
WANDB_MODE="${WANDB_MODE:-$([ -n "${WANDB_API_KEY}" ] && echo online || echo offline)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIPFORCING_OUTPUT_ROOT="${LIPFORCING_OUTPUT_ROOT:-./outputs}"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1
export NEG_TEXT_EMB_PATH="${NEG_TEXT_EMB_PATH:-/path/to/neg_text_emb.pt}"

# Checkpoints. DF init = the Stage-1 output; teacher = the bidirectional 14B;
# student and fake_score both initialize from the 14B V2V adapter.
# ${VAR-default} (no colon) on the DF ckpt preserves an explicitly-empty value.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/path/to/df_init_checkpoint.pth}"
if [[ -n "${OMNIAVATAR_DF_CKPT}" && ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

export OMNIAVATAR_TEACHER_CKPT="${OMNIAVATAR_TEACHER_CKPT:-/path/to/omniavatar_14b_init.pt}"
if [[ ! -f "${OMNIAVATAR_TEACHER_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_TEACHER_CKPT does not exist: ${OMNIAVATAR_TEACHER_CKPT}" >&2
    exit 1
fi

export OMNIAVATAR_STUDENT_CKPT_14B="${OMNIAVATAR_STUDENT_CKPT_14B:-/path/to/omniavatar_14b_init.pt}"
if [[ ! -f "${OMNIAVATAR_STUDENT_CKPT_14B}" ]]; then
    echo "ERROR: OMNIAVATAR_STUDENT_CKPT_14B does not exist: ${OMNIAVATAR_STUDENT_CKPT_14B}" >&2
    exit 1
fi

# Reward checkpoints (Re-DMD). Required: the SF config enables the sync-C reward.
export SYNCNET_CKPT="${SYNCNET_CKPT:-/path/to/syncnet_v2.model}"
if [[ ! -f "${SYNCNET_CKPT}" ]]; then
    echo "ERROR: SYNCNET_CKPT does not exist: ${SYNCNET_CKPT}" >&2
    exit 1
fi
export TAEW_CKPT="${TAEW_CKPT:-/path/to/taew2_1.pth}"
if [[ ! -f "${TAEW_CKPT}" ]]; then
    echo "ERROR: TAEW_CKPT does not exist: ${TAEW_CKPT}" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-sf_14b_lora_t769}"
LOG_GROUP="${LOG_GROUP:-stage2_sf}"
LR_SUMMARY="${LR_SUMMARY:-student 2e-6, critic 2e-6 (matched)}"
EFFECTIVE_BATCH=$((1 * NGPU * 4))
# Additional user overrides, appended last (so they win on conflict).
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
CKPT_DIR="${LIPFORCING_OUTPUT_ROOT}/LipForcing/${LOG_GROUP}/${RUN_NAME}/checkpoints"

echo "============================================="
echo "  Stage 2 — SF Re-DMD beta=2 + TAEW, 14B LoRA t769"
echo "============================================="
echo "  Config:          ${CONFIG_PATH}"
echo "  GPUs:            ${NGPU}"
echo "  DF init ckpt:    ${OMNIAVATAR_DF_CKPT}"
echo "  Teacher ckpt:    ${OMNIAVATAR_TEACHER_CKPT}"
echo "  Student init:    ${OMNIAVATAR_STUDENT_CKPT_14B}"
echo "  SyncNet ckpt:    ${SYNCNET_CKPT}"
echo "  TAEW ckpt:       ${TAEW_CKPT}"
echo "  Run name:        ${RUN_NAME}"
echo "  Log group:       ${LOG_GROUP}"
echo "  Output root:     ${LIPFORCING_OUTPUT_ROOT}"
echo "  Checkpoints:     ${CKPT_DIR}"
echo "  LRs:             ${LR_SUMMARY}"
echo "  Effective batch: ${EFFECTIVE_BATCH}  (BS=1 x ${NGPU} GPUs x GA=4)"
echo "  Schedule:        t_list=[0.999, 0.769, 0.0], student_sample_steps=2"
echo "  Max iter:        ${MAX_ITER}"
echo "  Save every:      ${SAVE_EVERY}"
echo "  Resume:          ${RESUME}"
echo "============================================="
echo ""

torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=${CONFIG_PATH} \
    - trainer.resume=${RESUME} \
    trainer.max_iter=${MAX_ITER} \
    trainer.save_ckpt_iter=${SAVE_EVERY} \
    log_config.group="${LOG_GROUP}" \
    log_config.name="${RUN_NAME}" \
    log_config.project="LipForcing" \
    log_config.wandb_entity="${WANDB_ENTITY:-}" \
    log_config.wandb_mode="${WANDB_MODE}" \
    ${EXTRA_OVERRIDES}
