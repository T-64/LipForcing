#!/usr/bin/env bash
# Run one video with demo.wav -> /root/code/data/lipforcing_runs/<name>_demo203.mp4
# Usage: bash run_one.sh <video_stem>
set -u
source /root/code/LipForcing/.venv/bin/activate
cd /root/code/LipForcing
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
NAME="$1"
OUTDIR=/root/code/data/lipforcing_runs
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${NAME}_demo203.mp4"
python -u scripts/inference/inference_streaming.py \
  --ckpt_path weights/lipforcing_14b.pth \
  --vae_path weights/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --wav2vec_path weights/wav2vec2-base-960h \
  --mask_path weights/mask.png \
  --taehv_ckpt weights/taew2_1.pth \
  --text_embeds_path weights/text_emb.pt \
  --face_cache_dir weights/face_cache \
  --video_path "/root/code/data/videos/${NAME}.mp4" \
  --audio_path /root/code/data/audios/demo.wav \
  --output_path "$OUT" \
  2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s)" "$l"; done > "/root/code/LipForcing/bench/log_${NAME}.txt"
echo "${NAME}_INFER_EXIT=${PIPESTATUS[0]}"
ls -lh "$OUT" 2>/dev/null
