#!/usr/bin/env bash
# Real-footage A/B: baseline vs occlusion-aware paste-back.
# Left=baseline (穿模 possible), Right=occlusion-aware (occluder preserved).
# Usage: bash run_occl_ab.sh <clip_mp4> <audio_wav> <out_stem>
set -u
source /root/code/LipForcing/.venv/bin/activate
cd /root/code/LipForcing
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1

CLIP="${1:?usage: run_occl_ab.sh <clip_mp4> <audio_wav> <out_stem>}"
AUDIO="${2:?audio wav}"
STEM="${3:?out stem}"
OUTDIR=/root/code/LipForcing/output/occl_test
mkdir -p "$OUTDIR"
LOG=/root/code/LipForcing/bench/log_occl_ab_${STEM}.txt

COMMON="--ckpt_path weights/lipforcing_14b.pth \
  --vae_path weights/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --wav2vec_path weights/wav2vec2-base-960h \
  --mask_path weights/mask.png \
  --taehv_ckpt weights/taew2_1.pth \
  --text_embeds_path weights/text_emb.pt \
  --face_cache_dir weights/face_cache \
  --video_path $CLIP --audio_path $AUDIO"

ts() { date +%s; }

{
echo "$(ts) === [$STEM] BASELINE (no occlusion) ==="
python -u scripts/inference/inference_segmentwise.py $COMMON \
  --output_path "$OUTDIR/${STEM}_baseline.mp4"
echo "$(ts) BASELINE_EXIT=$?"

echo "$(ts) === [$STEM] OCCLUSION-AWARE ==="
python -u scripts/inference/inference_segmentwise.py $COMMON \
  --occlusion_aware \
  --output_path "$OUTDIR/${STEM}_occl.mp4"
echo "$(ts) OCCL_EXIT=$?"

FF=$(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
echo "$(ts) === [$STEM] SIDE-BY-SIDE (left=baseline, right=occlusion-aware) ==="
"$FF" -y -i "$OUTDIR/${STEM}_baseline.mp4" -i "$OUTDIR/${STEM}_occl.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" -map 0:a? -c:v libx264 -crf 18 -preset fast -loglevel error \
  "$OUTDIR/${STEM}_compare.mp4"
echo "$(ts) COMPARE_EXIT=$?"
ls -lh "$OUTDIR"/${STEM}_*.mp4
} > "$LOG" 2>&1
