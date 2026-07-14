#!/usr/bin/env bash
# Real-footage A/B: baseline vs occlusion-aware on a 20s clip from jiadianziwei.
# Left=baseline (穿模 possible), Right=occlusion-aware (occluder preserved).
set -u
source /root/code/LipForcing/.venv/bin/activate
cd /root/code/LipForcing
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1

CLIP=/root/code/data/occl_demo/jz_clip.mp4
AUDIO=/root/code/data/occl_demo/demo20.wav
OUTDIR=/root/code/LipForcing/output/occl_test
mkdir -p "$OUTDIR"
LOG=/root/code/LipForcing/bench/log_occl_ab.txt

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
echo "$(ts) === BASELINE (no occlusion) ==="
python -u scripts/inference/inference_segmentwise.py $COMMON \
  --output_path "$OUTDIR/real_baseline.mp4"
echo "$(ts) BASELINE_EXIT=$?"

echo "$(ts) === OCCLUSION-AWARE ==="
python -u scripts/inference/inference_segmentwise.py $COMMON \
  --occlusion_aware \
  --output_path "$OUTDIR/real_occl.mp4"
echo "$(ts) OCCL_EXIT=$?"

FF=$(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
echo "$(ts) === SIDE-BY-SIDE (left=baseline, right=occlusion-aware) ==="
"$FF" -y -i "$OUTDIR/real_baseline.mp4" -i "$OUTDIR/real_occl.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" -map 0:a? -c:v libx264 -crf 18 -preset fast -loglevel error \
  "$OUTDIR/real_compare.mp4"
echo "$(ts) COMPARE_EXIT=$?"
ls -lh "$OUTDIR"/real_*.mp4
} > "$LOG" 2>&1
