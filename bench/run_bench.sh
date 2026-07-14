#!/usr/bin/env bash
# Benchmark: single vs 2-concurrent LipForcing streams, with dmon SM/mem/power sampling.
set -u
source /root/code/LipForcing/.venv/bin/activate
cd /root/code/LipForcing
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
BD=/root/code/LipForcing/bench; mkdir -p $BD
DREF=test_ref.mp4; DAUD=bench_audio_10s.wav; DTE=weights/text_emb.pt; DFC=weights/face_cache
COMMON="--ckpt_path weights/lipforcing_14b.pth --vae_path weights/Wan2.1-T2V-14B/Wan2.1_VAE.pth --wav2vec_path weights/wav2vec2-base-960h --mask_path weights/mask.png --taehv_ckpt weights/taew2_1.pth --text_embeds_path $DTE --face_cache_dir $DFC --video_path $DREF --audio_path $DAUD"

run_one() { # $1=out $2=log
  stdbuf -oL -eL python -u scripts/inference/inference_streaming.py $COMMON --output_path "$1" \
    2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$2"
}

echo "===== TEST 1: SINGLE STREAM ====="
nvidia-smi dmon -s pu -d 1 > "$BD/dmon_single.txt" 2>&1 &
DPID=$!
run_one "$BD/single.mp4" "$BD/log_single.txt"
kill $DPID 2>/dev/null
echo "TEST1 done"

echo "===== TEST 2: TWO CONCURRENT STREAMS ====="
nvidia-smi dmon -s pu -d 1 > "$BD/dmon_conc.txt" 2>&1 &
DPID=$!
run_one "$BD/conc_a.mp4" "$BD/log_a.txt" &
PA=$!
run_one "$BD/conc_b.mp4" "$BD/log_b.txt" &
PB=$!
wait $PA; wait $PB
kill $DPID 2>/dev/null
echo "TEST2 done"
echo "BENCH_DONE"
