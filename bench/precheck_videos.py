#!/usr/bin/env python
"""Pre-check new videos: cv2 decode + buffalo_l face-detection robustness.
Samples 1 frame/sec over the first ~170s (the window a 203s/5077-frame run
actually reads). Reports per-video detect-rate and where failures occur.
A single undetectable frame aborts the whole inference run, so this flags
which videos are safe for a long run.
"""
import os, sys, time
sys.path.insert(0, "/root/code/LipForcing")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import cv2
import numpy as np

MASK = "/root/code/LipForcing/weights/mask.png"
# 203s (demo.wav) @25fps output -> 5077 ref frames -> 169.2s @30fps source
AUDIO_SEC = 203.1
OUT_FPS = 25
NUM_REF_FRAMES = int(AUDIO_SEC * OUT_FPS)   # 5077

from scripts.inference._common import load_image_processor

print("loading insightface (buffalo_l) ...", flush=True)
ip = load_image_processor(MASK, "cuda")
print("loaded.", flush=True)

VIDEOS = [
    "/root/code/data/videos/jiadianziwei.mp4",
    "/root/code/data/videos/hanshu.mp4",
    "/root/code/data/videos/zhangxiaoquan.mp4",
]

for vp in VIDEOS:
    print("\n" + "=" * 64, flush=True)
    print(f"VIDEO: {os.path.basename(vp)}", flush=True)
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print("  [DECODE FAIL] cv2 cannot open this file (codec unsupported?)", flush=True)
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  cv2 opened OK | fps={fps:.2f} | total_frames={total} | "
          f"need first {NUM_REF_FRAMES} ({NUM_REF_FRAMES/fps:.0f}s)", flush=True)

    window = min(total, NUM_REF_FRAMES)
    stride = max(1, int(round(fps)))   # ~1 sample per second
    samples = list(range(0, window, stride))
    ok = 0; fails = []
    t0 = time.time()
    for idx in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            fails.append((idx, idx / fps, "decode-read-fail"))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            ip.affine_transform(frame_rgb)   # raises RuntimeError("Face not detected")
            ok += 1
        except Exception as e:
            fails.append((idx, idx / fps, str(e)[:40]))
    cap.release()
    dt = time.time() - t0
    rate = 100.0 * ok / max(1, len(samples))
    verdict = "SAFE (no fails in window)" if not fails else f"{len(fails)} fail(s) -> would ABORT full run"
    print(f"  samples={len(samples)} detected={ok} detect_rate={rate:.1f}% time={dt:.1f}s", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    if fails:
        for fi, ts, msg in fails[:12]:
            print(f"    fail @ frame {fi} ({ts:.1f}s): {msg}", flush=True)
        if len(fails) > 12:
            print(f"    ... +{len(fails)-12} more", flush=True)

print("\nPRECHECK_DONE", flush=True)
