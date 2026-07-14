#!/usr/bin/env python3
"""Scan a video for mouth-region occlusion using BiSeNet face parsing.

Cheap (no diffusion inference): for sampled frames it detects+aligns the
face (buffalo_l), runs BiSeNet, and reports how much of the mouth ROI is
NON-face (i.e. covered by a hand / object / food). High score = a good
clip to demonstrate occlusion-aware paste-back.

Usage:
    python bench/scan_occlusion.py --video X.mp4 [--stride_sec 0.5] [--top 15]
"""
import argparse, os
import numpy as np, torch, imageio.v3 as iio, imageio

from OmniAvatar.utils.latentsync.image_processor import ImageProcessor, load_fixed_mask
from OmniAvatar.utils.latentsync.occlusion_mask import OcclusionMasker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mask_path", default="weights/mask.png")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--stride_sec", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mask_tensor = load_fixed_mask(512, mask_image_path=args.mask_path) if args.mask_path else None
    ip = ImageProcessor(
        resolution=512, device=args.device, mask_image=mask_tensor,
        insightface_root=os.path.join(repo, "checkpoints", "auxiliary"),
    )
    masker = OcclusionMasker(weight_path=args.ckpt, device=args.device)

    # mouth ROI in 512 aligned space (lower-center)
    S = ip.resolution
    r0, r1 = int(0.60 * S), int(0.84 * S)
    c0, c1 = int(0.38 * S), int(0.66 * S)

    reader = imageio.get_reader(args.video)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0) or 25.0)
    total = reader.count_frames()
    step = max(1, int(round(args.stride_sec * fps)))
    idxs = list(range(0, total, step))
    print(f"video={args.video} fps={fps:.1f} frames={total} sampling {len(idxs)} @ {args.stride_sec}s")

    buf, ts = [], []
    results = []
    for fi, idx in enumerate(idxs):
        frame = reader.get_data(idx)  # RGB HxWx3 uint8
        try:
            face, box, aff = ip.affine_transform(frame)
        except RuntimeError:
            continue
        if face is None:
            continue
        if isinstance(face, torch.Tensor):
            face_np = face.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC RGB
        else:
            face_np = np.asarray(face)
        face_np = np.ascontiguousarray(face_np.astype(np.uint8))
        if face_np.shape[0] != S or face_np.shape[1] != S:
            import cv2 as _cv2
            face_np = _cv2.resize(face_np, (S, S))
        buf.append(face_np[:, :, :3])
        ts.append(idx / fps)

        if len(buf) == 16 or fi == len(idxs) - 1:
            skin = masker.compute_aligned(np.stack(buf, 0))  # [N,S,S] 1=face
            for j in range(len(buf)):
                mouth = (1.0 - skin[j])[r0:r1, c0:c1]
                results.append((ts[j], float(mouth.mean())))
            buf, ts = [], []

    reader.close()
    results.sort(key=lambda x: -x[1])
    print(f"\nTop {args.top} most-occluded moments (t_sec, mouth_nonface_frac):")
    for t, s in results[:args.top]:
        print(f"  t={t:7.2f}s   occ={s*100:5.1f}%")
    if results:
        # also print a smoothed "best window" hint (dense cluster of high-occ)
        thr = max(0.15, results[len(results)//4][1]) if len(results) > 4 else 0.15
        hot = sorted(t for t, s in results if s >= thr)
        if hot:
            print(f"\nhot timestamps (occ>={thr*100:.0f}%):",
                  ", ".join(f"{t:.1f}" for t in hot[:30]))


if __name__ == "__main__":
    main()
