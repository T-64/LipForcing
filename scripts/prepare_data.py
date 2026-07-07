#!/usr/bin/env python3
"""Prepare training samples from raw talking-head videos.

Turns raw videos into the sample-directory format the training dataloader
consumes with on-the-fly preprocessing (see DATA.md): one directory per
81-frame clip containing

    sub_clip.mp4   aligned 512x512 face crop, 25 fps
    audio.wav      matching audio slice, 16 kHz mono PCM
    prompt.txt     the text prompt

plus train/val list files (one sample directory per line) that
``OMNIAVATAR_DATA_LIST`` / ``OMNIAVATAR_VAL_LIST`` point at.

Face detection + 512x512 affine alignment uses the exact same LatentSync
pipeline as inference (scripts/inference/_common.py), so training samples
match what the model sees at test time. Videos are converted to 25 fps CFR
first when needed; segments where face detection fails are skipped.

Example:
    python scripts/prepare_data.py \
        --input_dir /path/to/raw_videos \
        --output_dir ./data/v2v_training_data \
        --mask_path /path/to/mask.png \
        --val_count 2
"""
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

# --- Path setup (mirror the inference scripts) ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPFORCING_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, LIPFORCING_ROOT)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "inference"))  # for `import _common`

from _common import _get_ffmpeg, load_image_processor, save_frames_as_video  # noqa: E402

FPS = 25  # training frame rate (OmniAvatar convention)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare 512x512 aligned training samples from raw talking-head videos."
    )
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of raw videos (*.mp4/*.mov/*.avi/*.mkv)")
    parser.add_argument("--videos", type=str, nargs="+", default=None,
                        help="Explicit list of raw video paths (alternative to --input_dir)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where sample directories and list files are written")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="Path to LatentSync mask.png (needed by the image processor)")
    parser.add_argument("--prompt", type=str, default="a person talking",
                        help="Prompt written to each sample's prompt.txt")
    parser.add_argument("--clip_frames", type=int, default=81,
                        help="Frames per training clip (81 = 21 VAE latents, the training length)")
    parser.add_argument("--max_clips_per_video", type=int, default=None,
                        help="Cap on clips extracted per video (default: all full clips)")
    parser.add_argument("--val_count", type=int, default=2,
                        help="Number of trailing sample dirs assigned to the val list")
    parser.add_argument("--train_list", type=str, default="train_list.txt",
                        help="Train list filename (written inside --output_dir)")
    parser.add_argument("--val_list", type=str, default="val_list.txt",
                        help="Val list filename (written inside --output_dir)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sample dirs that already contain all three files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for face detection")
    return parser.parse_args()


def list_videos(args):
    if (args.input_dir is None) == (args.videos is None):
        raise ValueError("Provide exactly one of --input_dir or --videos")
    if args.videos is not None:
        return list(args.videos)
    exts = (".mp4", ".mov", ".avi", ".mkv")
    return sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(exts)
    )


def ensure_25fps(video_path):
    """Return a 25 fps CFR version of *video_path* (temp file if converted).

    Returns:
        (path, tmp_path_or_None) — tmp_path must be deleted by the caller.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if abs(fps - FPS) < 0.05:
        return video_path, None
    print(f"  [fps] {fps:.2f} -> {FPS} CFR")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path, "-r", str(FPS), "-an",
        "-c:v", "libx264", "-crf", "13", tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.remove(tmp_path)
        raise RuntimeError(f"ffmpeg 25fps conversion failed:\n{result.stderr}")
    return tmp_path, tmp_path


def extract_audio_slice(video_path, start_s, dur_s, out_wav):
    """Extract a 16 kHz mono PCM slice [start_s, start_s+dur_s) from *video_path*."""
    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
        "-ss", f"{start_s:.4f}", "-t", f"{dur_s:.4f}", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def align_segment(frames, image_processor):
    """Affine-align a list of RGB frames to 512x512 face crops.

    Returns:
        [N, 512, 512, 3] uint8 array, or None if detection failed on any frame.
    """
    aligned = []
    for i, frame in enumerate(frames):
        try:
            face, _box, _matrix = image_processor.affine_transform(frame)
        except RuntimeError as e:
            print(f"    [skip segment] face detection failed on frame {i}: {e}")
            return None
        if isinstance(face, torch.Tensor):
            face = face.permute(1, 2, 0).cpu().numpy()
        aligned.append(np.asarray(face, dtype=np.uint8))
    return np.stack(aligned, axis=0)


def process_video(video_path, args, image_processor, sample_dirs):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n=== {stem} ===")

    cfr_path, tmp_cfr = ensure_25fps(video_path)
    clip_dur = args.clip_frames / FPS
    n_clips = 0
    try:
        cap = cv2.VideoCapture(cfr_path)
        if not cap.isOpened():
            print(f"  [skip video] cannot open {cfr_path}")
            return
        # Reset the aligner's temporal smoothing for each new video.
        image_processor.restorer.p_bias = None

        seg_idx = 0
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) == args.clip_frames or (not ret and frames):
                if len(frames) < args.clip_frames:
                    break  # drop the trailing partial clip
                sample_dir = os.path.join(args.output_dir, f"{stem}_{seg_idx:04d}")
                seg_start_s = seg_idx * clip_dur
                seg_idx += 1

                done = all(
                    os.path.isfile(os.path.join(sample_dir, f))
                    for f in ("sub_clip.mp4", "audio.wav", "prompt.txt")
                )
                if args.skip_existing and done:
                    print(f"  [exists] {sample_dir}")
                    sample_dirs.append(sample_dir)
                    n_clips += 1
                elif (aligned := align_segment(frames, image_processor)) is not None:
                    os.makedirs(sample_dir, exist_ok=True)
                    save_frames_as_video(aligned, os.path.join(sample_dir, "sub_clip.mp4"), fps=FPS)
                    # Audio comes from the ORIGINAL file — the 25fps CFR
                    # conversion preserves wall-clock time, so offsets match.
                    extract_audio_slice(video_path, seg_start_s, clip_dur,
                                        os.path.join(sample_dir, "audio.wav"))
                    with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                        f.write(args.prompt + "\n")
                    print(f"  [ok] {sample_dir} ({args.clip_frames} frames)")
                    sample_dirs.append(sample_dir)
                    n_clips += 1
                frames = []
                if args.max_clips_per_video and n_clips >= args.max_clips_per_video:
                    break
            if not ret:
                break
        cap.release()
    finally:
        if tmp_cfr is not None and os.path.exists(tmp_cfr):
            os.remove(tmp_cfr)
    if n_clips == 0:
        print(f"  [warn] no full {args.clip_frames}-frame clips extracted from {stem}")


def main():
    args = parse_args()
    videos = list_videos(args)
    if not videos:
        raise ValueError("No input videos found")
    print(f"Preparing {len(videos)} video(s) -> {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    image_processor = load_image_processor(args.mask_path, args.device)

    sample_dirs = []
    for video_path in videos:
        process_video(video_path, args, image_processor, sample_dirs)

    if not sample_dirs:
        raise RuntimeError("No samples were produced — check the input videos.")

    sample_dirs = [os.path.abspath(d) for d in sample_dirs]
    val_count = min(args.val_count, max(len(sample_dirs) - 1, 0))
    train_dirs = sample_dirs[:len(sample_dirs) - val_count] if val_count else sample_dirs
    val_dirs = sample_dirs[len(sample_dirs) - val_count:] if val_count else sample_dirs[-1:]

    train_list = os.path.join(args.output_dir, args.train_list)
    val_list = os.path.join(args.output_dir, args.val_list)
    with open(train_list, "w") as f:
        f.write("\n".join(train_dirs) + "\n")
    with open(val_list, "w") as f:
        f.write("\n".join(val_dirs) + "\n")

    print(f"\nWrote {len(train_dirs)} train / {len(val_dirs)} val samples")
    print(f"  OMNIAVATAR_DATA_LIST={train_list}")
    print(f"  OMNIAVATAR_VAL_LIST={val_list}")


if __name__ == "__main__":
    main()
