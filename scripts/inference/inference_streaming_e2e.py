#!/usr/bin/env python3
"""End-to-end streaming inference — face preprocessing runs inside the AR loop.

Like ``inference_streaming.py`` but with no upfront pass over the video: face
detection + 512x512 alignment (STAGE 1) are pulled into the AR loop and run
per block on just the frames that block needs (9 frames for block 0, 12 for
each block after). Encode, denoise, decode and compositing were already
per-block; this closes the last full-video stage, so

  * time-to-first-frame is independent of clip length (the first composited
    frames exist after reading only 9 input frames), and
  * host memory is bounded: original frames and aligned faces are released
    once composited instead of being held for the whole clip.

Audio is still encoded upfront ("Tier 1" streaming): wav2vec2 uses global
self-attention, so chunked audio encoding would diverge from the trained
encoder. The audio file is known ahead of time in this script's use case, and
one wav2vec pass over it is cheap; the streaming win is that no *video* pass
happens before generation starts.

Output is bit-identical to ``inference_streaming.py`` with its default
``--streamwise_encode``: per-frame alignment carries the same forward-only
smoothing state (``AlignRestore.p_bias``), the per-block VAE encode consumes
the same frame chunks through the same feat_cache stream, and compositing is
unchanged. Verified by scripts comparing composited frames elementwise.

Differences from inference_streaming.py:
  * No ``--face_cache_dir`` (an upfront detection cache contradicts streaming).
  * No ``--no_streamwise_encode`` (per-block encode is the point; always on).
  * Precomputed conditioning tensors in ``--input_dir`` subdirs are ignored —
    conditioning is always built from the raw video.
  * A mid-video face-detection failure no longer aborts the sample (the batch
    script drops the whole sample before generating anything). The frame keeps
    the previous aligned face for conditioning and the compositor passes the
    original frame through unchanged.

Usage:
    python scripts/inference/inference_streaming_e2e.py \
        --video_path /path/to/reference.mp4 \
        --output_path /path/to/output.mp4 \
        --ckpt_path /path/to/sf_trained_student.pth \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --wav2vec_path /path/to/wav2vec2-base-960h \
        --mask_path /path/to/mask.png \
        --taehv_ckpt /path/to/taew2_1.pth \
        --text_embeds_path /path/to/text_emb.pt
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from _common import (
    TAEHVDecoderWrapper,
    load_vae, load_wav2vec, load_or_encode_text,
    resolve_audio, compute_generation_length,
    load_image_processor,
    composite_with_latentsync_float,
    save_frames_as_video, mux_video_with_audio,
    encode_audio, frames_to_tensor, apply_spatial_mask, load_latentsync_mask,
    enumerate_samples,
)
# Use the 14B loader unconditionally — it dispatches on args.model_size
# (constructor_merge_lora flag + post-load PEFT merge gated to 14B).
# For 1.3B the LoRA-merge steps are skipped.
from _loader import load_diffusion_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end streaming inference: face preprocessing, encode, "
                    "denoise, decode and compositing all run per AR block.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model paths ---
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="SF-trained student checkpoint (.pth)")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to Wan2.1_VAE.pth")
    parser.add_argument("--wav2vec_path", type=str, required=True,
                        help="Path to wav2vec2-base-960h directory")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="Path to LatentSync mask.png")
    parser.add_argument("--base_model_paths", type=str, default=None,
                        help="Comma-separated safetensor paths for base Wan 2.1 T2V "
                             "(not needed with the released self-contained checkpoint)")
    parser.add_argument("--model_size", type=str, default="14B",
                        choices=["1.3B", "14B"],
                        help="Student model size; 14B uses PEFT LoRA path.")
    parser.add_argument("--merge_lora_post_load", action="store_true", default=True,
                        help="14B only: merge PEFT LoRA into base after load_state_dict.")
    parser.add_argument("--no_merge_lora_post_load", action="store_false",
                        dest="merge_lora_post_load",
                        help="Disable post-load LoRA merge (keep PEFT layers active)")
    parser.add_argument("--omniavatar_ckpt_path", type=str, default=None,
                        help="OmniAvatar LoRA+audio checkpoint "
                             "(not needed with the released self-contained checkpoint)")
    parser.add_argument("--text_embeds_path", type=str, default=None,
                        help="Pre-computed T5 embeddings .pt file")
    parser.add_argument("--text_encoder_path", type=str, default=None,
                        help="T5 model path for runtime encoding")
    parser.add_argument("--prompt", type=str, default="a person talking",
                        help="Text prompt (encoded when --text_encoder_path is set)")

    # --- TAEHV ---
    parser.add_argument("--taehv_ckpt", type=str, default=None,
                        help="Path to TAEHV taew2_1.pth (required by the default "
                             "streaming_taehv / batch_taehv decoders)")

    # --- Streaming decoder mode ---
    parser.add_argument("--streaming_decoder", type=str, default="streaming_taehv",
                        choices=["streaming_taehv", "batch_taehv", "wan_vae"],
                        help="Decoder mode for streaming pipeline.")

    # --- Input/output ---
    parser.add_argument("--video_path", type=str, default=None,
                        help="Reference video path (any resolution)")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Separate audio source (extracted from video if not provided)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of sample subdirs (each with sub_clip.mp4, audio.wav). "
                             "Mutually exclusive with --video_path.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose output already exists (for resume)")

    # --- Generation params ---
    parser.add_argument("--t_list", nargs="+", type=float, default=[0.999, 0.769, 0.0],
                        help="Denoising schedule. Must match the checkpoint's distillation "
                             "schedule: the released 14B student is a 2-step t769 model "
                             "(0.999 -> 0.769 -> 0.0).")
    parser.add_argument("--chunk_size", type=int, default=3,
                        help="Number of latent frames per AR chunk")
    parser.add_argument("--num_latent_frames", type=int, default=None,
                        help="Override generation length (must be multiple of chunk_size)")
    parser.add_argument("--min_latent_frames", type=int, default=0,
                        help="Floor on num_latent; shorter audio is padded via zero-audio "
                             "+ ping-pong video. 0 disables.")
    parser.add_argument("--context_noise", type=float, default=0.0,
                        help="Noise added to context frames during AR generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Output video FPS")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")

    # --- Attention ---
    parser.add_argument("--local_attn_size", type=int, default=7,
                        help="Rolling local attention window in latent frames. Default 7 "
                             "(the trained window: 1 sink + 6 rolling) keeps VRAM constant "
                             "for any clip length. -1 = full attention over the whole clip "
                             "(VRAM grows with clip length).")
    parser.add_argument("--sink_size", type=int, default=1,
                        help="Number of initial latent frames always kept in the attention "
                             "window (default 1, matching training)")
    parser.add_argument("--use_dynamic_rope", action="store_true", default=True,
                        help="Window-local dynamic RoPE (default: on, matching training)")
    parser.add_argument("--no_dynamic_rope", action="store_false", dest="use_dynamic_rope",
                        help="Disable window-local dynamic RoPE (absolute positions; "
                             "pair with --local_attn_size -1)")

    # --- Compositing ---
    parser.add_argument("--composite_full_face", action="store_true",
                        help="Composite the entire generated 512x512 face back into the "
                             "original frame. Default: blend only the mouth region of the "
                             "generated face; the rest stays from the input video.")
    parser.add_argument("--defer_composite", action="store_true",
                        help="Skip per-block compositing inside the AR loop; "
                             "concat all decoded chunks and run "
                             "composite_with_latentsync_float once after the "
                             "loop ends. Improves throughput (no per-block "
                             ".cpu() sync) at the cost of first-frame latency "
                             "and bounded memory (all frames stay buffered).")

    # --- torch.compile ---
    parser.add_argument("--compile", action="store_true",
                        help="Wrap diffusion model + Wan VAE encoder/decoder + "
                             "TAEHV (when present) with torch.compile.")

    return parser.parse_args()


def validate_args(args):
    if args.input_dir is not None and args.video_path is not None:
        raise ValueError("--input_dir and --video_path are mutually exclusive")
    if args.input_dir is None and args.video_path is None:
        raise ValueError("Must provide either --input_dir or --video_path")
    if args.input_dir is not None and args.output_dir is None:
        raise ValueError("--input_dir requires --output_dir")
    if args.input_dir is None and args.output_path is None:
        raise ValueError("--video_path mode requires --output_path")
    if args.text_embeds_path is None and args.text_encoder_path is None:
        raise ValueError(
            "Text conditioning is required: pass --text_embeds_path "
            "(precomputed T5 embeddings) or --text_encoder_path "
            "(encodes --prompt at runtime)."
        )
    if args.streaming_decoder in ("streaming_taehv", "batch_taehv") and not args.taehv_ckpt:
        raise ValueError(f"--streaming_decoder {args.streaming_decoder} requires --taehv_ckpt")


# ===========================================================================
# Streaming face preprocessing (STAGE 1 inside the AR loop)
# ===========================================================================

class StreamingFaceSource:
    """Frame-causal replacement for ``preprocess_with_latentsync``.

    Reads the reference video frame by frame and runs InsightFace detection +
    LatentSync affine alignment per frame, in order, carrying the alignment
    smoothing state (``AlignRestore.p_bias``, a forward-only EMA) exactly as
    the upfront pass does — so the aligned faces and affine matrices are
    bit-identical to ``preprocess_with_latentsync`` over the same sequence.

    ``next_block(block_idx, chunk_size)`` ingests exactly the frames the AR
    block needs (``4*chunk_size - 3`` for block 0, ``4*chunk_size`` after) and
    returns them as a ``[1, 3, n, res, res]`` pixel tensor in [-1, 1]. The
    per-frame metadata the compositor consumes (original frame, box, affine
    matrix, aligned face) accumulates in ``self.metadata``;
    ``release_composited(upto)`` frees the heavy entries once composited so
    host memory stays bounded for any clip length.

    Ping-pong: when the audio-driven length exceeds the video, frames past EOF
    are served from a rewind buffer with the same cycle as
    ``pingpong_indices`` (0..n-1, n-2..1, repeat), and each served frame goes
    through detection again — matching the upfront pass, which detects over
    the extended sequence. The rewind buffer is only kept when the container's
    reported frame count says the video is shorter than the requested length;
    for longer videos no raw frames are retained beyond compositing.
    """

    MIN_FRAMES = 5  # same floor as preprocess_with_latentsync

    def __init__(self, video_path, image_processor, num_frames):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path
        self.image_processor = image_processor
        self.num_frames = num_frames

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Rewind buffer for ping-pong. Only kept when the container reports
        # fewer frames than requested (ping-pong likely); a long video streams
        # through without retaining raw frames here.
        reported = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._keep_rewind_buffer = reported < num_frames
        self._rewind_frames = []
        self._eof = False
        self._n_raw = None  # actual decodable frame count, known at EOF

        # Per-frame metadata, indexed by extended-sequence frame index. Grows
        # with ingest; heavy entries are freed by release_composited().
        self.original_frames = []
        self.aligned_faces = []
        self.boxes = []
        self.affine_matrices = []
        self.detection_failures = []
        self._released_upto = 0
        self._last_face = None  # fallback for mid-video detection failures

        # Reset temporal smoothing bias for the new video (same as the
        # upfront pass).
        image_processor.restorer.p_bias = None

    # -- frame supply ------------------------------------------------------

    def _next_raw_frame(self, idx):
        """Return RGB frame *idx* of the ping-pong-extended sequence.

        Must be called with consecutive indices (ingest is sequential).
        """
        if not self._eof:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self._keep_rewind_buffer:
                    self._rewind_frames.append(frame)
                return frame
            self._eof = True
            self._n_raw = idx
            self.cap.release()
            if self._n_raw < self.MIN_FRAMES:
                raise RuntimeError(
                    f"Too few frames ({self._n_raw}) in {self.video_path}"
                )
            if not self._keep_rewind_buffer:
                raise RuntimeError(
                    f"{self.video_path}: decoded only {self._n_raw} frames but "
                    f"{self.num_frames} are needed and the container reported "
                    f"enough — frame-count metadata is wrong. Remux the video "
                    f"(e.g. ffmpeg -i in.mp4 -c copy out.mp4) and retry."
                )

        # Past EOF: ping-pong over the buffered video. Same cycle as
        # pingpong_indices: 0..n-1, n-2..1, repeating (period 2n-2).
        n = self._n_raw
        if n == 1:
            return self._rewind_frames[0]
        cycle = 2 * n - 2
        ci = idx % cycle
        src = ci if ci < n else cycle - ci
        return self._rewind_frames[src]

    # -- ingest ------------------------------------------------------------

    def _ingest_frame(self):
        """Read + detect + align one frame; append its metadata."""
        idx = len(self.original_frames)
        frame = self._next_raw_frame(idx)
        self.original_frames.append(frame)
        try:
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            self._last_face = face
        except RuntimeError as e:
            if self._last_face is None:
                raise RuntimeError(
                    f"Face detection failed on the first frame of "
                    f"{self.video_path}: {e}"
                )
            # Keep the previous aligned face for conditioning; a None box makes
            # the compositor pass the original frame through unchanged.
            print(f"[StreamingFaceSource] Face detection failed for frame {idx}: "
                  f"{e} — reusing previous aligned face")
            face, box, affine_matrix = self._last_face, None, None
            self.detection_failures.append(idx)
        self.aligned_faces.append(face)
        self.boxes.append(box)
        self.affine_matrices.append(affine_matrix)
        return face

    def next_block(self, block_idx, chunk_size):
        """Ingest the next AR block's frames -> ``[1, 3, n, res, res]`` in [-1, 1].

        Block 0 consumes ``4*chunk_size - 3`` frames (Wan VAE: 1 frame for the
        first latent, 4 per latent after); later blocks consume
        ``4*chunk_size``.
        """
        n = 4 * chunk_size - 3 if block_idx == 0 else 4 * chunk_size
        start = len(self.original_frames)
        assert start + n <= self.num_frames, (
            f"Block {block_idx} would ingest past num_frames "
            f"({start}+{n} > {self.num_frames})"
        )
        faces = [self._ingest_frame() for _ in range(n)]
        faces_np = np.stack([
            f.permute(1, 2, 0).numpy() if isinstance(f, torch.Tensor) else f
            for f in faces
        ], axis=0)
        return frames_to_tensor(faces_np)  # [1, 3, n, res, res]

    # -- compositor interface ------------------------------------------------

    @property
    def metadata(self):
        """Dict in the ``preprocess_with_latentsync`` return format, over the
        frames ingested so far (compatible with composite_with_latentsync_float)."""
        return {
            "video_path": self.video_path,
            "original_frames": self.original_frames,
            "num_frames": self.num_frames,
            "aligned_faces": self.aligned_faces,
            "boxes": self.boxes,
            "affine_matrices": self.affine_matrices,
            "detection_failures": self.detection_failures,
        }

    def release_composited(self, upto):
        """Free original frames + aligned faces below index *upto* (composited).

        The compositor only ever reads indices >= its running frame_offset, so
        entries below it are dead. Boxes/matrices are tiny and kept.
        """
        for i in range(self._released_upto, min(upto, len(self.original_frames))):
            self.original_frames[i] = None
            self.aligned_faces[i] = None
        self._released_upto = max(self._released_upto, upto)

    def close(self):
        if self.cap is not None and not self._eof:
            self.cap.release()


def build_condition_e2e(wav2vec_model, wav2vec_extractor, audio_path, text_embeds,
                        mask_path, num_video_frames, device, dtype,
                        height=512, width=512):
    """Build the minimal condition dict for end-to-end streaming AR inference.

    Only audio is encoded upfront (Tier 1: wav2vec2 self-attention is global,
    so the trained encoder needs the whole waveform). Video-derived
    conditioning (ref_latent / ref_sequence / masked_video) is built
    incrementally inside the AR loop from the streaming face source.

    Returns:
        (condition, mask_pixel_binary) — the [H, W] float32 numpy pixel mask
        is applied per block to the aligned faces before the masked encode.
    """
    # ============================================================
    # STAGE 2: Wav2Vec2 audio encode (full audio at once)
    # ============================================================
    print("Encoding audio (full) ...")
    audio_emb = encode_audio(
        wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device
    )
    audio_emb = audio_emb.to(dtype=dtype)

    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != height or mask_np.shape[1] != width:
        mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)
    mask_pixel_binary = (mask_np > 0.5).astype(np.float32)

    latent_mask = load_latentsync_mask(mask_path, height // 8, width // 8)
    latent_mask = latent_mask.to(device=device, dtype=dtype)

    condition = {
        "text_embeds": text_embeds,
        "audio_emb": audio_emb,
        "ref_latent": None,       # set on block 0 from growing_ref_seq[..., :1]
        "mask": latent_mask,
        "masked_video": None,     # built incrementally in AR loop
        "ref_sequence": None,     # built incrementally in AR loop
    }
    return condition, mask_pixel_binary


@torch.no_grad()
def run_streaming_pipeline_e2e(
    model, decoder_vae, vae, condition, num_latent_frames, num_video_frames,
    args, face_source, mask_pixel_binary, image_processor,
    audio_path, output_path, device, dtype,
):
    """Run the end-to-end streaming pipeline: per-block preprocess → encode →
    denoise → decode → composite.

    Stage map (same labels as inference_streaming.py):
        STAGE 1  : Face detect + 512x512 alignment     (PER BLOCK, via
                   face_source.next_block — the one stage that was upfront
                   in inference_streaming.py)
        STAGE 2  : Wav2Vec2 audio encode                (upfront, in build_condition_e2e)
        STAGE 3b : per-block VAE encode of unmasked + masked streams
        STAGE 4  : Per-block 2-step DDIM denoise (CausalOmniAvatarWan)
        STAGE 5  : Per-block VAE decode (Wan VAE / TAEHV / StreamingTAEHV)
        STAGE 6  : Per-block compositing (paste lip region back into full-res frame)
        STAGE 7  : Per-block KV cache update (model forward with denoised x0 stored as cache)
        STAGE 8  : (streaming_taehv only) flush remaining buffered frames
        STAGE 9  : Save MP4 + ffmpeg audio mux

    Returns:
        composited_np: [N, H, W, 3] uint8 numpy array of composited frames.
    """
    # ============================================================
    # STAGE 5-prep: Decoder selection
    # ============================================================
    _use_streaming_dec = False
    if args.streaming_decoder == "streaming_taehv":
        try:
            from lipforcing.methods.reward.taehv import StreamingTAEHV
            if hasattr(decoder_vae, 'taehv') and decoder_vae.taehv is not None:
                streaming_dec = StreamingTAEHV(decoder_vae.taehv)
                _use_streaming_dec = True
                print("  Using StreamingTAEHV decoder (temporal state across chunks)")
        except Exception:
            pass
    if not _use_streaming_dec:
        if args.streaming_decoder == "wan_vae":
            print("  Using Wan VAE decoder per chunk")
        else:
            print("  Using batch TAEHV decoder per chunk")

    # ============================================================
    # STAGE 5-prep: Wan VAE streaming-decode cache reset
    # ============================================================
    _use_wan_streaming = (args.streaming_decoder == "wan_vae"
                          and hasattr(decoder_vae, "streaming_decode_chunk"))
    if _use_wan_streaming:
        print("  Wan VAE: streaming-decode mode (cache continuity across chunks)")
        decoder_vae.reset_decode_cache()

    # ============================================================
    # STAGE 3b-prep: Streamwise encode setup (always on in this script)
    # ============================================================
    # Two independent encoder feat_cache streams: one for unmasked
    # (ref_sequence) and one for masked (masked_latents). We share a single
    # VAE instance and swap cache state between calls.
    vae.reset_encode_cache()
    unmasked_state = vae.save_encode_cache_state()  # both empty
    masked_state = vae.save_encode_cache_state()
    # These grow by chunk_size latents per block. _build_y slices them by
    # absolute frame range, and they are small (16x64x64 bf16 = 128 KB per
    # latent frame, ~1.5 MB per second of video), so keeping the full history
    # is fine even for long clips.
    growing_ref_seq = None     # [1, 16, T_so_far, H_lat, W_lat]
    growing_masked = None
    original_vae_dtype = next(vae.parameters()).dtype
    vae.to(dtype=torch.bfloat16)

    def _latent_frame_chunks(block_idx):
        """Block-relative (start, end) pixel-frame ranges, one per latent.
        Block 0: 1 frame for the first latent, then 4 per latent.
        Block i>=1: 4 frames per latent.
        """
        if block_idx == 0:
            return [(0, 1)] + [(1 + 4 * k, 5 + 4 * k)
                               for k in range(args.chunk_size - 1)]
        return [(4 * k, 4 * k + 4) for k in range(args.chunk_size)]

    def _stream_encode_block(block_idx, block_pixel_tensor, prev_state):
        """Encode one AR block's pixel frames into chunk_size latents."""
        vae.load_encode_cache_state(prev_state)
        chunks = []
        for s, e in _latent_frame_chunks(block_idx):
            chunk = block_pixel_tensor[0, :, s:e].to(
                dtype=torch.bfloat16, device=device)
            chunks.append(vae.streaming_encode_chunk(chunk, device=device))
        new_state = vae.save_encode_cache_state()
        # chunks are [1, 16, 1, H, W]; concat along time
        new_lats = torch.cat([c.squeeze(0) for c in chunks], dim=1).unsqueeze(0)
        return new_lats.to(dtype=dtype), new_state

    # --- Prepare model ---
    model.total_num_frames = num_latent_frames
    model.clear_caches()
    B, C = 1, 16
    H_lat = image_processor.resolution // 8
    W_lat = image_processor.resolution // 8
    t_list_t = torch.tensor(args.t_list, device=device, dtype=torch.float64)

    # Pre-generate all noise at once (must match the non-streaming pipeline)
    torch.manual_seed(args.seed)
    all_noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat, device=device, dtype=dtype)

    num_blocks = num_latent_frames // args.chunk_size
    all_composited_frames = []
    # When --defer_composite is on, we skip per-block compositing and stash
    # the raw decoded chunk_float tensors here, then composite once after
    # the AR loop ends. (This also disables freeing of composited frames.)
    all_decoded_chunks_cpu = []
    video_frame_offset = 0

    # ============================================================
    # STAGE 1/3b/4/5/6/7: AR streaming loop (repeats num_blocks times)
    # ============================================================
    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * args.chunk_size

        # ----------------------------------------------------------
        # STAGE 1: per-block face detect + align (this script's change)
        # ----------------------------------------------------------
        # Pull exactly this block's frames through detection + alignment.
        # Detection state (p_bias EMA) carries forward frame-to-frame, so
        # the result matches the upfront pass bit for bit.
        block_pixels = face_source.next_block(block_idx, args.chunk_size)
        block_masked = apply_spatial_mask(
            block_pixels, mask_pixel_binary, mask_all_frames=True
        )

        # ----------------------------------------------------------
        # STAGE 3b: per-block VAE encode (two feat_cache streams)
        # ----------------------------------------------------------
        new_unmasked, unmasked_state = _stream_encode_block(
            block_idx, block_pixels, unmasked_state)
        new_masked, masked_state = _stream_encode_block(
            block_idx, block_masked, masked_state)

        growing_ref_seq = (new_unmasked if growing_ref_seq is None
                           else torch.cat([growing_ref_seq, new_unmasked], dim=2))
        growing_masked = (new_masked if growing_masked is None
                          else torch.cat([growing_masked, new_masked], dim=2))

        condition["ref_sequence"] = growing_ref_seq
        condition["masked_video"] = growing_masked
        condition["ref_latent"] = growing_ref_seq[:, :, :1].contiguous()

        # ----------------------------------------------------------
        # STAGE 4: 2-step DDIM denoise on this block's latents
        # ----------------------------------------------------------
        noisy_input = all_noise[:, :, cur_start_frame:cur_start_frame + args.chunk_size]
        for step_idx in range(len(t_list_t) - 1):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]
            x0_pred = model(
                noisy_input, t_cur.expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False, is_ar=True,
                fwd_pred_type="x0", use_gradient_checkpointing=False,
            )
            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B))
            else:
                noisy_input = x0_pred

        # ----------------------------------------------------------
        # STAGE 5: per-block VAE decode (chunk_size latents -> video frames)
        # ----------------------------------------------------------
        if _use_streaming_dec:
            chunk_latent = x0_pred[0].to(device, dtype=torch.float16)
            chunk_latent_ntchw = chunk_latent.permute(1, 0, 2, 3).unsqueeze(0)
            chunk_frames = []
            for t in range(chunk_latent_ntchw.shape[1]):
                latent_t = chunk_latent_ntchw[:, t:t+1]
                frame = streaming_dec.decode(latent_t)
                while frame is not None:
                    chunk_frames.append(frame)
                    frame = streaming_dec.decode()
            if chunk_frames:
                chunk_float = torch.cat(chunk_frames, dim=1).squeeze(0)
            else:
                chunk_float = None
        elif _use_wan_streaming:
            # Stream one latent at a time so feat_cache state advances
            # exactly as in the per-latent inner loop of VideoVAE_.decode.
            vae_dtype = next(decoder_vae.parameters()).dtype
            chunk_latent = x0_pred[0].to(vae_dtype)  # [16, T, h_lat, w_lat]
            video_chunks = []
            for t in range(chunk_latent.shape[1]):
                latent_t = chunk_latent[:, t:t+1]  # [16, 1, h_lat, w_lat]
                v = decoder_vae.streaming_decode_chunk(latent_t, device=device)
                video_chunks.append(v)
            chunk_decoded = torch.cat(video_chunks, dim=2)  # [1, 3, T, H, W]
            chunk_float = chunk_decoded[0].permute(1, 0, 2, 3)
            chunk_float = ((chunk_float + 1) / 2).clamp(0, 1)
        else:
            chunk_latent = x0_pred[0].to(torch.float32)
            chunk_decoded = decoder_vae.decode([chunk_latent], device=device)
            chunk_decoded = chunk_decoded.clamp(-1, 1)
            chunk_float = chunk_decoded[0].permute(1, 0, 2, 3)
            chunk_float = ((chunk_float + 1) / 2).clamp(0, 1)

        # ----------------------------------------------------------
        # STAGE 6: per-block compositing (CPU)
        # ----------------------------------------------------------
        if chunk_float is not None:
            if args.defer_composite:
                all_decoded_chunks_cpu.append(chunk_float.cpu())
                video_frame_offset += all_decoded_chunks_cpu[-1].shape[0]
            else:
                composited = composite_with_latentsync_float(
                    chunk_float.cpu(), face_source.metadata, image_processor,
                    use_mouth_only_compositing=not args.composite_full_face,
                    frame_offset=video_frame_offset,
                )
                all_composited_frames.append(composited)
                video_frame_offset += composited.shape[0]
                # Everything below the composite offset is dead — free it so
                # host memory stays bounded for any clip length.
                face_source.release_composited(video_frame_offset)

        # ----------------------------------------------------------
        # STAGE 7: KV cache update (extra model forward per block)
        # ----------------------------------------------------------
        cache_input = x0_pred
        t_cache = torch.full((B,), args.context_noise, device=device, dtype=torch.float64)
        if args.context_noise > 0:
            cache_eps = torch.randn_like(x0_pred)
            cache_input = model.noise_scheduler.forward_process(
                x0_pred, cache_eps,
                torch.tensor(args.context_noise, device=device, dtype=torch.float64).expand(B))
        model(cache_input, t_cache, condition=condition,
              cur_start_frame=cur_start_frame, store_kv=True, is_ar=True,
              fwd_pred_type="x0", use_gradient_checkpointing=False)

        if (block_idx + 1) % 5 == 0 or block_idx == num_blocks - 1:
            print(f"  Streaming block {block_idx + 1}/{num_blocks} done")

    model.clear_caches()
    vae.to(dtype=original_vae_dtype)

    # ============================================================
    # STAGE 8: Flush remaining buffered frames (streaming_taehv only)
    # ============================================================
    if _use_streaming_dec:
        flush_frames = streaming_dec.flush_decoder()
        if flush_frames:
            flush_float = torch.cat(flush_frames, dim=1).squeeze(0)
            flush_cpu = flush_float.cpu()
            if args.defer_composite:
                all_decoded_chunks_cpu.append(flush_cpu)
                video_frame_offset += flush_cpu.shape[0]
            else:
                composited = composite_with_latentsync_float(
                    flush_cpu, face_source.metadata, image_processor,
                    use_mouth_only_compositing=not args.composite_full_face,
                    frame_offset=video_frame_offset,
                )
                all_composited_frames.append(composited)
                video_frame_offset += composited.shape[0]
                face_source.release_composited(video_frame_offset)

    # ============================================================
    # STAGE 6 (deferred): one-shot compositing over all decoded frames
    # ============================================================
    if args.defer_composite and all_decoded_chunks_cpu:
        all_decoded = torch.cat(all_decoded_chunks_cpu, dim=0)
        composited_np = composite_with_latentsync_float(
            all_decoded, face_source.metadata, image_processor,
            use_mouth_only_compositing=not args.composite_full_face,
            frame_offset=0,
        )
    else:
        composited_np = np.concatenate(all_composited_frames, axis=0)

    # ============================================================
    # STAGE 9: Save MP4 + ffmpeg audio mux (CPU)
    # ============================================================
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_frames_as_video(composited_np, output_path, fps=args.fps)
    video_duration = composited_np.shape[0] / args.fps
    tmp_composited = output_path + ".tmp.mp4"
    os.rename(output_path, tmp_composited)
    mux_video_with_audio(tmp_composited, audio_path, output_path, duration_s=video_duration)
    if os.path.exists(tmp_composited):
        os.remove(tmp_composited)

    return composited_np


def main():
    args = parse_args()
    validate_args(args)

    # Activate @conditional_compile decorators in network_causal.py BEFORE
    # the model class is imported (which happens later via load_diffusion_model).
    if args.compile:
        os.environ["LIPFORCING_COMPILE"] = "true"

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load models ---
    print("Loading diffusion model ...")
    model = load_diffusion_model(args, device, dtype)

    print("Loading VAE ...")
    vae = load_vae(args.vae_path, device)

    # Decoder selection based on streaming_decoder mode
    if args.streaming_decoder in ("streaming_taehv", "batch_taehv"):
        if not args.taehv_ckpt:
            raise ValueError(f"--streaming_decoder {args.streaming_decoder} requires --taehv_ckpt")
        print(f"Loading TAEHV decoder from {args.taehv_ckpt} ...")
        decoder_vae = TAEHVDecoderWrapper(args.taehv_ckpt, device)
    else:
        decoder_vae = vae

    encoder_vae = vae

    # Eagerly load Wav2Vec + text
    print("Loading Wav2Vec2 (eager) ...")
    wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)
    # OmniAvatar Wav2VecModel requires seq_len + output_hidden_states.
    _dummy_audio = np.zeros(16000, dtype=np.float32)
    _dummy_input = wav2vec_extractor(_dummy_audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        wav2vec_model(
            _dummy_input.input_values.to(device),
            seq_len=25, output_hidden_states=True,
        )
    print("Wav2Vec2 warmed up.")

    print("Loading text embeddings ...")
    text_embeds = load_or_encode_text(args, device, dtype)

    # LatentSync ImageProcessor (face detection + alignment; always on —
    # this pipeline detects, aligns and composites every block on the fly).
    image_processor = load_image_processor(args.mask_path, device)

    # ===================================================================
    # Optional torch.compile wrapping (compile time absorbed by warmup)
    # ===================================================================
    if args.compile:
        print("[--compile] Hot functions decorated with @conditional_compile.")
        _compile_kw = dict(mode=None, backend="inductor", dynamic=None)

        # Wan VAE decoder compile (skip if decoder is TAEHV — its internals
        # do enumerate(self.decoder) which breaks under OptimizedModule)
        if not isinstance(decoder_vae, TAEHVDecoderWrapper):
            if hasattr(decoder_vae, 'model') and hasattr(decoder_vae.model, 'decoder'):
                decoder_vae.model.decoder = torch.compile(
                    decoder_vae.model.decoder, **_compile_kw)
                print("[--compile] Wan VAE decoder compiled.")

        # Wan VAE encoder compile
        if hasattr(encoder_vae, 'model') and hasattr(encoder_vae.model, 'encoder'):
            encoder_vae.model.encoder = torch.compile(
                encoder_vae.model.encoder, **_compile_kw)
            print("[--compile] Wan VAE encoder compiled.")

    # --- Loop over samples ---
    samples = list(enumerate_samples(args))
    succeeded, failed, skipped = [], [], []

    for sample_idx, (name, video_path, audio_path_sample, _precomputed) in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"[{sample_idx+1}/{len(samples)}] {name}")
        print(f"{'='*60}")

        if args.input_dir is not None:
            output_path = os.path.join(args.output_dir, f"{name}.mp4")
        else:
            output_path = args.output_path

        if args.skip_existing and os.path.isfile(output_path):
            print(f"  [Skip] {output_path}")
            skipped.append(name)
            continue

        tmp_audio = None
        face_source = None
        try:
            audio_path, tmp_audio = resolve_audio(
                audio_path=audio_path_sample, video_path=video_path,
            )

            num_latent_frames, num_video_frames = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps,
                min_latent_frames=args.min_latent_frames,
            )

            # STAGE 1 runs inside the AR loop — the source only opens the
            # video here; no frames are read or detected yet.
            face_source = StreamingFaceSource(
                video_path, image_processor, num_video_frames,
            )

            # STAGE 2: audio (upfront) + mask; video conditioning is built
            # per block inside the pipeline.
            print("Building conditioning (audio + mask) ...")
            condition, mask_pixel_binary = build_condition_e2e(
                wav2vec_model, wav2vec_extractor, audio_path, text_embeds,
                args.mask_path, num_video_frames, device, dtype,
                height=image_processor.resolution,
                width=image_processor.resolution,
            )

            print(f"Running e2e streaming pipeline ({args.streaming_decoder}) ...")
            run_streaming_pipeline_e2e(
                model, decoder_vae, encoder_vae, condition,
                num_latent_frames, num_video_frames,
                args, face_source, mask_pixel_binary, image_processor,
                audio_path, output_path, device, dtype,
            )

            succeeded.append(name)
            print(f"  Done: {output_path}")

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)
        finally:
            if face_source is not None:
                face_source.close()
            if tmp_audio is not None and os.path.exists(tmp_audio):
                os.remove(tmp_audio)
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
