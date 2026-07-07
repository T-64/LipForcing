#!/usr/bin/env python3
"""Segment-wise inference — block-wise AR generation, decode after the full rollout.

Generates lip-synced video from a reference video and audio using the
CausalOmniAvatarWan student model (14B by default; ``--model_size 1.3B`` also
supported) trained via Self-Forcing. The whole clip is denoised chunk-by-chunk
first, then decoded and composited in one pass — the maximum-quality path
(use ``inference_streaming.py`` for decode-as-you-go / low-latency output).

Decoding uses the full Wan VAE by default; pass ``--taehv_ckpt`` to swap in
the TAEHV tiny decoder for throughput (``--taehv_streaming`` /
``--taehv_encode`` further extend that).

Face detection + alignment + compositing (the LatentSync preprocessing
pipeline) runs by default so arbitrary talking-head videos work as input.
Pass ``--skip_preprocessing`` only when inputs are already 512x512 aligned
face crops.

Usage:
    python scripts/inference/inference_segmentwise.py \
        --video_path /path/to/reference.mp4 \
        --output_path /path/to/output.mp4 \
        --ckpt_path /path/to/sf_trained_student.pth \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --wav2vec_path /path/to/wav2vec2-base-960h \
        --mask_path /path/to/mask.png \
        --text_embeds_path /path/to/text_emb.pt
"""

import argparse
import os

import numpy as np
import torch

from _common import (
    TAEHVDecoderWrapper, StreamingTAEHVDecoderWrapper,
    load_vae, load_wav2vec, load_or_encode_text,
    resolve_audio, compute_generation_length,
    load_and_adjust_video,
    load_image_processor, preprocess_with_latentsync,
    build_condition, build_condition_from_precomputed,
    composite_with_latentsync_float,
    save_frames_as_video, mux_video_with_audio,
    enumerate_samples,
)
from _loader import load_diffusion_model  # (shared; see scripts/inference/_loader.py)


# ===========================================================================
# CLI argument parsing
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment-wise causal OmniAvatar inference (block-wise AR with audio conditioning)"
    )

    # --- Single-sample mode ---
    parser.add_argument("--video_path", type=str, default=None,
                        help="Reference video path (any resolution; must be 512x512 "
                             "only with --skip_preprocessing)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="SF-trained student checkpoint (.pth)")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to Wan2.1_VAE.pth")
    parser.add_argument("--taehv_ckpt", type=str, default=None,
                        help="Optional path to TAEHV taew2_1.pth. If set, uses the TAEHV "
                             "tiny decoder for latent->pixel decoding (full Wan VAE is still "
                             "used for encoding driving video unless --taehv_encode is set).")
    parser.add_argument("--taehv_encode", action="store_true",
                        help="Also use TAEHV for encoding the driving video (requires --taehv_ckpt). "
                             "Default: full Wan VAE encoder.")
    parser.add_argument("--taehv_streaming", action="store_true",
                        help="Use StreamingTAEHV for decoding (feeds latents one at a time). "
                             "Requires --taehv_ckpt.")
    parser.add_argument("--wav2vec_path", type=str, required=True,
                        help="Path to wav2vec2-base-960h directory")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="Path to LatentSync mask.png")

    # --- Optional model paths ---
    parser.add_argument("--base_model_paths", type=str, default=None,
                        help="Comma-separated safetensor paths for base Wan 2.1 T2V (1.3B or 14B)")
    parser.add_argument("--omniavatar_ckpt_path", type=str, default=None,
                        help="OmniAvatar LoRA+audio checkpoint")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Separate audio source (extracted from video if not provided)")

    # --- Generation parameters ---
    parser.add_argument("--num_latent_frames", type=int, default=None,
                        help="Override generation length (must be multiple of chunk_size)")
    parser.add_argument("--min_latent_frames", type=int, default=0,
                        help="Floor on num_latent; if audio is shorter, pad via zero-audio + ping-pong "
                             "video. 0 disables. 21 corresponds to the 81-frame (21 latent) generation length.")
    parser.add_argument("--prompt", type=str, default="a person talking",
                        help="Text prompt")
    parser.add_argument("--text_embeds_path", type=str, default=None,
                        help="Pre-computed T5 embeddings .pt file")
    parser.add_argument("--text_encoder_path", type=str, default=None,
                        help="T5 model path for runtime encoding")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory with precomputed .pt files (vae_latents_mask_all.pt, "
                             "ref_latents.pt, audio_emb_omniavatar.pt, text_emb.pt). "
                             "Bypasses VAE/Wav2Vec encoding — uses exact training-style tensors.")

    # --- Batch inference ---
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of sample subdirs (each with sub_clip.mp4, audio.wav). "
                             "Mutually exclusive with --video_path. Training-style sample "
                             "dirs contain pre-aligned 512x512 crops — combine with "
                             "--skip_preprocessing (face detection fails on tight crops).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose output already exists (for resume)")

    # --- Preprocessing (face detection + alignment + compositing) ---
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip the face detection + 512x512 alignment + compositing "
                             "pipeline. Requires inputs that are ALREADY 512x512 aligned "
                             "face crops; the output is the raw generated video (no "
                             "paste-back into the original frames).")
    parser.add_argument("--face_cache_dir", type=str, default=None,
                        help="Optional directory for face-detection caches; speeds up "
                             "repeated runs over the same videos. Unset = no caching.")
    parser.add_argument("--composite_full_face", action="store_true",
                        help="Composite the entire generated 512x512 face back into the "
                             "original frame. Default: blend only the mouth region of the "
                             "generated face; the rest stays from the input video.")
    parser.add_argument("--save_aligned", action="store_true",
                        help="Additionally save the raw generated 512x512 aligned face "
                             "video as <output>_aligned.mp4 (before compositing).")

    parser.add_argument("--t_list", type=float, nargs="+",
                        default=[0.999, 0.769, 0.0],
                        help="Denoising schedule. Must match the checkpoint's distillation "
                             "schedule: the released 14B student is a 2-step t769 model "
                             "(0.999 -> 0.769 -> 0.0).")
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
    parser.add_argument("--model_size", type=str, default="14B",
                        choices=["1.3B", "14B"],
                        help="Student model size. 14B is the default for SF LoRA runs.")
    parser.add_argument("--merge_lora_post_load", action="store_true", default=True,
                        help="After loading the SF trainable LoRA values, merge them into "
                             "base for inference speed. The model is constructed with "
                             "merge_lora=False (to expose lora_A/lora_B keys for the "
                             "trainable-filtered SF state_dict), then merged in-place "
                             "after load_state_dict.  Set --no_merge_lora_post_load to keep "
                             "PEFT layers active (slower forward, useful for debugging).")
    parser.add_argument("--no_merge_lora_post_load", action="store_false",
                        dest="merge_lora_post_load",
                        help="Disable post-load LoRA merge (keep PEFT layers active).")
    parser.add_argument("--chunk_size", type=int, default=3,
                        help="Number of latent frames per AR chunk")
    parser.add_argument("--context_noise", type=float, default=0.0,
                        help="Noise added to context frames during AR generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model dtype")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output video FPS")

    # --- torch.compile ---
    parser.add_argument("--compile", action="store_true",
                        help="Wrap diffusion model + Wan VAE encoder/decoder + "
                             "TAEHV (when present) with torch.compile. First "
                             "warmup clip absorbs the compile time; subsequent "
                             "clips run on the compiled graphs.")

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


# ===========================================================================
# Inference & post-processing
# ===========================================================================

@torch.no_grad()
def run_inference(
    model, condition, num_latent_frames, t_list,
    chunk_size, context_noise, seed, device, dtype,
):
    """Block-wise AR inference loop.

    Adapted from Self-Forcing's rollout_with_gradient but inference-only:
    - No gradients, no random exit steps
    - Full denoising per block (all steps in t_list)
    - KV cache updated after each block with denoised output
    - Rolling window eviction handled internally by CausalSelfAttention

    Args:
        model: CausalOmniAvatarWan (1.3B student)
        condition: dict with text_embeds, audio_emb, ref_latent, mask, etc.
        num_latent_frames: total latent frames to generate
        t_list: denoising timestep schedule (e.g. [0.999, 0.9, 0.75, 0.5, 0.0])
        chunk_size: frames per AR block (3)
        context_noise: noise level for cache updates (0 = clean)
        seed: random seed
        device: torch device
        dtype: torch dtype

    Returns:
        output: [1, 16, num_latent_frames, H_lat, W_lat] denoised latents
    """
    # Update model's total_num_frames for correct cache allocation
    model.total_num_frames = num_latent_frames
    model.clear_caches()

    # Determine spatial dims from ref_latent
    ref_latent = condition["ref_latent"]  # [1, 16, 1, H_lat, W_lat]
    B = ref_latent.shape[0]
    C = 16
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]

    num_blocks = num_latent_frames // chunk_size
    assert num_latent_frames % chunk_size == 0

    # Generate noise
    torch.manual_seed(seed)
    noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat, device=device, dtype=dtype)

    # Convert t_list to tensor
    t_list_t = torch.tensor(t_list, device=device, dtype=torch.float64)

    # Output accumulator
    output = torch.zeros_like(noise)

    print(f"  {num_blocks} blocks x {len(t_list) - 1} denoising steps")
    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size

        # Slice noise for this chunk
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        # Multi-step denoising
        for step_idx in range(len(t_list_t) - 1):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]

            # Forward pass — model.forward() handles _build_y, rescale_t, _forward_ar internally
            # Keep timesteps in float64 for numerically stable scheduling
            x0_pred = model(
                noisy_input,
                t_cur.expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False,
                is_ar=True,
                fwd_pred_type="x0",
                use_gradient_checkpointing=False,
            )

            if t_next > 0:
                # Add noise for next step (SDE: fresh random noise)
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B),
                )
            else:
                # Final step — clean output
                noisy_input = x0_pred

        # Store denoised chunk
        output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x0_pred

        # Update KV cache with denoised output (context for next block)
        cache_input = x0_pred
        t_cache = torch.full((B,), context_noise, device=device, dtype=torch.float64)
        if context_noise > 0:
            cache_eps = torch.randn_like(x0_pred)
            cache_input = model.noise_scheduler.forward_process(
                x0_pred, cache_eps,
                torch.tensor(context_noise, device=device, dtype=torch.float64).expand(B),
            )

        model(
            cache_input,
            t_cache,
            condition=condition,
            cur_start_frame=cur_start_frame,
            store_kv=True,
            is_ar=True,
            fwd_pred_type="x0",
            use_gradient_checkpointing=False,
        )

        if (block_idx + 1) % 10 == 0 or block_idx == num_blocks - 1:
            print(f"  Block {block_idx + 1}/{num_blocks} done")

    model.clear_caches()
    return output


@torch.no_grad()
def decode_and_save(vae, output_latents, audio_path, output_path, fps, device):
    """VAE decode latents -> save silent video -> mux with audio."""
    import imageio.v3 as iio

    # VAE decode — expects list of [C, T_lat, H_lat, W_lat] in float32
    latent_for_vae = output_latents[0].to(torch.float32)  # [16, T_lat, H_lat, W_lat]
    video_tensor = vae.decode([latent_for_vae], device=device)  # [1, 3, T_video, H, W]
    video_tensor = video_tensor.clamp(-1, 1)

    # Convert to uint8 frames: [T, H, W, 3]
    video_np = video_tensor[0]  # [3, T, H, W]
    video_np = video_np.permute(1, 2, 3, 0)  # [T, H, W, 3]
    video_np = ((video_np.float() + 1) * 127.5).clamp(0, 255).cpu().to(torch.uint8).numpy()

    # Save silent video to temp file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_silent = output_path + ".silent.mp4"
    iio.imwrite(
        tmp_silent,
        video_np,
        fps=fps,
        codec="libx264",
        output_params=["-loglevel", "quiet", "-crf", "18"],
    )
    print(f"  Silent video: {video_np.shape[0]} frames at {fps}fps")

    # Mux with audio
    video_duration = video_np.shape[0] / fps
    mux_video_with_audio(tmp_silent, audio_path, output_path, duration_s=video_duration)

    # Cleanup
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    validate_args(args)
    use_preprocessing = not args.skip_preprocessing

    # Activate per-function torch.compile decorators in network_causal.py
    # BEFORE the model class is imported (which happens later via
    # load_diffusion_model). Done by setting LIPFORCING_COMPILE=true.
    if args.compile:
        os.environ["LIPFORCING_COMPILE"] = "true"

    # --- Resolve dtype ---
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # --- Seed ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ===================================================================
    # Load models once (expensive — minutes for 14B/1.3B weights)
    # ===================================================================
    print("Loading diffusion model ...")
    model = load_diffusion_model(args, device, dtype)

    print("Loading VAE ...")
    vae = load_vae(args.vae_path, device)

    # Decoder selection: full Wan VAE stays loaded for encoding the driving video;
    # decoding swaps to TAEHV tiny decoder if --taehv_ckpt is provided.
    if args.taehv_streaming:
        if not args.taehv_ckpt:
            raise ValueError("--taehv_streaming requires --taehv_ckpt")
        print(f"Loading StreamingTAEHV decoder from {args.taehv_ckpt} ...")
        decoder_vae = StreamingTAEHVDecoderWrapper(args.taehv_ckpt, device)
    elif args.taehv_ckpt:
        print(f"Loading TAEHV tiny decoder from {args.taehv_ckpt} ...")
        decoder_vae = TAEHVDecoderWrapper(args.taehv_ckpt, device)
    else:
        decoder_vae = vae

    # Encoder selection: default to full Wan VAE. If --taehv_encode is set,
    # reuse the same TAEHV model (it implements both encode and decode).
    if args.taehv_encode:
        if not args.taehv_ckpt:
            raise ValueError("--taehv_encode requires --taehv_ckpt")
        print("Using TAEHV tiny encoder for driving video encoding.")
        encoder_vae = decoder_vae
    else:
        encoder_vae = vae

    # Eagerly load Wav2Vec + text
    wav2vec_model = wav2vec_extractor = None
    if args.wav2vec_path:
        print("Loading Wav2Vec2 (eager) ...")
        wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)
        # Warmup forward pass to compile CUDA kernels.
        # OmniAvatar Wav2VecModel requires seq_len + output_hidden_states.
        _dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz → 25 video-frames
        _dummy_input = wav2vec_extractor(_dummy_audio, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            wav2vec_model(
                _dummy_input.input_values.to(device),
                seq_len=25, output_hidden_states=True,
            )
        print("Wav2Vec2 warmed up.")
    print("Loading text embeddings (eager) ...")
    text_embeds = load_or_encode_text(args, device, dtype)

    # LatentSync ImageProcessor (face detection + alignment; on by default)
    image_processor = None
    if use_preprocessing:
        image_processor = load_image_processor(args.mask_path, device)

    # ===================================================================
    # Optional torch.compile wrapping (compile time absorbed by warmup)
    # ===================================================================
    # Hot DiT functions are decorated via @conditional_compile (activated by
    # LIPFORCING_COMPILE=true env var, set above before model imports).  Here
    # we additionally wrap the VAE / TAEHV encode + decode forwards.
    if args.compile:
        print("[--compile] Hot DiT functions decorated with @conditional_compile. "
              "Warmup clip will trigger Dynamo trace.")
        _compile_kw = dict(mode=None, backend="inductor", dynamic=None)

        # Compile VAE encode/decode paths.
        # TAEHV is skipped — its internals do `b = model[i]` which breaks
        # when the Sequential is wrapped in an OptimizedModule.  The DiT
        # gets compile via @conditional_compile (LIPFORCING_COMPILE=true above).

        # Wan VAE decoder compile (skip if decoder is TAEHV)
        if not isinstance(decoder_vae, (TAEHVDecoderWrapper, StreamingTAEHVDecoderWrapper)):
            if hasattr(decoder_vae, 'model') and hasattr(decoder_vae.model, 'decoder'):
                decoder_vae.model.decoder = torch.compile(
                    decoder_vae.model.decoder, **_compile_kw)
                print("[--compile] Wan VAE decoder compiled.")

        # Wan VAE encoder compile (skip if encoder is TAEHV)
        if not isinstance(encoder_vae, (TAEHVDecoderWrapper, StreamingTAEHVDecoderWrapper)):
            if hasattr(encoder_vae, 'model') and hasattr(encoder_vae.model, 'encoder'):
                if not isinstance(encoder_vae.model.encoder, torch._dynamo.eval_frame.OptimizedModule):
                    encoder_vae.model.encoder = torch.compile(
                        encoder_vae.model.encoder, **_compile_kw)
                    print("[--compile] Wan VAE encoder compiled.")

    # ===================================================================
    # Loop over samples
    # ===================================================================
    samples = list(enumerate_samples(args))
    succeeded, failed, skipped = [], [], []

    for sample_idx, (name, video_path, audio_path_sample, precomputed_dir) in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"[{sample_idx+1}/{len(samples)}] {name}")
        print(f"{'='*60}")

        # --- Determine output path ---
        if args.input_dir is not None:
            output_path = os.path.join(args.output_dir, f"{name}.mp4")
        else:
            output_path = args.output_path

        # --- Skip existing ---
        if args.skip_existing and os.path.isfile(output_path):
            print(f"  [Skip] Output exists: {output_path}")
            skipped.append(name)
            continue

        tmp_audio = None
        try:
            # --- Resolve audio ---
            audio_path, tmp_audio = resolve_audio(
                audio_path=audio_path_sample, video_path=video_path,
            )

            # --- Compute generation length ---
            num_latent_frames, num_video_frames = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps,
                min_latent_frames=args.min_latent_frames,
            )

            # --- Face detection + alignment (default preprocessing) ---
            latentsync_metadata = None
            if use_preprocessing:
                print("Running LatentSync face detection ...")
                latentsync_metadata = preprocess_with_latentsync(
                    video_path, image_processor, args.face_cache_dir,
                    num_frames=num_video_frames,
                )
                if latentsync_metadata is None:
                    print(f"  [FAIL] LatentSync preprocessing failed, skipping {name}")
                    failed.append(name)
                    continue

            # --- Build conditioning ---
            if precomputed_dir is not None:
                condition = build_condition_from_precomputed(
                    precomputed_dir, args.mask_path,
                    num_latent_frames, device, dtype,
                )
            else:
                # Wav2Vec + text already loaded eagerly before the loop

                # Reference frames: aligned faces from LatentSync or raw video
                if use_preprocessing and latentsync_metadata is not None:
                    aligned_faces = latentsync_metadata["aligned_faces"]
                    ref_frames_np = np.stack([
                        f.permute(1, 2, 0).numpy() if isinstance(f, torch.Tensor) else f
                        for f in aligned_faces[:num_video_frames]
                    ], axis=0)
                else:
                    ref_frames_np = load_and_adjust_video(video_path, num_video_frames)

                print("Building conditioning ...")
                condition = build_condition(
                    encoder_vae, wav2vec_model, wav2vec_extractor, ref_frames_np,
                    audio_path, text_embeds, args.mask_path,
                    num_video_frames, num_latent_frames, device, dtype,
                )

            # --- Run inference ---
            print("Running inference ...")
            output_latents = run_inference(
                model, condition, num_latent_frames, args.t_list,
                args.chunk_size, args.context_noise, args.seed, device, dtype,
            )

            # --- Post-processing: decode + save ---
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            if use_preprocessing and latentsync_metadata is not None:
                # LatentSync compositing path — float-space decode + composite
                print("VAE decoding (float) ...")
                latent_for_vae = output_latents[0].to(torch.float32)
                video_decoded = decoder_vae.decode([latent_for_vae], device=device)
                video_decoded = video_decoded.clamp(-1, 1)

                # [1, 3, T_video, H, W] -> [T, 3, H, W] in [0, 1]
                generated_float = video_decoded[0].permute(1, 0, 2, 3)  # [3,T,H,W] -> [T,3,H,W]
                generated_float = ((generated_float + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]

                # Composite onto original frames
                print("Compositing ...")
                composited_np = composite_with_latentsync_float(
                    generated_float.cpu(), latentsync_metadata, image_processor,
                    use_mouth_only_compositing=not args.composite_full_face,
                )

                # Save composited video (original resolution) with audio
                composited_path = output_path
                save_frames_as_video(composited_np, composited_path, fps=args.fps)
                video_duration = composited_np.shape[0] / args.fps
                tmp_composited = composited_path + ".tmp.mp4"
                os.rename(composited_path, tmp_composited)
                mux_video_with_audio(tmp_composited, audio_path, composited_path,
                                     duration_s=video_duration)
                if os.path.exists(tmp_composited):
                    os.remove(tmp_composited)

                print(f"  Saved composited: {composited_path}")

                # Optionally also save the raw generated aligned (512x512) video
                if args.save_aligned:
                    aligned_path = output_path.replace(".mp4", "_aligned.mp4")
                    aligned_np = ((generated_float.permute(0, 2, 3, 1).cpu().float()) * 255
                                  ).clamp(0, 255).to(torch.uint8).numpy()
                    save_frames_as_video(aligned_np, aligned_path, fps=args.fps)
                    tmp_aligned = aligned_path + ".tmp.mp4"
                    os.rename(aligned_path, tmp_aligned)
                    mux_video_with_audio(tmp_aligned, audio_path, aligned_path,
                                         duration_s=video_duration)
                    if os.path.exists(tmp_aligned):
                        os.remove(tmp_aligned)
                    print(f"  Saved aligned:    {aligned_path}")
            else:
                # Standard decode + save (no LatentSync)
                print("Decoding and saving ...")
                decode_and_save(decoder_vae, output_latents, audio_path, output_path,
                                args.fps, device)

            succeeded.append(name)
            print(f"  Done: {output_path}")

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

        finally:
            # Cleanup per-sample temp audio
            if tmp_audio is not None and os.path.exists(tmp_audio):
                os.remove(tmp_audio)

            # Free per-sample GPU memory
            torch.cuda.empty_cache()

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped "
          f"(out of {len(samples)} total)")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()
