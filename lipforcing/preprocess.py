# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared on-the-fly preprocessing / encoding for LipForcing.

Single source of truth for turning raw ``(reference video, audio, prompt)`` into
the exact tensors the OmniAvatar V2V model consumes. The functions here are
imported by BOTH

  * the training dataloader on-the-fly / cache path
    (``lipforcing.datasets.omniavatar_dataloader``), and
  * the inference scripts (via ``scripts/inference/_common.py``)

so the encode logic is defined once and shared by training and inference. The
produced tensors follow the precomputed ``.pt`` training format.

Precomputed ``.pt`` format reproduced here (per sample directory):

    vae_latents_mask_all.pt : {input_latents  [16,21,64,64] bf16,
                               masked_latents [16,21,64,64] bf16}
    audio_emb_omniavatar.pt : {audio_emb [T,10752] f32, metadata}   (T = video frames)
    ref_latents.pt          : {ref_sequence_latents [16,21,64,64] bf16, metadata}
    text_emb.pt             : [1,512,4096] f32   (or {key: tensor})

Encode mechanics:

  * Video frames: cv2 BGR->RGB, ``cv2.resize`` to 512x512, normalize ``/127.5 - 1``,
    short clips padded by repeating the last frame; first ``num_frames`` frames are
    the GT segment (start_frame=0); the reference segment starts at
    ``num_frames`` when ``total_frames >= 2*num_frames`` else ``max(0, total-num)``.
  * Spatial mask: LatentSync PNG, ``convert('L')`` /255, ``cv2.INTER_LINEAR`` resize
    to pixel res, ``> 0.5`` binarize. Convention **1 = keep, 0 = mask**. Masking is
    applied to ALL frames (including frame 0) in pixel space before VAE encode.
  * VAE: ``WanVideoVAE`` (z=16); encoded in **bf16** (bf16 weights + bf16 input);
    81 px frames -> 21 latent frames, 512 -> 64.
  * Audio: OmniAvatar custom ``Wav2VecModel`` called with ``seq_len=`` and
    ``output_hidden_states=True``; the 10752 feature is ``last_hidden_state``
    concatenated with all 13 ``hidden_states`` (14 x 768). The CNN features are
    linearly interpolated to ``seq_len`` frames, so the audio is encoded at the
    full video's frame count and sliced to ``num_video_frames`` downstream.
  * Text: ``WanTextEncoder`` (UMT5-XXL) via ``WanPrompter`` -> [1,512,4096].
"""

import hashlib
import math
import os
import tempfile

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Default media geometry (OmniAvatar V2V 512x512 @ 25fps, Wan VAE 4x temporal).
DEFAULT_NUM_FRAMES = 81
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_FPS = 25
WAV2VEC_SR = 16000


# ===========================================================================
# Encoder model loaders (frozen, eval) — load once, reuse for the whole run.
# ===========================================================================

def load_vae(vae_path, device):
    """Load the Wan 2.1 Video VAE (``WanVideoVAE``, z=16) in eval mode on *device*.

    Mirrors the inference loader: tolerates ``model.``-prefixed, ``model_state``
    (CivitAI) and flat key formats.
    """
    from OmniAvatar.models.wan_video_vae import WanVideoVAE

    vae = WanVideoVAE(z_dim=16)
    state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)

    if any(k.startswith("model.") for k in state_dict):
        vae.load_state_dict(state_dict, strict=True)
    elif "model_state" in state_dict:
        converter = WanVideoVAE.state_dict_converter()
        vae.load_state_dict(converter.from_civitai(state_dict), strict=True)
    else:
        prefixed = {"model." + k: v for k, v in state_dict.items()}
        vae.load_state_dict(prefixed, strict=True)

    vae = vae.to(device=device)
    vae.eval()
    return vae


def load_wav2vec(wav2vec_path, device):
    """Load the OmniAvatar custom ``Wav2VecModel`` + feature extractor.

    Returns ``(model, extractor)`` with the model frozen in eval/float32 on
    *device*. The feature extractor (CNN) must stay float32.
    """
    from transformers import Wav2Vec2FeatureExtractor

    from OmniAvatar.models.wav2vec import Wav2VecModel

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
    model = Wav2VecModel.from_pretrained(wav2vec_path, attn_implementation="eager")
    model.feature_extractor.requires_grad_(False)
    model = model.to(device).float()
    model.eval()
    model.requires_grad_(False)
    return model, extractor


def _resolve_tokenizer_path(text_encoder_path, tokenizer_path=None):
    """Locate the UMT5 tokenizer directory for a Wan text-encoder checkpoint.

    Wan layouts keep the tokenizer either directly beside the ``.pth`` or in a
    ``google/umt5-xxl`` subdirectory. Prefers an explicit *tokenizer_path*, then
    the ``google/umt5-xxl`` subdir, then the checkpoint's own directory.
    """
    if tokenizer_path is not None:
        return tokenizer_path
    base = os.path.dirname(text_encoder_path)
    subdir = os.path.join(base, "google", "umt5-xxl")
    if os.path.isdir(subdir):
        return subdir
    return base


def load_text_encoder(text_encoder_path, device, dtype=torch.bfloat16, tokenizer_path=None):
    """Load the Wan UMT5-XXL text encoder + prompter for prompt encoding.

    Returns ``(text_encoder, prompter)``. The tokenizer is resolved via
    :func:`_resolve_tokenizer_path` (explicit, ``google/umt5-xxl`` subdir, or the
    checkpoint directory).
    """
    from OmniAvatar.models.wan_video_text_encoder import WanTextEncoder
    from OmniAvatar.prompters.wan_prompter import WanPrompter

    text_encoder = WanTextEncoder()
    te_state = torch.load(text_encoder_path, map_location="cpu", weights_only=False)
    converter = WanTextEncoder.state_dict_converter()
    te_state = converter.from_civitai(te_state)
    text_encoder.load_state_dict(te_state, strict=True)
    text_encoder = text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)

    prompter = WanPrompter(
        tokenizer_path=_resolve_tokenizer_path(text_encoder_path, tokenizer_path),
        text_len=512,
    )
    prompter.fetch_models(text_encoder=text_encoder)
    return text_encoder, prompter


def load_encoders(
    vae_path,
    wav2vec_path,
    device,
    dtype=torch.bfloat16,
    text_encoder_path=None,
    load_text=True,
):
    """Load all frozen encoders needed for on-the-fly preprocessing.

    Returns a dict with keys ``vae``, ``wav2vec``, ``wav2vec_extractor`` and
    (when *load_text* and *text_encoder_path* are given) ``text_encoder`` and
    ``prompter``. The text encoder (~11B UMT5-XXL) is optional so callers that
    only need VAE/audio (or that serve every prompt from cache) can skip it.
    """
    encoders = {"device": device, "dtype": dtype}
    encoders["vae"] = load_vae(vae_path, device)
    wav2vec, extractor = load_wav2vec(wav2vec_path, device)
    encoders["wav2vec"] = wav2vec
    encoders["wav2vec_extractor"] = extractor
    if load_text and text_encoder_path is not None:
        text_encoder, prompter = load_text_encoder(text_encoder_path, device, dtype)
        encoders["text_encoder"] = text_encoder
        encoders["prompter"] = prompter
    else:
        encoders["text_encoder"] = None
        encoders["prompter"] = None
    return encoders


# ===========================================================================
# Frame loading + spatial mask
# ===========================================================================

def read_video_frames_pixel(video_path, num_frames, height=DEFAULT_HEIGHT,
                            width=DEFAULT_WIDTH, start_frame=0):
    """Read *num_frames* frames -> ``[T, H, W, 3]`` float32 in ``[-1, 1]``.

    Reads frames with cv2, BGR->RGB, ``cv2.resize`` to (W, H), normalizes
    ``/127.5 - 1``, pads to *num_frames* by repeating the last frame. Also
    returns the video's total
    frame count (``CAP_PROP_FRAME_COUNT``) which drives the reference-segment
    offset and the audio sequence length.

    Returns:
        (frames [T, H, W, 3] float32 in [-1, 1], total_frames int)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frame = frame.astype(np.float32) / 127.5 - 1.0
            frames.append(frame)
        else:
            break
    cap.release()

    if len(frames) == 0:
        frames.append(np.zeros((height, width, 3), dtype=np.float32))
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames), total_frames


def frames_pixel_to_tensor(frames_thwc):
    """``[T, H, W, 3]`` (numpy or tensor) -> ``[3, T, H, W]`` float32 tensor.

    Matches the precompute layout: ``torch.from_numpy(frames).permute(3,0,1,2)``.
    """
    if isinstance(frames_thwc, np.ndarray):
        t = torch.from_numpy(frames_thwc)
    else:
        t = frames_thwc
    return t.permute(3, 0, 1, 2).contiguous().float()


def binarize_pixel_mask(mask_path, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
    """Load LatentSync mask -> binary pixel mask ``[H, W]`` float32, 1=keep 0=mask.

    Replicates the precompute worker: ``Image.open(...).convert('L')`` /255,
    ``cv2.INTER_LINEAR`` resize to (W, H) when needed, ``> 0.5`` binarize.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != height or mask_np.shape[1] != width:
        mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy((mask_np > 0.5).astype(np.float32))  # [H, W]


def load_latent_mask(mask_path, latent_h=64, latent_w=64):
    """Load LatentSync mask and resize to latent resolution -> ``[h, w]`` float.

    Convention 1=keep, 0=mask. Matches the dataloader's ``self.mask`` (bilinear
    interpolate to latent res, ``> 0.5``) and inference's ``load_latentsync_mask``.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_resized = F.interpolate(
        mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False
    )
    return (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [h, w]


# ---------------------------------------------------------------------------
# Inference-side frame helpers. These use the ``/255*2-1`` normalization and
# ping-pong length extension of the inference path, mathematically equivalent in
# pixel space to the ``/127.5-1`` normalization above (``x/255*2-1 == x/127.5-1``).
# ---------------------------------------------------------------------------

def pingpong_indices(n, target):
    """Frame indices extending a length-*n* sequence to *target* via ping-pong."""
    if n >= target:
        return list(range(target))
    if n == 1:
        return [0] * target
    cycle = list(range(n)) + list(range(n - 2, 0, -1))
    indices = []
    while len(indices) < target:
        indices.extend(cycle)
    return indices[:target]


def adjust_video_length(frames_np, target):
    """Adjust ``[N,H,W,3]`` video to exactly *target* frames (ping-pong / clip)."""
    n = len(frames_np)
    if n >= target:
        return frames_np[:target]
    return frames_np[pingpong_indices(n, target)]


def frames_to_tensor(frames_np):
    """``[N,H,W,3]`` uint8 -> ``[1,3,N,H,W]`` float in ``[-1,1]`` (inference path)."""
    t = torch.from_numpy(frames_np).float() / 255.0
    t = t.permute(0, 3, 1, 2)
    t = t * 2.0 - 1.0
    t = t.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return t


def apply_spatial_mask(video_tensor, mask_np, mask_all_frames=True):
    """Apply a ``[H,W]`` 1=keep/0=mask binary mask to ``[1,3,N,H,W]`` video."""
    mask_t = torch.from_numpy(mask_np).float()[None, None, None, :, :]
    masked = video_tensor.clone()
    if mask_all_frames:
        masked *= mask_t
    else:
        masked[:, :, 1:, :, :] *= mask_t
    return masked


# ===========================================================================
# VAE encode
# ===========================================================================

def vae_encode_pixels(vae, pixel_tensor, device, dtype=torch.bfloat16):
    """VAE-encode a pixel-space video tensor in bf16 (weights + input).

    Args:
        vae: ``WanVideoVAE`` instance.
        pixel_tensor: ``[3, T, H, W]`` (single) or ``[B, 3, T, H, W]`` (batch),
            float in ``[-1, 1]``.
        device, dtype: compute device and output dtype.

    Returns:
        ``[16, T_lat, H_lat, W_lat]`` (single) or ``[B, 16, ...]`` (batch).

    The VAE is temporarily cast to bf16 for the encode and restored afterwards,
    matching the precompute pipeline (bf16 VAE + bf16 input). ``WanVideoVAE.encode``
    iterates its argument and unsqueezes each element, so a single ``[3,T,H,W]``
    is passed as a one-element list and a ``[B,3,T,H,W]`` batch is passed directly.
    """
    single = pixel_tensor.dim() == 4
    original_dtype = next(vae.parameters()).dtype
    vae.to(dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            if single:
                latents = vae.encode([pixel_tensor.to(dtype=torch.bfloat16)], device=device)
            else:
                latents = vae.encode(pixel_tensor.to(dtype=torch.bfloat16), device=device, tiled=False)
    finally:
        vae.to(dtype=original_dtype)
    latents = latents.to(dtype=dtype)
    if single:
        latents = latents[0]  # drop batch dim -> [16, T_lat, H_lat, W_lat]
    return latents


def encode_vae_masked(vae, gt_pixel, pixel_mask, device, dtype=torch.bfloat16):
    """Encode GT + spatially-masked video -> ``input_latents`` and ``masked_latents``.

    The mouth region is masked on **every** frame (including frame 0) in a single
    encode pass.

    Args:
        vae: ``WanVideoVAE``.
        gt_pixel: ``[3, T, H, W]`` float in ``[-1, 1]`` (unmasked GT frames).
        pixel_mask: ``[H, W]`` binary, 1=keep, 0=mask.

    Returns:
        (input_latents [16, T_lat, H, W], masked_latents [16, T_lat, H, W]) in *dtype*.
    """
    mask_2d = pixel_mask.view(1, 1, pixel_mask.shape[-2], pixel_mask.shape[-1]).to(gt_pixel.dtype)
    input_latents = vae_encode_pixels(vae, gt_pixel, device, dtype)
    masked_latents = vae_encode_pixels(vae, gt_pixel * mask_2d, device, dtype)
    return input_latents, masked_latents


def ref_segment_start(total_frames, num_frames=DEFAULT_NUM_FRAMES):
    """Reference-segment start frame (matches precompute):

    ``num_frames`` if ``total_frames >= 2*num_frames`` else ``max(0, total-num)``.
    """
    if total_frames >= 2 * num_frames:
        return num_frames
    return max(0, total_frames - num_frames)


def encode_ref(vae, video_path, num_frames=DEFAULT_NUM_FRAMES,
               height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, device="cuda",
               dtype=torch.bfloat16, total_frames=None):
    """Encode the reference segment -> ``ref_sequence_latents [16, T_lat, H, W]``.

    Reads *num_frames* unmasked frames starting at :func:`ref_segment_start` and
    VAE-encodes them. Byte-matches ``ref_latents.pt:ref_sequence_latents``.
    """
    if total_frames is None:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    ref_start = ref_segment_start(total_frames, num_frames)
    ref_frames, _ = read_video_frames_pixel(
        video_path, num_frames, height, width, start_frame=ref_start
    )
    ref_pixel = frames_pixel_to_tensor(ref_frames)
    ref_latents = vae_encode_pixels(vae, ref_pixel, device, dtype)
    return ref_latents, ref_start, total_frames


# ===========================================================================
# Audio encode (OmniAvatar custom Wav2Vec, 10752-dim concat)
# ===========================================================================

def audio_natural_frames(audio, sample_rate=WAV2VEC_SR, fps=DEFAULT_FPS):
    """Natural frame count of an audio source: ``ceil(num_samples / (sr // fps))``.

    *audio* is a path or a 1-D numpy waveform (already at *sample_rate*).
    """
    if isinstance(audio, str):
        wav, _ = librosa.load(audio, sr=sample_rate)
        n = len(wav)
    else:
        n = len(audio)
    samples_per_frame = sample_rate // fps
    return int(math.ceil(n / samples_per_frame))


def encode_audio_omniavatar(wav2vec_model, wav2vec_extractor, audio, seq_len,
                            device, sample_rate=WAV2VEC_SR, fps=DEFAULT_FPS):
    """Encode audio -> ``[1, seq_len, 10752]`` via the OmniAvatar custom Wav2Vec.

    This is the shared core used by both training and inference. *audio* is a
    path or a 1-D numpy waveform at *sample_rate*. The waveform is normalized by
    the feature extractor, zero-padded up to ``seq_len * (sr // fps)`` samples,
    and the CNN features are linearly interpolated to *seq_len* frames; the
    10752 feature is ``last_hidden_state`` concatenated with all 13
    ``hidden_states`` (14 x 768).

    Note: the wav2vec model runs in float32, so the output is float32 (the
    precompute ``audio_emb`` dtype). Callers cast to bf16 downstream.
    """
    if isinstance(audio, str):
        audio, _ = librosa.load(audio, sr=sample_rate)

    input_values = np.squeeze(
        wav2vec_extractor(audio, sampling_rate=sample_rate).input_values
    )
    input_values = torch.from_numpy(input_values).float().to(device=device)
    input_values = input_values.unsqueeze(0)

    samples_per_frame = sample_rate // fps  # 640 at 16kHz/25fps
    target_samples = seq_len * samples_per_frame
    if input_values.shape[1] < target_samples:
        input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

    with torch.no_grad():
        hidden_states = wav2vec_model(
            input_values, seq_len=seq_len, output_hidden_states=True
        )
    audio_emb = hidden_states.last_hidden_state
    for hs in hidden_states.hidden_states:
        audio_emb = torch.cat((audio_emb, hs), -1)
    return audio_emb  # [1, seq_len, 10752]


# ===========================================================================
# Text encode
# ===========================================================================

def encode_text(prompter, prompt, device, dtype=torch.bfloat16):
    """Encode a prompt string -> ``[1, 512, 4096]`` text embedding in *dtype*.

    Uses ``WanPrompter.encode_prompt`` (UMT5-XXL). Matches the inference and
    precompute text-embedding format.
    """
    with torch.no_grad():
        text_embeds = prompter.encode_prompt(prompt, positive=True, device=device)
    if text_embeds.dim() == 2:
        text_embeds = text_embeds.unsqueeze(0)
    return text_embeds.to(dtype=dtype)


def compute_prompt_hash(prompt):
    """Content address for a prompt string: ``sha1(prompt)`` hex digest."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


# ===========================================================================
# Raw sample preparation (CPU, dataloader-worker safe) + GPU encode
# ===========================================================================

def prepare_raw_sample(video_path, audio_path, prompt, pixel_mask,
                       num_video_frames=DEFAULT_NUM_FRAMES,
                       height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                       sample_rate=WAV2VEC_SR, fps=DEFAULT_FPS,
                       load_audio_waveform=True):
    """Decode raw inputs on CPU into the payload the GPU encode step consumes.

    Safe to run inside dataloader workers (no models, no GPU). Reads the GT and
    reference frame segments, loads the audio waveform, and carries the prompt
    plus its content hash and the sequence lengths needed to reproduce the
    precompute tensors.

    Args:
        pixel_mask: ``[H, W]`` binary mask, 1=keep 0=mask (shared, precomputed by
            :func:`binarize_pixel_mask`). Passed through unchanged.

    Returns a dict with CPU tensors / scalars:
        gt_pixel [3, T, H, W], ref_pixel [3, T, H, W], enc_waveform [L] (or None),
        audio_path, prompt, prompt_hash, total_frames, ref_start, seq_len,
        num_video_frames, pixel_mask.

    ``enc_waveform`` is the raw 16 kHz waveform for wav2vec encoding; it is named
    distinctly so it never collides with the reward path's ``audio_waveform``.
    """
    gt_frames, total_frames = read_video_frames_pixel(
        video_path, num_video_frames, height, width, start_frame=0
    )
    gt_pixel = frames_pixel_to_tensor(gt_frames)

    ref_start = ref_segment_start(total_frames, num_video_frames)
    ref_frames, _ = read_video_frames_pixel(
        video_path, num_video_frames, height, width, start_frame=ref_start
    )
    ref_pixel = frames_pixel_to_tensor(ref_frames)

    waveform = None
    if load_audio_waveform and audio_path is not None and os.path.exists(audio_path):
        wav, _ = librosa.load(audio_path, sr=sample_rate)
        waveform = torch.from_numpy(wav)

    # Audio is encoded at the full video frame count (precompute convention: the
    # wav2vec features are interpolated to the video frame grid), guarded to be at
    # least num_video_frames so the downstream [:num_video_frames] slice is valid.
    seq_len = max(total_frames, num_video_frames)

    return {
        "gt_pixel": gt_pixel,
        "ref_pixel": ref_pixel,
        "enc_waveform": waveform,
        "audio_path": audio_path if audio_path is not None else "",
        "prompt": prompt,
        "prompt_hash": compute_prompt_hash(prompt),
        "total_frames": int(total_frames),
        "ref_start": int(ref_start),
        "seq_len": int(seq_len),
        "num_video_frames": int(num_video_frames),
        "pixel_mask": pixel_mask,
    }


def _load_text_cache_disk(text_cache_dir, phash):
    path = os.path.join(text_cache_dir, f"{phash}.pt") if text_cache_dir else None
    if path and os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
    return None


def _save_text_cache_disk(text_cache_dir, phash, text_emb):
    if not text_cache_dir:
        return
    try:
        os.makedirs(text_cache_dir, exist_ok=True)
        path = os.path.join(text_cache_dir, f"{phash}.pt")
        # Unique temp name so concurrent ranks sharing the cache dir don't
        # clobber each other's .tmp before the atomic rename.
        fd, tmp = tempfile.mkstemp(dir=text_cache_dir, suffix=".tmp")
        os.close(fd)
        torch.save(text_emb.cpu(), tmp)
        os.replace(tmp, path)
    except Exception:
        pass  # best-effort cache


def encode_prepared(encoders, raw, device=None, dtype=torch.bfloat16,
                    text_emb_cache=None, text_cache_dir=None):
    """GPU-encode a prepared raw sample into the precompute ``.pt`` tensors.

    Runs in the main training process (encoders live here, never in workers).

    Args:
        encoders: dict from :func:`load_encoders` (vae, wav2vec[+extractor],
            optionally prompter).
        raw: dict from :func:`prepare_raw_sample`.
        text_emb_cache: optional ``{prompt_hash: tensor}`` in-memory cache so each
            unique prompt is encoded by UMT5 only once per run.
        text_cache_dir: optional directory for a disk, content-addressed text
            cache (``{sha1(prompt)}.pt``) reused across epochs and runs.

    Resolution order for ``text_emb``: a value preloaded on the raw sample
    (``raw['text_emb']``), then the in-memory cache, then the disk cache, then a
    fresh UMT5 encode (when a prompter is loaded), else None.

    Returns a dict mirroring the precomputed files (full, unsliced):
        input_latents, masked_latents, ref_sequence_latents [16, T_lat, H, W] bf16,
        audio_emb [seq_len, 10752] f32, text_emb [1, 512, 4096] (in *dtype*),
        plus prompt_hash / total_frames / ref_start / seq_len / num_video_frames.
    """
    device = device or encoders.get("device")
    vae = encoders["vae"]
    wav2vec = encoders["wav2vec"]
    extractor = encoders["wav2vec_extractor"]

    input_latents, masked_latents = encode_vae_masked(
        vae, raw["gt_pixel"].to(device), raw["pixel_mask"].to(device), device, dtype,
    )
    ref_latents = vae_encode_pixels(vae, raw["ref_pixel"].to(device), device, dtype)

    audio_src = (raw["enc_waveform"].numpy() if raw.get("enc_waveform") is not None
                 else raw["audio_path"])
    audio_emb = encode_audio_omniavatar(
        wav2vec, extractor, audio_src, raw["seq_len"], device
    ).squeeze(0).to(torch.float32)

    # Text: content-addressed by prompt hash so UMT5 runs once per unique prompt.
    phash = raw["prompt_hash"]
    text_emb = raw.get("text_emb")  # preloaded from sample dir, if any
    if text_emb is None and text_emb_cache is not None and phash in text_emb_cache:
        text_emb = text_emb_cache[phash]
    if text_emb is None:
        text_emb = _load_text_cache_disk(text_cache_dir, phash)
    if text_emb is None and encoders.get("prompter") is not None:
        text_emb = encode_text(encoders["prompter"], raw["prompt"], device, dtype)
        _save_text_cache_disk(text_cache_dir, phash, text_emb)
    if text_emb is not None:
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)
        text_emb = text_emb.to(dtype)
        if text_emb_cache is not None:
            text_emb_cache[phash] = text_emb

    return {
        "input_latents": input_latents.cpu(),
        "masked_latents": masked_latents.cpu(),
        "ref_sequence_latents": ref_latents.cpu(),
        "audio_emb": audio_emb.cpu(),
        "text_emb": text_emb.cpu() if text_emb is not None else None,
        "prompt_hash": phash,
        "total_frames": raw["total_frames"],
        "ref_start": raw["ref_start"],
        "seq_len": raw["seq_len"],
        "num_video_frames": raw["num_video_frames"],
    }


def encode_sample_from_files(encoders, video_path, audio_path, prompt, mask_path,
                             num_video_frames=DEFAULT_NUM_FRAMES,
                             height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                             device=None, dtype=torch.bfloat16):
    """Convenience: decode + encode a sample straight from files.

    Composes :func:`prepare_raw_sample` and :func:`encode_prepared`. Used for any
    standalone encode of a raw ``(video, audio, prompt)`` triple.
    """
    device = device or encoders.get("device")
    pixel_mask = binarize_pixel_mask(mask_path, height, width)
    raw = prepare_raw_sample(
        video_path, audio_path, prompt, pixel_mask,
        num_video_frames=num_video_frames, height=height, width=width,
    )
    return encode_prepared(encoders, raw, device, dtype)
