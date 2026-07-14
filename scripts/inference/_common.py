"""Shared helpers for the inference scripts.

Everything that ``inference_segmentwise.py`` and ``inference_streaming.py``
have in common lives here: model/encoder loading, conditioning construction,
LatentSync face preprocessing + compositing, TAEHV decoder wrappers, audio
handling, and video I/O.
"""

import math
import os
import subprocess
import sys
import tempfile

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPFORCING_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, LIPFORCING_ROOT)

# Shared preprocessing/encode helpers (single source of truth, shared with training).
from lipforcing import preprocess as pp  # noqa: E402


def _get_ffmpeg():
    """Return path to ffmpeg binary (system or imageio_ffmpeg fallback)."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or pip install imageio-ffmpeg.")


# ===========================================================================
# TAEHV decoder wrappers
# ===========================================================================

class TAEHVDecoderWrapper:
    """Drop-in decode-only replacement that mimics WanVideoVAE.decode().

    TAEHV convention:
      - Input:  diffusion-space latents (same space the denoiser works in).
                No mean/std scaling is applied here — TAEHV was distilled to
                consume these directly.
      - Output: pixels in [0, 1], NTCHW layout.
    WanVideoVAE convention:
      - decode() returns pixels in [-1, 1], shape [1, 3, T_video, H, W] (NCTHW).
    This wrapper converts TAEHV output to Wan's range/layout so downstream
    code (decode_and_save, LatentSync path) works unchanged.
    """
    def __init__(self, checkpoint_path, device):
        from lipforcing.methods.reward.taehv import TAEHV
        self.device = device
        # trim_output=False: this wrapper trims frames itself to match Wan's convention.
        self.taehv = TAEHV(checkpoint_path=checkpoint_path, trim_output=False).to(device, torch.float16).eval()

    @torch.no_grad()
    def decode(self, latents_list, device=None):
        # latents_list: list of one [C=16, T_lat, H, W] tensor (matches WanVideoVAE.decode signature)
        target_device = device if device is not None else self.device
        lat = latents_list[0].to(target_device, dtype=torch.float16)      # [16, T, H, W]
        lat = lat.permute(1, 0, 2, 3).unsqueeze(0)                        # [1, T, 16, H, W]  NTCHW
        vid = self.taehv.decode_video(lat, parallel=False)                # [1, T*4, 3, H', W']  in [0, 1]
        # Front-trim is disabled here, so match Wan's
        # temporal length convention: num_video = 1 + (num_latent - 1) * 4 = T_lat*4 - frames_to_trim.
        vid = vid[:, self.taehv.frames_to_trim:]                          # [1, T_lat*4 - 3, 3, H', W']
        vid = vid.mul(2).sub(1)                                           # -> [-1, 1] (match Wan)
        return vid.permute(0, 2, 1, 3, 4).float()                         # [1, 3, T_video, H', W']  NCTHW

    @torch.no_grad()
    def encode(self, videos_list, device=None):
        """Drop-in replacement for WanVideoVAE.encode().

        Wan convention: input list of [3, T, H, W] in [-1, 1]; returns [N, 16, T_lat, H_lat, W_lat]
        with T_lat = 1 + (T-1)//4 = ⌈T/4⌉.
        TAEHV: wants NTCHW in [0, 1], its temporal compression is floor(T/4). We pad the
        INPUT video to the next multiple of 4 so floor(T_pad/4) = ⌈T/4⌉, matching Wan's T_lat
        naturally — no latent-side duplication needed.
        """
        target_device = device if device is not None else self.device
        outs = []
        for vid in videos_list:
            T = vid.shape[1]
            T_pad = ((T + 3) // 4) * 4                                    # round up to multiple of 4
            if T_pad > T:
                # PREPEND copies of the first frame so TAEHV's latent 0 pools [f0,f0,f0,f0]
                # = encoding of the static starting frame. This matches Wan's convention where
                # latent 0 encodes frame 0 alone; latents i>0 encode groups of 4 consecutive frames.
                pad = vid[:, :1].expand(-1, T_pad - T, -1, -1).contiguous()
                vid = torch.cat([pad, vid], dim=1)
            x = vid.to(target_device, dtype=torch.float16)
            x = x.add(1).div(2)                                           # [-1,1] -> [0,1]
            x = x.permute(1, 0, 2, 3).unsqueeze(0)                        # [1, T_pad, 3, H, W]  NTCHW
            lat = self.taehv.encode_video(x, parallel=False, show_progress_bar=False)  # [1, T_pad/4, 16, H', W']
            lat = lat.permute(0, 2, 1, 3, 4).float()                      # [1, 16, T_pad/4, H', W']
            outs.append(lat.squeeze(0))                                   # [16, T_pad/4, H', W']
        return torch.stack(outs)                                          # [N, 16, T_pad/4, H', W']


class StreamingTAEHVDecoderWrapper(TAEHVDecoderWrapper):
    """Drop-in decoder using StreamingTAEHV — feeds latents one at a time and
    collects pixel frames as they emerge.

    Same signature as TAEHVDecoderWrapper.decode() so downstream code works
    unchanged; encode() is inherited (used when --taehv_encode is combined
    with --taehv_streaming).
    """
    def __init__(self, checkpoint_path, device):
        from lipforcing.methods.reward.taehv import StreamingTAEHV
        super().__init__(checkpoint_path, device)  # sets self.taehv
        self.streaming = StreamingTAEHV(self.taehv)

    @torch.no_grad()
    def decode(self, latents_list, device=None):
        target_device = device if device is not None else self.device
        self.streaming.reset()
        lat = latents_list[0].to(target_device, dtype=torch.float16)      # [16, T_lat, H, W]
        lat = lat.permute(1, 0, 2, 3).unsqueeze(0)                        # [1, T_lat, 16, H, W]  NTCHW

        frames = []
        for t in range(lat.shape[1]):
            latent_t = lat[:, t:t+1]                                       # [1, 1, 16, H, W]
            frame = self.streaming.decode(latent_t)
            while frame is not None:
                frames.append(frame)
                frame = self.streaming.decode()

        for frame in self.streaming.flush_decoder():
            frames.append(frame)

        # Stack [N1CHW, ...] → [1, T, C, H, W] NTCHW, convert to NCTHW [-1, 1]
        vid = torch.cat(frames, dim=1)                                     # [1, T, 3, H', W']
        vid = vid.mul(2).sub(1)                                            # [0,1] → [-1,1]
        return vid.permute(0, 2, 1, 3, 4).float()                         # [1, 3, T, H', W']


# ===========================================================================
# Model loading functions
# ===========================================================================

def load_vae(vae_path, device):
    """Load the Wan 2.1 Video VAE.

    Returns:
        WanVideoVAE instance in eval mode on *device*.
    """
    from OmniAvatar.models.wan_video_vae import WanVideoVAE

    vae = WanVideoVAE(z_dim=16)

    print(f"Loading VAE from {vae_path} ...")
    state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)

    # Handle both 'model.xxx' prefixed and flat key formats
    if any(k.startswith("model.") for k in state_dict):
        # Already has model. prefix — load directly into WanVideoVAE
        vae.load_state_dict(state_dict, strict=True)
    elif "model_state" in state_dict:
        # CivitAI format: state_dict['model_state'] with flat keys
        converter = WanVideoVAE.state_dict_converter()
        converted = converter.from_civitai(state_dict)
        vae.load_state_dict(converted, strict=True)
    else:
        # Flat keys — add 'model.' prefix
        prefixed = {"model." + k: v for k, v in state_dict.items()}
        vae.load_state_dict(prefixed, strict=True)

    vae = vae.to(device=device)
    vae.eval()
    return vae


def load_wav2vec(wav2vec_path, device):
    """Load wav2vec2-base-960h model and feature extractor.

    Returns:
        (wav2vec_model, wav2vec_extractor) — model in eval/float32 on *device*.
    """
    from transformers import Wav2Vec2FeatureExtractor
    from OmniAvatar.models.wav2vec import Wav2VecModel

    print(f"Loading Wav2Vec2 from {wav2vec_path} ...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
    model = Wav2VecModel.from_pretrained(wav2vec_path, attn_implementation="eager")

    # Freeze feature extractor (CNN) — must stay float32
    model.feature_extractor.requires_grad_(False)
    model = model.to(device).float()
    model.eval()
    return model, extractor


def load_or_encode_text(args, device, dtype):
    """Get text embeddings — either from file or by encoding the prompt.

    Returns:
        text_embeds: [1, 512, 4096] tensor on *device* in *dtype*.
    """
    if args.text_embeds_path is not None:
        print(f"Loading text embeddings from {args.text_embeds_path} ...")
        data = torch.load(args.text_embeds_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            # Handle dict formats: {'context': tensor} or {'text_emb': tensor}
            for key in ("context", "text_emb", "prompt_emb"):
                if key in data:
                    text_embeds = data[key]
                    break
            else:
                # Take first tensor value
                text_embeds = next(iter(data.values()))
        else:
            text_embeds = data

        # Ensure shape [1, 512, 4096]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)
        assert text_embeds.shape == (1, 512, 4096), (
            f"Expected text_embeds shape [1, 512, 4096], got {text_embeds.shape}"
        )
        return text_embeds.to(device=device, dtype=dtype)

    elif args.text_encoder_path is not None:
        print(f"Loading T5 text encoder from {args.text_encoder_path} ...")
        from OmniAvatar.models.wan_video_text_encoder import WanTextEncoder
        from OmniAvatar.prompters.wan_prompter import WanPrompter

        # Load text encoder
        text_encoder = WanTextEncoder()
        te_state = torch.load(args.text_encoder_path, map_location="cpu", weights_only=False)
        converter = WanTextEncoder.state_dict_converter()
        te_state = converter.from_civitai(te_state)
        text_encoder.load_state_dict(te_state, strict=True)
        text_encoder = text_encoder.to(device).eval()

        # Set up prompter; resolve the tokenizer (handles the google/umt5-xxl subdir layout)
        tokenizer_path = pp._resolve_tokenizer_path(args.text_encoder_path)
        prompter = WanPrompter(tokenizer_path=tokenizer_path, text_len=512)
        prompter.fetch_models(text_encoder=text_encoder)

        # Encode
        with torch.no_grad():
            text_embeds = prompter.encode_prompt(
                args.prompt, positive=True, device=device
            )

        # Ensure shape [1, 512, 4096]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)

        # Cleanup to free VRAM
        del text_encoder, prompter
        torch.cuda.empty_cache()

        return text_embeds.to(dtype=dtype)

    else:
        raise ValueError(
            "Must provide either --text_embeds_path or --text_encoder_path "
            "to obtain text embeddings."
        )


# ===========================================================================
# Input preprocessing functions
# ===========================================================================

def resolve_audio(audio_path=None, video_path=None, args=None):
    """Determine the audio source path.

    Accepts explicit *audio_path* / *video_path* for batch mode, or falls
    back to reading from *args* for single-sample backward-compatibility.

    Returns:
        (audio_path, tmp_path_or_None) — tmp_path is set when a temp file
        was created and must be cleaned up later.
    """
    if audio_path is None and args is not None:
        audio_path = getattr(args, "audio_path", None)
    if video_path is None and args is not None:
        video_path = getattr(args, "video_path", None)

    if audio_path is not None:
        return audio_path, None

    if video_path is None:
        raise ValueError("resolve_audio: need either audio_path or video_path")

    # Extract audio from video
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed:\n{result.stderr}"
        )
    print(f"Extracted audio to {tmp_path}")
    return tmp_path, tmp_path


def get_audio_duration(audio_path):
    """Get audio duration in seconds.

    Returns:
        float — duration in seconds.
    """
    # Use librosa instead of ffprobe (ffprobe may not be installed)
    duration = librosa.get_duration(filename=audio_path)
    return duration


def compute_generation_length(audio_path, override_frames, chunk_size, fps,
                              min_latent_frames=0):
    """Compute generation length in both latent and video frames.

    The VAE temporal compression is: num_latent = 1 + (num_video - 1) // 4.
    We round DOWN num_latent to the nearest multiple of chunk_size so the AR
    loop produces complete chunks.

    If ``min_latent_frames`` > 0 and the audio-derived num_latent is shorter,
    we pad up to ``min_latent_frames``: audio zero-pads via
    wav2vec; video frames are ping-pong extended in adjust_video_length.

    Args:
        audio_path: path to audio file (for duration)
        override_frames: explicit num_latent_frames (or None)
        chunk_size: AR chunk size in latent frames
        fps: video frames per second
        min_latent_frames: floor on num_latent; 0 disables padding.

    Returns:
        (num_latent_frames, num_video_frames)
    """
    duration = get_audio_duration(audio_path)
    num_video_raw = int(duration * fps)  # floor
    num_latent_raw = 1 + (num_video_raw - 1) // 4

    if override_frames is not None:
        num_latent = override_frames
        if num_latent % chunk_size != 0:
            raise ValueError(
                f"--num_latent_frames ({num_latent}) must be a multiple of "
                f"chunk_size ({chunk_size})"
            )
    else:
        # Round DOWN to multiple of chunk_size
        num_latent = (num_latent_raw // chunk_size) * chunk_size
        num_latent = max(num_latent, chunk_size)  # at least one chunk

    if min_latent_frames and num_latent < min_latent_frames:
        print(f"  Audio too short ({duration:.2f}s → {num_latent} latent frames), "
              f"padding to {min_latent_frames}")
        num_latent = min_latent_frames

    # Inverse: num_video = 1 + (num_latent - 1) * 4
    num_video = 1 + (num_latent - 1) * 4

    print(f"Generation length: {num_latent} latent frames, {num_video} video frames")
    return num_latent, num_video


def load_video_frames(video_path, max_frames=None):
    """Load video frames as [N, H, W, 3] uint8 numpy array.

    Validates that frames are 512x512.

    Args:
        video_path: path to video file
        max_frames: if set, read at most this many frames

    Returns:
        frames: [N, H, W, 3] uint8 numpy array
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    checked_size = False
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not checked_size:
            h, w = frame.shape[:2]
            if h != 512 or w != 512:
                cap.release()
                raise ValueError(
                    f"Video must be 512x512, got {w}x{h}. "
                    "Resize the input, or drop --skip_preprocessing so the "
                    "face-alignment pipeline handles arbitrary resolutions."
                )
            checked_size = True
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    return np.stack(frames, axis=0)  # [N, H, W, 3] uint8


def pingpong_indices(n, target):
    """Frame indices that extend a length-*n* sequence to *target* via ping-pong.

    Plays forward then backward then forward: 0,1,...,n-1,n-2,...,1,0,1,...
    For n == 1 this is all zeros; for n >= target it clips to range(target).

    This is the "rewind" extension used when the audio-driven generation length
    exceeds the reference video: the mouth syncs over the looped/rewound video
    instead of freezing on the last frame. The cycle is independent of *target*,
    so a longer index list is always a prefix-superset of a shorter one.
    """
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
    """Adjust video to exactly *target* frames via ping-pong extension or clipping.

    Args:
        frames_np: [N, H, W, 3] uint8
        target: desired number of frames

    Returns:
        [target, H, W, 3] uint8
    """
    n = len(frames_np)
    if n >= target:
        return frames_np[:target]
    return frames_np[pingpong_indices(n, target)]


def load_and_adjust_video(video_path, num_video_frames):
    """Load video and adjust to exactly *num_video_frames* frames.

    Returns:
        [num_video_frames, H, W, 3] uint8 numpy array.
    """
    frames = load_video_frames(video_path)
    return adjust_video_length(frames, num_video_frames)


def frames_to_tensor(frames_np):
    """Convert [N, H, W, 3] uint8 numpy → [1, 3, N, H, W] float tensor in [-1, 1].

    Normalizes to [-1, 1] and reorders to [1, 3, N, H, W].
    """
    t = torch.from_numpy(frames_np).float() / 255.0  # [N, H, W, 3] in [0, 1]
    t = t.permute(0, 3, 1, 2)  # [N, 3, H, W]
    t = t * 2.0 - 1.0  # [-1, 1]
    t = t.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, N, H, W]
    return t


def load_latentsync_mask(mask_path, latent_h, latent_w):
    """Load LatentSync mask and resize to latent resolution.

    Returns:
        [H_lat, W_lat] float tensor.  1=keep, 0=mask (LatentSync convention).
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0  # 1=keep, 0=mask
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_resized = F.interpolate(
        mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False
    )
    return (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [H_lat, W_lat]


def apply_spatial_mask(video_tensor, mask_np, mask_all_frames=True):
    """Apply LatentSync spatial mask to a normalized video tensor.

    Matches training convention: normalize first (already done), then mask.
    Masked region becomes 0.0 in [-1,1] space (mid-gray).

    Args:
        video_tensor: [1, 3, N, H, W] float in [-1, 1]
        mask_np: [H, W] float32, 1=keep, 0=mask (LatentSync convention)
        mask_all_frames: if True, mask ALL frames including frame 0

    Returns:
        masked_tensor: [1, 3, N, H, W] float in [-1, 1]
    """
    mask_t = torch.from_numpy(mask_np).float()  # [H, W]
    mask_t = mask_t[None, None, None, :, :]  # [1, 1, 1, H, W]
    masked = video_tensor.clone()
    if mask_all_frames:
        masked *= mask_t
    else:
        masked[:, :, 1:, :, :] *= mask_t
    return masked


def encode_reference_video(vae, video_frames_np, mask_path, device, dtype):
    """Encode reference video through VAE (both unmasked and masked).

    Args:
        vae: WanVideoVAE instance
        video_frames_np: [N, H, W, 3] uint8
        mask_path: path to LatentSync mask
        device: torch device
        dtype: torch dtype

    Returns:
        (ref_latent, masked_latents, ref_sequence, latent_mask) where:
        - ref_latent: [1, 16, 1, H_lat, W_lat] — first frame latent
        - masked_latents: [1, 16, T_lat, H_lat, W_lat] — spatially masked
        - ref_sequence: [1, 16, T_lat, H_lat, W_lat] — unmasked full video
        - latent_mask: [H_lat, W_lat] float (LatentSync convention)
    """
    H, W = 512, 512

    # Convert to tensor
    video_tensor = frames_to_tensor(video_frames_np)  # [1, 3, N, H, W]

    # Load pixel-level mask — use cv2 bilinear (INTER_LINEAR) resize to match
    # the latent-mask preprocessing.
    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != H or mask_np.shape[1] != W:
        mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_pixel_binary = (mask_np > 0.5).astype(np.float32)

    # Apply spatial mask (all frames)
    masked_video_tensor = apply_spatial_mask(video_tensor, mask_pixel_binary, mask_all_frames=True)

    # VAE encode. Wan VAE runs in bf16 (cast temporarily); TAEHV runs in fp16 natively.
    is_taehv = isinstance(vae, TAEHVDecoderWrapper)

    N = video_tensor.shape[2]

    if is_taehv:
        # TAEHV batch-encodes all frames at once and OOMs on long clips, so long
        # videos are encoded in GRID-ALIGNED chunks: the first chunk is 81 frames
        # (1 + 20x4 latents, matching Wan's ``1 + 4x`` temporal grid) and every
        # later chunk is 80 frames (20 four-frame groups). Chunking at these
        # boundaries keeps the latent<->frame alignment exact. Naive fixed-size
        # chunking must NOT be used here: it emits one extra "first-frame" latent
        # per chunk, progressively time-shifting the conditioning against the
        # audio/rollout timeline (~3 frames per 81-frame chunk).
        FIRST, REST = 81, 80

        def _encode(vt):
            if N <= FIRST:
                return vae.encode([vt[0]], device=device)
            out = [vae.encode([vt[:, :, :FIRST][0]], device=device)]
            for s in range(FIRST, N, REST):
                e = min(s + REST, N)
                out.append(vae.encode([vt[:, :, s:e][0]], device=device))
            return torch.cat(out, dim=2)

        with torch.no_grad():
            source_latents = _encode(video_tensor)
            masked_latents = _encode(masked_video_tensor)
    else:
        # The Wan VAE processes time sequentially with an internal feature
        # cache, so the full clip is encoded in ONE pass regardless of length
        # (matching the original research pipeline). Do not chunk this path:
        # naive chunk boundaries break the ``1 + 4x`` latent grid and
        # progressively desynchronize the conditioning on long videos.
        original_dtype = next(vae.parameters()).dtype
        vae.to(dtype=torch.bfloat16)
        video_tensor = video_tensor.to(dtype=torch.bfloat16)
        masked_video_tensor = masked_video_tensor.to(dtype=torch.bfloat16)
        with torch.no_grad():
            source_latents = vae.encode([video_tensor[0]], device=device)
            masked_latents = vae.encode([masked_video_tensor[0]], device=device)
        vae.to(dtype=original_dtype)

    ref_latent = source_latents[:, :, :1].to(dtype=dtype)  # [1, 16, 1, H_lat, W_lat]
    ref_sequence = source_latents.to(dtype=dtype)  # [1, 16, T_lat, H_lat, W_lat]
    masked_latents = masked_latents.to(dtype=dtype)
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
    latent_mask = load_latentsync_mask(mask_path, H_lat, W_lat).to(device=device, dtype=dtype)

    return ref_latent, masked_latents, ref_sequence, latent_mask


def encode_audio(wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device):
    """Encode audio to wav2vec2 features matching OmniAvatar's encode_audio.

    Encodes at the FULL audio's natural frame count, then slices to
    num_video_frames. This preserves the temporal grid that the model was
    trained on.

    Args:
        wav2vec_model: Wav2VecModel instance (on device, float32)
        wav2vec_extractor: Wav2Vec2FeatureExtractor
        audio_path: path to audio file
        num_video_frames: number of video frames to produce embeddings for
        device: torch device

    Returns:
        audio_emb: [1, num_video_frames, 10752] float tensor
    """
    wav2vec_sr = 16000  # Wav2Vec2 native sample rate
    fps = 25  # OmniAvatar default

    audio, sr = librosa.load(audio_path, sr=wav2vec_sr)
    input_values = np.squeeze(
        wav2vec_extractor(audio, sampling_rate=wav2vec_sr).input_values
    )
    input_values = torch.from_numpy(input_values).float().to(device=device)
    input_values = input_values.unsqueeze(0)

    # Compute the full audio's natural frame count
    samples_per_frame = wav2vec_sr // fps  # 640 at 16kHz/25fps
    total_audio_frames = math.ceil(input_values.shape[1] / samples_per_frame)
    total_audio_frames = max(total_audio_frames, num_video_frames)  # at least num_frames

    # Pad to align with total_audio_frames
    target_samples = total_audio_frames * samples_per_frame
    if input_values.shape[1] < target_samples:
        input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

    # Encode at the full audio length, then slice to num_video_frames.
    with torch.no_grad():
        hidden_states = wav2vec_model(
            input_values, seq_len=total_audio_frames, output_hidden_states=True
        )
    audio_emb = hidden_states.last_hidden_state
    for hs in hidden_states.hidden_states:
        audio_emb = torch.cat((audio_emb, hs), -1)
    # audio_emb: [1, total_audio_frames, 10752]

    # Slice to num_video_frames (matches training: full_emb[:num_training_frames])
    audio_emb = audio_emb[:, :num_video_frames, :]
    return audio_emb  # [1, num_video_frames, 10752]


def build_condition(vae, wav2vec_model, wav2vec_extractor, video_frames_np,
                    audio_path, text_embeds, mask_path, num_video_frames,
                    num_latent_frames, device, dtype):
    """Build the full conditioning dictionary for the causal model.

    Args:
        vae: WanVideoVAE
        wav2vec_model: Wav2VecModel
        wav2vec_extractor: Wav2Vec2FeatureExtractor
        video_frames_np: [N, H, W, 3] uint8
        audio_path: path to audio file
        text_embeds: [1, 512, 4096] tensor
        mask_path: path to LatentSync mask
        num_video_frames: total video frames
        num_latent_frames: total latent frames
        device: torch device
        dtype: torch dtype

    Returns:
        dict with keys: text_embeds, audio_emb, ref_latent, mask,
        masked_video, ref_sequence
    """
    print("Encoding audio ...")
    audio_emb = encode_audio(
        wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device
    )
    audio_emb = audio_emb.to(dtype=dtype)

    print("Encoding reference video ...")
    ref_latent, masked_latents, ref_sequence, latent_mask = encode_reference_video(
        vae, video_frames_np, mask_path, device, dtype
    )

    return {
        "text_embeds": text_embeds,
        "audio_emb": audio_emb,
        "ref_latent": ref_latent.to(device=device, dtype=dtype),
        "mask": latent_mask.to(device=device),
        "masked_video": masked_latents.to(device=device, dtype=dtype),
        "ref_sequence": ref_sequence.to(device=device, dtype=dtype),
    }


def build_condition_from_precomputed(precomputed_dir, mask_path, num_latent_frames, device, dtype):
    """Build conditioning dict from pre-computed .pt files (exact training format).

    This bypasses VAE/Wav2Vec encoding and uses the same tensors the model was
    trained on, enabling direct comparison.
    """
    print(f"Loading precomputed tensors from {precomputed_dir} ...")

    # VAE latents (input + masked)
    vae_data = torch.load(
        os.path.join(precomputed_dir, "vae_latents_mask_all.pt"),
        map_location="cpu", weights_only=False,
    )
    input_latents = vae_data["input_latents"].to(dtype=dtype)  # [16, T, H, W]
    masked_latents = vae_data["masked_latents"].to(dtype=dtype)

    # ref_latent = first frame of input video
    ref_latent = input_latents[:, :1].unsqueeze(0)  # [1, 16, 1, H, W]

    # Slice to num_latent_frames
    input_latents = input_latents[:, :num_latent_frames].unsqueeze(0)  # [1, 16, T, H, W]
    masked_latents = masked_latents[:, :num_latent_frames].unsqueeze(0)

    # ref_sequence (from separate file)
    ref_path = os.path.join(precomputed_dir, "ref_latents.pt")
    if os.path.exists(ref_path):
        ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
        ref_seq_key = "ref_sequence_latents" if "ref_sequence_latents" in ref_data else list(ref_data.keys())[0]
        ref_sequence = ref_data[ref_seq_key].to(dtype=dtype)[:, :num_latent_frames].unsqueeze(0)
    else:
        print("  Warning: ref_latents.pt not found, using input_latents as ref_sequence")
        ref_sequence = input_latents

    # Audio (video-frame-rate)
    audio_data = torch.load(
        os.path.join(precomputed_dir, "audio_emb_omniavatar.pt"),
        map_location="cpu", weights_only=False,
    )
    audio_emb = audio_data["audio_emb"] if isinstance(audio_data, dict) else audio_data
    # Training slices to num_video_frames = 1 + (num_latent - 1) * 4
    num_video_frames = 1 + (num_latent_frames - 1) * 4
    audio_emb = audio_emb[:num_video_frames].unsqueeze(0).to(dtype=dtype)  # [1, V, 10752]
    print(f"  audio_emb: {audio_emb.shape} (sliced to {num_video_frames} video frames)")

    # Text
    text_data = torch.load(
        os.path.join(precomputed_dir, "text_emb.pt"),
        map_location="cpu", weights_only=False,
    )
    if isinstance(text_data, dict):
        text_embeds = next(v for v in text_data.values() if isinstance(v, torch.Tensor))
    else:
        text_embeds = text_data
    if text_embeds.dim() == 2:
        text_embeds = text_embeds.unsqueeze(0)
    text_embeds = text_embeds.to(dtype=dtype)

    # Mask
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
    latent_mask = load_latentsync_mask(mask_path, H_lat, W_lat)

    print(f"  ref_latent: {ref_latent.shape}, masked_video: {masked_latents.shape}")
    print(f"  ref_sequence: {ref_sequence.shape}, mask: {latent_mask.shape}")

    return {
        "text_embeds": text_embeds.to(device),
        "audio_emb": audio_emb.to(device),
        "ref_latent": ref_latent.to(device),
        "mask": latent_mask.to(device=device, dtype=dtype),
        "masked_video": masked_latents.to(device),
        "ref_sequence": ref_sequence.to(device),
    }


# ===========================================================================
# LatentSync preprocessing / compositing
# ===========================================================================

def load_image_processor(mask_path, device):
    """Load LatentSync ImageProcessor for face detection and alignment.

    Initializes the LatentSync ImageProcessor:
    - mask_image loaded via load_fixed_mask (uses caller-specified path)
    - insightface_root anchored to the repo root so inference can run from any
      working directory (insightface auto-downloads buffalo_l there on first use)
    - device passed as string for InsightFace compatibility
    """
    import os as _os
    _os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
    from OmniAvatar.utils.latentsync.image_processor import ImageProcessor, load_fixed_mask
    print("Loading LatentSync ImageProcessor ...")
    device_str = str(device) if isinstance(device, torch.device) else device
    mask_tensor = load_fixed_mask(512, mask_image_path=mask_path) if mask_path else None
    processor = ImageProcessor(
        resolution=512,
        device=device_str,
        mask_image=mask_tensor,
        insightface_root=os.path.join(LIPFORCING_ROOT, "checkpoints", "auxiliary"),
    )
    return processor


def preprocess_with_latentsync(video_path, image_processor, face_detection_cache_dir=None, num_frames=81):
    """Detect faces, align to 512x512 via affine transform.

    When *face_detection_cache_dir* is set, detection results are cached there
    (``<video_stem>_face_cache.pt``) and reused on later runs over the same
    video; ``None`` disables caching.
    """
    if not os.path.exists(video_path):
        print(f"[LatentSync] WARNING: Video not found: {video_path}")
        return None

    try:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        if video_basename in ("sub_clip", "video"):
            video_stem = os.path.basename(os.path.dirname(video_path))
        else:
            video_stem = video_basename
        face_cache_path = (
            os.path.join(face_detection_cache_dir, f"{video_stem}_face_cache.pt")
            if face_detection_cache_dir else None
        )

        face_cache_loaded = False
        original_frames = None
        if face_cache_path and os.path.isfile(face_cache_path):
            try:
                face_cache = torch.load(face_cache_path, weights_only=False)
                cached_frames = face_cache.get("num_frames")
                if cached_frames is None:
                    cached_frames = len(face_cache.get("aligned_faces", []))
                # Reuse only if resolution matches, the cache was built with the
                # ping-pong padding (not the legacy last-frame freeze), and it is
                # long enough for this request. The ping-pong cycle is target-
                # independent, so a longer cache is a valid prefix-superset.
                cache_ok = (
                    face_cache.get("resolution") == image_processor.resolution
                    and face_cache.get("pad_mode") == "pingpong"
                    and cached_frames >= num_frames
                )
                if cache_ok:
                    boxes = face_cache["boxes"]
                    affine_matrices = face_cache["affine_matrices"]
                    aligned_faces = face_cache["aligned_faces"]
                    detection_failures = []
                    face_cache_loaded = True
                    print(f"[LatentSync] Loaded face cache: {face_cache_path}")
                else:
                    print(f"[LatentSync] Cache stale "
                          f"(res={face_cache.get('resolution')}/{image_processor.resolution}, "
                          f"pad={face_cache.get('pad_mode')}, "
                          f"frames={cached_frames}/{num_frames}), recomputing...")
            except Exception as e:
                print(f"[LatentSync] Cache corrupt ({e}), recomputing...")
                os.remove(face_cache_path)

        if not face_cache_loaded:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if len(frames) < 5:
                print(f"[LatentSync] Too few frames ({len(frames)}) in {video_path}")
                return None

            if len(frames) < num_frames:
                # Ping-pong (rewind) extend to the audio-driven length instead of
                # freezing the last frame, so face detection and the VAE
                # conditioning run over the rewound video.
                frames = [frames[i] for i in pingpong_indices(len(frames), num_frames)]

            original_frames = np.stack(frames, axis=0)
            boxes = []
            affine_matrices = []
            aligned_faces = []
            detection_failures = []

            # Reset temporal smoothing bias for new video
            image_processor.restorer.p_bias = None

            # Robustness: reuse the last good detection for brief detection blips
            # (motion blur, head turn, momentary occlusion) instead of aborting the
            # whole clip on a handful of frames. The face barely moves over a few
            # frames, so reusing the previous alignment/crop is visually
            # imperceptible. Only hard-fail when there is no reference yet, or the
            # gap is long enough (>=60 consecutive frames, ~2s) to be a real loss.
            consecutive_fail = 0
            reused = 0
            last_good = None  # (face, box, affine_matrix)
            for i, frame in enumerate(frames):
                try:
                    face, box, affine_matrix = image_processor.affine_transform(frame)
                    boxes.append(box)
                    affine_matrices.append(affine_matrix)
                    aligned_faces.append(face)
                    last_good = (face, box, affine_matrix)
                    consecutive_fail = 0
                except RuntimeError as e:
                    consecutive_fail += 1
                    if last_good is not None and consecutive_fail < 60:
                        gf, gb, ga = last_good
                        boxes.append(gb)
                        affine_matrices.append(ga)
                        aligned_faces.append(gf)
                        reused += 1
                    else:
                        print(f"[LatentSync] Face detection failed for frame {i}: {e}")
                        boxes.append(None)
                        affine_matrices.append(None)
                        detection_failures.append(i)

            if reused:
                print(f"[LatentSync] Reused last-good face for {reused} frames "
                      f"(brief detection blips)")

            if detection_failures:
                print(f"[LatentSync] Face detection failed for {len(detection_failures)} frames, skipping")
                return None

            if face_cache_path:
                os.makedirs(face_detection_cache_dir, exist_ok=True)
                torch.save({
                    "aligned_faces": aligned_faces,
                    "boxes": boxes,
                    "affine_matrices": affine_matrices,
                    "resolution": image_processor.resolution,
                    "num_frames": len(original_frames),
                    "pad_mode": "pingpong",
                }, face_cache_path)
                print(f"[LatentSync] Saved face cache: {face_cache_path}")

        return {
            "video_path": video_path,
            "original_frames": original_frames,
            "num_frames": num_frames,
            "aligned_faces": aligned_faces,
            "boxes": boxes,
            "affine_matrices": affine_matrices,
            "detection_failures": detection_failures if not face_cache_loaded else [],
        }

    except Exception as e:
        print(f"[LatentSync] Preprocessing failed for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def composite_with_latentsync_float(generated_float, latentsync_metadata, image_processor,
                                     use_mouth_only_compositing=False, frame_offset=0,
                                     occlusion_masker=None):
    """Composite generated faces back onto original video, staying in float space.

    Keeps the model output in float space (no uint8 quantization) before
    compositing for maximum precision.

    Args:
        generated_float: [T, C, H, W] float tensor in [0, 1]
        frame_offset: offset into the metadata arrays (for per-chunk streaming)
        occlusion_masker: optional OcclusionMasker instance. When provided, runs
            face parsing on the original frames covered by this batch and uses
            the per-frame skin mask to protect occluding objects (hands,
            billboards, cloth) from being overwritten by the generated face.
    """
    import torchvision.transforms.functional as TF_v

    original_frames = latentsync_metadata["original_frames"]
    if original_frames is None:
        video_path = latentsync_metadata["video_path"]
        num_frames = latentsync_metadata.get("num_frames", 81)
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) < num_frames:
            # Ping-pong (rewind) extend to match the conditioning frames so the
            # composite background tracks the same rewound video.
            frames = [frames[i] for i in pingpong_indices(len(frames), num_frames)]
        original_frames = np.stack(frames, axis=0)

    boxes = latentsync_metadata["boxes"]
    affine_matrices = latentsync_metadata["affine_matrices"]
    detection_failures = latentsync_metadata.get("detection_failures", [])
    aligned_faces = latentsync_metadata.get("aligned_faces", None)

    skin_masks_aligned = None
    if occlusion_masker is not None:
        # Run BiSeNet on the aligned face crops (its training distribution).
        # We warp the resulting mask back to the original frame per-crop below.
        num_out = int(generated_float.shape[0])
        gi_lo = int(frame_offset)
        gi_hi = min(gi_lo + num_out, len(original_frames))
        if gi_hi > gi_lo and aligned_faces is not None:
            aligned_batch = []
            for gi in range(gi_lo, gi_hi):
                af = aligned_faces[gi]
                if isinstance(af, torch.Tensor):
                    af_np = af.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                else:
                    af_np = np.asarray(af, dtype=np.uint8)
                aligned_batch.append(af_np)
            aligned_batch = np.stack(aligned_batch, axis=0)
            skin_masks_aligned = occlusion_masker.compute_aligned(aligned_batch)

    composite_frames = []

    for i in range(generated_float.shape[0]):
        gi = i + frame_offset
        if gi >= len(original_frames):
            break
        if gi in detection_failures or boxes[gi] is None:
            composite_frames.append(original_frames[gi])
            continue

        face = generated_float[i]  # [C, H, W] float [0,1]

        if use_mouth_only_compositing and aligned_faces is not None:
            mouth_mask = image_processor.mask_image.float()
            original_aligned_float = aligned_faces[gi].float() / 255.0
            face = face * (1 - mouth_mask) + original_aligned_float * mouth_mask

        x1, y1, x2, y2 = boxes[gi]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face_resized = TF_v.resize(
            face, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        face_resized = face_resized * 2.0 - 1.0

        ext_mask = None
        if skin_masks_aligned is not None:
            aligned_mask = skin_masks_aligned[i]
            # face_resized is in the AlignRestore crop space (face_size), so
            # first resize the mask to that space, then let restore_img invert
            # the same affine.
            m_resized = cv2.resize(
                aligned_mask,
                (image_processor.restorer.face_size[0],
                 image_processor.restorer.face_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )
            H_orig, W_orig = original_frames[gi].shape[:2]
            ext_mask = occlusion_masker.warp_to_original(
                m_resized, affine_matrices[gi], (H_orig, W_orig),
            )

        try:
            restored_frame = image_processor.restorer.restore_img(
                original_frames[gi], face_resized, affine_matrices[gi],
                external_mask=ext_mask,
            )
            composite_frames.append(restored_frame)
        except Exception as e:
            print(f"[LatentSync] Restoration failed for frame {gi}: {e}")
            composite_frames.append(original_frames[gi])

    return np.stack(composite_frames)


def save_frames_as_video(frames_np, output_path, fps=25):
    """Save [N, H, W, 3] uint8 numpy array as mp4 video.

    Encodes with libx264 at CRF 13 and macro_block_size=None.
    """
    import imageio
    writer = imageio.get_writer(
        output_path, fps=fps, codec='libx264',
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
        ffmpeg_log_level="error",
    )
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def mux_video_with_audio(video_path, audio_path, output_path, duration_s=None):
    """Mux silent video with audio via ffmpeg."""
    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-q:v", "0", "-q:a", "0",
    ]
    if duration_s is not None:
        cmd.extend(["-t", f"{duration_s:.4f}"])
    cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr}")


# ===========================================================================
# Batch enumeration
# ===========================================================================

def enumerate_samples(args):
    """Yield (name, video_path, audio_path, precomputed_dir) per sample.

    Batch mode (--input_dir): one sample per subdir containing sub_clip.mp4 +
    audio.wav; training-style precomputed tensors are picked up automatically
    when present. Single-sample mode: --video_path (+ optional --audio_path /
    --precomputed_dir where the script supports it).
    """
    if args.input_dir is not None:
        for entry in sorted(os.listdir(args.input_dir)):
            sample_dir = os.path.join(args.input_dir, entry)
            if not os.path.isdir(sample_dir):
                continue
            video_path = os.path.join(sample_dir, "sub_clip.mp4")
            if not os.path.isfile(video_path):
                continue
            audio_path = os.path.join(sample_dir, "audio.wav")
            if not os.path.isfile(audio_path):
                print(f"[Skip] No audio.wav in {sample_dir}")
                continue
            precomputed = sample_dir if os.path.isfile(
                os.path.join(sample_dir, "vae_latents_mask_all.pt")
            ) else None
            yield entry, video_path, audio_path, precomputed
    else:
        name = os.path.splitext(os.path.basename(args.video_path))[0]
        yield name, args.video_path, args.audio_path, getattr(args, "precomputed_dir", None)
