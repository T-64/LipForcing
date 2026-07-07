# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Dataset for OmniAvatar V2V precomputed training data.

Each sample directory contains precomputed .pt files:
    - vae_latents_mask_all.pt: {input_latents [16,21,64,64], masked_latents [16,21,64,64]}
    - audio_emb_omniavatar.pt: {audio_emb [N,10752]} where N >= 81
    - text_emb.pt: tensor [1,512,4096]
    - ref_latents.pt: {ref_sequence_latents [16,21,64,64], metadata}
    - ode_path.pt: [4, 16, 21, 64, 64] (ODE trajectories for KD, optional)
"""

import hashlib
import os
import tempfile
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def _load_raw_waveform_for_reward(
    audio_path: str,
    target_sample_rate: int,
    target_length: int,
) -> torch.Tensor:
    """Load a wav, mono-collapse, resample to target_sample_rate, pad/truncate to target_length.

    Returns a 1-D float32 tensor of exactly target_length samples. Used by the
    SyncCScorer-based Re-DMD reward.
    """
    import scipy.io.wavfile as wavfile
    from scipy import signal

    sr, wav = wavfile.read(audio_path)

    # Convert to float32 in [-1, 1] range
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # Convert to torch tensor
    wav = torch.from_numpy(wav)

    # Handle stereo -> mono
    if wav.ndim == 2:
        wav = wav.mean(dim=1)

    # Resample if needed
    if sr != target_sample_rate:
        # Use scipy resample for simplicity
        num_samples_new = int(len(wav) * target_sample_rate / sr)
        wav_np = wav.numpy()
        wav_np = signal.resample(wav_np, num_samples_new)
        wav = torch.from_numpy(wav_np)

    # Pad or truncate to target_length
    L = target_length
    if wav.shape[0] < L:
        wav = torch.nn.functional.pad(wav, (0, L - wav.shape[0]))
    else:
        wav = wav[:L]

    return wav.to(torch.float32)


class OmniAvatarDataset(Dataset):
    """
    Dataset for OmniAvatar V2V training data with precomputed tensors.

    Returns dict with:
        real: [16, 21, 64, 64] — clean video latents (bf16)
        masked_video: [16, 21, 64, 64] — mouth-masked video latents (bf16)
        audio_emb: [81, 10752] — Wav2Vec2 audio features (bf16)
        text_embeds: [1, 512, 4096] — T5 text embedding (bf16)
        ref_sequence: [16, 21, 64, 64] — reference sequence latents (bf16, optional)
        mask: [64, 64] — spatial mask, LatentSync convention: 1=keep, 0=mask (float32)
        neg_text_embeds: [1, 512, 4096] — negative text embedding for CFG (bf16)
        path: [4, 16, 21, 64, 64] — ODE trajectory (bf16, optional, for KD training)
    """

    # Precomputed tensor filenames (the fast path).
    PRECOMPUTED_FILES = ["vae_latents_mask_all.pt", "audio_emb_omniavatar.pt", "text_emb.pt"]
    # Raw inputs required to encode a sample on the fly.
    RAW_FILES = ["sub_clip.mp4", "audio.wav", "prompt.txt"]

    def __init__(
        self,
        data_list_path: str,
        latentsync_mask_path: str,
        neg_text_emb_path: str = None,
        use_ref_sequence: bool = True,
        load_ode_path: bool = False,
        num_video_frames: int = 81,
        latent_h: int = 64,
        latent_w: int = 64,
        load_raw_audio: bool = False,
        raw_audio_sample_rate: int = 16000,
        raw_audio_num_frames: int = 81,
        raw_audio_fps: float = 25.0,
        on_the_fly: bool = False,
        cache_encoded: bool = True,
        cache_dir: str = None,
        height: int = 512,
        width: int = 512,
    ):
        self.use_ref_sequence = use_ref_sequence
        self.load_ode_path = load_ode_path
        self.num_video_frames = num_video_frames
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.load_raw_audio = load_raw_audio
        self.raw_audio_sample_rate = raw_audio_sample_rate
        self.raw_audio_num_frames = raw_audio_num_frames
        self.raw_audio_fps = raw_audio_fps
        self.raw_audio_length = int(self.raw_audio_num_frames / self.raw_audio_fps * self.raw_audio_sample_rate)

        # On-the-fly encoding: when True, samples without precomputed .pt are
        # encoded from raw (sub_clip.mp4 + audio.wav + prompt.txt) — the heavy
        # encoders run in the MAIN training process (see encode_on_the_fly_batch),
        # never in dataloader workers. When False, behavior is unchanged: only
        # fully-precomputed samples are kept (the precompute fast path).
        self.on_the_fly = on_the_fly
        self.cache_encoded = cache_encoded
        self.cache_dir = cache_dir
        self.height = height
        self.width = width
        self.latentsync_mask_path = latentsync_mask_path

        # Read sample directories from text file
        with open(data_list_path) as f:
            all_dirs = [line.strip() for line in f if line.strip()]

        # Filter samples. Fast path requires the precomputed .pt; on-the-fly only
        # requires the raw inputs (a sample with precomputed .pt still uses the
        # fast path, otherwise it is encoded on the fly).
        self.dirs = []
        for d in all_dirs:
            if self._has_precomputed(d):
                self.dirs.append(d)
            elif self.on_the_fly and self._has_raw(d):
                self.dirs.append(d)
            else:
                missing = [fn for fn in self.PRECOMPUTED_FILES if not os.path.exists(os.path.join(d, fn))]
                if self.on_the_fly:
                    missing = [fn for fn in self.RAW_FILES if not os.path.exists(os.path.join(d, fn))]
                warnings.warn(f"Skipping {d}: missing {missing}")

        if len(self.dirs) < len(all_dirs):
            print(
                f"[OmniAvatarDataset] Kept {len(self.dirs)}/{len(all_dirs)} samples "
                f"({len(all_dirs) - len(self.dirs)} skipped due to missing files)"
            )
        if self.on_the_fly:
            n_pre = sum(self._has_precomputed(d) for d in self.dirs)
            print(
                f"[OmniAvatarDataset] on_the_fly=True: {n_pre}/{len(self.dirs)} samples "
                f"have precomputed tensors (fast path); the rest are encoded on the fly "
                f"(cache_encoded={self.cache_encoded}, cache_dir={self.cache_dir})."
            )

        # Load spatial mask once: PNG (256x256 RGB) -> single channel -> resize to latent res -> threshold
        # LatentSync convention: 1=keep (upper face), 0=mask (mouth region)
        mask_img = Image.open(latentsync_mask_path)
        mask_arr = np.array(mask_img, dtype=np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]  # take first channel
        mask_arr = mask_arr / 255.0  # normalize to [0, 1]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask_resized = F.interpolate(
            mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False
        )
        self.mask = (mask_resized.squeeze() > 0.5).float()  # [64, 64], float32

        # Pixel-resolution binary mask (1=keep, 0=mask) for on-the-fly VAE masking.
        # Lazily built on first use; binarizes the mask at pixel resolution.
        self._pixel_mask = None

        # Load negative text embedding (for CFG)
        if neg_text_emb_path is not None and os.path.exists(neg_text_emb_path):
            neg_emb = torch.load(neg_text_emb_path, map_location="cpu", weights_only=False)
            if isinstance(neg_emb, dict):
                # Handle dict format if needed
                neg_emb = next(v for v in neg_emb.values() if isinstance(v, torch.Tensor))
            self.neg_text_embeds = neg_emb.to(torch.bfloat16)
        else:
            self.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

        # Ensure correct shape
        if self.neg_text_embeds.dim() == 2:
            self.neg_text_embeds = self.neg_text_embeds.unsqueeze(0)

    # ------------------------------------------------------------------
    # Fast-path / on-the-fly helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_all(d, files):
        return d is not None and all(os.path.exists(os.path.join(d, f)) for f in files)

    def _has_raw(self, sample_dir):
        return self._has_all(sample_dir, self.RAW_FILES)

    def _cache_sample_dir(self, sample_dir):
        """Per-sample cache directory (``cache_dir/<sample_id>``) or None."""
        if self.cache_dir is None:
            return None
        norm = os.path.normpath(sample_dir)
        # basename + hash of the full path: readable, but collision-free across
        # different source dirs that share a leaf name (e.g. vidA/clip0, vidB/clip0).
        sample_id = f"{os.path.basename(norm)}_{hashlib.sha1(norm.encode()).hexdigest()[:8]}"
        return os.path.join(self.cache_dir, sample_id)

    def _precomputed_source(self, sample_dir):
        """Directory holding this sample's precomputed .pt, or None.

        Prefers the sample directory itself, then the per-sample cache directory
        (written by a previous on-the-fly encode). Returning non-None means the
        fast path applies.
        """
        if self._has_all(sample_dir, self.PRECOMPUTED_FILES):
            return sample_dir
        cdir = self._cache_sample_dir(sample_dir)
        if self._has_all(cdir, self.PRECOMPUTED_FILES):
            return cdir
        return None

    def _has_precomputed(self, sample_dir):
        return self._precomputed_source(sample_dir) is not None

    @property
    def pixel_mask(self):
        """Pixel-resolution binary mask [H, W] (1=keep, 0=mask), built once."""
        if self._pixel_mask is None:
            from lipforcing import preprocess as pp
            self._pixel_mask = pp.binarize_pixel_mask(
                self.latentsync_mask_path, self.height, self.width
            )
        return self._pixel_mask

    def _getitem_raw(self, sample_dir) -> dict:
        """On-the-fly path: CPU-decode raw inputs for main-process encoding.

        Runs in dataloader workers — no models, no GPU. Returns a dict flagged
        with ``_needs_encode`` that :func:`encode_on_the_fly_batch` turns into the
        model-ready tensors in the main training process.
        """
        from lipforcing import preprocess as pp

        video = os.path.join(sample_dir, "sub_clip.mp4")
        audio = os.path.join(sample_dir, "audio.wav")
        try:
            with open(os.path.join(sample_dir, "prompt.txt")) as f:
                prompt = f.read().strip()
            raw = pp.prepare_raw_sample(
                video, audio, prompt, self.pixel_mask,
                num_video_frames=self.num_video_frames,
                height=self.height, width=self.width,
                load_audio_waveform=True,
            )
        except Exception as e:
            warnings.warn(f"Error preparing raw sample {sample_dir}: {e}")
            return None

        raw["_needs_encode"] = True
        raw["sample_id"] = os.path.basename(os.path.normpath(sample_dir))
        raw["cache_sample_dir"] = self._cache_sample_dir(sample_dir) or sample_dir
        raw["use_ref_sequence"] = self.use_ref_sequence
        raw["mask"] = self.mask  # shared latent mask [64, 64]
        raw["neg_text_embeds"] = self.neg_text_embeds.clone()
        raw["audio_path"] = audio if os.path.exists(audio) else ""
        if self.load_raw_audio and os.path.exists(audio):
            raw["audio_waveform"] = _load_raw_waveform_for_reward(
                audio,
                target_sample_rate=self.raw_audio_sample_rate,
                target_length=self.raw_audio_length,
            )
        return raw

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx) -> dict:
        sample_dir = self.dirs[idx]

        # Resolve the precomputed source (sample dir or cache). When absent and
        # on-the-fly is enabled, yield raw inputs for main-process encoding.
        src = self._precomputed_source(sample_dir)
        if src is None:
            if self.on_the_fly:
                return self._getitem_raw(sample_dir)
            warnings.warn(f"No precomputed tensors for {sample_dir} and on_the_fly=False")
            return None

        try:
            # --- VAE latents ---
            vae_data = torch.load(
                os.path.join(src, "vae_latents_mask_all.pt"),
                map_location="cpu",
                weights_only=False,
            )
            real = vae_data["input_latents"].to(torch.bfloat16)  # [16, 21, 64, 64]
            masked_video = vae_data["masked_latents"].to(torch.bfloat16)  # [16, 21, 64, 64]

            # --- Audio embeddings ---
            audio_data = torch.load(
                os.path.join(src, "audio_emb_omniavatar.pt"),
                map_location="cpu",
                weights_only=False,
            )
            audio_emb = audio_data["audio_emb"][: self.num_video_frames]  # [81, 10752]
            audio_emb = audio_emb.to(torch.bfloat16)

            # --- Text embedding ---
            text_emb = torch.load(
                os.path.join(src, "text_emb.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(text_emb, dict):
                text_emb = next(v for v in text_emb.values() if isinstance(v, torch.Tensor))
            text_emb = text_emb.to(torch.bfloat16)
            # Ensure shape [1, 512, 4096]
            if text_emb.dim() == 2:
                text_emb = text_emb.unsqueeze(0)

            result = {
                "real": real,
                "masked_video": masked_video,
                "audio_emb": audio_emb,
                "text_embeds": text_emb,
                "mask": self.mask,  # shared across all samples, float32
                "neg_text_embeds": self.neg_text_embeds.clone(),
            }

            # Audio file path for wandb video logging with audio muxing.
            # Always include key so default_collate doesn't fail on mixed-key batches.
            audio_wav_path = os.path.join(sample_dir, "audio.wav")
            result["audio_path"] = audio_wav_path if os.path.exists(audio_wav_path) else ""

            # Load raw audio waveform if requested (for SyncCScorer reward)
            if self.load_raw_audio and isinstance(result.get("audio_path"), str) and os.path.exists(result["audio_path"]):
                result["audio_waveform"] = _load_raw_waveform_for_reward(
                    result["audio_path"],
                    target_sample_rate=self.raw_audio_sample_rate,
                    target_length=self.raw_audio_length,
                )

            # --- Reference sequence latents (optional) ---
            if self.use_ref_sequence:
                ref_path = os.path.join(src, "ref_latents.pt")
                if os.path.exists(ref_path):
                    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
                    result["ref_sequence"] = ref_data["ref_sequence_latents"].to(torch.bfloat16)
                else:
                    # Fallback: zeros with same shape as real latents
                    result["ref_sequence"] = torch.zeros_like(real)

            # --- ODE trajectory (optional, for KD training) ---
            if self.load_ode_path:
                ode_path_file = os.path.join(src, "ode_path.pt")
                if os.path.exists(ode_path_file):
                    result["path"] = torch.load(ode_path_file, map_location="cpu", weights_only=False).to(
                        torch.bfloat16
                    )
                else:
                    # Also check path.pth (alternative filename)
                    alt_path_file = os.path.join(src, "path.pth")
                    if os.path.exists(alt_path_file):
                        result["path"] = torch.load(alt_path_file, map_location="cpu", weights_only=False).to(
                            torch.bfloat16
                        )

            return result

        except Exception as e:
            warnings.warn(f"Error loading sample {sample_dir}: {e}")
            # Return None; collate_fn should filter these out
            return None


class OmniAvatarDataLoader:
    """Infinite-iterator DataLoader wrapper with DistributedSampler support.

    The trainer expects an infinite iterator. This class wraps a standard
    DataLoader and yields batches indefinitely, cycling through the dataset.
    """

    def __init__(
        self,
        data_list_path: str = None,
        datatags: list = None,
        latentsync_mask_path: str = None,
        batch_size: int = 1,
        num_workers: int = 4,
        load_ode_path: bool = False,
        vae_path: str = None,
        wav2vec_path: str = None,
        text_encoder_path: str = None,
        **kwargs,
    ):
        # Support both data_list_path and datatags (list of paths)
        if data_list_path is None and datatags is not None:
            data_list_path = datatags[0] if isinstance(datatags, list) else datatags
        assert data_list_path is not None, "Must provide data_list_path or datatags"
        assert latentsync_mask_path is not None, "Must provide latentsync_mask_path"

        # Encoder checkpoints for on-the-fly preprocessing. These are consumed by
        # the MAIN training process (see encode_on_the_fly_batch / the trainer),
        # never inside dataloader workers, so they are stored on the loader rather
        # than forwarded to the (worker-side) dataset.
        self.encoder_config = {
            "vae_path": vae_path,
            "wav2vec_path": wav2vec_path,
            "text_encoder_path": text_encoder_path,
            "latentsync_mask_path": latentsync_mask_path,
        }
        self.on_the_fly = bool(kwargs.get("on_the_fly", False))
        self.cache_encoded = bool(kwargs.get("cache_encoded", True))
        self.cache_dir = kwargs.get("cache_dir", None)
        self.num_video_frames = int(kwargs.get("num_video_frames", 81))

        self.dataset = OmniAvatarDataset(
            data_list_path=data_list_path,
            latentsync_mask_path=latentsync_mask_path,
            load_ode_path=load_ode_path,
            **kwargs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Use DistributedSampler for multi-GPU training
        if dist.is_initialized():
            self._sampler = DistributedSampler(self.dataset, shuffle=True)
            shuffle = False
        else:
            self._sampler = None
            shuffle = True

        def collate_fn(batch):
            """Filter out None samples from failed loads, detach tensors for worker compatibility.

            If any sample needs on-the-fly encoding, the batch is returned as an
            unstacked ``{"_samples": [...]}`` list so the main process can run the
            encoders (see :func:`encode_on_the_fly_batch`) before stacking. Fully
            precomputed batches take the original fast path (detach + collate).
            """
            valid = [s for s in batch if s is not None]
            if not valid:
                return {}
            if any(isinstance(s, dict) and s.get("_needs_encode") for s in valid):
                return {"_samples": valid}
            # Detach tensors to avoid autograd collation errors with num_workers>0
            detached = []
            for sample in valid:
                d = {}
                for k, v in sample.items():
                    d[k] = v.detach() if isinstance(v, torch.Tensor) and v.requires_grad else v
                detached.append(d)
            return torch.utils.data.default_collate(detached)

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self._sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def __iter__(self):
        """Infinite iterator — cycles through the dataset."""
        epoch = 0
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)
            yield from self._dataloader
            epoch += 1

    def __len__(self):
        return len(self.dataset)


# Keep the simple function for backward compatibility
def create_omniavatar_dataloader(
    data_list_path: str,
    latentsync_mask_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    load_ode_path: bool = False,
    shuffle: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for OmniAvatar training data (non-infinite, no DDP support).

    For training, prefer OmniAvatarDataLoader which provides infinite iteration
    and DistributedSampler support.
    """
    dataset = OmniAvatarDataset(
        data_list_path=data_list_path,
        latentsync_mask_path=latentsync_mask_path,
        load_ode_path=load_ode_path,
        **kwargs,
    )

    def collate_fn(batch):
        """Filter out None samples; defer stacking when on-the-fly encoding is needed."""
        valid = [s for s in batch if s is not None]
        if not valid:
            return {}
        if any(isinstance(s, dict) and s.get("_needs_encode") for s in valid):
            return {"_samples": valid}
        return torch.utils.data.default_collate(valid)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


# ===========================================================================
# Main-process on-the-fly encode (runs in the training process, not workers)
# ===========================================================================

def _assemble_model_sample(enc, raw, num_video_frames, dtype=torch.bfloat16):
    """Build the model-ready per-sample dict from an :func:`encode_prepared` result.

    Produces exactly the keys/dtypes the precompute fast path
    (``OmniAvatarDataset.__getitem__``) yields, so on-the-fly and precomputed
    samples are interchangeable downstream.
    """
    out = {
        "real": enc["input_latents"].to(dtype),
        "masked_video": enc["masked_latents"].to(dtype),
        "audio_emb": enc["audio_emb"][:num_video_frames].to(dtype),
        "mask": raw["mask"],
        "neg_text_embeds": raw["neg_text_embeds"],
        "audio_path": raw.get("audio_path", ""),
    }
    text = enc.get("text_emb")
    if text is not None:
        if text.dim() == 2:
            text = text.unsqueeze(0)
        out["text_embeds"] = text.to(dtype)
    if raw.get("use_ref_sequence", True):
        out["ref_sequence"] = enc["ref_sequence_latents"].to(dtype)
    if "audio_waveform" in raw:
        out["audio_waveform"] = raw["audio_waveform"]
    return out


def _save_atomic(obj, path):
    """torch.save to a unique temp file then atomically rename: no partial .pt on
    crash, and no cross-rank .tmp collision when the cache dir is shared."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _write_sample_cache(cache_dir, enc):
    """Write the precompute-format .pt for one encoded sample into *cache_dir*.

    Writes the precompute-format .pt filenames so a later epoch finds them
    via the normal fast path. Text is written only when available (so the fast
    path's text_emb.pt requirement is met).
    """
    _save_atomic(
        {"input_latents": enc["input_latents"], "masked_latents": enc["masked_latents"]},
        os.path.join(cache_dir, "vae_latents_mask_all.pt"),
    )
    _save_atomic(
        {
            "ref_sequence_latents": enc["ref_sequence_latents"],
            "metadata": {
                "ref_start": enc["ref_start"],
                "total_frames": enc["total_frames"],
                "num_frames": enc["num_video_frames"],
                "height": 512,
                "width": 512,
            },
        },
        os.path.join(cache_dir, "ref_latents.pt"),
    )
    _save_atomic(
        {
            "audio_emb": enc["audio_emb"],
            "metadata": {
                "total_video_frames": enc["total_frames"],
                "seq_len": enc["seq_len"],
                "fps": 25,
                "feature_dim": int(enc["audio_emb"].shape[-1]),
            },
        },
        os.path.join(cache_dir, "audio_emb_omniavatar.pt"),
    )
    if enc.get("text_emb") is not None:
        _save_atomic(enc["text_emb"], os.path.join(cache_dir, "text_emb.pt"))


def encode_on_the_fly_batch(
    encoders,
    batch,
    num_video_frames=81,
    device=None,
    dtype=torch.bfloat16,
    cache_encoded=True,
    text_emb_cache=None,
    text_cache_dir=None,
):
    """Encode a raw on-the-fly batch into a stacked, model-ready batch.

    Called once per batch in the main training process. A batch produced by the
    fast path (no ``_samples`` key) is returned unchanged, so this is a no-op for
    precomputed data. For raw batches each sample is GPU-encoded (and optionally
    cached as precompute-format .pt for reuse), then the now-uniform list is
    stacked with ``default_collate``.

    Args:
        encoders: dict from :func:`lipforcing.preprocess.load_encoders`.
        batch: either a stacked model-ready dict (fast path) or ``{"_samples": [...]}``.
        text_emb_cache: optional ``{prompt_hash: tensor}`` so each unique prompt is
            encoded by UMT5 only once across the whole run.
    """
    if not isinstance(batch, dict) or "_samples" not in batch:
        return batch  # fast path / empty — nothing to encode

    from lipforcing import preprocess as pp

    device = device or encoders.get("device")
    text_emb_cache = text_emb_cache if text_emb_cache is not None else {}

    model_samples = []
    for s in batch["_samples"]:
        if not (isinstance(s, dict) and s.get("_needs_encode")):
            # Already model-ready (mixed batch): strip any internal markers.
            model_samples.append({k: v for k, v in s.items() if not k.startswith("_")})
            continue
        enc = pp.encode_prepared(
            encoders, s, device=device, dtype=dtype,
            text_emb_cache=text_emb_cache, text_cache_dir=text_cache_dir,
        )
        model_samples.append(_assemble_model_sample(enc, s, num_video_frames, dtype))
        if cache_encoded and s.get("cache_sample_dir"):
            try:
                _write_sample_cache(s["cache_sample_dir"], enc)
            except Exception as e:  # caching is best-effort; never fail training
                warnings.warn(f"Failed to cache encoded sample {s.get('sample_id')}: {e}")

    detached = []
    for sample in model_samples:
        d = {
            k: (v.detach() if isinstance(v, torch.Tensor) and v.requires_grad else v)
            for k, v in sample.items()
        }
        detached.append(d)
    return torch.utils.data.default_collate(detached)
