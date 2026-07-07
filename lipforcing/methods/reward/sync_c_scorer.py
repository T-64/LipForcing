# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SyncNet-v2 sync-C reward scorer for talking-head lip-sync."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from lipforcing.methods.reward.syncnet_v2 import SyncNetV2


class SyncCScorer(nn.Module):
    """Drop-in replacement for VideoVLMRewardInference.

    Scores talking-head lip-sync using SyncNet-v2's offset-margin confidence,
    as a detached scalar multiplier on the DMD MSE term.

    Pre-conditions:
      - input video frames are face-aligned (centered); no detection here
      - audio waveforms are the driving audio, temporally aligned with frame 0
      - generator emits 25 fps; scorer does NO FPS conversion
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_fps: float = 25.0,
        audio_sample_rate: int = 16000,
        face_crop_size: int = 224,
        vshift: int = 15,
        mfcc_n: int = 13,
        mfcc_hop_ms: float = 10.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert input_fps == 25.0, "SyncNet-v2 is native 25 fps; resample upstream"

        self.net = SyncNetV2()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, nn.Module):
            state = state.state_dict()
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(device=device, dtype=dtype)
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.audio_sample_rate = audio_sample_rate
        self.target_sample_rate = 16000
        self.face_crop_size = face_crop_size
        self.vshift = vshift
        self.mfcc_n = mfcc_n

        # MFCC is computed via python_speech_features in _prep_audio with these
        # settings (nfilt=26, preemph=0.97, ceplifter=22, appendEnergy=True,
        # rectangular window) because they are what the audio encoder's BN
        # running stats expect. torchaudio's defaults differ in 5 of those params
        # and meaningfully shift the cepstral distribution away from those stats.

    def _device(self):
        return next(self.net.parameters()).device

    def _dtype(self):
        return next(self.net.parameters()).dtype

    def _prep_video(self, video: torch.Tensor) -> torch.Tensor:
        """[F, 3, H, W] uint8 (RGB) -> [1, 3, F, 224, 224] float in [0, 255] (BGR).

        Feeds float pixels in [0, 255] in BGR order (no /255), matching the channel
        order and value range the SyncNet conv/BatchNorm stats were trained on. The
        first Conv3d + BatchNorm3d carry running stats keyed to that distribution.
        Two corrections vs naive feed:
          1. No /255 — keep float in [0, 255] so BN running stats apply correctly.
             Otherwise post-BN activations collapse and sync-C drops ~10-100x.
          2. RGB -> BGR — match cv2.imread channel order so first-conv R/B filters
             see the data they were trained on (lip color is a strong sync cue).
        """
        video = video.to(self._device()).float()
        video = F.interpolate(
            video, size=(self.face_crop_size, self.face_crop_size),
            mode="bilinear", align_corners=False,
        )
        video = video[:, [2, 1, 0], :, :]  # RGB -> BGR
        return video.permute(1, 0, 2, 3).unsqueeze(0)

    def _prep_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """[L] float waveform in [-1, 1] -> [1, 1, 13, M] MFCC matching joonson eval.

        Uses python_speech_features.mfcc with library defaults: nfilt=26,
        preemph=0.97, ceplifter=22, appendEnergy=True (C0 = log frame energy),
        rectangular window.

        Waveform is scaled by 32768 before MFCC: joonson's pipeline reads
        wavfile.read() output (int16 magnitudes), and appendEnergy makes C0
        magnitude-dependent. Without this scaling, C0 is ~21 nats lower than
        what the audio encoder's BatchNorm running stats expect.
        """
        from python_speech_features import mfcc as _psf_mfcc

        assert audio.dim() == 1, f"expected 1-D waveform, got shape {tuple(audio.shape)}"
        audio = audio.to(self._device()).float()
        if self.audio_sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, self.audio_sample_rate, self.target_sample_rate,
            )
        # python_speech_features is numpy-only — CPU round-trip ~10ms,
        # negligible vs ~5s training step.
        audio_np = (audio.detach().cpu().numpy() * 32768.0).astype("float32")
        mfcc_np = _psf_mfcc(audio_np, samplerate=self.target_sample_rate, numcep=self.mfcc_n)  # [T, 13]
        mfcc = torch.from_numpy(mfcc_np).to(device=self._device(), dtype=self._dtype())
        mfcc = mfcc.t().contiguous().unsqueeze(0)  # [1, 13, T]
        return mfcc.unsqueeze(1)                    # [1, 1, 13, T]

    def _lip_windows(self, video: torch.Tensor) -> torch.Tensor:
        """[1, 3, F, 224, 224] -> [F-4, 3, 5, 224, 224] — 5-frame stride-1 lip windows."""
        F_ = video.shape[2]
        if F_ < 5:
            raise ValueError(f"Need at least 5 frames, got {F_}")
        w = video.unfold(2, 5, 1).squeeze(0)            # [3, N, 224, 224, 5]
        return w.permute(1, 0, 4, 2, 3).contiguous()

    def _aud_windows(self, mfcc: torch.Tensor) -> torch.Tensor:
        """[1, 1, 13, M] -> [N, 1, 13, 20] — 20-MFCC stride-4 audio windows.

        Stride 4 because MFCC is 100 fps, video is 25 fps (4 MFCC frames per video
        frame), so consecutive audio windows align with consecutive lip windows.
        """
        M = mfcc.shape[-1]
        if M < 20:
            raise ValueError(f"Need at least 20 MFCC frames, got {M}")
        w = mfcc.unfold(-1, 20, 4).squeeze(0).squeeze(0)  # [13, N, 20]
        return w.permute(1, 0, 2).unsqueeze(1)

    def _offset_search(self, lip_emb: torch.Tensor, aud_emb: torch.Tensor) -> torch.Tensor:
        """[N, 1024] x [N, 1024] -> scalar sync-C margin.

        Slides audio relative to lip across `[-vshift, +vshift]` frames, computes
        the mean pairwise L2 distance at each shift, and returns
        `median(mean_dists) - min(mean_dists)` — higher = more confident sync.
        """
        N = min(lip_emb.shape[0], aud_emb.shape[0])
        lip_emb, aud_emb = lip_emb[:N], aud_emb[:N]
        dists = []
        for shift in range(-self.vshift, self.vshift + 1):
            if shift < 0:
                l, a = lip_emb[-shift:], aud_emb[:N + shift]
            elif shift > 0:
                l, a = lip_emb[:N - shift], aud_emb[shift:]
            else:
                l, a = lip_emb, aud_emb
            d = F.pairwise_distance(l, a).mean()
            dists.append(d)
        mean_dists = torch.stack(dists, dim=0)
        return mean_dists.median() - mean_dists.min()

    @torch.no_grad()
    def reward_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        audio_tensors: List[torch.Tensor],
        prompts: Optional[List[str]] = None,
        use_norm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Score a batch of talking-head clips.

        Args:
            video_tensors: list of `[F, 3, H, W]` uint8 face-aligned frames.
            audio_tensors: list of `[L]` float waveforms, time-aligned with frame 0.
            prompts: unused, kept for VideoVLMRewardInference interface compat.
            use_norm: unused, kept for interface compat.

        Returns:
            dict with:
              - `sync_c`: `[B]` scalar tensor, higher = more confident lip sync.
              - `MQ`: alias of `sync_c` so existing Re-DMD callers need no change.
        """
        assert len(video_tensors) == len(audio_tensors), "video/audio batch mismatch"
        confs = [self._score_single(v, a) for v, a in zip(video_tensors, audio_tensors)]
        sync_c = torch.stack(confs, dim=0)
        return {"sync_c": sync_c, "MQ": sync_c}

    def _score_single(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """End-to-end scoring for one sample."""
        video = self._prep_video(video)
        mfcc = self._prep_audio(audio)
        lip_windows = self._lip_windows(video)
        aud_windows = self._aud_windows(mfcc)
        lip_emb = self.net.forward_lip(lip_windows.to(self._dtype()))
        aud_emb = self.net.forward_aud(aud_windows.to(self._dtype()))
        return self._offset_search(lip_emb, aud_emb)
