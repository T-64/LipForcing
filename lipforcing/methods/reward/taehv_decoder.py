# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAEHV decoder wrapper that mimics WanVideoVAE.decode() for the Re-DMD reward path.

WanVideoVAE.decode contract (what _decode_gen_to_pixels consumes):
  input:  list of [C=16, T_lat, H, W] float tensors
  output: [N, 3, T_pix, H_pix, W_pix] float32 tensor in [-1, 1], NCTHW

TAEHV's decode_video contract:
  input:  NTCHW tensor in the raw diffusion latent space (no mean/std scaling)
  output: NTCHW RGB tensor in [0, 1], already trimmed to (T_lat - 1) * t_upscale + 1 frames
         via the built-in frames_to_trim slice at the end of decode_video.

Transformation applied here:
  1. stack list -> [N, 16, T_lat, H, W] and permute to NTCHW: [N, T_lat, 16, H, W]
  2. run TAEHV.decode_video(parallel=True, show_progress_bar=False)
  3. rescale [0, 1] -> [-1, 1]  via  x.mul(2).sub(1)
  4. permute back to NCTHW: [N, 3, T_pix, H_pix, W_pix]
  5. .float() for downstream compatibility (the scorer expects float32)
"""

from __future__ import annotations
from typing import List, Optional

import torch

from lipforcing.methods.reward.taehv import TAEHV


class TAEHVDecoderWrapper:
    """Drop-in WanVideoVAE.decode replacement backed by TAEHV.

    Runs in fp16 internally for speed; returns fp32 to match WanVideoVAE's
    float contract with the sync-C scorer (which re-casts to float anyway).
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self._taehv = TAEHV(checkpoint_path=checkpoint_path).to(device, torch.float16).eval()
        for p in self._taehv.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def decode(self, latents_list: List[torch.Tensor], device: Optional[str] = None) -> torch.Tensor:
        target_device = device if device is not None else self.device
        # Stack list of [C, T, H, W] into [N, C, T, H, W], then NCTHW -> NTCHW.
        batched = torch.stack([lat.to(target_device, dtype=torch.float16) for lat in latents_list], dim=0)
        batched = batched.permute(0, 2, 1, 3, 4).contiguous()  # [N, T_lat, C, H, W]
        vid = self._taehv.decode_video(batched, parallel=True, show_progress_bar=False)
        # vid: [N, T_pix, 3, H_pix, W_pix] in [0, 1] — already trimmed by decode_video.
        vid = vid.mul(2.0).sub(1.0)  # [0, 1] -> [-1, 1], match WanVideoVAE contract
        return vid.permute(0, 2, 1, 3, 4).float().contiguous()  # [N, 3, T_pix, H_pix, W_pix]
