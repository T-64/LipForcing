# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar Diffusion Forcing model for Stage 1 initialization.

Alternative to ODE-based KD (CausalKDModel). Instead of pre-computing
ODE trajectories from the teacher, this adds Gaussian noise to real data at
inhomogeneous block-wise timesteps and trains the student to denoise with L2 loss.
No teacher model or ODE generation needed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, TYPE_CHECKING, Callable
from functools import partial

import torch
import torch.nn.functional as F

from lipforcing.methods.knowledge_distillation.KD import KDModel
from lipforcing.methods.distribution_matching.causvid import CausVidModel
import lipforcing.utils.logging_utils as logger

if TYPE_CHECKING:
    from lipforcing.configs.config import BaseModelConfig as ModelConfig


class OmniAvatarDiffusionForcingModel(KDModel):
    """Diffusion Forcing on real data — alternative to ODE KD for Stage 1.

    Adds noise to real data at inhomogeneous block-wise timesteps.
    Student denoises -> L2 loss vs clean data. No teacher ODE needed.

    Inheritance: OmniAvatarDiffusionForcingModel -> KDModel -> FastGenModel
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build_model(self):
        """Build model and optionally load VAE for visual logging."""
        super().build_model()
        # FastGenModel.build_model (model.py:260) unconditionally does
        # `self.net.train().requires_grad_(True)` after instantiating the
        # network, which destroys the LoRA freeze that PEFT's
        # inject_adapter_in_model set up inside _load_weights.  Re-apply
        # the freeze here so the optimizer (constructed downstream by
        # init_optimizers, which filters on requires_grad) only sees the
        # intended trainable subset (LoRA A/B + user-listed unfreeze
        # modules).  No-op when merge_lora=True or when PEFT injection
        # didn't actually run (apply_lora_freeze handles both).
        if hasattr(self.net, "apply_lora_freeze"):
            self.net.apply_lora_freeze()

        vae_path = getattr(self.config, "vae_path", "") or ""
        if vae_path and os.path.exists(vae_path):
            self._load_vae(vae_path)

    def init_optimizers(self):
        """Initialize optimizers, with a defensive LoRA freeze re-apply.

        Belt-and-suspenders against any post-build_model code path that
        might also reset requires_grad (e.g., FSDP wrap converting params
        to DTensors and not preserving requires_grad in some PyTorch
        versions).  apply_lora_freeze is idempotent.
        """
        if hasattr(self.net, "apply_lora_freeze"):
            self.net.apply_lora_freeze()
        super().init_optimizers()

    def _load_vae(self, vae_path: str):
        """Load WanVideoVAE for decoding generated samples in wandb visual logging.

        The wandb callback calls model.net.vae.decode(tensor) with a single [B,C,T,H,W] tensor.
        WanVideoVAE.decode expects (hidden_states_list, device). We wrap it for compatibility.
        """
        from OmniAvatar.models.wan_video_vae import WanVideoVAE

        # Load VAE weights — checkpoint keys lack "model." prefix, use converter
        raw_vae = WanVideoVAE(z_dim=16)
        vae_state = torch.load(vae_path, map_location="cpu", weights_only=False)
        # Add "model." prefix to match WanVideoVAE's self.model attribute
        if any(k.startswith("encoder.") for k in vae_state):
            vae_state = {f"model.{k}": v for k, v in vae_state.items()}
        raw_vae.load_state_dict(vae_state)
        device_str = f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
        raw_vae = raw_vae.to(device_str).eval()

        # Wrap decode to match wandb callback's expected interface: decode(tensor) -> tensor
        class VAEWrapper:
            def __init__(self, vae, device):
                self._vae = vae
                self._device = device

            def decode(self, x):
                """Decode [B, C, T, H, W] latent to [B, 3, T*4, H*8, W*8] pixel video."""
                with torch.no_grad():
                    # WanVideoVAE.decode expects list of [C,T,H,W] tensors in float32
                    return self._vae.decode([xi.float() for xi in x], self._device)

            def to(self, *args, **kwargs):
                return self

        self.net.vae = VAEWrapper(raw_vae, device_str)
        logger.info(f"Loaded WanVideoVAE from {vae_path} for visual logging")

    # Use CausVidModel's AR sample loop for visualization (chunk-by-chunk with KV cache).
    # Without this, FastGenModel._student_sample_loop processes the entire video as one
    # bidirectional pass, which doesn't reflect actual AR inference behavior.
    _student_sample_loop = CausVidModel._student_sample_loop

    def _build_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build OmniAvatar condition dict from data batch.

        Expected shapes (after collation, with batch dim):
            text_embeds:    [B, 1, 512, 4096] or [B, 512, 4096]
            audio_emb:      [B, 81, audio_dim]
            mask:           [B, H, W] or [H, W]
            masked_video:   [B, 16, T, H, W]
            ref_sequence:   [B, 16, T, H, W] (optional)

        Args:
            data: Batch from OmniAvatarDataset.

        Returns:
            Condition dict for OmniAvatar networks.
        """
        for key in ("real", "text_embeds", "audio_emb", "mask", "masked_video"):
            assert key in data, f"Missing required key '{key}' in data batch"

        real_data = data["real"]
        ref_latent = real_data[:, :, :1, :, :]  # [B, 16, 1, H, W]

        mask = data["mask"]
        if mask.dim() == 3:
            mask = mask[0]

        text_embeds = data["text_embeds"]
        if text_embeds.dim() == 4:
            assert text_embeds.shape[1] == 1, (
                f"text_embeds dim 1 must be 1 for squeeze, got shape {list(text_embeds.shape)}"
            )
            text_embeds = text_embeds.squeeze(1)

        condition = {
            "text_embeds": text_embeds,
            "audio_emb": data["audio_emb"],
            "ref_latent": ref_latent,
            "mask": mask,
            "masked_video": data["masked_video"],
        }
        if "ref_sequence" in data:
            condition["ref_sequence"] = data["ref_sequence"]

        return condition

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        has_vae = hasattr(self.net, "vae")
        if not has_vae:
            logger.debug("No VAE loaded on net — visual logging disabled")
        if has_vae and condition is not None:
            noise = torch.randn_like(gen_data, dtype=self.precision)
            context_noise = getattr(self.config, "context_noise", 0)
            gen_rand_func = partial(
                CausVidModel.generator_fn,
                net=self.net_inference,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                context_noise=context_noise,
                precision_amp=self.precision_amp_infer,
            )
            return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data.detach()}
        return {"gen_rand_train": gen_data.detach()}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Single training step using diffusion forcing on real data.

        Instead of gathering from pre-computed ODE trajectories (as in CausalKDModel),
        this adds Gaussian noise to real data at inhomogeneous block-wise timesteps.
        """
        real_data = data["real"]  # [B, 16, 21, 64, 64]
        condition = self._build_condition(data)

        batch_size, num_frames = real_data.shape[0], real_data.shape[2]
        chunk_size = self.net.chunk_size

        # Sample inhomogeneous block-wise timesteps
        t_inhom, _ = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            device=self.device,
            dtype=real_data.dtype,
        )  # [B, T]

        # Diffusion forcing: add noise to real data at sampled timesteps
        eps = torch.randn_like(real_data)
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # [B, 1, T, 1, 1]
        noisy_data = self.net.noise_scheduler.forward_process(real_data, eps, t_inhom_expanded)

        # Student denoise
        gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

        # L2 loss
        loss = 0.5 * F.mse_loss(gen_data, real_data, reduction="mean")

        # Outputs for logging (detached to avoid holding autograd references)
        outputs = self._get_outputs(gen_data.detach(), condition=condition)

        loss_map = {"total_loss": loss, "recon_loss": loss.detach()}
        return loss_map, outputs
