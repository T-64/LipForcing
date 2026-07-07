# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""OmniAvatar Self-Forcing with Re-DMD reward weighting.

Overrides `_student_update_step` to scale the VSD loss by
`exp(beta * sync_c_detached)`, matching the Reward-Forcing formulation
(arXiv:2512.04678):

    L_gen = 0.5 * exp(beta * r) * MSE(gen_latent, (gen_latent - DMD_grad).detach())

where r is a scalar sync-C from SyncNet-v2 (detached, no gradient). The
scorer consumes face-cropped uint8 frames and the audio waveform and returns
a per-sample sync-C reward used to weight the VSD loss.
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

import lipforcing.utils.logging_utils as logger
from lipforcing.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
from lipforcing.methods.reward.sync_c_scorer import SyncCScorer


def _reduce(x: torch.Tensor, op) -> torch.Tensor:
    """Reduce a 0-d tensor across ranks, returning a 0-d tensor.

    No-op when torch.distributed is not initialized (single-rank / unit test
    path). For SUM, divides by world_size to get a mean; for MIN/MAX, leaves
    the value as-is.
    """
    y = x.detach().clone().float().reshape(())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(y, op=op)
        if op == dist.ReduceOp.SUM:
            y = y / dist.get_world_size()
    return y


class OmniAvatarSelfForcingReDMD(OmniAvatarSelfForcingModel):
    """Rewarded variant of OmniAvatar Self-Forcing.

    All Re-DMD-specific logic (reward scorer instantiation, reward-weighting of
    the VSD loss) lives in this subclass. The base class stays unchanged.
    """

    def __init__(self, config):
        # NOTE: don't initialize self.reward_scorer / self._reward_running_mean
        # here — base Model.__init__ calls self.build_model() during super().__init__,
        # and build_model sets these attrs. A post-super `= None` would clobber
        # the scorer back to None and silently disable the reward path.
        super().__init__(config)

    def build_model(self):
        """Build base-class components, then load the SyncCScorer."""
        super().build_model()

        # Initialize per-run state here (runs every time build_model is called)
        self._reward_running_mean: Optional[float] = None

        rcfg = getattr(self.config, "reward", None)
        if rcfg is None or not getattr(rcfg, "enabled", True):
            self.reward_scorer = None
            logger.info("Re-DMD reward disabled — running as vanilla OmniAvatar SF.")
            return

        device_str = (
            f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
        )
        self.reward_scorer = SyncCScorer(
            checkpoint_path=rcfg.checkpoint_path,
            input_fps=getattr(rcfg, "input_fps", 25.0),
            audio_sample_rate=getattr(rcfg, "audio_sample_rate", 16000),
            vshift=getattr(rcfg, "vshift", 15),
            device=device_str,
            dtype=torch.float32,
        )
        logger.info(
            f"SyncCScorer loaded: beta={self.config.reward_beta}, "
            f"vshift={getattr(rcfg, 'vshift', 15)}, ckpt={rcfg.checkpoint_path}"
        )

        # Opt-in TAEW decoder for the reward path.
        self._maybe_init_taew_decoder()

    def _maybe_init_taew_decoder(self):
        """Build self._taew_decoder if config.reward.decoder_kind == 'taew'.

        Otherwise leave it as None so _decode_gen_to_pixels falls back to
        self.net.vae.decode() — default behavior, bitwise-identical to
        what the VAE baseline runs.
        """
        rcfg = getattr(self.config, "reward", None)
        kind = getattr(rcfg, "decoder_kind", "vae") if rcfg is not None else "vae"
        if kind == "vae":
            self._taew_decoder = None
            return
        if kind == "taew":
            ckpt = getattr(rcfg, "taew_checkpoint_path", "")
            if not ckpt:
                raise ValueError(
                    "config.reward.decoder_kind='taew' requires "
                    "config.reward.taew_checkpoint_path to be set."
                )
            from lipforcing.methods.reward.taehv_decoder import TAEHVDecoderWrapper
            device_str = (
                f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
            )
            self._taew_decoder = TAEHVDecoderWrapper(checkpoint_path=ckpt, device=device_str)
            logger.info(f"TAEHVDecoderWrapper loaded for reward path: ckpt={ckpt}")
            return
        raise ValueError(
            f"unknown config.reward.decoder_kind={kind!r} "
            f"(expected 'vae' or 'taew')"
        )

    def _apply_reward_weighting(
        self,
        vsd_loss: torch.Tensor,
        videos: Any,
        audios: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply the per-sample reward weights to per-sample vsd_loss.

        `vsd_loss` MUST be a [B] tensor. The combination is controlled by
        `config.reward_weighting_mode`:

            "per_sample" (default):
                weighted = mean_i(w_i * vsd_loss_i)
                Per-sample coupling without partition-function normalization.
                Reward magnitude contributes to loss scale; at high beta or
                non-zero-centered r, loss can grow with exp(beta * mean_r).

            "self_normalized":
                weighted = sum_i(w_i * vsd_loss_i) / sum_j(w_j)
                Self-normalized importance-sampling estimator. Implements the
                Z(c) normalization from the Re-DMD paper's Eq. 11. Shift-
                invariant: `r_i -> r_i + c` cancels in numerator and
                denominator. Loss bounded in [min(L), max(L)]. Preferred
                when mixing with auxiliary losses (GAN, R1) whose weights
                were tuned at unit-magnitude loss scale.

            "legacy_batch_mean":
                weighted = mean_i(w_i) * mean_i(L_i)
                Original Reward-Forcing-style batch-mean collapse. At batch
                size 1 equivalent to `per_sample`; at batch > 1 decouples
                reward from loss (each sample's reward scales the entire
                batch-mean loss equally). Provided for direct comparison
                with upstream; not recommended for batch > 1 training.

        Returns:
            (weighted_loss, log_map). `weighted_loss` is a scalar.
        """
        with torch.no_grad():
            reward = self.reward_scorer.reward_from_frames(videos, audios)
        sync_c = reward["sync_c"].detach().float()  # [B]

        assert vsd_loss.dim() == 1 and vsd_loss.shape == sync_c.shape, (
            f"_apply_reward_weighting expects per-sample vsd_loss shape "
            f"{tuple(sync_c.shape)}, got {tuple(vsd_loss.shape)}"
        )

        beta = float(self.config.reward_beta)

        # Optional centering (EMA subtraction — keeps mean weight ≈ 1)
        if getattr(self.config, "center_reward", False):
            ema_alpha = 0.9
            batch_mean = sync_c.mean().item()
            if self._reward_running_mean is None:
                self._reward_running_mean = batch_mean
            else:
                self._reward_running_mean = (
                    ema_alpha * self._reward_running_mean
                    + (1.0 - ema_alpha) * batch_mean
                )
            sync_c = sync_c - self._reward_running_mean

        # Optional clamping (bounds exp(beta * r))
        clamp = getattr(self.config, "clamp_reward", None)
        if clamp is not None:
            sync_c = sync_c.clamp(clamp[0], clamp[1])

        weight = torch.exp(beta * sync_c)  # [B], detached (sync_c is detached)
        # Cast vsd_loss to the same dtype/device as weight so element-wise
        # multiplication doesn't silently upcast/downcast away from the
        # student's compute dtype. vsd_loss shape matches weight.
        vsd_per_sample = vsd_loss.to(dtype=weight.dtype, device=weight.device)

        mode = getattr(self.config, "reward_weighting_mode", "per_sample")
        if mode == "per_sample":
            # mean_i(w_i * L_i). Per-sample coupling; reward magnitude
            # contributes to loss scale.
            weighted = (weight * vsd_per_sample).mean()
        elif mode == "self_normalized":
            # sum_i(w_i * L_i) / sum_j(w_j). Self-normalized IS estimator;
            # shift-invariant and bounded in [min(L), max(L)].
            Z = weight.sum().detach() + 1e-8
            weighted = (weight * vsd_per_sample).sum() / Z
        elif mode == "legacy_batch_mean":
            # mean_i(w_i) * mean_i(L_i). Original Reward-Forcing batch-mean
            # collapse — at batch > 1 reward and loss become decoupled.
            weighted = weight.mean() * vsd_per_sample.mean()
        else:
            raise ValueError(
                f"unknown reward_weighting_mode={mode!r} "
                f"(expected 'per_sample', 'self_normalized', or 'legacy_batch_mean')"
            )

        sync_c_mean_r = _reduce(sync_c.mean(), dist.ReduceOp.SUM)
        sync_c_min_r  = _reduce(sync_c.min(),  dist.ReduceOp.MIN)
        sync_c_max_r  = _reduce(sync_c.max(),  dist.ReduceOp.MAX)
        weight_mean_r = _reduce(weight.mean(), dist.ReduceOp.SUM)
        weight_min_r  = _reduce(weight.min(),  dist.ReduceOp.MIN)
        weight_max_r  = _reduce(weight.max(),  dist.ReduceOp.MAX)

        log_map = {
            "reward_sync_c_mean": float(sync_c_mean_r.item()),
            "reward_sync_c_min": float(sync_c_min_r.item()),
            "reward_sync_c_max": float(sync_c_max_r.item()),
            "reward_weight_mean": float(weight_mean_r.item()),
            "reward_weight_min": float(weight_min_r.item()),
            "reward_weight_max": float(weight_max_r.item()),
            "vsd_loss_unweighted": float(vsd_per_sample.detach().float().mean().item()),
            "vsd_loss_weighted": float(weighted.detach().item()),
        }

        # Per-rank first-sample values via all_gather (gives rank-level visibility in wandb)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            local_sync_c = sync_c[0:1].detach().float()    # [1] — rank-local first sample
            local_weight = weight[0:1].detach().float()    # [1]
            gathered_sync_c = [torch.zeros_like(local_sync_c) for _ in range(world_size)]
            gathered_weight = [torch.zeros_like(local_weight) for _ in range(world_size)]
            dist.all_gather(gathered_sync_c, local_sync_c)
            dist.all_gather(gathered_weight, local_weight)
            for r in range(world_size):
                log_map[f"reward_sync_c_r{r}"] = float(gathered_sync_c[r].item())
                log_map[f"reward_weight_r{r}"] = float(gathered_weight[r].item())
        else:
            # Single-rank fallback (unit tests and 1-GPU runs) — log all batch items as r0..rN-1
            for b in range(sync_c.shape[0]):
                log_map[f"reward_sync_c_r{b}"] = float(sync_c[b].item())
                log_map[f"reward_weight_r{b}"] = float(weight[b].item())

        return weighted, log_map

    # ------------------------------------------------------------------
    # single_train_step override: stash iteration for _maybe_save_debug_video
    # ------------------------------------------------------------------

    def single_train_step(self, data: Dict[str, Any], iteration: int):
        self._current_iteration = iteration
        return super().single_train_step(data, iteration)

    # ------------------------------------------------------------------
    # Optional MP4 debug save
    # ------------------------------------------------------------------

    def _maybe_save_debug_video(self, pixels: torch.Tensor, iteration: int) -> None:
        """Save the first sample's decoded video to disk for manual inspection.

        Only fires on rank 0, only when config.save_reward_debug_video is True.
        Writes an MP4 at config.reward_debug_dir/gen_iter{iteration:06d}.mp4.
        """
        if not getattr(self.config, "save_reward_debug_video", False):
            return
        # Rank-0 only
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        debug_dir = getattr(self.config, "reward_debug_dir", "logs/redmd_debug")
        os.makedirs(debug_dir, exist_ok=True)

        # pixels: [B, 3, T_pix, H, W] in [-1, 1]
        sample = pixels[0].clamp(-1.0, 1.0)             # [3, T, H, W]
        sample = ((sample + 1.0) * 127.5).to(torch.uint8)  # uint8 [3, T, H, W]
        sample = sample.permute(1, 2, 3, 0).contiguous().cpu()  # [T, H, W, 3] for torchvision.io

        out_path = os.path.join(debug_dir, f"gen_iter{iteration:06d}.mp4")
        try:
            from torchvision.io import write_video
            write_video(out_path, sample, fps=25)
            logger.info(f"[Re-DMD debug] saved decoded video at {out_path} ({tuple(sample.shape)})")
        except Exception as e:
            logger.warning(f"[Re-DMD debug] failed to save debug video: {e}")

    # ------------------------------------------------------------------
    # _student_update_step with VAE decode + sync-C reward
    # ------------------------------------------------------------------

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ):
        """Re-DMD override: standard VSD loss, then scaled by exp(beta * sync_c).

        When ``self.reward_scorer is None`` (reward disabled), falls back to the
        base DMD2 behavior (no VAE decode, no reward keys in loss_map).

        Reproduces DMD2Model._student_update_step inline so we can intervene
        between vsd_loss and total_loss without fragile post-hoc rescaling.

        NOTE: per-rank EMA on ``self._reward_running_mean`` means multi-rank runs
        will have independent running means. For the typical OmniAvatar setup
        (batch_size=1 per GPU, 4 GPUs) this is close enough to a batch mean.
        """
        from lipforcing.methods.common_loss import variational_score_distillation_loss

        # ---- reproduce the DMD2Model._student_update_step body ----
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)

        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        with torch.no_grad():
            fake_score_x0 = self.fake_score(
                perturbed_data, t, condition=condition, fwd_pred_type="x0"
            )

        # When reward is active, compute per-sample losses so reward weights
        # couple to the combined (VSD + GAN) per-sample loss before reduction.
        reward_active = self.reward_scorer is not None and "audio_waveform" in data
        per_sample = "none" if reward_active else "mean"

        teacher_x0, gan_loss_gen = self._compute_teacher_prediction_gan_loss(
            perturbed_data, t, condition=condition, gan_reduction=per_sample,
        )

        if getattr(self.config, "guidance_scale", None) is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed_data, t, teacher_x0, neg_condition=neg_condition
            )

        vsd_loss = variational_score_distillation_loss(
            gen_data, teacher_x0, fake_score_x0, reduction=per_sample,
        )

        # ---- Re-DMD reward weighting on the combined per-sample loss ----
        reward_log: Dict[str, Any] = {}
        if reward_active:
            with torch.no_grad():
                pixels = self._decode_gen_to_pixels(gen_data)           # [B, 3, T_pix, H, W] in [-1, 1]
                self._maybe_save_debug_video(pixels, iteration=getattr(self, "_current_iteration", 0))
                videos_u8 = self._pixels_to_uint8_face_crop(pixels)     # list of B [T_pix, 3, H, W] uint8
                audios = list(data["audio_waveform"].unbind(0))         # list of B [L]
            # Combine per-sample VSD + GAN into a single per-sample loss, then
            # apply reward weighting once. This keeps the VSD/GAN relative weight
            # (config.gan_loss_weight_gen) invariant to reward magnitude:
            #   loss = mean(w_i * (vsd_i + gan_loss_weight_gen * gan_i))
            combined_per_sample = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen
            weighted_loss, reward_log = self._apply_reward_weighting(
                combined_per_sample, videos_u8, audios,
            )
            vsd_loss_scalar = vsd_loss.detach().float().mean()
            gan_loss_gen_scalar = gan_loss_gen.detach().float().mean()
        else:
            weighted_loss = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen
            vsd_loss_scalar = vsd_loss.detach()
            gan_loss_gen_scalar = (
                gan_loss_gen.detach()
                if torch.is_tensor(gan_loss_gen)
                else torch.tensor(float(gan_loss_gen))
            )
            reward_log = {"vsd_loss_unweighted": float(vsd_loss_scalar.item())}

        loss_map: Dict[str, Any] = {
            "total_loss": weighted_loss,
            "vsd_loss": vsd_loss_scalar,
            "vsd_loss_weighted": weighted_loss.detach(),
            "gan_loss_gen": gan_loss_gen_scalar,
            **reward_log,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs

    def _decode_gen_to_pixels(self, gen_latent: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, 16, T_lat, H_lat, W_lat]`` latents to ``[B, 3, T_pix, H_pix, W_pix]``.

        Uses the VAEWrapper loaded by the base class's ``_load_vae``. That wrapper's
        ``decode`` takes a list of per-sample tensors and returns a list of decoded
        pixel tensors in ``[-1, 1]`` range (WanVideoVAE native output range).

        Raises a clear RuntimeError when ``self.net.vae`` is missing, so the user
        knows to set ``config.vae_path`` in the rewarded config.
        """
        # TAEW opt-in path — defined when config.reward.decoder_kind == "taew".
        if getattr(self, "_taew_decoder", None) is not None:
            return self._taew_decoder.decode(
                [gen_latent[b].float() for b in range(gen_latent.shape[0])]
            )
        # Default: Wan 2.1 full VAE.
        if not hasattr(self.net, "vae") or self.net.vae is None:
            raise RuntimeError(
                "Re-DMD needs a VAE for reward decode, but self.net.vae is unset. "
                "Either ensure the base OmniAvatar model loads a VAE, or set "
                "config.reward.decoder_kind='taew' + taew_checkpoint_path."
            )
        decoded = self.net.vae.decode(
            [gen_latent[b].float() for b in range(gen_latent.shape[0])]
        )
        # WanVideoVAE.decode returns a stacked Tensor [B, C, T, H, W],
        # not a list, so guard against both cases.
        if isinstance(decoded, torch.Tensor):
            return decoded
        return torch.stack(decoded, dim=0)

    def _pixels_to_uint8_face_crop(self, pixels: torch.Tensor) -> list:
        """``[B, 3, T_pix, H_pix, W_pix]`` float in ``[-1, 1]`` -> list of B tensors
        each ``[T_pix, 3, H, W]`` uint8.

        Face alignment is already done upstream (pre-aligned training data); we
        only range-map and permute to match the ``[F, 3, H, W]`` uint8 format
        expected by ``SyncCScorer.reward_from_frames``.
        """
        pixels = pixels.clamp(-1.0, 1.0)
        u8 = ((pixels + 1.0) * 127.5).to(torch.uint8)          # [B, 3, T, H, W]
        u8 = u8.permute(0, 2, 1, 3, 4).contiguous()            # [B, T, 3, H, W]
        return [u8[b] for b in range(u8.shape[0])]
