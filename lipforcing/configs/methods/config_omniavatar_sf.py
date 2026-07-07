# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for OmniAvatar Self-Forcing distillation."""

import attrs
from omegaconf import DictConfig

from lipforcing.utils import LazyCall as L
from typing import Optional

from lipforcing.configs.methods.config_self_forcing import (
    Config as SFConfig,
    ModelConfig as SFModelConfig,
)
from lipforcing.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
from lipforcing.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
    EMA_CALLBACK,
)


@attrs.define(slots=False)
class RewardConfig:
    """Config for the Re-DMD reward scorer (SyncNet-v2 sync-C)."""
    enabled: bool = True
    checkpoint_path: str = ""
    input_fps: float = 25.0
    audio_sample_rate: int = 16000
    vshift: int = 15
    # Opt-in TAEW decoder. Default "vae" preserves WanVideoVAE.decode behavior.
    # When "taew", the Re-DMD model loads a TAEHVDecoderWrapper from
    # taew_checkpoint_path and uses it in place of self.net.vae for the
    # reward-path pixel decode.
    decoder_kind: str = "vae"
    taew_checkpoint_path: str = ""


@attrs.define(slots=False)
class OmniAvatarModelConfig(SFModelConfig):
    # Separate fake_score config (allows 1.3B fake_score with 14B teacher)
    fake_score: Optional[DictConfig] = None

    # Wan VAE path for decoding validation videos in the wandb callback.
    # Logging-only; declared here so it survives config serialization.
    vae_path: str = ""

    # Re-DMD reward config. None = reward disabled (vanilla SF).
    reward: Optional[RewardConfig] = None
    reward_beta: float = 0.25
    center_reward: bool = False
    clamp_reward: Optional[list] = None

    # How to combine per-sample reward weights with per-sample VSD losses.
    #   - "per_sample" (default): mean_i(exp(beta*r_i) * L_i). Per-sample
    #       coupling without normalization. Preserves reward magnitude as
    #       a loss-scale signal.
    #   - "self_normalized": sum_i(exp(beta*r_i) * L_i) / sum_j(exp(beta*r_j)).
    #       Self-normalized importance sampling (paper's Z(c) partition).
    #       Shift-invariant; loss bounded in [min(L), max(L)]. Better for
    #       additive combination with GAN/other aux losses.
    #   - "legacy_batch_mean": mean_i(exp(beta*r_i)) * mean_i(L_i). Batch-mean
    #       form that decouples reward from loss at batch > 1; not recommended.
    reward_weighting_mode: str = "per_sample"

    # Diagnostic video dump at every generator step (rank 0 only).
    save_reward_debug_video: bool = False
    reward_debug_dir: str = "logs/redmd_debug"


@attrs.define(slots=False)
class Config(SFConfig):
    model: OmniAvatarModelConfig = attrs.field(factory=OmniAvatarModelConfig)
    model_class: DictConfig = L(OmniAvatarSelfForcingModel)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **EMA_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
