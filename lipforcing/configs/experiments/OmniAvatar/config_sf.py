# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF (Re-DMD beta=2 + TAEW) for the 14B causal student + 14B fake_score,
LoRA + selective unfreeze, t769 schedule, matched critic LR 2e-6.

Self-contained Self-Forcing experiment config for the 14B causal student +
14B critic (Re-DMD, beta=2, TAEW decoder, t769 2-step schedule).

Combines:
- 14B student (causal) + 14B fake_score (bidirectional), both LoRA +
  selective unfreeze, initialized from a pretrained 14B audio adapter checkpoint.
- 14B teacher = OmniAvatar-LS (the lip-sync-finetuned OmniAvatar teacher,
  paper App. B.2; bidirectional, frozen, merge_lora=True).
- Sliding-window attention: sink=1 + window=7, dynamic RoPE.
- Re-DMD reward path (SyncNet-v2 sync-C), beta=2, TAEW decoder.
- t769 2-step schedule: t_list=[0.999, 0.769, 0.0], student_sample_steps=2.
- Effective batch 16 = 1/GPU * 4 GPUs * grad_accum=4; FSDP bf16/fp32.
- Critic (fake_score) LR matched to the student at 2e-6 (paper D.1).
"""

import os


from lipforcing.utils import LazyCall as L

# Methods-level configs (define the attrs Config/ModelConfig classes, network
# classes, RewardConfig, and the Re-DMD model class). These are NOT experiment
# chain files and are kept.
import lipforcing.configs.methods.config_omniavatar_sf as config_sf_default
from lipforcing.configs.methods.config_omniavatar_sf import RewardConfig

from lipforcing.networks.OmniAvatar.network import OmniAvatarWan
from lipforcing.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
from lipforcing.datasets.omniavatar_dataloader import OmniAvatarDataLoader, create_omniavatar_dataloader
from lipforcing.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD


# ---- Paths (override via CLI or env) ----
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "./OmniAvatar")
DATA_ROOT = os.getenv("OMNIAVATAR_DATA_ROOT", "./data/v2v_training_data")
TEACHER_CKPT = os.getenv(
    "OMNIAVATAR_TEACHER_CKPT",
    "/path/to/omniavatar_teacher.pt",
)
# Pretrained 1.3B student checkpoint (used by the base network specs below).
# The 14B release loads STUDENT_CKPT_14B instead; this is kept so the 1.3B
# student can be enabled later without restructuring the config.
STUDENT_CKPT_1_3B = os.getenv(
    "OMNIAVATAR_STUDENT_CKPT_1_3B",
    "/path/to/omniavatar_1.3b.pt",
)

# Initial 14B checkpoint (same as the teacher): provides initial LoRA values
# plus audio/patch-embedding weights for both the student and the fake_score
# before training.
STUDENT_CKPT_14B = os.getenv(
    "OMNIAVATAR_STUDENT_CKPT_14B",
    "/path/to/omniavatar_14b_init.pt",
)

# Directory holding the base Wan2.1-T2V-14B diffusion safetensors shards.
WAN_BASE_DIR = os.getenv(
    "OMNIAVATAR_WAN_BASE", f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B"
)
WAN_14B_BASE = ",".join(
    f"{WAN_BASE_DIR}/diffusion_pytorch_model-{i:05d}-of-00006.safetensors"
    for i in range(1, 7)
)

# Submodule paths (relative to each network) to keep fully trainable
# alongside the LoRA A/B matrices on the transformer blocks. Different
# prefix between the causal and bidirectional classes:
#   - CausalOmniAvatarWan (student): WanModel lives at self._core
#   - OmniAvatarWan (fake_score):    WanModel lives at self.model
STUDENT_UNFREEZE = [
    "_core.audio_proj",
    "_core.audio_cond_projs",
    "_core.patch_embedding",
]
FAKE_SCORE_UNFREEZE = [
    "model.audio_proj",
    "model.audio_cond_projs",
    "model.patch_embedding",
]

# Reward checkpoints: SyncNet scorer + TAEW decoder for the reward-path pixel decode.
SYNCNET_CKPT = os.getenv("SYNCNET_CKPT", "/path/to/syncnet_v2.model")
TAEW_CKPT = os.getenv("TAEW_CKPT", "/path/to/taew2_1.pth")

# Critic (fake_score) LR: matched to the student LR (paper D.1).
CRITIC_LR = 2e-6
RUN_NAME = "sf_14b_lora_t769"


# ---- Network configs ----
OmniAvatar_V2V_14B_Teacher: dict = L(OmniAvatarWan)(
    model_size="14B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    base_model_paths=WAN_14B_BASE,
    omniavatar_ckpt_path=TEACHER_CKPT,
    merge_lora=True,
    net_pred_type="flow",
    schedule_type="rf",
)

OmniAvatar_V2V_1_3B_FakeScore: dict = L(OmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT_1_3B,
    merge_lora=False,  # Fake score is trainable, keep LoRA separate
    net_pred_type="flow",
    schedule_type="rf",
)

CausalOmniAvatar_V2V_1_3B_Student: dict = L(CausalOmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    chunk_size=3,
    total_num_frames=21,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT_1_3B,
    net_pred_type="flow",
    schedule_type="rf",
    # Sliding window attention for AR rollout.
    # Defaults: full causal (no window constraint during SF).
    local_attn_size=-1,
    sink_size=0,
    use_dynamic_rope=True,
)


def create_config():
    # ================================================================== #
    # Base: methods-level OmniAvatar Self-Forcing config                  #
    # ================================================================== #
    config = config_sf_default.create_config()

    # ================================================================== #
    # Self-Forcing overrides: LR, optimizer, FSDP                         #
    # ================================================================== #
    # Learning rates and optimizer (Adam beta1=0.0, as used by Self-Forcing)
    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.betas = (0.0, 0.999)
    config.model.fake_score_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.betas = (0.0, 0.999)

    # Multi-GPU: FSDP required (DDP OOMs on student update at ~79GB/GPU)
    config.trainer.fsdp = True

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 512x512 @ 81 frames -> latent [16, 21, 64, 64]
    config.model.input_shape = [16, 21, 64, 64]
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 4.5
    # Sync-Window DMD (SW-DMD; paper Sec. 4.3 / Eq. 6): teacher CFG is gated to the
    # sync window. t_lo/t_hi = the shifted-timestep band [0.556, 0.882] = teacher ODE
    # steps j in [20, 40] (the sync-favoring band from the trajectory analysis).
    config.model.sync_window_cfg.enabled = False  # base default; enabled in the sliding-window block below
    config.model.sync_window_cfg.t_lo = 0.556
    config.model.sync_window_cfg.t_hi = 0.882

    # Networks (base specs): 14B teacher + 1.3B student + 1.3B fake_score.
    # net and fake_score are switched to 14B in the 14B block below.
    config.model.net = CausalOmniAvatar_V2V_1_3B_Student
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = OmniAvatar_V2V_14B_Teacher  # OmniAvatar-LS (paper App. B.2)
    config.model.fake_score_net = OmniAvatar_V2V_1_3B_FakeScore

    # GAN disabled by default.
    config.model.gan_loss_weight_gen = 0
    config.model.student_update_freq = 5  # 1:5 ratio

    # Student weights: do NOT copy 14B teacher weights onto 1.3B student.
    config.model.load_student_weights = False
    # Load the DF-initialized student from the Stage-1 checkpoint. This DF init
    # must match the sliding-window (sink=1, window=7) architecture set below.
    config.trainer.checkpointer.pretrained_ckpt_path = os.getenv(
        "OMNIAVATAR_DF_CKPT",
        "/path/to/df_init_checkpoint.pth",
    )
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}

    # Timestep schedule — shift=5.0 matches OmniAvatar's inference scheduler
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Self-Forcing specific
    config.model.enable_gradient_in_rollout = True
    config.model.start_gradient_frame = 0
    config.model.same_step_across_blocks = True
    config.model.context_noise = 0.0

    # Dataloader (OmniAvatarDataLoader provides infinite iteration)
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=os.getenv("OMNIAVATAR_DATA_LIST", f"{DATA_ROOT}/train_list.txt"),
        latentsync_mask_path=os.getenv(
            "MASK_PATH",
            "/path/to/mask.png",
        ),
        batch_size=8,
        num_workers=2,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
    )

    # Validation dataloader — 10 fixed samples, finite iterator, batch_size=1
    VAL_LIST = os.getenv("OMNIAVATAR_VAL_LIST", f"{DATA_ROOT}/val_list.txt")
    VAE_PATH = os.getenv(
        "OMNIAVATAR_VAE_PATH",
        os.path.join(OMNIAVATAR_ROOT, "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
    )
    config.dataloader_val = L(create_omniavatar_dataloader)(
        data_list_path=VAL_LIST,
        latentsync_mask_path=os.getenv(
            "MASK_PATH",
            "/path/to/mask.png",
        ),
        batch_size=1,
        num_workers=2,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
        load_ode_path=False,
    )
    config.model.vae_path = VAE_PATH

    # ---- Optional: on-the-fly preprocessing (opt-in via env) ----
    # When OMNIAVATAR_ON_THE_FLY=1, samples lacking precomputed .pt are encoded
    # from raw (sub_clip.mp4 + audio.wav + prompt.txt) by frozen encoders in the
    # MAIN training process; encoded tensors are optionally cached for reuse. The
    # precompute fast path is used by default, and with the env unset this block
    # is a no-op.
    if os.getenv("OMNIAVATAR_ON_THE_FLY", "0") == "1":
        config.dataloader_train.on_the_fly = True
        config.dataloader_train.cache_encoded = os.getenv("OMNIAVATAR_CACHE_ENCODED", "1") == "1"
        config.dataloader_train.cache_dir = os.getenv("OMNIAVATAR_CACHE_DIR", None)
        config.dataloader_train.vae_path = VAE_PATH
        config.dataloader_train.wav2vec_path = os.getenv("OMNIAVATAR_WAV2VEC_PATH", None)
        config.dataloader_train.text_encoder_path = os.getenv("OMNIAVATAR_TEXT_ENCODER_PATH", None)

    # Training — 1.3B base: bs=8, grad_accum=2 -> eff batch 64 (14B override below = eff 16)
    config.trainer.grad_accum_rounds = 2
    config.trainer.max_iter = 600  # released 14B checkpoint = step 600 (paper App. D.1)
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 100
    config.trainer.validation_iter = 100
    config.trainer.skip_initial_validation = True

    # Wandb sample logging (video generation) every 100 steps
    config.trainer.callbacks.wandb.sample_logging_iter = 100
    config.trainer.callbacks.wandb.fps = 25  # OmniAvatar is 25 fps
    config.trainer.callbacks.wandb.syncnet_checkpoint_path = SYNCNET_CKPT

    config.log_config.group = "stage2_sf"
    config.log_config.wandb_entity = ""

    # ================================================================== #
    # Sliding-window attention (sink=1, window=7)                         #
    # ================================================================== #
    # Sliding window: 1 sink + 6 rolling = 7 total visible frames
    config.model.net.local_attn_size = 7
    config.model.net.sink_size = 1
    config.model.net.use_dynamic_rope = True

    # 2-step distillation: t_list[0] and t_list[2] from the 4-step schedule
    config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]
    config.model.student_sample_steps = 2

    # Enable Sync-Window DMD (SW-DMD) for this experiment
    config.model.sync_window_cfg.enabled = True

    # ================================================================== #
    # Re-DMD reward weighting (sync-C, beta=2)                            #
    # ================================================================== #
    # Switch model class to the Re-DMD variant.
    config.model_class._target_ = OmniAvatarSelfForcingReDMD

    # Reward sub-config (RewardConfig attrs class survives OmegaConf serialization)
    config.model.reward = RewardConfig(
        enabled=True,
        checkpoint_path=SYNCNET_CKPT,
        input_fps=25.0,
        audio_sample_rate=16000,
        vshift=15,
    )

    # Top-level reward knobs (read by _apply_reward_weighting)
    config.model.reward_beta = 2
    config.model.center_reward = False
    config.model.clamp_reward = None

    # VAE path (required for reward decode in _decode_gen_to_pixels)
    assert getattr(config.model, "vae_path", "") != "", (
        "config.model.vae_path must be set for Re-DMD — the reward path VAE-decodes "
        "the generator output to pixels. Either set OMNIAVATAR_VAE_PATH env or "
        "ensure the default OmniAvatar install path is present."
    )

    # Data: raw waveform loading (required for audio_waveform in batch)
    config.dataloader_train.load_raw_audio = True
    config.dataloader_train.raw_audio_sample_rate = 16000
    config.dataloader_train.raw_audio_num_frames = 81
    config.dataloader_train.raw_audio_fps = 25.0

    # ================================================================== #
    # TAEW decoder for reward-path pixel decode                           #
    # ================================================================== #
    config.model.reward.decoder_kind = "taew"
    config.model.reward.taew_checkpoint_path = TAEW_CKPT

    # ================================================================== #
    # 14B student + critic: LoRA, t769 schedule                           #
    # ================================================================== #
    # ---- Student: 1.3B causal -> 14B causal LoRA (omit to keep the 1.3B student) ----
    config.model.net.model_size = "14B"
    config.model.net.base_model_paths = WAN_14B_BASE
    config.model.net.omniavatar_ckpt_path = STUDENT_CKPT_14B
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = STUDENT_UNFREEZE
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # ---- Fake_score: 1.3B bidirectional -> 14B bidirectional LoRA ----
    config.model.fake_score_net.model_size = "14B"
    config.model.fake_score_net.base_model_paths = WAN_14B_BASE
    config.model.fake_score_net.omniavatar_ckpt_path = STUDENT_CKPT_14B
    config.model.fake_score_net.merge_lora = False
    config.model.fake_score_net.unfreeze_modules = FAKE_SCORE_UNFREEZE
    config.model.fake_score_net.lora_rank = 128
    config.model.fake_score_net.lora_alpha = 64

    # ---- Teacher: stays 14B + merge_lora=True (frozen, full state) ----
    # No changes.

    # ---- Schedule: t769 ----
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]
    config.model.student_sample_steps = 2

    # ---- Effective batch 16 = 1/GPU * 4 GPUs * grad_accum=4 ----
    config.dataloader_train.batch_size = 1
    config.trainer.grad_accum_rounds = 4

    # ---- FSDP knobs (mirror 14B DF LoRA setup) ----
    config.trainer.ddp = False
    config.trainer.fsdp = True
    config.trainer.fsdp_min_num_params = int(1e8)
    config.trainer.fsdp_cpu_offload = False
    config.trainer.fsdp_sharding_group_size = None
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"
    # Meta-init enabled (3-network 14B run); RoPE buffers are re-materialized via
    # reset_parameters() on both network classes.
    config.model.fsdp_meta_init = True

    # ================================================================== #
    # Critic LR = student LR (matched, 2e-6)                             #
    # ================================================================== #
    # Keep the student optimizer unchanged; reduce only the critic.
    config.model.fake_score_optimizer.lr = CRITIC_LR

    config.log_config.name = RUN_NAME

    assert abs(config.model.net_optimizer.lr - 2e-6) < 1e-12, (
        f"Expected student LR to stay at 2e-6, got {config.model.net_optimizer.lr}"
    )
    assert abs(config.model.fake_score_optimizer.lr - CRITIC_LR) < 1e-12, (
        f"Expected critic LR {CRITIC_LR}, got {config.model.fake_score_optimizer.lr}"
    )
    assert config.model.sample_t_cfg.t_list == [0.999, 0.769, 0.0], (
        f"Expected t769 schedule, got {config.model.sample_t_cfg.t_list}"
    )
    assert config.model.student_sample_steps == 2, (
        f"Expected 2 student sample steps, got {config.model.student_sample_steps}"
    )

    return config


config = create_config()
