# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) for the 14B causal student — LoRA + selective unfreeze + t769 schedule.

Flattened, self-contained Diffusion-Forcing config for the 14B causal
student (shift=5, LoRA + selective unfreeze, t769 schedule).

Effect: the DF student is only ever trained at the noise levels used at
inference (input on step 1 = t=0.999, input on step 2 = t=0.769),
avoiding a train/test schedule mismatch.

Combines:

1) 14B LoRA + selective unfreeze student:
   - model_size="14B" student
   - pretrained 14B student checkpoint (set via OMNIAVATAR_STUDENT_CKPT_14B)
   - merge_lora=False (PEFT injects LoRA on transformer blocks)
   - unfreeze_modules on the audio path + patch embedding
   - lora_rank=128, lora_alpha=64
   - FSDP + bf16 fwd / fp32 master+optim

2) t769 schedule narrowing:
   - sample_t_cfg.t_list = [0.999, 0.769, 0.0]
   - student_sample_steps = 2
"""

import os


from lipforcing.utils import LazyCall as L

# Methods-level config (defines the attrs Config/ModelConfig classes and the
# default create_config()). We keep importing this — it is NOT an experiment
# chain file.
import lipforcing.configs.methods.config_omniavatar_df as config_df_default

from lipforcing.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
from lipforcing.datasets.omniavatar_dataloader import OmniAvatarDataLoader, create_omniavatar_dataloader


# ---- Paths (override via env vars) ----
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "./OmniAvatar")
DATA_ROOT = os.getenv("OMNIAVATAR_DATA_ROOT", "./data/v2v_training_data")
# Pretrained 1.3B student checkpoint (used by the base network specs below).
# The 14B release loads STUDENT_CKPT_14B instead; this is kept so the 1.3B
# student can be enabled later without restructuring the config.
STUDENT_CKPT_1_3B = os.getenv(
    "OMNIAVATAR_STUDENT_CKPT_1_3B",
    "/path/to/omniavatar_1.3b.pt",
)
DATA_LIST = os.getenv("OMNIAVATAR_DATA_LIST", f"{DATA_ROOT}/train_list.txt")
VAL_LIST = os.getenv("OMNIAVATAR_VAL_LIST", f"{DATA_ROOT}/val_list.txt")
MASK_PATH = os.getenv(
    "MASK_PATH",
    "/path/to/mask.png",
)
VAE_PATH = os.getenv(
    "OMNIAVATAR_VAE_PATH",
    os.path.join(OMNIAVATAR_ROOT, "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
)

# Pretrained 14B student checkpoint. Override via OMNIAVATAR_STUDENT_CKPT_14B
# to use a different checkpoint.
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

# Submodules to keep fully trainable alongside LoRA on the transformer blocks.
# Paths are dotted, relative to the CausalOmniAvatarWan instance (so they
# include the "_core." prefix where the actual modules live).
DEFAULT_UNFREEZE_MODULES = [
    "_core.audio_proj",
    "_core.audio_cond_projs",
    "_core.patch_embedding",
]


# ---- Student network config (1.3B base — overridden to 14B below) ----
CausalOmniAvatar_V2V_1_3B_Config: dict = L(CausalOmniAvatarWan)(
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
    use_dynamic_rope=False,
    stochastic_attn_configs=[
        {"local_attn_size": 7,  "sink_size": 1, "weight": 0.2},   # sink=1, window=6
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.2},   # sink=1, window=9
        {"local_attn_size": 13, "sink_size": 1, "weight": 0.2},   # sink=1, window=12
        {"local_attn_size": 9,  "sink_size": 3, "weight": 0.2},   # sink=3, window=6
        {"local_attn_size": 12, "sink_size": 3, "weight": 0.2},   # sink=3, window=9
    ],
)


def create_config():
    # ================================================================== #
    # Base: methods-level OmniAvatar Diffusion Forcing config             #
    # ================================================================== #
    config = config_df_default.create_config()

    # ================================================================== #
    # DF (shift=5) overrides: LR, precision                               #
    # ================================================================== #
    # Learning rate
    config.model.net_optimizer.lr = 1e-5

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 512x512 @ 81 frames -> latent [16, 21, 64, 64]
    config.model.input_shape = [16, 21, 64, 64]

    # Student network
    config.model.net = CausalOmniAvatar_V2V_1_3B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Timestep schedule — shift=5.0 matches OmniAvatar's default scheduler
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Diffusion forcing settings
    config.model.student_sample_steps = 4

    # Dataloader
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=DATA_LIST,
        latentsync_mask_path=MASK_PATH,
        batch_size=2,
        num_workers=4,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
        load_ode_path=False,
    )

    # Validation dataloader — 10 fixed samples, finite iterator, batch_size=1
    config.dataloader_val = L(create_omniavatar_dataloader)(
        data_list_path=VAL_LIST,
        latentsync_mask_path=MASK_PATH,
        batch_size=1,
        num_workers=2,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
        load_ode_path=False,
    )

    # VAE for visual logging (decodes latents to video for wandb)
    config.model.vae_path = VAE_PATH

    # ---- Optional: on-the-fly preprocessing (opt-in via env) ----
    # When OMNIAVATAR_ON_THE_FLY=1, samples lacking precomputed .pt are encoded
    # from raw (sub_clip.mp4 + audio.wav + prompt.txt) by frozen encoders in the
    # MAIN training process; encoded tensors are optionally cached for reuse. The
    # precompute fast path is unchanged, and with the env unset this block is a
    # no-op so the precompute-based data path is used unchanged.
    if os.getenv("OMNIAVATAR_ON_THE_FLY", "0") == "1":
        config.dataloader_train.on_the_fly = True
        config.dataloader_train.cache_encoded = os.getenv("OMNIAVATAR_CACHE_ENCODED", "1") == "1"
        config.dataloader_train.cache_dir = os.getenv("OMNIAVATAR_CACHE_DIR", None)
        config.dataloader_train.vae_path = VAE_PATH
        config.dataloader_train.wav2vec_path = os.getenv("OMNIAVATAR_WAV2VEC_PATH", None)
        config.dataloader_train.text_encoder_path = os.getenv("OMNIAVATAR_TEXT_ENCODER_PATH", None)

    # Training (5K steps — matches the released DF run and paper App. D.1)
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 500
    config.trainer.skip_initial_validation = True
    config.trainer.callbacks.wandb.sample_logging_iter = 500

    config.log_config.group = "stage1_df"
    config.log_config.name = "df_14b_lora_t769"

    # ================================================================== #
    # Switch student to 14B + FSDP                                        #
    # ================================================================== #
    # ---- Switch student to 14B (omit this block for the 1.3B base student) ----
    config.model.net.model_size = "14B"
    config.model.net.base_model_paths = WAN_14B_BASE
    config.model.net.omniavatar_ckpt_path = STUDENT_CKPT_14B
    # The 14B teacher uses merge_lora=True to fuse the adapter into the base
    # before training. Mirror it for the 14B student so DF starts from the
    # fused state, not a base+LoRA stacked state.
    config.model.net.merge_lora = True

    # ---- DDP -> FSDP ----
    config.trainer.ddp = False
    config.trainer.fsdp = True
    config.trainer.fsdp_min_num_params = int(1e8)
    config.trainer.fsdp_cpu_offload = False
    config.trainer.fsdp_sharding_group_size = None  # default = world_size
    # Mixed-precision FSDP: bf16 fwd/bwd, fp32 master+optim.
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"
    # Meta-init disabled (RoPE Python-attr issue at 14B DF).
    config.model.fsdp_meta_init = False

    # ---- Effective batch 16 = 2/GPU * 4 GPUs * grad_accum 2 (matches train_stage1_df.sh) ----
    config.trainer.grad_accum_rounds = 2

    # ================================================================== #
    # Full fine-tune -> LoRA + selective unfreeze                         #
    # ================================================================== #
    # ---- Switch from full FT to LoRA + selective unfreeze ----
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = DEFAULT_UNFREEZE_MODULES

    # LoRA hyperparameters. Match the V2V adapter we're loading from
    # (rank=128, alpha=64).
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # ================================================================== #
    # t769 2-step schedule                                                #
    # ================================================================== #
    # 2-step schedule overrides. With student_sample_steps=2, the student
    # is trained on the two intervals (0.999 -> 0.769) and (0.769 -> 0.0)
    # — the exact intervals SF t769 inference uses.
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]
    config.model.student_sample_steps = 2

    return config


config = create_config()
