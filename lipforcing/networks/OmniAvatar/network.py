# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bidirectional OmniAvatar wrapper for LipForcing.

Used as both the teacher (frozen 14B) and fake_score (trainable 1.3B DSM) networks
in the Self-Forcing distillation pipeline.  Wraps the standalone WanModel DiT from
``lipforcing.networks.OmniAvatar.wan_model`` behind the ``FastGenNetwork`` abstract
interface so that LipForcing's training loop can call it uniformly.

Weight loading:
    1. Base Wan 2.1 T2V safetensor weights  ->  WanModel (via smart_load_weights)
    2. OmniAvatar LoRA + audio + patch_embedding checkpoint on top
    3. (optional) merge LoRA into base weights for inference
"""

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open

from torch.distributed.fsdp import fully_shard

from lipforcing.networks.network import FastGenNetwork
from lipforcing.networks.noise_schedule import NET_PRED_TYPES
from lipforcing.networks.OmniAvatar.wan_model import WanModel, precompute_freqs_cis_3d
import lipforcing.utils.logging_utils as logger

# ---------------------------------------------------------------------------
# Model size configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "14B": dict(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40),
    "1.3B": dict(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30),
}

# Common constants shared across model sizes
_COMMON_CFG = dict(
    out_dim=16,
    text_dim=4096,
    freq_dim=256,
    eps=1e-6,
    patch_size=(1, 2, 2),
)

# LoRA target modules in OmniAvatar's DiT (same for 14B and 1.3B)
LORA_TARGET_MODULES = ["q", "k", "v", "o", "ffn.0", "ffn.2"]


# ---------------------------------------------------------------------------
# Weight-loading utilities (self-contained, no OmniAvatar repo imports)
# ---------------------------------------------------------------------------

def _load_safetensors(path: str, dtype: torch.dtype = None) -> Dict[str, torch.Tensor]:
    """Load a single safetensors file into a CPU state dict."""
    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if dtype is not None:
                t = t.to(dtype)
            state[k] = t
    return state


def _load_torch_file(path: str, dtype: torch.dtype = None) -> Dict[str, torch.Tensor]:
    """Load a .pt / .pth / .bin checkpoint into a CPU state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if dtype is not None:
        state = {k: (v.to(dtype) if isinstance(v, torch.Tensor) else v) for k, v in state.items()}
    return state


def _load_state_dict(path: str, dtype: torch.dtype = None) -> Dict[str, torch.Tensor]:
    """Dispatch between safetensors and torch checkpoint formats."""
    if path.endswith(".safetensors"):
        return _load_safetensors(path, dtype)
    else:
        return _load_torch_file(path, dtype)


def _smart_load_weights(
    model: nn.Module,
    ckpt_sd: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str]]:
    """Load weights with shape-aware expansion (copy existing, zero-init new channels).

    Mirrors ``OmniAvatar/utils/io_utils.py::smart_load_weights`` — when model param
    shape >= ckpt param shape along every dimension, creates a zero-initialised tensor
    and pastes the ckpt weights into the leading slice.

    Returns:
        (missing_keys, unexpected_keys) from the final ``load_state_dict`` call.
    """
    model_sd = model.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}

    for name, param in model_sd.items():
        if name not in ckpt_sd:
            continue
        ckpt_param = ckpt_sd[name]
        if param.shape == ckpt_param.shape:
            new_sd[name] = ckpt_param
        elif all(p >= c for p, c in zip(param.shape, ckpt_param.shape)):
            logger.info(f"[smart_load] Expanding {name}: ckpt {list(ckpt_param.shape)} -> model {list(param.shape)}")
            new_param = param.clone()
            slices = tuple(slice(0, s) for s in ckpt_param.shape)
            new_param[slices] = ckpt_param
            new_sd[name] = new_param
        else:
            logger.warning(
                f"[smart_load] Skipping {name}: ckpt {list(ckpt_param.shape)} vs model {list(param.shape)}"
            )

    missing, unexpected = model.load_state_dict(new_sd, assign=True, strict=False)
    return missing, unexpected


# ---------------------------------------------------------------------------
# Key-name mapping for original Wan 2.1 T2V -> WanModel
# ---------------------------------------------------------------------------

# The base Wan 2.1 safetensors use *diffusers* naming.  Our WanModel uses the
# *original Wan* naming convention (matching OmniAvatar's DiT).  This mapping
# converts diffusers keys to WanModel keys so we can load the pretrained base.

_DIFFUSERS_TO_WANMODEL_RENAMES = [
    # Timestep / text embeddings
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    ("condition_embedder.time_proj", "time_projection.1"),
    # Head (top-level only, not inside blocks)
    # NOTE: scale_shift_table -> head.modulation is handled specially below for head vs blocks
    ("proj_out", "head.head"),
    # Attention
    ("attn1.to_q", "self_attn.q"),
    ("attn1.to_k", "self_attn.k"),
    ("attn1.to_v", "self_attn.v"),
    ("attn1.to_out.0", "self_attn.o"),
    ("attn1.norm_q", "self_attn.norm_q"),
    ("attn1.norm_k", "self_attn.norm_k"),
    ("attn2.to_q", "cross_attn.q"),
    ("attn2.to_k", "cross_attn.k"),
    ("attn2.to_v", "cross_attn.v"),
    ("attn2.to_out.0", "cross_attn.o"),
    ("attn2.norm_q", "cross_attn.norm_q"),
    ("attn2.norm_k", "cross_attn.norm_k"),
    # I2V extras (for completeness; not used in T2V base)
    ("attn2.add_k_proj", "cross_attn.k_img"),
    ("attn2.add_v_proj", "cross_attn.v_img"),
    ("attn2.norm_added_k", "cross_attn.norm_k_img"),
    # Image embedder
    ("condition_embedder.image_embedder.norm1", "img_emb.proj.0"),
    ("condition_embedder.image_embedder.ff.net.0.proj", "img_emb.proj.1"),
    ("condition_embedder.image_embedder.ff.net.2", "img_emb.proj.3"),
    ("condition_embedder.image_embedder.norm2", "img_emb.proj.4"),
    ("condition_embedder.image_embedder.pos_embed", "img_emb.emb_pos"),
    # FFN  (diffusers -> original naming)
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2", "ffn.2"),
]

def _convert_diffusers_key_to_wanmodel(key: str) -> Optional[str]:
    """Convert a diffusers WanTransformer3DModel key to our WanModel key format.

    Returns None if the key should be skipped (e.g. RoPE buffers, non-persistent).
    """
    # Strip common prefixes
    for prefix in ("transformer.", "model.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]

    # Skip RoPE buffers (recomputed at init)
    if "freqs" in key and ("rope" in key or "freqs_cos" in key or "freqs_sin" in key):
        return None

    # Apply rename mappings (order matters for nested replacements)
    for diffusers_name, wan_name in _DIFFUSERS_TO_WANMODEL_RENAMES:
        if diffusers_name in key:
            key = key.replace(diffusers_name, wan_name)

    # Handle block-level specifics
    if "blocks." in key:
        # Norm swap: diffusers norm2 = original norm3, diffusers norm3 = original norm2
        # In diffusers: norm1=self-attn, norm2=cross-attn(learnable), norm3=FFN
        # In original:  norm1=self-attn, norm2=FFN, norm3=cross-attn(learnable)
        # After the general renames above, norm names are still diffusers-style.
        # Direct swap using regex: match ".normX." where X is 2 or 3
        def _swap_norms(m):
            n = m.group(1)
            return ".norm3." if n == "2" else ".norm2."
        key = re.sub(r"\.norm([23])\.", _swap_norms, key)

        # Block-level scale_shift_table -> modulation
        key = key.replace(".scale_shift_table", ".modulation")

    else:
        # Top-level scale_shift_table -> head.modulation
        if key.startswith("scale_shift_table"):
            key = key.replace("scale_shift_table", "head.modulation")

    return key


def _convert_diffusers_state_dict(
    diffusers_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert an entire diffusers state dict to WanModel key format."""
    converted = {}
    for k, v in diffusers_sd.items():
        new_k = _convert_diffusers_key_to_wanmodel(k)
        if new_k is not None:
            converted[new_k] = v
    return converted


# ---------------------------------------------------------------------------
# OmniAvatar checkpoint key mapping
# ---------------------------------------------------------------------------

def _map_omniavatar_lora_keys(
    ckpt_sd: Dict[str, torch.Tensor],
    use_peft: bool = True,
) -> Dict[str, torch.Tensor]:
    """Map OmniAvatar LoRA checkpoint keys to PEFT-injected model keys.

    OmniAvatar saves ``lora_A.weight`` / ``lora_B.weight``.
    After ``inject_adapter_in_model``, PEFT creates ``lora_A.default.weight``.

    When ``use_peft=False`` (i.e. merge_lora mode), no mapping needed since we
    load raw A/B matrices directly.
    """
    if not use_peft:
        return ckpt_sd

    mapped = {}
    for k, v in ckpt_sd.items():
        new_k = k
        if "lora_A.weight" in k:
            new_k = k.replace("lora_A.weight", "lora_A.default.weight")
        if "lora_B.weight" in k:
            new_k = k.replace("lora_B.weight", "lora_B.default.weight")
        mapped[new_k] = v
    return mapped


# ---------------------------------------------------------------------------
# LoRA merging
# ---------------------------------------------------------------------------

def _merge_lora_into_model(
    model: nn.Module,
    lora_sd: Dict[str, torch.Tensor],
    lora_rank: int,
    lora_alpha: int,
    target_modules: List[str],
) -> None:
    """Merge LoRA weights into the base model in-place.

    For each target module with LoRA adapters A (down) and B (up):
        W_merged = W_base + (alpha / rank) * B @ A

    This modifies model parameters directly so no LoRA overhead at inference.
    """
    scaling = lora_alpha / lora_rank
    merged_count = 0

    # Collect all LoRA pairs
    lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # base_key -> (A, B)
    for k, v in lora_sd.items():
        if "lora_A" in k:
            base_key = k.split(".lora_A")[0]
            if base_key not in lora_pairs:
                lora_pairs[base_key] = [None, None]
            lora_pairs[base_key][0] = v
        elif "lora_B" in k:
            base_key = k.split(".lora_B")[0]
            if base_key not in lora_pairs:
                lora_pairs[base_key] = [None, None]
            lora_pairs[base_key][1] = v

    # Apply merging
    model_sd = dict(model.named_parameters())
    for base_key, (A, B) in lora_pairs.items():
        if A is None or B is None:
            logger.warning(f"[merge_lora] Incomplete LoRA pair for {base_key}, skipping")
            continue

        weight_key = f"{base_key}.weight"
        if weight_key not in model_sd:
            logger.warning(f"[merge_lora] Base weight {weight_key} not found in model, skipping")
            continue

        param = model_sd[weight_key]
        # LoRA: W' = W + scaling * B @ A
        # Do matmul on GPU for speed (14B LoRA merge takes 40+ min on CPU)
        merge_device = torch.device("cuda") if torch.cuda.is_available() else param.device
        delta = scaling * (B.to(device=merge_device, dtype=param.dtype) @ A.to(device=merge_device, dtype=param.dtype))
        param.data.add_(delta.to(param.device))
        merged_count += 1

    logger.info(f"[merge_lora] Merged {merged_count} LoRA pairs with scaling={scaling}")


# ---------------------------------------------------------------------------
# OmniAvatarWan: FastGenNetwork wrapper
# ---------------------------------------------------------------------------

class OmniAvatarWan(FastGenNetwork):
    """Bidirectional OmniAvatar wrapper for teacher and fake_score.

    Wraps the standalone ``WanModel`` DiT behind LipForcing's ``FastGenNetwork``
    interface.  Handles:
      - Model construction from size config
      - Multi-stage weight loading (base Wan 2.1 + OmniAvatar LoRA/audio)
      - V2V conditioning tensor assembly
      - Prediction-type conversion
    """

    def __init__(
        self,
        model_size: str = "1.3B",
        in_dim: int = 49,
        mode: str = "v2v",
        use_audio: bool = True,
        audio_hidden_size: int = 32,
        base_model_paths: Optional[str] = None,
        omniavatar_ckpt_path: Optional[str] = None,
        merge_lora: bool = True,
        lora_rank: int = 128,
        lora_alpha: int = 64,
        unfreeze_modules: Optional[List[str]] = None,  # Used with merge_lora=False; see apply_lora_freeze
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        mask_all_frames: bool = True,
        disable_grad_ckpt: bool = False,
        dtype: str = "bf16",
        expand_audio_checkpoint_scope: bool = False,
        **kwargs,
    ):
        """
        Args:
            model_size: ``"14B"`` or ``"1.3B"``.
            in_dim: Input channels to patch embedding.
                33 for I2V (16 noise + 16 ref + 1 mask),
                49 for V2V (+ 16 masked_video),
                65 for V2V + ref_sequence (+ 16 ref_seq).
            mode: ``"i2v"`` or ``"v2v"``.
            use_audio: Whether to include audio conditioning modules.
            audio_hidden_size: Hidden dim of AudioPack (default 32, same for all sizes).
            base_model_paths: Comma-separated safetensor paths for base Wan 2.1 T2V weights.
                e.g. for 1.3B: ``"path/diffusion_pytorch_model.safetensors"``
                     for 14B:  ``"path/diffusion_pytorch_model-00001-of-00006.safetensors,...,path/diffusion_pytorch_model-00006-of-00006.safetensors"``
                If None, model is randomly initialised (e.g. fresh fake_score).
            omniavatar_ckpt_path: Path to OmniAvatar checkpoint (``.pt``/``.pth``)
                containing LoRA, audio modules, and patch_embedding weights.
                If None, only base weights are loaded (or random init).
            merge_lora: If True, merge LoRA weights into base model. If False, keep
                LoRA weights separate (requires PEFT injection, not yet implemented here).
            lora_rank: LoRA rank (default 128).
            lora_alpha: LoRA alpha (default 64).
            net_pred_type: Network prediction type (default ``"flow"`` for rectified flow).
            schedule_type: Noise schedule type (default ``"rf"``).
            mask_all_frames: If True, apply spatial mask to all frames including frame 0.
                If False, frame 0 is unmasked (reference frame).
            dtype: Default dtype string (``"bf16"``, ``"fp16"``, ``"fp32"``).
            **kwargs: Passed to ``FastGenNetwork.__init__`` (noise schedule kwargs).
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **kwargs)

        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_size '{model_size}'. Choose from {list(MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.in_dim = in_dim
        self.mode = mode
        self.use_audio = use_audio
        self.audio_hidden_size = audio_hidden_size
        self.merge_lora = merge_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        # Paths (relative to self, dotted) of submodules whose parameters
        # should keep requires_grad=True even when PEFT injection has frozen
        # the rest of the base model.  Used with merge_lora=False to enable
        # selective full fine-tuning of specific components alongside LoRA
        # on the transformer blocks.  For OmniAvatarWan (bidirectional), the
        # WanModel lives at self.model, so paths look like "model.audio_proj"
        # rather than the causal class's "_core.audio_proj".  Ignored when
        # merge_lora=True.
        self.unfreeze_modules: List[str] = list(unfreeze_modules) if unfreeze_modules else []
        if self.unfreeze_modules and self.merge_lora:
            logger.warning(
                f"[OmniAvatarWan] unfreeze_modules={self.unfreeze_modules} is set "
                f"but merge_lora=True; the unfreeze list will be ignored. "
                f"Set merge_lora=False to use selective unfreeze with LoRA injection."
            )
        self.mask_all_frames = mask_all_frames

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._default_dtype = dtype_map.get(dtype, torch.bfloat16)

        cfg = MODEL_CONFIGS[model_size]

        # head_dim is stored on self so reset_parameters can recompute
        # self.model.freqs after FSDP's to_empty under fsdp_meta_init=True.
        # WanModel internally also computes head_dim = dim // num_heads in
        # its __init__ to set self.freqs, but doesn't store either dim or
        # num_heads in a way reset_parameters can easily reach.
        self._head_dim = cfg["dim"] // cfg["num_heads"]

        # Build the WanModel DiT
        self.model = WanModel(
            dim=cfg["dim"],
            in_dim=in_dim,
            ffn_dim=cfg["ffn_dim"],
            out_dim=_COMMON_CFG["out_dim"],
            text_dim=_COMMON_CFG["text_dim"],
            freq_dim=_COMMON_CFG["freq_dim"],
            eps=_COMMON_CFG["eps"],
            patch_size=_COMMON_CFG["patch_size"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            use_audio=use_audio,
            audio_hidden_size=audio_hidden_size,
            has_image_input=False,  # OmniAvatar always uses T2V base, not I2V
        )

        # Gradient checkpointing: enabled by default (like T2V's Wan network)
        self._use_gradient_checkpointing = not disable_grad_ckpt
        # Opt-in memory optimization: move the audio-conditioning add INSIDE
        # the checkpoint boundary so its pre-add x is recomputed rather than
        # retained. Off by default — turn on
        # for memory-constrained regimes (GAN on 14B teacher, V2V with 64x64+
        # latents).
        self._expand_audio_checkpoint_scope = expand_audio_checkpoint_scope

        # Load weights (unless we are in meta device context for FSDP).
        #
        # On non-meta-init ranks: full path — load base safetensors + V2V
        # LoRA ckpt, optionally PEFT-inject, apply selective unfreeze, cast.
        #
        # On meta-init ranks: SKIP the disk loads (saves host RAM, the whole
        # point of meta-init) but STILL run PEFT structural injection so the
        # module graph matches rank 0's.  Without this, set_model_state_dict
        # post-FSDP-wrap would fail with "Missing key(s)" because rank 0 has
        # 'q.base_layer.weight' (LoraLinear-wrapped) and ranks 1+ have
        # plain 'q.weight'.  See _inject_peft_structure_for_meta for details.
        if not self._is_in_meta_context():
            self._load_weights(base_model_paths, omniavatar_ckpt_path)
            self.model.to(self._default_dtype)
        else:
            self._inject_peft_structure_for_meta()

    def _inject_peft_structure_for_meta(self) -> None:
        """Apply PEFT injection on meta-init ranks without touching disk.

        Symmetric to ``CausalOmniAvatarWan._inject_peft_structure_for_meta``.
        Mirrors the structural transformation rank 0 performs in
        ``_load_weights``'s ``merge_lora=False`` branch, so all ranks have
        matching state_dict keys before FSDP's broadcast.

        No-op when ``merge_lora=True`` (full FT — no LoRA layers exist).
        """
        if self.merge_lora:
            return

        from peft import LoraConfig, inject_adapter_in_model

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights=True,
            target_modules=LORA_TARGET_MODULES,
        )
        inject_adapter_in_model(lora_config, self.model)
        self._apply_unfreeze(self.unfreeze_modules)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(
        self,
        base_model_paths: Optional[str],
        omniavatar_ckpt_path: Optional[str],
    ) -> None:
        """Multi-stage weight loading.

        Stage 1: Load base Wan 2.1 T2V safetensor weights (diffusers format).
                 Converts keys to WanModel naming and uses smart_load_weights
                 for shape-mismatch handling (patch_embedding 16ch -> in_dim ch).

        Stage 2: Load OmniAvatar checkpoint (LoRA + audio + patch_embedding).
                 - Map LoRA keys for PEFT compatibility
                 - Handle patch_embedding expansion (33ch -> 49ch -> 65ch)
                 - Either merge LoRA into base or load as PEFT adapters
        """
        # --- Stage 1: Base Wan 2.1 weights ---
        if base_model_paths is not None:
            paths = [p.strip() for p in base_model_paths.split(",") if p.strip()]
            logger.info(f"[OmniAvatarWan] Loading base Wan 2.1 from {len(paths)} file(s)")

            # Load and merge all safetensor shards
            base_sd: Dict[str, torch.Tensor] = {}
            for p in paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Base model weight file not found: {p}")
                base_sd.update(_load_state_dict(p, dtype=self._default_dtype))

            # Detect format: diffusers or original Wan
            sample_key = next(iter(base_sd.keys()), "")
            is_diffusers = any(
                marker in sample_key
                for marker in ("condition_embedder", "attn1", "attn2", "ffn.net")
            )

            if is_diffusers:
                logger.info("[OmniAvatarWan] Detected diffusers format, converting keys")
                base_sd = _convert_diffusers_state_dict(base_sd)
            else:
                # Original Wan format — strip common prefixes
                cleaned = {}
                for k, v in base_sd.items():
                    clean_k = k
                    for prefix in ("model.", "module.", "transformer."):
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                    cleaned[clean_k] = v
                base_sd = cleaned

            missing, unexpected = _smart_load_weights(self.model, base_sd)
            loaded = len(base_sd) - len(unexpected)
            logger.info(
                f"[OmniAvatarWan] Base weights: {loaded} loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )
            if missing:
                # Expected missing: audio modules (not in base), expanded patch_embedding channels
                audio_missing = [k for k in missing if "audio" in k]
                other_missing = [k for k in missing if "audio" not in k]
                if audio_missing:
                    logger.info(f"[OmniAvatarWan] Expected missing (audio): {len(audio_missing)} keys")
                if other_missing:
                    logger.warning(f"[OmniAvatarWan] Unexpected missing keys: {other_missing[:10]}...")
        else:
            logger.info("[OmniAvatarWan] No base_model_paths provided, using random init")

        # --- Stage 2: OmniAvatar checkpoint ---
        if omniavatar_ckpt_path is not None:
            if not os.path.isfile(omniavatar_ckpt_path):
                raise FileNotFoundError(f"OmniAvatar checkpoint not found: {omniavatar_ckpt_path}")

            logger.info(f"[OmniAvatarWan] Loading OmniAvatar checkpoint: {omniavatar_ckpt_path}")
            ckpt_sd = _load_state_dict(omniavatar_ckpt_path, dtype=self._default_dtype)

            # Separate LoRA weights from non-LoRA weights
            lora_sd: Dict[str, torch.Tensor] = {}
            non_lora_sd: Dict[str, torch.Tensor] = {}
            for k, v in ckpt_sd.items():
                if "lora_A" in k or "lora_B" in k:
                    lora_sd[k] = v
                else:
                    non_lora_sd[k] = v

            logger.info(
                f"[OmniAvatarWan] Checkpoint: {len(lora_sd)} LoRA params, "
                f"{len(non_lora_sd)} non-LoRA params"
            )

            # Handle patch_embedding expansion
            pe_key = "patch_embedding.weight"
            if pe_key in non_lora_sd:
                model_pe = self.model.patch_embedding.weight
                if non_lora_sd[pe_key].shape != model_pe.shape:
                    old_in_ch = non_lora_sd[pe_key].shape[1]
                    new_in_ch = model_pe.shape[1]
                    logger.info(
                        f"[OmniAvatarWan] Expanding patch_embedding: "
                        f"{list(non_lora_sd[pe_key].shape)} -> {list(model_pe.shape)}"
                    )
                    new_pe = torch.zeros_like(model_pe.data)
                    slices = tuple(slice(0, s) for s in non_lora_sd[pe_key].shape)
                    new_pe[slices] = non_lora_sd[pe_key]
                    non_lora_sd[pe_key] = new_pe
                    logger.info(
                        f"[OmniAvatarWan] Channels 0-{old_in_ch - 1}: from checkpoint, "
                        f"{old_in_ch}-{new_in_ch - 1}: zero-initialized"
                    )

            pe_bias_key = "patch_embedding.bias"
            if pe_bias_key in non_lora_sd and self.model.patch_embedding.bias is not None:
                model_pe_bias = self.model.patch_embedding.bias
                if non_lora_sd[pe_bias_key].shape != model_pe_bias.shape:
                    new_bias = model_pe_bias.data.clone()
                    new_bias[: non_lora_sd[pe_bias_key].shape[0]] = non_lora_sd[pe_bias_key]
                    non_lora_sd[pe_bias_key] = new_bias

            # Load non-LoRA weights (audio modules, patch_embedding, etc.)
            if non_lora_sd:
                missing, unexpected = self.model.load_state_dict(non_lora_sd, strict=False)
                loaded = len(non_lora_sd) - len(unexpected)
                logger.info(
                    f"[OmniAvatarWan] Non-LoRA weights: {loaded} loaded, "
                    f"{len(missing)} missing, {len(unexpected)} unexpected"
                )

            # Handle LoRA weights
            if lora_sd:
                if self.merge_lora:
                    logger.info("[OmniAvatarWan] Merging LoRA weights into base model")
                    _merge_lora_into_model(
                        self.model,
                        lora_sd,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        target_modules=LORA_TARGET_MODULES,
                    )
                else:
                    # Load as PEFT adapters — requires peft's inject_adapter_in_model
                    logger.info("[OmniAvatarWan] Loading LoRA as PEFT adapters (not merged)")
                    mapped_lora_sd = _map_omniavatar_lora_keys(lora_sd, use_peft=True)
                    try:
                        from peft import LoraConfig, inject_adapter_in_model

                        lora_config = LoraConfig(
                            r=self.lora_rank,
                            lora_alpha=self.lora_alpha,
                            init_lora_weights=True,
                            target_modules=LORA_TARGET_MODULES,
                        )
                        self.model = inject_adapter_in_model(lora_config, self.model)
                        missing, unexpected = self.model.load_state_dict(
                            mapped_lora_sd, strict=False
                        )
                        logger.info(
                            f"[OmniAvatarWan] PEFT LoRA: {len(mapped_lora_sd) - len(unexpected)} loaded, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected"
                        )
                    except ImportError:
                        raise ImportError(
                            "merge_lora=False requires the `peft` package. "
                            "Install with: pip install peft"
                        )
        else:
            logger.info("[OmniAvatarWan] No omniavatar_ckpt_path provided, using base/random init only")

    # ------------------------------------------------------------------
    # LoRA freeze utilities (used by merge_lora=False training paths)
    # ------------------------------------------------------------------

    def _apply_unfreeze(self, unfreeze_modules: List[str]) -> None:
        """Re-enable ``requires_grad`` on parameters of specific submodules.

        Used in conjunction with PEFT LoRA injection (``merge_lora=False``):
        ``inject_adapter_in_model`` freezes the entire base and only leaves
        LoRA A/B trainable.  This helper selectively un-freezes the
        submodules listed in ``unfreeze_modules`` so they participate in
        training as full fine-tunes.

        Paths are dotted module paths relative to ``self``.  For
        ``OmniAvatarWan`` (bidirectional) the WanModel lives at ``self.model``
        (not ``self._core`` as in the causal class), so the typical paths
        look like:

        - ``"model.audio_proj"``
        - ``"model.audio_cond_projs"``
        - ``"model.patch_embedding"``

        Failure mode is intentionally strict: a non-resolvable path raises
        ``AttributeError`` so a config typo is caught at construction time.

        Args:
            unfreeze_modules: list of dotted module paths relative to self.
                Empty list or ``None`` is a no-op.
        """
        if not unfreeze_modules:
            return
        for path in unfreeze_modules:
            module = self.get_submodule(path)  # raises AttributeError if missing
            n_params_unfrozen = 0
            for p in module.parameters():
                p.requires_grad_(True)
                n_params_unfrozen += p.numel()
            logger.info(
                f"[OmniAvatarWan] unfreeze: re-enabled requires_grad on "
                f"{n_params_unfrozen / 1e6:.2f}M params in '{path}'"
            )

    def reset_parameters(self) -> None:
        """Recompute non-persistent state after FSDP's `to_empty(device=cuda)`.

        Symmetric to ``CausalOmniAvatarWan.reset_parameters``.  Required when
        ``fsdp_meta_init=True``: ranks 1+ build the model on the meta device,
        so ``self.model.freqs`` (a Python tuple of complex tensors, not a
        registered buffer) is constructed as meta tensors.  After ``to_empty``
        materializes params/buffers and ``set_model_state_dict`` broadcasts
        rank 0's state, the non-buffer ``freqs`` attribute is never restored
        on non-rank-0 ranks unless we recompute it here.

        Without this hook, non-rank-0 ranks would have meta freqs at forward
        time and ``meta.to(cuda)`` allocates uninitialized memory -> wrong
        RoPE rotations -> training silently broken.
        """
        self.model.freqs = precompute_freqs_cis_3d(self._head_dim)
        super().reset_parameters()  # FastGenNetwork.reset_parameters reseeds noise scheduler

    def apply_lora_freeze(self) -> None:
        """Re-apply PEFT's "only LoRA + selective-unfreeze trainable" freeze.

        Symmetric counterpart to ``CausalOmniAvatarWan.apply_lora_freeze``.
        While the bidirectional ``OmniAvatarWan`` is not currently subject
        to the generator-only re-instantiation in ``FastGenModel.build_model``
        (that path only operates on ``self.net``, not ``self.fake_score`` or
        ``self.teacher``), this method exists for symmetry and as a defensive
        hook for the SF method class to call uniformly on both networks without
        dispatching on type.

        Idempotent and safe to call repeatedly.  No-op when:
        - ``self.merge_lora=True`` (full FT mode, no LoRA layers exist)
        - PEFT injection didn't actually run (no ``"lora_"`` params present)
        """
        if self.merge_lora:
            return
        has_lora = any("lora_" in n for n, _ in self.named_parameters())
        if not has_lora:
            return

        n_lora = 0
        n_frozen = 0
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                n_lora += param.numel()
            else:
                param.requires_grad_(False)
                n_frozen += param.numel()

        self._apply_unfreeze(self.unfreeze_modules)

        logger.info(
            f"[OmniAvatarWan] apply_lora_freeze: LoRA {n_lora/1e6:.2f}M "
            f"trainable, base {n_frozen/1e6:.2f}M frozen, then "
            f"unfreeze_modules re-enabled on top"
        )

    # ------------------------------------------------------------------
    # V2V conditioning
    # ------------------------------------------------------------------

    def _build_y(
        self,
        condition: Dict[str, torch.Tensor],
        T: int,
    ) -> torch.Tensor:
        """Build the V2V conditioning tensor ``y`` from condition components.

        Returns:
            y: ``[B, C_y, T, H, W]`` where C_y = in_dim - 16 (noise channels excluded).
                For V2V (in_dim=49): C_y=33  = 16 ref + 1 mask + 16 masked_video
                For V2V+ref (in_dim=65): C_y=49 = 16 ref + 1 mask + 16 masked_video + 16 ref_seq
        """
        ref_latent = condition["ref_latent"]      # [B, 16, 1, H, W]
        mask = condition["mask"]                   # [H_lat, W_lat]  (LatentSync: 1=keep, 0=mask)
        masked_video = condition.get("masked_video")    # [B, 16, T, H, W] or None
        ref_sequence = condition.get("ref_sequence")    # [B, 16, T, H, W] or None

        B = ref_latent.shape[0]
        H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
        device = ref_latent.device
        dtype = ref_latent.dtype

        # Reference frame repeated across time
        ref_repeated = ref_latent.repeat(1, 1, T, 1, 1)  # [B, 16, T, H, W]

        # Build spatial mask channel
        # OmniAvatar convention: 0=keep, 1=generate
        # LatentSync convention: 1=keep, 0=mask
        # Invert: mouth (0 in LatentSync) -> 1 (generate in OmniAvatar)
        inverted_mask = 1.0 - mask.to(dtype=dtype)  # [H_lat, W_lat]
        mask_ch = torch.zeros(B, 1, T, H_lat, W_lat, device=device, dtype=dtype)
        if self.mask_all_frames:
            mask_ch[:, :, :] = inverted_mask[None, None, None]
        else:
            # Frame 0: keep all (reference frame, mask=0 everywhere)
            mask_ch[:, :, 0] = 0
            # Frames 1+: apply spatial mask
            mask_ch[:, :, 1:] = inverted_mask[None, None, None]

        parts = [ref_repeated, mask_ch]

        if masked_video is not None:
            parts.append(masked_video)  # [B, 16, T, H, W]

        if ref_sequence is not None:
            parts.append(ref_sequence)  # [B, 16, T, H, W]

        y = torch.cat(parts, dim=1)  # [B, C_y, T, H, W]
        return y

    # ------------------------------------------------------------------
    # Forward pass (FastGenNetwork interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the OmniAvatar diffusion score model.

        Args:
            x_t: Noisy latent ``[B, 16, T, H, W]``.
            t: Timestep in ``[0, 1)`` range, shape ``[B]``.
            condition: Dict with keys:
                - ``text_embeds``:   ``[B, 512, 4096]``
                - ``audio_emb``:     ``[B, num_video_frames, 10752]``
                - ``ref_latent``:    ``[B, 16, 1, H, W]``
                - ``mask``:          ``[H_lat, W_lat]`` (LatentSync convention: 1=keep)
                - ``masked_video``:  ``[B, 16, T, H, W]`` (optional for V2V)
                - ``ref_sequence``:  ``[B, 16, T, H, W]`` (optional for V2V+refseq)
            r: Reserved for mean-flow models (unused here).
            return_features_early: If True, return intermediate features and exit.
            feature_indices: Set of block indices to extract features from.
            return_logvar: Unused (no logvar head in OmniAvatar DiT).
            fwd_pred_type: Override prediction type for output conversion.
            **fwd_kwargs: Additional kwargs (``use_gradient_checkpointing``, etc.).

        Returns:
            Model output tensor, or features list, depending on flags.
        """
        if feature_indices is None:
            feature_indices = set()

        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            if fwd_pred_type not in NET_PRED_TYPES:
                raise ValueError(f"Unsupported fwd_pred_type '{fwd_pred_type}'. Supported: {NET_PRED_TYPES}")

        # Unpack condition dict
        assert isinstance(condition, dict), f"condition must be a dict, got {type(condition)}"
        text_embeds = condition["text_embeds"]   # [B, 512, 4096]
        audio_emb = condition.get("audio_emb")   # [B, num_video_frames, 10752] or None

        # Build V2V y-tensor
        T = x_t.shape[2]  # Number of latent frames
        y = self._build_y(condition, T)

        # Rescale timestep: LipForcing uses t in [0, 1), WanModel expects t * num_steps
        # The RF noise schedule rescale_t does: t * 1000
        timestep = self.noise_scheduler.rescale_t(t)

        # Forward through the DiT
        use_gradient_checkpointing = fwd_kwargs.get("use_gradient_checkpointing", self._use_gradient_checkpointing)
        has_features = feature_indices is not None and len(feature_indices) > 0

        model_output = self.model(
            x=x_t,
            timestep=timestep,
            context=text_embeds,
            clip_feature=None,
            y=y,
            use_gradient_checkpointing=use_gradient_checkpointing,
            audio_emb=audio_emb,
            feature_indices=feature_indices if has_features else None,
            return_features_early=return_features_early,
            expand_audio_checkpoint_scope=self._expand_audio_checkpoint_scope,
        )

        # Early exit: model returned just the unpatchified features
        if return_features_early and has_features:
            return model_output  # List of [B, dim, T, H, W] feature tensors

        # Unpack if model returned (output, features) tuple
        features = None
        if has_features and isinstance(model_output, tuple):
            model_output, features = model_output

        # Convert prediction type
        out = self.noise_scheduler.convert_model_output(
            x_t, model_output, t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        # Return format depends on what was requested
        if features is not None:
            if return_logvar:
                logvar = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)
                return [out, features], logvar
            return [out, features]

        if return_logvar:
            logvar = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)
            return out, logvar

        return out

    def fully_shard(self, **kwargs):
        """Fully shard the network for FSDP2.

        Shards ``self.model`` (WanModel) instead of ``self`` because the wrapper
        class inherits from ABC, which causes issues with FSDP2's ``__class__``
        assignment.

        We keep manual gradient checkpointing (``torch.utils.checkpoint.checkpoint``
        in ``WanModel.forward``) rather than ``apply_fsdp_checkpointing`` because
        audio conditioning is injected between blocks in the parent forward — the
        ``checkpoint_wrapper`` boundary around DiTBlock can't account for these
        inter-block audio tensors, causing tensor-count mismatches during recompute.
        Manual checkpointing wraps exactly ``block(x, context, t_mod, freqs)`` after
        audio injection, keeping the checkpoint boundary clean.
        """
        if self._use_gradient_checkpointing:
            logger.info(
                "OmniAvatarWan: keeping manual gradient checkpointing "
                "(checkpoint_wrapper incompatible with inter-block audio injection)"
            )

        for block in self.model.blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.model, **kwargs)
