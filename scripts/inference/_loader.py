"""Shared student-model loader for the inference scripts.

``inference_segmentwise.py`` and ``inference_streaming.py`` both import
``load_diffusion_model`` from here so the 3-source assembly (base Wan 2.1 T2V +
OmniAvatar-LS adapter + Self-Forcing student) and the post-load LoRA merge live
in exactly one place.

This module intentionally keeps only light top-level deps (``os``, ``torch``);
the heavy model import is lazy (inside the function), so tools that merely need
the loader (e.g. ``scripts/export_merged_checkpoint.py``) don't pull in the
video/audio stack.
"""
import os

import torch


def load_diffusion_model(args, device, dtype):
    """Load the CausalOmniAvatarWan student model.

    Constructs the model, loads base Wan + OmniAvatar weights (if paths given),
    then overlays the Self-Forcing trained checkpoint on top.

    Returns:
        CausalOmniAvatarWan model in eval mode on *device*.
    """
    from lipforcing.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    # 14B SF LoRA: construct with merge_lora=False so PEFT is injected and
    # the model exposes lora_A/lora_B keys that the trainable-filtered SF
    # state_dict expects.  After load_state_dict the LoRA values are
    # populated; we then merge them into base for inference speed (unless
    # --no_merge_lora_post_load is passed).
    #
    # For 1.3B, merge_lora=True folds the OmniAvatar V2V LoRA into base at
    # construction, so the SF checkpoint is expected to have plain
    # (already-merged) keys.
    constructor_merge_lora = (args.model_size == "1.3B")

    model = CausalOmniAvatarWan(
        model_size=args.model_size,
        in_dim=65,
        mode="v2v",
        use_audio=True,
        audio_hidden_size=32,
        chunk_size=args.chunk_size,
        total_num_frames=21,
        base_model_paths=args.base_model_paths,
        omniavatar_ckpt_path=args.omniavatar_ckpt_path,
        merge_lora=constructor_merge_lora,
        lora_rank=128,
        lora_alpha=64,
        net_pred_type="flow",
        schedule_type="rf",
        mask_all_frames=True,
        dtype=args.dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        use_dynamic_rope=args.use_dynamic_rope,
    )

    # Load Self-Forcing checkpoint on top
    # Supports: regular .pt/.pth, FSDP distcp directory, or .pth + adjacent distcp dir
    print(f"Loading SF checkpoint from {args.ckpt_path} ...")

    state_dict = None

    # Check for FSDP distributed checkpoint: look for .net_model/ directory
    ckpt_stem = args.ckpt_path.replace(".pth", "")
    fsdp_net_dir = ckpt_stem + ".net_model"
    if os.path.isdir(fsdp_net_dir):
        # FSDP2 distributed checkpoint — load via torch.distributed.checkpoint
        print(f"  Loading FSDP distributed checkpoint from {fsdp_net_dir} ...")
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

        reader = FileSystemReader(fsdp_net_dir)
        md = reader.read_metadata()
        state_dict = {}
        for key, meta in md.state_dict_metadata.items():
            if hasattr(meta, "size"):
                state_dict[key] = torch.empty(meta.size)
        dcp_load(state_dict, storage_reader=reader, no_dist=True)
        print(f"  Loaded {len(state_dict)} tensors from FSDP distcp")
    elif os.path.isdir(args.ckpt_path):
        # Direct distcp directory path
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

        reader = FileSystemReader(args.ckpt_path)
        md = reader.read_metadata()
        state_dict = {}
        for key, meta in md.state_dict_metadata.items():
            if hasattr(meta, "size"):
                state_dict[key] = torch.empty(meta.size)
        dcp_load(state_dict, storage_reader=reader, no_dist=True)
        print(f"  Loaded {len(state_dict)} tensors from distcp directory")
    else:
        # Regular .pt/.pth checkpoint
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict) and "net" in ckpt["model"]:
                state_dict = ckpt["model"]["net"]
            elif "net" in ckpt:
                state_dict = ckpt["net"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

    # Keys in checkpoint use plain names (e.g. "patch_embedding.weight")
    # Model wraps everything under _core, so keys are "_core.xxx"
    # Try loading directly first, if too many missing, try adding _core prefix
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > len(state_dict) * 0.5 and not any(k.startswith("_core.") for k in state_dict):
        # Try with _core prefix
        prefixed_sd = {"_core." + k: v for k, v in state_dict.items()}
        missing2, unexpected2 = model.load_state_dict(prefixed_sd, strict=False)
        if len(missing2) < len(missing):
            missing, unexpected = missing2, unexpected2
            print("  Applied _core. prefix for key matching")

    print(f"  SF checkpoint: {len(state_dict)} params, {len(missing)} missing, {len(unexpected)} unexpected")

    # Merge LoRA into base post-load (14B path only; the 1.3B path constructs
    # with merge_lora=True so there's nothing to merge here).
    #
    # Uses PEFT's per-layer merge() which fuses LoRA A/B into the wrapped
    # base_layer.weight in-place.  After merge, LoraLinear.forward dispatches
    # to base_layer.forward and the LoRA path is a no-op — same numeric
    # result as not having LoRA at all.  We keep the LoraLinear wrappers in
    # place (no unload) because that's a structural change and not necessary
    # for correctness; only the merge_count is reported.
    if args.model_size == "14B" and args.merge_lora_post_load:
        print("  Merging LoRA into base for inference speed...")
        from peft.tuners.lora import LoraLayer
        merge_count = 0
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module.merge()
                merge_count += 1
        if merge_count > 0:
            print(f"  Merged {merge_count} LoRA layers.")
        elif args.base_model_paths is None and args.omniavatar_ckpt_path is None:
            # Consolidated single-file checkpoint: no base/adapter supplied and
            # the .pth already carries fully-merged plain weights, so there are
            # no PEFT layers to merge. This is the expected published-weights path.
            print("  Checkpoint is already consolidated (merged); nothing to merge.")
        else:
            print("  WARN: no LoraLayer instances found; model has no PEFT "
                  "adapters to merge (was --model_size set correctly?).")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


