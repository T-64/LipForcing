#!/usr/bin/env python3
r"""Export a self-contained, merged single-file student checkpoint for inference.

Why this exists
---------------
The Self-Forcing student is *saved as a trainable-only delta* (LoRA +
selectively-unfrozen modules — ~5 GB at 14B; see ``checkpointer.py``), so
running it normally requires assembling THREE weight sources at load time:

    base Wan 2.1 T2V  +  OmniAvatar-LS V2V adapter  +  this SF delta

This script performs that assembly once, merges the LoRA into the base, and
writes ONE fully-baked ``.pth`` (plain keys, no PEFT wrappers) that the
inference scripts load with ``--ckpt_path`` ALONE — no ``--base_model_paths``
or ``--omniavatar_ckpt_path`` needed.  This is the recommended artifact to
publish for the inference-only audience.

Run on a machine that has the three sources + enough RAM/VRAM (e.g. one H200).
It re-uses the exact assembly path from ``scripts/inference/_loader.py``
so the exported weights are numerically identical to a normal 3-source load.

Example
-------
    python scripts/export_merged_checkpoint.py \
        --model_size 14B \
        --ckpt_path            /path/to/0000600.pth \          # SF delta (.pth or .net_model dir)
        --base_model_paths     /path/wan2.1_t2v_14b-00001.safetensors,/path/...-00002.safetensors \
        --omniavatar_ckpt_path /path/to/omniavatar_ls_14b.pt \
        --output_path          /path/to/lipforcing_14b.pth

Then inference needs only:
    python scripts/inference/inference_streaming.py --model_size 14B \
        --ckpt_path /path/to/lipforcing_14b.pth \
        --vae_path ... --wav2vec_path ... --mask_path ... --taehv_ckpt ...
"""
import argparse
import os
import sys

import torch

# --- Path setup (mirror the inference scripts) ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPFORCING_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, LIPFORCING_ROOT)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "inference"))  # for `import _loader`
os.environ.setdefault("OMNIAVATAR_ROOT", LIPFORCING_ROOT)

from _loader import load_diffusion_model  # noqa: E402

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _to_plain_merged_state_dict(core: torch.nn.Module) -> dict:
    """Map a post-merge PEFT ``_core`` state_dict to plain (unwrapped) keys.

    After ``LoraLayer.merge()`` the merged weights live in
    ``<module>.base_layer.weight`` while the now-redundant ``lora_A``/``lora_B``
    tensors remain.  A model constructed WITHOUT the OmniAvatar adapter has no
    PEFT wrappers, so its keys are plain (``<module>.weight``).  We therefore
    drop the lora_* tensors and strip the ``base_layer.`` infix.

    (For the 1.3B path the OmniAvatar LoRA is already folded at construction, so
    ``_core`` has plain keys to begin with and this is a near no-op.)
    """
    raw = core.state_dict()
    out, n_drop, n_unwrap, n_plain = {}, 0, 0, 0
    for k, v in raw.items():
        if ".lora_A." in k or ".lora_B." in k or k.endswith(".lora_A") or k.endswith(".lora_B"):
            n_drop += 1
            continue
        if ".base_layer." in k:
            k = k.replace(".base_layer.", ".")
            n_unwrap += 1
        else:
            n_plain += 1
        out[k] = v
    print(f"  merged state_dict: {len(out)} tensors "
          f"({n_unwrap} unwrapped from PEFT base_layer, {n_plain} plain, {n_drop} lora_* dropped)")
    return out


def _verify_reload(plain_sd: dict, args) -> bool:
    """Build a fresh model WITHOUT base/omni and confirm it accepts the merged
    state_dict with no missing parameters and no unexpected keys.  This is the
    exact structure the inference scripts construct when only ``--ckpt_path`` is
    given, so a clean result here means single-flag inference will load fully.
    """
    from lipforcing.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
    probe = CausalOmniAvatarWan(
        model_size=args.model_size, in_dim=65, mode="v2v", use_audio=True,
        audio_hidden_size=32, chunk_size=args.chunk_size, total_num_frames=21,
        base_model_paths=None, omniavatar_ckpt_path=None, merge_lora=False,
        lora_rank=128, lora_alpha=64, net_pred_type="flow", schedule_type="rf",
        mask_all_frames=True, dtype=args.dtype, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, use_dynamic_rope=args.use_dynamic_rope,
    )
    prefixed = {"_core." + k: v for k, v in plain_sd.items()}
    missing, unexpected = probe.load_state_dict(prefixed, strict=False)
    param_keys = set(dict(probe.named_parameters()).keys())
    missing_params = [m for m in missing if m in param_keys]
    print(f"  verify reload: {len(missing_params)} missing params, {len(unexpected)} unexpected keys")
    if missing_params:
        print(f"    WARN missing params (first 10): {missing_params[:10]}")
    if unexpected:
        print(f"    WARN unexpected keys (first 10): {unexpected[:10]}")
    return not missing_params and not unexpected


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_size", default="14B", choices=["14B", "1.3B"])
    ap.add_argument("--ckpt_path", required=True,
                    help="Self-Forcing student delta: .pth, .net_model distcp dir, or .pth + sibling dir.")
    ap.add_argument("--base_model_paths", required=True,
                    help="Comma-separated base Wan 2.1 T2V safetensor paths (1.3B or 14B).")
    ap.add_argument("--omniavatar_ckpt_path", required=True,
                    help="OmniAvatar-LS V2V adapter checkpoint (.pt/.pth) — LoRA + audio + patch-embed.")
    ap.add_argument("--output_path", required=True, help="Output single-file merged .pth.")
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES),
                    help="Compute dtype for assembly (default bf16, matches training/inference).")
    ap.add_argument("--save_dtype", default=None, choices=list(_DTYPES),
                    help="Dtype to store floating tensors in (default: same as --dtype).")
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--local_attn_size", type=int, default=-1)
    ap.add_argument("--sink_size", type=int, default=0)
    ap.add_argument("--use_dynamic_rope", action="store_true", default=False)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_verify", action="store_true", help="Skip the reload sanity check.")
    args = ap.parse_args()
    args.merge_lora_post_load = True  # force the LoRA->base merge for 14B

    dtype = _DTYPES[args.dtype]
    device = torch.device(args.device)

    print(f"Assembling {args.model_size} student (base Wan + OmniAvatar-LS + SF delta) ...")
    model = load_diffusion_model(args, device, dtype)  # constructs + loads 3 sources + merges LoRA

    plain = _to_plain_merged_state_dict(model._core)
    save_dtype = _DTYPES[args.save_dtype] if args.save_dtype else dtype
    plain = {k: (v.to(save_dtype) if v.is_floating_point() else v).cpu() for k, v in plain.items()}

    if not args.no_verify:
        ok = _verify_reload(plain, args)
        print(f"  verify: {'OK — single-flag inference will load fully.' if ok else 'MISMATCH — inspect the warnings above before publishing.'}")

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Writing merged checkpoint -> {args.output_path} ...")
    torch.save(plain, args.output_path)
    sz = os.path.getsize(args.output_path) / 1e9
    print(f"Done. {len(plain)} tensors, {sz:.1f} GB. Load it with --ckpt_path alone (no base/omni flags).")


if __name__ == "__main__":
    main()
