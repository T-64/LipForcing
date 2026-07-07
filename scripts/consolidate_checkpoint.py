#!/usr/bin/env python3
"""Consolidate an FSDP distributed checkpoint into a single flat ``.pth``.

Training saves the student weights as a sharded distributed-checkpoint (DCP)
directory ``<step>.net_model/`` (alongside a small ``<step>.pth`` metadata
stub). The inference scripts can load that DCP directory **directly** (they
detect the sibling ``.net_model/`` dir), so this step is optional for local
use — but it is convenient for distributing a single-file checkpoint
(e.g. uploading one ``.pth`` to the Hugging Face Hub) that ``--ckpt_path``
also accepts.

Usage:
    python scripts/consolidate_checkpoint.py <step>.net_model  out.pth
    # or pass the <step>.pth stub; the sibling <step>.net_model/ dir is used:
    python scripts/consolidate_checkpoint.py <step>.pth         out.pth
"""
import argparse
import os

import torch  # noqa: F401  (ensures torch.distributed is importable)
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="Path to the <step>.net_model DCP directory (or the <step>.pth stub next to it).")
    ap.add_argument("output", help="Output single-file .pth path.")
    args = ap.parse_args()

    dcp_dir = args.ckpt
    if dcp_dir.endswith(".pth"):
        dcp_dir = dcp_dir[:-len(".pth")] + ".net_model"
    if not os.path.isdir(dcp_dir):
        raise SystemExit(f"DCP directory not found: {dcp_dir}")

    print(f"Consolidating DCP '{dcp_dir}' -> '{args.output}' ...")
    dcp_to_torch_save(dcp_dir, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
