#!/usr/bin/env python
"""Precompute T5 text embeddings for the default prompt and save as text_emb.pt.
Reuses load_or_encode_text so the format matches exactly ([1, 512, 4096])."""
import sys, argparse, torch
sys.path.insert(0, "/root/code/LipForcing")
from scripts.inference._common import load_or_encode_text

TE = "/root/code/LipForcing/weights/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
OUT = "/root/code/LipForcing/weights/text_emb.pt"

args = argparse.Namespace(text_embeds_path=None, text_encoder_path=TE, prompt="a person talking")
emb = load_or_encode_text(args, "cuda", torch.bfloat16)   # [1, 512, 4096] on cuda
print("text_embeds shape:", tuple(emb.shape), "dtype:", emb.dtype)
torch.save(emb.cpu(), OUT)
print("saved ->", OUT)
