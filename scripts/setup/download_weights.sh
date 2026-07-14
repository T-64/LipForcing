#!/usr/bin/env bash
# Download all LipForcing inference weights (~45GB) via corporate proxy.
source /root/code/LipForcing/proxy.env
source /root/code/LipForcing/.venv/bin/activate
cd /root/code/LipForcing
mkdir -p weights

echo "### [1/5] lipforcing_14b.pth (~28GB)"
hf download JinhyukJang/lipforcing lipforcing_14b.pth --local-dir weights

echo "### [2/5] Wan2.1 VAE + UMT5 text encoder + tokenizer"
hf download Wan-AI/Wan2.1-T2V-14B \
  Wan2.1_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json \
  --local-dir weights/Wan2.1-T2V-14B

echo "### [3/5] wav2vec2-base-960h audio encoder"
hf download facebook/wav2vec2-base-960h --local-dir weights/wav2vec2-base-960h

echo "### [4/5] TAEW tiny decoder + LatentSync mask (github raw)"
curl -fL --retry 3 -o weights/taew2_1.pth https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth
curl -fL --retry 3 -o weights/mask.png https://raw.githubusercontent.com/bytedance/LatentSync/main/latentsync/utils/mask.png

echo "### [5/5] BiSeNet face-parsing weights (occlusion-aware compositing)"
# Used by --occlusion_aware. Mirrors zllrunning/face-parsing.PyTorch 79999_iter.pth.
mkdir -p OmniAvatar/utils/face_parsing
hf download jonathandinu/face-parsing 79999_iter.pth --local-dir OmniAvatar/utils/face_parsing

echo "WEIGHTS_DONE"
