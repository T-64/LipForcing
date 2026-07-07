# Licenses for LipForcing and third-party code

LipForcing itself is released under the Apache License 2.0 (see the repository-root
[`LICENSE`](../LICENSE)). The codebase derives from
[NVIDIA FastGen](https://github.com/NVlabs/FastGen) (Apache License 2.0), whose copyright
notices are retained in the root `LICENSE` and in per-file SPDX headers.

## Third-party code

1. [Diffusers](diffusers/LICENSE) ([github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)): [Apache License 2.0](https://github.com/huggingface/diffusers/blob/8d415a6f481ff1b26168c046267628419650f930/LICENSE) - runtime dependency (no source vendored); license included for completeness.
2. [Wan](Wan/LICENSE) ([github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)): [Apache License 2.0](https://github.com/Wan-Video/Wan2.1/blob/841fe5237bbf8724e08870b1e83213e669cda0d1/LICENSE.txt) - base diffusion model and VAE.
3. [Detectron2](detectron2/LICENSE) ([github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)): [Apache License 2.0](https://github.com/facebookresearch/detectron2/blob/fd27788985af0f4ca800bca563acdb700bb890e2/LICENSE) - the registry adapted in `lipforcing/utils/registry.py` and the `locate`/`instantiate` helpers in `lipforcing/utils/__init__.py`.
4. [OmniAvatar](OmniAvatar/LICENSE.txt) ([github.com/Omni-Avatar/OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar)): Apache License 2.0 - the vendored V2V teacher / encoder / preprocessing pipeline under `OmniAvatar/`. Parts of that pipeline originate from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache License 2.0) via OmniAvatar.
5. [SyncNet](SyncNet/LICENSE) ([github.com/joonson/syncnet_python](https://github.com/joonson/syncnet_python)): MIT License - the sync model vendored in `lipforcing/methods/reward/syncnet_v2.py`.
6. [LatentSync](LatentSync/LICENSE) ([github.com/bytedance/LatentSync](https://github.com/bytedance/LatentSync)): Apache License 2.0 - the face detection / affine-alignment / compositing pipeline vendored under `OmniAvatar/utils/latentsync/` (including the StyleSync-derived affine utilities it carries).
7. [TAEHV](TAEHV/LICENSE) ([github.com/madebyollin/taehv](https://github.com/madebyollin/taehv)): MIT License - the tiny video autoencoder vendored in `lipforcing/methods/reward/taehv.py`, used as the fast decoder for streaming inference and the Re-DMD reward path.
