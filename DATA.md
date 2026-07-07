# Data & Preprocessing

Training runs on **sample directories** of raw inputs (`sub_clip.mp4`, `audio.wav`, `prompt.txt`)
and encodes them **on the fly** with the same frozen encoders inference uses (one shared module,
`lipforcing/preprocess.py`; enabled with `OMNIAVATAR_ON_THE_FLY=1`, see step 2) - no separate
precompute step needed.

## 1 · Prepare training data from raw videos

```bash
python scripts/prepare_data.py \
  --input_dir /path/to/raw_videos \
  --output_dir ./data/v2v_training_data \
  --mask_path weights/mask.png \
  --val_count 2
```

Each video is converted to 25 fps CFR if needed, face-aligned with the **same** LatentSync
detection + 512×512 affine pipeline as inference, and split into consecutive 81-frame clips -
one sample dir per clip (`sub_clip.mp4` aligned crop · `audio.wav` 16 kHz mono slice ·
`prompt.txt`). It also writes `train_list.txt` / `val_list.txt` (one sample dir per line).

## 2 · Wire the training configs

Point the env vars at your weights and lists (the training scripts set placeholder defaults):

| Env var | What |
|---------|------|
| `OMNIAVATAR_DATA_LIST` / `OMNIAVATAR_VAL_LIST` | the list files from step 1 |
| `OMNIAVATAR_ON_THE_FLY=1` | enable on-the-fly encoding of raw sample dirs |
| `OMNIAVATAR_STUDENT_CKPT_14B` | student + fake-score init (OmniAvatar-LS adapter) |
| `OMNIAVATAR_TEACHER_CKPT` | frozen distillation teacher (Stage 2) |
| `OMNIAVATAR_DF_CKPT` | Stage-1 checkpoint that initializes Stage 2 |
| `OMNIAVATAR_WAN_BASE` | directory with the six Wan2.1-T2V-14B `diffusion_pytorch_model-*.safetensors` shards |
| `OMNIAVATAR_VAE_PATH` | Wan VAE (also used by the reward decode) |
| `OMNIAVATAR_WAV2VEC_PATH` | wav2vec2-base-960h dir |
| `OMNIAVATAR_TEXT_ENCODER_PATH` | UMT5-XXL `.pth` (unless every sample has a cached `text_emb.pt`) |
| `MASK_PATH` | LatentSync mouth mask |
| `SYNCNET_CKPT` / `TAEW_CKPT` | SyncNet reward + TAEW reward decoder (Stage 2) |

Then run the stage scripts as shown in the [README](README.md#training). Download sources for all
weights are in the README's [Weights](README.md#weights) section.

<details>
<summary><b>Reference: encoding details, tensor format, caching</b></summary>

### On-the-fly encoding

With `OMNIAVATAR_ON_THE_FLY=1`, any sample lacking precomputed `.pt` tensors is decoded on CPU in
the dataloader workers and encoded on GPU in the **main training process** by frozen VAE / wav2vec /
UMT5 encoders (loaded once - never per-worker; UMT5-XXL is ~11B). Samples that *do* have
precomputed `.pt` take the fast path, so mixed datasets work.

**Caching**: with `OMNIAVATAR_CACHE_ENCODED=1` (default) each sample is encoded once and written
back as `.pt` (`OMNIAVATAR_CACHE_DIR` overrides the destination; default is the sample dir), so
later epochs/runs hit the fast path - this doubles as the "precompute" step; there is no separate
CLI. Caching is content-addressed: `text_emb` is keyed by `sha1(prompt)` (each unique prompt is
encoded once), latents/audio by sample id. Optionally, `NEG_TEXT_EMB_PATH` supplies a precomputed
negative-prompt embedding for the teacher's CFG (zero tensor when unset).

### Per-sample tensor format

The dataloader (`lipforcing/datasets/omniavatar_dataloader.py`) reads one directory per sample:

| File | Key(s) | Shape | dtype | Notes |
|------|--------|-------|-------|-------|
| `vae_latents_mask_all.pt` | `input_latents`, `masked_latents` | `[16, 21, 64, 64]` each | bf16 | Wan VAE latents (21 latent frames from 81 pixel frames, 512→64). **Mask-all**: the mouth region is masked on **all** frames (incl. frame 0), before the VAE encode. |
| `audio_emb_omniavatar.pt` | `audio_emb` | `[N, 10752]`, `N ≥ 81` | f32→bf16 | wav2vec features: `last_hidden_state` concatenated with all 13 hidden states = `14 × 768 = 10752`. Sliced to the first 81 frames. |
| `text_emb.pt` | (bare tensor) | `[1, 512, 4096]` | f32→bf16 | UMT5-XXL text-encoder output. |
| `ref_latents.pt` *(optional)* | `ref_sequence_latents` | `[16, 21, 64, 64]` | bf16 | Reference-sequence VAE latents. Falls back to zeros if absent. |
| `audio.wav` *(optional for the fast path)* | raw waveform | - | - | Used by the reward path (raw-audio SyncNet); resampled to 16 kHz. |

The mouth **mask** is a single static PNG (LatentSync convention, `1 = keep`, `0 = mask`), resized
to the latent resolution - not stored per sample. Pass it via `--mask_path` (inference) or the
`latentsync_mask_path` config field (training).

> The `10752`-dim audio packing and the mask-all masking come from OmniAvatar's **custom**
> `OmniAvatar/models/wav2vec.py::Wav2VecModel` and the mask-all VAE encode path. A plain
> `transformers` wav2vec produces 768-dim features and will **not** match - always use the
> vendored encoder.

### Training the 1.3B student

The released configs are hard-wired to the **14B** student (the "Switch student to 14B" block at
the bottom of each config). To train the 1.3B student, omit/comment out that block -
`OMNIAVATAR_STUDENT_CKPT_1_3B` then supplies its checkpoint path (setting the env var alone does
**not** change the model size). `OMNIAVATAR_DATA_ROOT` (default `./data/v2v_training_data`) sets
the default location for the data list files.

</details>
