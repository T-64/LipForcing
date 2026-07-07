# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Callable, TYPE_CHECKING
import gc

import numpy as np


import torch
import torchvision
from torchvision.transforms import functional as tv_F

import wandb
import wandb.util

from lipforcing.callbacks.callback import Callback
from lipforcing.configs.config_utils import serialize_config
from lipforcing.utils import basic_utils

from lipforcing.utils.distributed import rank0_only, synchronize, world_size
from lipforcing.utils import logging_utils as logger

if TYPE_CHECKING:
    from lipforcing.configs.config import BaseConfig
    from lipforcing.methods import FastGenModel


def tensor_to_wandb_video_with_audio(
    video_tensor: torch.Tensor,
    audio_path: str,
    fps: int = 25,
    vid_format: str = "mp4",
    caption: str | None = None,
) -> wandb.Video:
    """Convert a [B, T, C, H, W] uint8 video tensor + audio file to wandb.Video with audio.

    Takes the first sample in the batch. Writes video to a temp file, muxes audio
    with ffmpeg, and returns a wandb.Video from the muxed output.

    Args:
        video_tensor: uint8 tensor of shape [B, T, C, H, W] (already in 0-255 range).
        audio_path: Path to the audio .wav file.
        fps: Video frame rate (default 25 for OmniAvatar).
        vid_format: Video format (default "mp4").
        caption: Optional caption for wandb.Video.

    Returns:
        wandb.Video with muxed audio, or silent video if muxing fails.
    """
    try:
        # Take first sample: [T, C, H, W]
        vid = video_tensor[0] if video_tensor.dim() == 5 else video_tensor
        # Convert to [T, H, W, C] uint8 on CPU
        vid = vid.permute(0, 2, 3, 1).cpu()

        tmpdir = tempfile.mkdtemp()
        silent_path = os.path.join(tmpdir, f"silent.{vid_format}")
        muxed_path = os.path.join(tmpdir, f"muxed.{vid_format}")

        # Write silent video — try torchvision.io first, fall back to raw ffmpeg pipe
        T, H, W, C = vid.shape
        try:
            torchvision.io.write_video(silent_path, vid, fps=fps, video_codec="libx264")
        except Exception:
            # Fallback: pipe raw frames to ffmpeg
            write_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                silent_path,
            ]
            proc = subprocess.run(write_cmd, input=vid.numpy().tobytes(), capture_output=True, timeout=60)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg raw write failed: {proc.stderr.decode()}")

        # Mux audio with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", silent_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-loglevel", "error",
            muxed_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode == 0 and os.path.exists(muxed_path):
            return wandb.Video(muxed_path, fps=fps, format=vid_format, caption=caption)
        else:
            logger.warning(f"ffmpeg muxing failed (rc={result.returncode}): {result.stderr.decode()}")
            return wandb.Video(video_tensor[:1].cpu().numpy(), fps=fps, format=vid_format, caption=caption)

    except Exception as e:
        logger.warning(f"Audio muxing failed, falling back to silent video: {e}")
        return wandb.Video(video_tensor[:1].cpu().numpy(), fps=fps, format=vid_format, caption=caption)


def to_wandb(
    tensor: torch.Tensor,
    rgb_range: float = 255.0,
    normalized: bool = False,
    max_plot_img: int = 16,
    max_plot_vid: int = 2,
    fps: int = 16,
    channel_before_time: bool = True,
    caption: str | None = None,
    vid_format: str = "mp4",
) -> wandb.Image | wandb.Video:
    """
    Convert a tensor to a wandb.Image or wandb.Video.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B,C,H,W], [B,T,C,H,W], or [B,T,C,H,W,D].
        rgb_range (float, optional): Output target RGB range (can almost definitely be kept as 255).
            Defaults to 255.0.
        normalized (bool, optional): Whether the tensor is normalized to [0,1]. Defaults to False which assumes [-1,1] range.
        max_plot_img (int, optional): Max number of images to plot. Defaults to 16.
        max_plot_vid (int, optional): Max number of videos to plot. Defaults to 2.
        fps (int, optional): Frames per second. Defaults to 8.
        channel_before_time (bool, optional): Whether the tensor is in the format [B,C,T,..]. Set False if the [B,T,C,..] format is used.
        caption (str, optional): Caption for the image or video. Defaults to None.
        vid_format (str, optional): Format of the video file. Defaults to "mp4".

    Returns:
        wandb.Image | wandb.Video: Format a tensor for logging to W&B.
    """

    if tensor.ndim == 5:
        max_plot = max_plot_vid
        if channel_before_time:
            tensor = tensor.permute(0, 2, 1, 3, 4)
    elif tensor.ndim == 4:
        max_plot = max_plot_img
    else:
        raise ValueError(f"Tensor must be 4 or 5 dimensional, but got {tensor.ndim} dimensions")

    # slice and adjust range
    if normalized:
        factor = rgb_range
        offset = 0.0
    else:
        factor = rgb_range / 2.0
        offset = rgb_range / 2.0
    tensor = tensor[:max_plot].mul(factor).add(offset).clip_(0, rgb_range).to(torch.uint8)

    # convert to wandb.Image or wandb.Video
    assert tensor.shape[-3] == 3, "Make sure that the data is in ..., C, H, W format"
    if tensor.ndim == 5:
        return wandb.Video(tensor.cpu().numpy(), fps=fps, format=vid_format, caption=caption)
    else:
        image_grid = torchvision.utils.make_grid(tensor, nrow=4, pad_value=1)
        image_grid = tv_F.to_pil_image(image_grid)
        return wandb.Image(image_grid, caption=caption)


def _to_wandb_with_audio(
    tensor: torch.Tensor,
    audio_path: str,
    fps: int = 25,
    rgb_range: float = 255.0,
    normalized: bool = False,
    vid_format: str = "mp4",
    caption: str | None = None,
    channel_before_time: bool = True,
) -> wandb.Video:
    """Convert a video tensor to wandb.Video with audio muxed in.

    Handles the same normalization as to_wandb, then delegates to
    tensor_to_wandb_video_with_audio for ffmpeg muxing.

    Args:
        tensor: [B, C, T, H, W] or [B, T, C, H, W] video tensor in [-1,1] or [0,1] range.
        audio_path: Path to audio .wav file.
        fps: Frame rate for the output video.
        rgb_range: Target RGB range (255).
        normalized: Whether tensor is in [0,1] (True) or [-1,1] (False).
        vid_format: Video file format.
        caption: Optional caption.
        channel_before_time: Whether tensor is [B,C,T,H,W] (True) or [B,T,C,H,W] (False).

    Returns:
        wandb.Video with audio.
    """
    if channel_before_time:
        tensor = tensor.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] -> [B,T,C,H,W]

    # Normalize to uint8
    if normalized:
        factor = rgb_range
        offset = 0.0
    else:
        factor = rgb_range / 2.0
        offset = rgb_range / 2.0
    tensor = tensor[:1].mul(factor).add(offset).clip_(0, rgb_range).to(torch.uint8)

    return tensor_to_wandb_video_with_audio(
        tensor, audio_path, fps=fps, vid_format=vid_format, caption=caption,
    )


def _load_audio_waveform(audio_path: str, target_sr: int = 16000, num_frames: int = 81, fps: float = 25.0) -> Optional[torch.Tensor]:
    """Load audio waveform from .wav for SyncCScorer. Returns [L] float32 tensor or None."""
    if not audio_path or not os.path.isfile(audio_path):
        return None
    try:
        import scipy.io.wavfile as wavfile
        from scipy import signal
        sr, wav = wavfile.read(audio_path)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        wav = torch.from_numpy(wav)
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
        if sr != target_sr:
            num_samples_new = int(len(wav) * target_sr / sr)
            wav = torch.from_numpy(signal.resample(wav.numpy(), num_samples_new))
        target_length = int(num_frames / fps * target_sr)
        if wav.shape[0] < target_length:
            wav = torch.nn.functional.pad(wav, (0, target_length - wav.shape[0]))
        else:
            wav = wav[:target_length]
        return wav.to(torch.float32)
    except Exception as e:
        logger.warning(f"[SyncEval] Failed to load audio from {audio_path}: {e}")
        return None


@rank0_only
def init_wandb(config: BaseConfig):
    # wandb login
    wandb_credential = config.log_config.wandb_credential
    if os.path.isfile(wandb_credential):
        os.environ["WANDB_API_KEY"] = open(wandb_credential, encoding="utf-8").read().strip("\n")
        logger.info(f"Loading WANDB_API_KEY from {wandb_credential}")

    wandb_config = config.log_config

    # Resume with or generate a wandb id
    logger.info(f"wandb_config.save_path: {wandb_config.save_path}")
    os.makedirs(wandb_config.save_path, exist_ok=True)
    wandb_id_path = f"{wandb_config.save_path}/wandb_id.txt"
    resuming = getattr(config.trainer, "resume", True)
    if os.path.isfile(wandb_id_path) and resuming:
        wandb_id = open(wandb_id_path, encoding="utf-8").read().strip()
        logger.info(f"Resuming with an existing wandb id: {wandb_id}")
    else:
        wandb_id = wandb.util.generate_id()
        with open(wandb_id_path, "w", encoding="utf-8") as f:
            f.write(f"{wandb_id}\n")
        logger.info(f"Generating a wandb id: {wandb_id}")

    # Get config as plain dict
    config_resolved = serialize_config(config, return_type="dict")

    # Initialize the wandb library.
    wandb.init(
        id=wandb_id,
        project=wandb_config.project,
        group=wandb_config.group,
        name=wandb_config.name,
        entity=getattr(wandb_config, "wandb_entity", None),
        config=config_resolved,
        dir=wandb_config.save_path,
        resume="allow",
        mode=wandb_config.wandb_mode,
    )

    # Save a copy of code to a wandb Artifact (this can be slow)
    # Make code upload optional to avoid distributed training delays
    upload_code = basic_utils.str2bool(os.getenv("WANDB_UPLOAD_CODE", "false"))
    if upload_code:
        logger.info("Uploading code to wandb (this may take a few minutes)...")
        wandb.run.log_code(".")
        logger.info("Code upload to wandb completed")
    else:
        logger.info("Wandb code upload disabled (set WANDB_UPLOAD_CODE=true to enable)")


@dataclass
class _LossDictRecord:
    loss_dict: dict = field(default_factory=dict)
    iter_count_dict: dict = field(default_factory=dict)

    def add(self, loss_dict: Optional[Dict[str, torch.Tensor]]) -> None:
        if loss_dict is not None:
            for loss_name, loss_val in loss_dict.items():
                scalar = loss_val.float().item() if torch.is_tensor(loss_val) else float(loss_val)
                self.loss_dict[loss_name] = self.loss_dict.get(loss_name, 0.0) + scalar
                self.iter_count_dict[loss_name] = self.iter_count_dict.get(loss_name, 0) + 1

    def reset(self) -> None:
        self.loss_dict = {}
        self.iter_count_dict = {}

    def gather_dict(self, dictionary: Dict[str, float | int]) -> Dict[str, float | int]:
        n_ranks = world_size()
        if n_ranks > 1:
            dict_list = [None for _ in range(n_ranks)]
            torch.distributed.all_gather_object(dict_list, dictionary)
            # from list of dicts to dict of summed values
            dictionary = {}
            for d in dict_list:
                for key, value in d.items():
                    dictionary[key] = dictionary.get(key, 0.0) + value
        return dictionary

    def get_stat(self) -> Dict[str, float]:
        # number of ranks that logged this loss
        rank_dict = self.gather_dict({k: 1 for k in self.loss_dict.keys()})
        # number of times this loss was computed
        count_dict = self.gather_dict(self.iter_count_dict)
        # sum of all losses
        loss_dict = self.gather_dict(self.loss_dict)

        avg_loss_dict = {}
        for loss_name, loss_val in loss_dict.items():
            count = count_dict.get(loss_name, 0)
            ranks = rank_dict.get(loss_name, 1)
            iter_count = count / ranks
            avg_loss = (loss_val / count) * (ranks / world_size()) if count > 0 else 0.0
            logger.info(f"avg_{loss_name}: {avg_loss:.4f}".ljust(30) + f"iter count: {iter_count}")
            avg_loss_dict[loss_name] = avg_loss
        self.reset()
        return avg_loss_dict


class WandbCallback(Callback):
    """
    The callback gets precision for data from model
    """

    def __init__(
        self,
        *args,
        validation_logging_step: int = 1,
        sample_logging_iter: Optional[int] = None,
        vid_format: str = "mp4",
        fps: int = 16,
        syncnet_checkpoint_path: Optional[str] = None,
        syncnet_vshift: int = 15,
        syncnet_audio_sr: int = 16000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.validation_logging_step = validation_logging_step
        self.sample_logging_iter = sample_logging_iter
        self.val_sample_map = None
        self._val_gen_videos: list[torch.Tensor] = []
        self._val_gt_videos: list[torch.Tensor] = []
        self._val_audio_paths: list[str | None] = []
        self.vid_format = vid_format
        self.fps = fps
        self.syncnet_checkpoint_path = syncnet_checkpoint_path
        self.syncnet_vshift = syncnet_vshift
        self.syncnet_audio_sr = syncnet_audio_sr
        self._syncnet_scorer = None
        self._val_sync_c_scores: list[float] = []
        self.loss_dict_record = _LossDictRecord()
        self.val_loss_dict_record = _LossDictRecord()

    def on_app_begin(self) -> None:
        assert hasattr(self, "config"), "Missing config in WandbCallback."
        init_wandb(self.config)
        self.offload_module_in_decoding = self.config.trainer.offload_module_in_decoding
        # disable offloading if using FSDP
        if self.config.trainer.fsdp:
            self.offload_module_in_decoding = False
        if self.sample_logging_iter is None:
            self.sample_logging_iter = self.config.trainer.logging_iter
        synchronize()

    def _get_syncnet_scorer(self):
        if self._syncnet_scorer is not None:
            return self._syncnet_scorer
        if not self.syncnet_checkpoint_path:
            return None
        try:
            from lipforcing.methods.reward.sync_c_scorer import SyncCScorer
            self._syncnet_scorer = SyncCScorer(
                checkpoint_path=self.syncnet_checkpoint_path,
                input_fps=25.0,
                audio_sample_rate=self.syncnet_audio_sr,
                vshift=self.syncnet_vshift,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float32,
            )
            logger.info(f"[WandbCallback] Loaded SyncCScorer from {self.syncnet_checkpoint_path}")
        except Exception as e:
            logger.warning(f"[WandbCallback] Failed to load SyncCScorer: {e}")
            self.syncnet_checkpoint_path = None
        return self._syncnet_scorer

    def on_dataloader_init_end(
        self, model: FastGenModel, dataloader_train, dataloader_val, iteration: int = 0
    ) -> None:
        """Upload GT validation videos at the start so they're always available for comparison.

        Not decorated with @rank0_only — all ranks must enter this method to stay
        synchronized (synchronize() calls dist.barrier). Only rank 0 does the actual
        VAE decode and wandb upload.
        """
        if dataloader_val is None:
            return
        # Skip GT upload if SKIP_GT_VAL_UPLOAD env var is set (avoids NCCL timeout)
        if os.environ.get("SKIP_GT_VAL_UPLOAD", "0") == "1":
            if wandb.run:
                logger.info("SKIP_GT_VAL_UPLOAD=1 — skipping GT val video upload")
            synchronize()
            return
        if iteration > 0:
            if wandb.run:
                logger.info("Resuming from checkpoint — skipping GT val video upload (already logged)")
            synchronize()
            return
        if not hasattr(model.net, "vae"):
            if wandb.run:
                logger.info("No VAE loaded — skipping GT val video upload")
            synchronize()
            return

        # Only rank 0 decodes and uploads; other ranks wait at the barrier below
        if wandb.run:
            logger.info("Uploading GT validation videos to wandb...")
            device = model.device
            try:
                gt_videos = []
                audio_paths = []
                with torch.no_grad(), basic_utils.inference_mode(
                    precision_amp=model.precision_amp_enc, device_type=device.type
                ):
                    for step, data in enumerate(dataloader_val):
                        real = data["real"].to(device)  # [1, 16, 21, 64, 64]
                        decoded = model.net.vae.decode(real[:1])  # [1, C, T, H, W]
                        gt_videos.append(self._to_uint8_video(decoded))
                        ap = None
                        if "audio_path" in data:
                            raw = data["audio_path"]
                            if isinstance(raw, (list, tuple)) and len(raw) > 0 and raw[0]:
                                ap = raw[0] if os.path.isfile(raw[0]) else None
                        audio_paths.append(ap)
                gt_list = []
                for v, ap in zip(gt_videos, audio_paths):
                    if ap:
                        gt_list.append(tensor_to_wandb_video_with_audio(v, ap, fps=self.fps))
                    else:
                        gt_list.append(wandb.Video(v[0].numpy(), fps=self.fps, format="mp4"))
                wandb.log({"val_gt/videos": gt_list}, step=0)
                logger.info(f"Uploaded {len(gt_videos)} GT validation videos to wandb")
            except Exception as e:
                logger.warning(f"Failed to upload GT val videos: {e}")
        synchronize()

    @rank0_only
    def on_optimizer_step_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        assert hasattr(self, "config"), "Missing config in WandbCallback."
        if iteration % self.config.trainer.logging_iter == 0:
            for name, scheduler in model.scheduler_dict.items():
                wandb.log({f"optimizer/lr_{name}": scheduler.get_last_lr()[0]}, step=iteration)

    def get_sample_map(
        self, model: FastGenModel, data_batch: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor | Callable]
    ) -> dict[str, wandb.Image | wandb.Video]:
        # Collect generated and real data and create copies to avoid modifying the original dicts
        sample_map = {}
        gen_rand = output_batch["gen_rand"]
        if isinstance(gen_rand, Callable):
            synchronize()
            gen_rand = gen_rand()
            synchronize()

        # Avoid modifying the original dicts
        data_batch = data_batch.copy()
        output_batch = output_batch.copy()

        # Decide whether we want to visualize multistep teacher generation
        if self.config.trainer.visualize_teacher:
            assert "input_rand" in output_batch, "We need to know the noise to visualize teacher generation"
            teacher_output = model.sample(
                model.teacher,
                output_batch["input_rand"][0:1],
                data_batch["condition"][0:1],  # e.g. text condition encoded by the text encoder
                data_batch["neg_condition"][0:1],  # e.g. negative text condition encoded by the text encoder
            )
            output_batch["gen_teacher"] = teacher_output

        # Decode to pixel if it's in latent space
        if hasattr(model.net, "init_preprocessors"):
            torch.cuda.empty_cache()
            device_nets = model.device

            has_vae = hasattr(model.net, "vae")
            if not has_vae:
                model.net.init_vae()
                model.net.vae.to(device=device_nets, dtype=model.precision)

            if self.offload_module_in_decoding:
                # offload the unneeded models to CPU (enable it if hitting OOM here)
                logger.info(
                    f"GPU Memory BEFORE moving nets to CPU: {torch.cuda.memory_allocated(device_nets) / 1024 ** 2:.2f} MB"
                )
                if hasattr(model, "fake_score"):
                    model.fake_score = model.fake_score.to("cpu")
                if hasattr(model, "teacher"):
                    model.teacher = model.teacher.to("cpu")
                logger.info(
                    f"GPU Memory AFTER moving nets to CPU: {torch.cuda.memory_allocated(device_nets) / 1024 ** 2:.2f} MB"
                )
                synchronize()

            with basic_utils.inference_mode(precision_amp=model.precision_amp_enc, device_type=device_nets.type):
                if "real" in data_batch:
                    # only generate one sample for video
                    limit = 1 if len(data_batch["real"].shape) == 5 else len(data_batch["real"])
                    data_batch["real"] = model.net.vae.decode(data_batch["real"][:limit])
                if isinstance(gen_rand, dict):
                    for k in gen_rand:
                        limit = 1 if len(gen_rand[k].shape) == 5 else len(gen_rand[k])
                        gen_rand[k] = model.net.vae.decode(gen_rand[k][:limit])
                else:
                    limit = 1 if len(gen_rand.shape) == 5 else len(gen_rand)
                    gen_rand = model.net.vae.decode(gen_rand[:limit])

                if "gen_teacher" in output_batch:
                    output_batch["gen_teacher"] = model.net.vae.decode(output_batch["gen_teacher"][:limit])
                if logger.LOG_LEVEL == "DEBUG" and "gen_rand_train" in output_batch:
                    output_batch["gen_rand_train"] = model.net.vae.decode(output_batch["gen_rand_train"][:limit])

            if not has_vae:
                del model.net.vae

            if self.offload_module_in_decoding:
                # move back fake_score to gpu
                if hasattr(model, "fake_score"):
                    model.fake_score = model.fake_score.to(device_nets)
                if hasattr(model, "teacher"):
                    model.teacher = model.teacher.to(device_nets)
                logger.info(
                    f"GPU Memory AFTER moving nets back to GPU: {torch.cuda.memory_allocated(device_nets) / 1024 ** 2:.2f} MB"
                )
                synchronize()

        if wandb.run:
            if (
                "condition_raw" in data_batch
                and isinstance(data_batch["condition_raw"], (list, tuple))
                and isinstance(data_batch["condition_raw"][0], str)
            ):
                caption = "\n".join(data_batch["condition_raw"][: len(gen_rand)])
            else:
                caption = None

            # Check for audio path (from OmniAvatar dataloader) for audio-muxed video logging.
            # The dataloader sets audio_path="" when audio.wav doesn't exist.
            audio_path = None
            if "audio_path" in data_batch:
                ap = data_batch["audio_path"]
                # default_collate turns strings into a list
                if isinstance(ap, (list, tuple)) and len(ap) > 0:
                    audio_path = ap[0] if ap[0] else None
                elif isinstance(ap, str):
                    audio_path = ap if ap else None
                # Verify the file actually exists
                if audio_path and not os.path.isfile(audio_path):
                    logger.warning(f"audio_path does not exist, logging silent video: {audio_path}")
                    audio_path = None

            if isinstance(gen_rand, dict):
                for k in gen_rand:
                    sample_map[f"student/generation/{k}"] = to_wandb(
                        gen_rand[k], caption=caption, vid_format=self.vid_format
                    )
            else:
                if audio_path and gen_rand.ndim == 5:
                    sample_map["student/generation"] = _to_wandb_with_audio(
                        gen_rand, audio_path, fps=self.fps, vid_format=self.vid_format, caption=caption,
                    )
                else:
                    sample_map["student/generation"] = to_wandb(gen_rand, caption=caption, vid_format=self.vid_format, fps=self.fps)
            if "real" in data_batch:
                if audio_path and data_batch["real"].ndim == 5:
                    sample_map["data/real"] = _to_wandb_with_audio(
                        data_batch["real"], audio_path, fps=self.fps, vid_format=self.vid_format, caption=caption,
                    )
                else:
                    sample_map["data/real"] = to_wandb(data_batch["real"], caption=caption, vid_format=self.vid_format, fps=self.fps)
            if "gen_teacher" in output_batch:
                sample_map["teacher/generation"] = to_wandb(
                    output_batch["gen_teacher"], caption=caption, vid_format=self.vid_format
                )
            if logger.LOG_LEVEL == "DEBUG" and "gen_rand_train" in output_batch:
                sample_map["student/generation_train"] = to_wandb(
                    output_batch["gen_rand_train"], caption=caption, vid_format=self.vid_format
                )

        return sample_map

    def log_sample_map(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        suffix: str = "",
        iteration: int = 0,
        group: str = "train",
    ) -> None:
        sample_map = self.get_sample_map(model, data_batch, output_batch)
        sample_map = {f"{group}_media/{k}{suffix}": v for k, v in sample_map.items()}
        if wandb.run:
            wandb.log(sample_map, step=iteration)
        synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def log_stats(self, loss_dict_record: _LossDictRecord, iteration: int = 0, group: str = "train") -> None:
        logger.info(f"logging {group} stats at iteration {iteration}" + "-" * 20)
        # Collect distributed statistics
        avg_loss_dict = loss_dict_record.get_stat()
        stats = {f"{group}/{name}": val for name, val in avg_loss_dict.items()}
        base_info = {"optimizer/iteration": iteration}

        # log stats and base info
        if wandb.run:
            wandb.log(stats, step=iteration)
            wandb.log(base_info, step=iteration)

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        self.loss_dict_record.add(loss_dict)
        time_start = time.perf_counter()
        logged = False
        if iteration % self.config.trainer.logging_iter == 0 or iteration == 1:
            self.log_stats(self.loss_dict_record, iteration=iteration, group="train")
            logged = True
        skip_early_sample = os.environ.get("SKIP_EARLY_SAMPLE_LOG", "0") == "1"
        if iteration % self.sample_logging_iter == 0 or (iteration == 1 and not skip_early_sample):
            self.log_sample_map(model, data_batch, output_batch, iteration=iteration, group="train")
            logged = True
        if logged:
            time_taken = time.perf_counter() - time_start
            logger.info(f"WandB logging complete after {time_taken:.2f} seconds")

    @staticmethod
    def _to_uint8_video(tensor: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """Convert [B, C, T, H, W] float video to [B, T, C, H, W] uint8 on CPU."""
        t = tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
        if normalized:
            t = t.mul(255.0)
        else:
            t = t.mul(127.5).add(127.5)
        return t.clamp(0, 255).to(torch.uint8).cpu()

    def on_validation_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        self.val_loss_dict_record.add(loss_dict)

        if step % self.validation_logging_step == 0:
            has_vae = hasattr(model.net, "vae")
            if not has_vae:
                return

            # AR-generate the video
            gen_rand = output_batch.get("gen_rand")
            if gen_rand is not None and isinstance(gen_rand, Callable):
                synchronize()
                gen_rand = gen_rand()
                synchronize()

            if gen_rand is None:
                return

            # VAE decode + video collection — rank 0 only to avoid FSDP deadlock.
            # Other ranks wait at synchronize() below.
            _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if _rank == 0:
                device = model.device
                with torch.no_grad(), basic_utils.inference_mode(
                    precision_amp=model.precision_amp_enc, device_type=device.type
                ):
                    gen_decoded = model.net.vae.decode(gen_rand[:1])
                    gt_decoded = model.net.vae.decode(data_batch["real"][:1].to(device))

                self._val_gen_videos.append(self._to_uint8_video(gen_decoded))
                self._val_gt_videos.append(self._to_uint8_video(gt_decoded))

                # Extract audio path for muxing
                audio_path = None
                if "audio_path" in data_batch:
                    ap = data_batch["audio_path"]
                    if isinstance(ap, (list, tuple)) and len(ap) > 0 and ap[0]:
                        audio_path = ap[0] if os.path.isfile(ap[0]) else None
                self._val_audio_paths.append(audio_path)

                # SyncNet-v2 evaluation on generated video
                scorer = self._get_syncnet_scorer()
                if scorer is not None and audio_path is not None:
                    try:
                        audio_wav = _load_audio_waveform(
                            audio_path, target_sr=self.syncnet_audio_sr,
                        )
                        if audio_wav is not None:
                            pixels = gen_decoded.clamp(-1.0, 1.0)
                            u8 = ((pixels + 1.0) * 127.5).to(torch.uint8)
                            face_frames = u8[0].permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
                            sync_c = scorer._score_single(face_frames, audio_wav)
                            self._val_sync_c_scores.append(sync_c.item())
                            logger.info(f"[SyncEval] val sample {step}: sync_c={sync_c.item():.3f}")
                    except Exception as e:
                        logger.warning(f"[SyncEval] Failed on val sample {step}: {e}")

            synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def on_validation_end(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        self.log_stats(self.val_loss_dict_record, iteration=iteration, group=f"val{idx}")
        if wandb.run and self._val_gen_videos:
            gen_list = []
            gt_list = []
            for i, (gen_v, gt_v) in enumerate(zip(self._val_gen_videos, self._val_gt_videos)):
                ap = self._val_audio_paths[i] if i < len(self._val_audio_paths) else None
                caption = None
                if i < len(self._val_sync_c_scores):
                    caption = f"sync_c={self._val_sync_c_scores[i]:.3f}"
                if ap:
                    gen_list.append(tensor_to_wandb_video_with_audio(gen_v, ap, fps=self.fps, caption=caption))
                    gt_list.append(tensor_to_wandb_video_with_audio(gt_v, ap, fps=self.fps))
                else:
                    gen_list.append(wandb.Video(gen_v[0].numpy(), fps=self.fps, format="mp4", caption=caption))
                    gt_list.append(wandb.Video(gt_v[0].numpy(), fps=self.fps, format="mp4"))
            wandb.log({
                f"val{idx}/generated": gen_list,
                f"val{idx}/reconstructed": gt_list,
            }, step=iteration)
            logger.info(f"Logged {len(self._val_gen_videos)} val videos at iteration {iteration}")
            if self._val_sync_c_scores:
                mean_sync_c = sum(self._val_sync_c_scores) / len(self._val_sync_c_scores)
                wandb.log({f"val{idx}/sync_c_mean": mean_sync_c}, step=iteration)
                for i, sc in enumerate(self._val_sync_c_scores):
                    wandb.log({f"val{idx}/sync_c_sample_{i}": sc}, step=iteration)
                logger.info(f"[SyncEval] val{idx} mean sync_c: {mean_sync_c:.3f} (n={len(self._val_sync_c_scores)})")
        self._val_gen_videos = []
        self._val_gt_videos = []
        self._val_audio_paths = []
        self._val_sync_c_scores = []
