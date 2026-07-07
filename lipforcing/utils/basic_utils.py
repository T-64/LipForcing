# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Dict
import contextlib
import attrs
from collections.abc import Mapping, Iterable
from contextlib import contextmanager
import random
from typing import TYPE_CHECKING, Any
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lipforcing.utils.distributed import world_size, get_rank
import lipforcing.utils.logging_utils as logger

if TYPE_CHECKING:
    from lipforcing.configs.config import BaseConfig

PRECISION_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def get_batch_size_total(config: BaseConfig):
    # accumulated batch size per GPU
    batch_size = config.dataloader_train.batch_size * config.trainer.grad_accum_rounds
    return batch_size * world_size()


def to_str(obj: Any) -> str | Dict[Any, str]:
    """Print the object in a readable format. Typically used for batches of data."""
    if isinstance(obj, torch.Tensor):
        return f"Tensor{list(obj.shape)}"
    elif isinstance(obj, str):
        dots = "..." if len(obj) > 10 else ""
        return f"{dots}{obj[-10:]}"
    elif isinstance(obj, Mapping):
        return {k: to_str(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return str([to_str(v) for v in obj])
    return str(obj)


@contextmanager
def inference_mode(*modules: torch.nn.Module, precision_amp: torch.dtype | None = None, device_type: str = "cuda"):
    """
    Wraps torch.inference_mode() and temporarily sets the provided modules
    to .eval() mode. If precision_amp is not None, it also wraps the context in torch.autocast().

    Args:
        *modules: Modules to set temporarily to eval mode.
        precision_amp: If not None, wraps the context in torch.autocast().
        device_type: Device type to use for autocast.

    Returns:
        Generator that yields the context manager.

    Upon exit, it restores the original .training state of each module.
    """
    # 1. Capture the original training state of each module
    #    (True if in train mode, False if in eval mode)
    modules = [mod for mod in modules if isinstance(mod, torch.nn.Module)]
    previous_states = [mod.training for mod in modules]

    try:
        # 2. Set all specific modules to eval mode
        #    This is crucial for layers like Dropout and BatchNorm
        for mod in modules:
            mod.eval()

        # 3. Enter strict inference mode (disables gradients, etc.) and autocast if needed
        with torch.inference_mode(), torch.autocast(
            dtype=precision_amp, device_type=device_type, enabled=precision_amp is not None
        ):
            yield

    finally:
        # 4. Restore the original state of each module
        for mod, was_training in zip(modules, previous_states):
            mod.train(was_training)


def set_random_seed(
    seed: int, iteration: int = 0, by_rank: bool = False, devices: List[torch.device | str | int] | None = None
) -> int:
    """Set random seed for `random, numpy, Pytorch, cuda`.

    Args:
        seed (int): Random seed.
        by_rank (bool): if set to true, each GPU will use a different random seed.
        devices (List[torch.device] | None): devices to set the seed on. If None, will set the seed on all devices.
    Returns:
        The final random seed for the current rank.
    """
    seed += iteration
    if by_rank:
        seed += get_rank()
    seed %= 1 << 31
    logger.info(f"Using random seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    if devices is None:
        # sets seed on the current CPU & all GPUs
        torch.manual_seed(seed)
    else:
        # set the seed on cpu
        torch.default_generator.manual_seed(seed)
        # set the seed on devices
        for device in devices:
            # get device index (as in torch.cuda.set_rng_state)
            if isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device("cuda", device)
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            torch.cuda.default_generators[idx].manual_seed(seed)
    return seed


@contextlib.contextmanager
def set_tmp_random_seed(
    seed, iteration: int = 0, by_rank: bool = False, devices: List[torch.device | str | int] | None = None
):
    """A context manager to temporarily set the random seeds.

    Args:
        seed (int): Random seed.
        iteration (int): Iteration number.
        by_rank (bool): if set to true, each GPU will use a different random seed.
        devices (List[torch.device] | None): devices to set the seed on. If None, will set the seed on all devices.
    """
    if seed is None:
        yield
        return

    # Save the original random states
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # Fork torch state
        with torch.random.fork_rng(devices=devices):
            # Set the new seeds
            set_random_seed(seed, iteration=iteration, by_rank=by_rank, devices=devices)
            yield
    finally:
        # Restore the original random states
        np.random.set_state(np_state)
        random.setstate(py_state)


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype, and/or memory_format.

    The input data can be a tensor, a list of tensors, a dict of tensors.
    See the documentation for torch.Tensor.to() for details.

    Args:
        data (Any): Input data.
        device (str | torch.device): GPU device (default: None).
        dtype (torch.dtype): data type (default: None).

    Returns:
        data (Any): Data cast to the specified device, dtype, and/or memory_format.
    """
    assert device is not None or dtype is not None, "at least one of device, dtype should be specified"
    if isinstance(data, torch.Tensor):
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        if data.dtype == torch.int64:
            # t variable is int64 for some networks (e.g. CogVideoX, Stable Diffusion)
            dtype = torch.int64

        data = data.to(
            device=device,
            dtype=dtype,
            non_blocking=(not is_cpu),
        )
        return data
    elif isinstance(data, (list, tuple)):
        return type(data)(to(d, device, dtype) for d in data)
    elif isinstance(data, dict):
        return {k: to(v, device, dtype) for k, v in data.items()}
    else:
        return data


def convert_cfg_to_dict(cfg) -> dict:
    """Convert config to dictionary, handling both OmegaConf and attrs cases.

    Args:
        cfg: Either a DictConfig (from OmegaConf/Hydra) or Config (attrs class)

    Returns:
        Dictionary representation of the config
    """
    if isinstance(cfg, DictConfig):
        # Production case: OmegaConf DictConfig
        return OmegaConf.to_container(cfg, resolve=True)
    else:
        # Test case: attrs SampleTConfig class
        return attrs.asdict(cfg)


def detach(
    data: Any,
) -> Any:
    """Recursively detach data if it is a tensor.

    Args:
        data (Any): Input data.
    Returns:
        data (Any): Data detached from the computation graph.
    """
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, (list, tuple)):
        return type(data)(detach(d) for d in data)
    elif isinstance(data, dict):
        return {k: detach(v) for k, v in data.items()}
    else:
        return data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")
