# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path
import io
import torch.nn
from typing import Optional, Dict, Union, Any

from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

from lipforcing.configs.config import BaseCheckpointerConfig
from lipforcing.utils.distributed.s3_filesystem import S3StorageWriter, S3StorageReader
from lipforcing.utils.io_utils import s3_load, s3_save, latest_checkpoint
import lipforcing.utils.logging_utils as logger
from lipforcing.utils.distributed import synchronize, is_rank0
from lipforcing.callbacks.callback import CallbackDict


class Checkpointer:
    """Class to save and load model checkpoints"""

    def __init__(self, config: BaseCheckpointerConfig):
        self.config = config

    def _save_checkpoint(self, save_dict: Dict[str, Any], path: str):
        assert path.endswith(".pth"), f"{path} does not end with .pth"
        if self.config.use_s3:
            assert path.startswith("s3://"), f"{path} does not start with s3:// when using s3 storage"
            logger.info(f"Saving the model to {path}")
            # convert the save_dict to bytes
            buffer = io.BytesIO()
            torch.save(save_dict, buffer)
            s3_save(path, buffer.getvalue())
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(save_dict, path)

    def _load_checkpoint(self, path: str, device: torch.device = "cpu") -> Dict[str, Any]:
        assert path.endswith(".pth"), f"{path} does not end with .pth"
        if self.config.use_s3:
            assert path.startswith("s3://"), f"{path} does not start with s3:// when using s3 storage"
            state = torch.load(s3_load(path), map_location=device)
        else:
            assert os.path.exists(path), f"{path} does not exist"
            state = torch.load(path, map_location=device, weights_only=False)
        return state

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """
        synchronize()
        model_state = {k: v.state_dict() for k, v in model_dict.items()}
        optim_state = None if optimizer_dict is None else {k: v.state_dict() for k, v in optimizer_dict.items()}
        scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
        grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
        callbacks_state = None if callbacks is None else callbacks.state_dict()
        save_dict = {
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": scheduler_state,
            "grad_scaler": grad_scaler_state,
            "callbacks": callbacks_state,
            "iteration": iteration,
        }

        if is_rank0():
            if path is None:
                if self.config.use_s3:
                    path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}.pth")
                else:
                    path = os.path.join(self.config.save_dir, f"{iteration:07d}.pth")

            if not self.config.use_s3:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Saving the model to {path}")
            self._save_checkpoint(save_dict, path)

        logger.success(f"Model saved at iteration {iteration}")
        synchronize()
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: Optional[torch.device] = "cpu",
    ) -> int:
        """Load the model checkpoint

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): The path to the checkpoint file
            device (Optional[torch.device]): The device to load the model to. Defaults to "cpu".

        Returns:
            int: The iteration number

        """
        # TODO: rank 0 load and broadcast to all other ranks
        if path is None:
            if self.config.use_s3:
                checkpoint_path = os.path.join(self.config.s3_container, self.config.save_dir)
            else:
                checkpoint_path = self.config.save_dir
            path = latest_checkpoint(checkpoint_path) + ".pth"
            if path == ".pth":
                # no checkpoint found, starting from iteration 0
                return 0
        if not os.path.exists(path):
            logger.critical(f"Checkpoint file not found at {path}")
            return 0
        logger.info(f"Loading model from {path}")
        state = self._load_checkpoint(path, device=device)

        logger.info("Loading the model_dict...")
        for k, v in model_dict.items():
            if k in state["model"] and v is not None:
                # strict is False to allow evaluating external checkpoints without, e.g., logvar parameters
                model_load_info = v.load_state_dict(state["model"][k], strict=False)
                logger.info(f"Model {k}, loading info: {model_load_info}")
            else:
                logger.warning(f"Model {k} not found in checkpoint.")

        if optimizer_dict is not None:
            logger.info("Loading the optimizer_dict...")
            for k, v in optimizer_dict.items():
                if k in state["optimizer"] and v is not None:
                    v.load_state_dict(state["optimizer"][k])
                else:
                    logger.warning(f"Optimizer {k} not found in checkpoint.")

        if scheduler_dict is not None:
            logger.info("Loading the scheduler_dict...")
            for k, v in scheduler_dict.items():
                if k in state["scheduler"] and v is not None:
                    v.load_state_dict(state["scheduler"][k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            # Check if saved grad_scaler state is non-empty (disabled scalers save empty state)
            if state.get("grad_scaler") and len(state["grad_scaler"]) > 0:
                grad_scaler.load_state_dict(state["grad_scaler"])
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            if "callbacks" in state:
                callbacks.load_state_dict(state["callbacks"])
            else:
                logger.warning("Callbacks not found in checkpoint.")

        if "iteration" not in state:
            logger.warning("Iteration not found in checkpoint.")
            return 0
        return state["iteration"]


class ModelWrapper(Stateful):
    """
    Wrapper for model state dict handling

    Code taken from this tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.
    """

    def __init__(self, model: torch.nn.Module, options: StateDictOptions | None = None):
        self.model = model
        # Default to strict=False so partial-state saves (e.g., LoRA +
        # selective-unfreeze runs that filter to trainable params only at
        # save time, see FSDPCheckpointer.save) can be loaded back without
        # PyTorch raising on missing frozen-base keys.  The frozen base
        # values are filled in during model construction (load base
        # safetensors + V2V adapter ckpt), so a strict=False load just
        # leaves them at their construction-time values — which is exactly
        # what we want for resume / inference.
        if options is None:
            options = StateDictOptions(strict=False)
        self.options = options

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, and sets the default state dict type to FSDP.SHARDED_STATE_DICT
        return get_model_state_dict(self.model, options=self.options)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_model_state_dict(self.model, model_state_dict=state_dict, options=self.options)


class OptimizerWrapper(Stateful):
    """
    Wrapper for optimizer state dict handling

    Code taken from this tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.
    """

    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, options: StateDictOptions | None = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.options = options

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, and sets the default state dict type to FSDP.SHARDED_STATE_DICT
        optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=self.options)
        return optimizer_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=state_dict, options=self.options)


class FSDPCheckpointer(Checkpointer):
    """Class to save and load model checkpoints"""

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.config.use_s3:
            return S3StorageWriter(
                credential_path=self.config.s3_credential,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.config.use_s3:
            return S3StorageReader(
                credential_path=self.config.s3_credential,
                path=checkpoint_path,
            )
        return FileSystemReader(checkpoint_path)

    def save(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        iteration: int = 0,
    ) -> str:
        """Save a checkpoint of the model (and optionally optimizer, scheduler, and grad scaler to resume training)

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to save
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to save
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to save
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to save
            path (str): The path to save the checkpoint file
            iteration (int): The iteration number

        Returns:
            str: The path to the saved checkpoint file
        """

        if path is None:
            if self.config.use_s3:
                path = os.path.join(self.config.s3_container, self.config.save_dir, f"{iteration:07d}")
            else:
                path = os.path.join(self.config.save_dir, f"{iteration:07d}")
                if not os.path.exists(self.config.save_dir):
                    os.makedirs(self.config.save_dir, exist_ok=True)
        if path.endswith(".pth"):  # In the case of autoresume
            path = path[:-4]
        logger.info(f"Saving FSDP model to prefix {path}")

        synchronize()
        # fsdp should save on all ranks
        for k, v in model_dict.items():
            model_state_dict = ModelWrapper(model=v).state_dict()
            # Filter to params with requires_grad=True (trainable-only) when
            # partial-freeze is in effect.  No-op for full-FT runs (where
            # every param has requires_grad=True; the filter keeps everything).
            # For LoRA + selective-unfreeze runs at 14B, this collapses save
            # size from ~56 GB (full base) to ~5 GB (LoRA + unfreeze).
            #
            # Non-parameter state (registered buffers, module-level Tensors
            # that aren't nn.Parameters) is preserved — only nn.Parameters
            # are filtered.  ``v.named_parameters()`` enumerates only
            # parameters; anything else in ``model_state_dict`` is kept.
            params_dict = dict(v.named_parameters())
            filtered = {
                key: tensor
                for key, tensor in model_state_dict.items()
                if key not in params_dict or params_dict[key].requires_grad
            }
            n_kept = len(filtered)
            n_total = len(model_state_dict)
            if n_kept < n_total:
                logger.info(
                    f"[FSDPCheckpointer] {k}_model: saving {n_kept}/{n_total} state-dict "
                    f"entries (filtered out {n_total - n_kept} frozen-param entries)"
                )
            storage_writer = self.get_storage_writer(checkpoint_path=f"{path}.{k}_model")
            dcp.save(filtered, storage_writer=storage_writer)

        if optimizer_dict is not None:
            for k, v in optimizer_dict.items():
                optim_state_dict = OptimizerWrapper(model=model_dict[k], optimizer=v).state_dict()
                storage_writer = self.get_storage_writer(checkpoint_path=f"{path}.{k}_optim")
                dcp.save(optim_state_dict, storage_writer=storage_writer)

        # other scalars only save on rank 0
        if is_rank0():
            scheduler_state = None if scheduler_dict is None else {k: v.state_dict() for k, v in scheduler_dict.items()}
            grad_scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
            callbacks_state = None if callbacks is None else callbacks.state_dict()
            save_dict = {
                "scheduler": scheduler_state,
                "grad_scaler": grad_scaler_state,
                "callbacks": callbacks_state,
                "iteration": iteration,
            }
            self._save_checkpoint(save_dict, f"{path}.pth")

        logger.success(f"Model saved at iteration {iteration}")
        synchronize()
        return path

    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: Optional[torch.device] = "cpu",
    ) -> int:
        """Load the model checkpoint

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): The path to the checkpoint file
            device (Optional[torch.device]): The device to load the model to. Defaults to "cpu".

        Returns:
            int: The iteration number

        """
        if path is None:
            if self.config.use_s3:
                checkpoint_path = os.path.join(self.config.s3_container, self.config.save_dir)
            else:
                checkpoint_path = self.config.save_dir
            path = latest_checkpoint(checkpoint_path)
            if path == "":
                # no checkpoint found, starting from iteration 0
                return 0
        if path.endswith(".pth"):
            # An FSDPCheckpointer-saved checkpoint stores model/optim shards
            # in sibling `<step>.{k}_model/` and `<step>.{k}_optim/` dirs and
            # writes only metadata (iteration, scheduler, callbacks, grad_scaler)
            # into the .pth file — there is NO `"model"` key in that .pth.
            # The regular `Checkpointer.load` at L156 indexes `state["model"]`
            # unconditionally and would KeyError.  Detect the FSDP layout by
            # looking for any sibling `*.<k>_model/` dir, and if present, fall
            # through to the FSDP load path below (treating the .pth as a
            # metadata stub for an FSDP-saved bundle).
            stripped_path = path[:-4]
            has_fsdp_layout = any(
                os.path.isdir(f"{stripped_path}.{k}_model") for k in model_dict.keys()
            )
            if has_fsdp_layout:
                logger.info(
                    f"Loading FSDP model (.pth metadata stub at {path}, "
                    f"sharded state in sibling *.{{k}}_model dirs)"
                )
                path = stripped_path
                # fall through to the FSDP code path below
            else:
                # regular DDP-saved single-file checkpoint
                logger.debug(f"Loading non-FSDP model from {path}")
                return super().load(
                    model_dict,
                    optimizer_dict=optimizer_dict,
                    scheduler_dict=scheduler_dict,
                    grad_scaler=grad_scaler,
                    callbacks=callbacks,
                    path=path,
                    device=device,
                )
        if not os.path.exists(f"{path}.pth"):
            logger.critical(f"Checkpoint file not found at {path}")
            return 0
        logger.info(f"Loading FSDP model from prefix {path}")

        for k, v in model_dict.items():
            logger.info(f"Loading the FSDP model dict for key {k}...")
            assert os.path.exists(f"{path}.{k}_model"), f"Key {k} does not exist in FSDP model dict"
            # Skip on ranks where parameters live on the meta device.  This
            # check MUST come before constructing the ModelWrapper / calling
            # state_dict(), because get_model_state_dict on a non-FSDP-wrapped
            # meta-init model can materialize the meta tensors as a side
            # effect, defeating the very memory savings we're trying to
            # protect.
            #
            # Context for the skip:
            # - This load path runs in two contexts: pretrained_ckpt_path
            #   (BEFORE FSDP wrap, model is unsharded) and resume (AFTER FSDP
            #   wrap, model is sharded with real per-rank tensors).
            # - Pre-FSDP-wrap, with fsdp_meta_init=True, only rank 0 holds
            #   real weights; ranks 1-3 hold meta tensors. dcp.load on a meta
            #   state_dict materializes those tensors, allocating real memory
            #   on every rank and defeating the meta-init memory savings —
            #   which combined with the bf16->fp32 cast in on_train_begin
            #   can push the cgroup over its memory cap.
            # - The skip is safe because model_to_fsdp(sync_module_states=True)
            #   broadcasts rank-0's loaded state to all ranks during FSDP wrap
            #   via set_model_state_dict(broadcast_from_rank0=True), filling
            #   the per-rank shards with the loaded values.
            # - Post-FSDP-wrap (resume path), parameters are real DTensors with
            #   per-rank shards — never on meta — so this check is False and
            #   the load proceeds normally.
            has_meta_params = any(p.is_meta for p in v.parameters())
            if has_meta_params:
                # Participate in DCP's collective planning protocol with an
                # empty state_dict so we don't allocate, but rank 0's dcp.load
                # doesn't deadlock waiting for our all_gather.  An empty dict
                # tells the planner this rank wants nothing — no allocation,
                # no disk read.
                storage_reader = self.get_storage_reader(checkpoint_path=f"{path}.{k}_model")
                logger.info(
                    f"[FSDPCheckpointer] {k}_model: meta-rank empty load "
                    f"(rank-0 broadcasts via sync_module_states later)"
                )
                dcp.load(state_dict={}, storage_reader=storage_reader)
                continue
            model_wrapper = ModelWrapper(model=v)
            model_state_dict = model_wrapper.state_dict()
            storage_reader = self.get_storage_reader(checkpoint_path=f"{path}.{k}_model")
            # Symmetric to save filter: only load trainable param entries.
            # Frozen base params remain at the model's current state (from
            # PEFT injection + pretrained init).  Non-parameter entries
            # (registered buffers) pass through.  No-op for full-FT runs.
            params_dict = dict(v.named_parameters())
            filtered_state_dict = {
                key: tensor
                for key, tensor in model_state_dict.items()
                if key not in params_dict or params_dict[key].requires_grad
            }
            n_kept = len(filtered_state_dict)
            n_total = len(model_state_dict)
            if n_kept < n_total:
                logger.info(
                    f"[FSDPCheckpointer] {k}_model: loading {n_kept}/{n_total} "
                    f"state-dict entries (skipping {n_total - n_kept} frozen-param entries)"
                )
            dcp.load(
                state_dict=filtered_state_dict,
                storage_reader=storage_reader,
            )
            model_wrapper.load_state_dict(model_state_dict)

        if optimizer_dict is not None:
            allow_missing_optim = os.environ.get("LIPFORCING_ALLOW_MISSING_FSDP_OPTIM", "").lower() in {
                "1",
                "true",
                "yes",
            }
            for k, v in optimizer_dict.items():
                logger.info(f"Loading the FSDP optimizer dict for key {k}...")
                optim_wrapper = OptimizerWrapper(model=model_dict[k], optimizer=v)
                # For fresh optimizers with no state, we need to initialize with fake gradients
                # that are DTensors (not regular Tensors) to avoid the mixed Tensor/DTensor error
                if len(v.state) == 0:
                    # Set fake DTensor gradients to initialize optimizer state
                    for param in model_dict[k].parameters():
                        if param.requires_grad and param.grad is None:
                            param.grad = torch.zeros_like(param)
                optim_state_dict = optim_wrapper.state_dict()
                assert os.path.exists(f"{path}.{k}_model"), f"Key {k} does not exist in FSDP model dict"
                optim_path = f"{path}.{k}_optim"
                if not os.path.exists(optim_path):
                    if allow_missing_optim:
                        logger.warning(
                            f"Optimizer checkpoint for {k} missing at {optim_path}; "
                            "LIPFORCING_ALLOW_MISSING_FSDP_OPTIM=1, so continuing with fresh optimizer state."
                        )
                        v.state.clear()
                        continue
                    raise FileNotFoundError(
                        f"Optimizer checkpoint for {k} not found at {optim_path}. "
                        "Set LIPFORCING_ALLOW_MISSING_FSDP_OPTIM=1 to resume model weights with fresh optimizer state."
                    )
                storage_reader = self.get_storage_reader(checkpoint_path=optim_path)

                try:
                    dcp.load(
                        state_dict=optim_state_dict,
                        storage_reader=storage_reader,
                    )
                    optim_wrapper.load_state_dict(optim_state_dict)
                    logger.success(f"Successfully loaded optimizer state for {k}")
                except Exception as e:
                    error_msg = str(e)
                    if (
                        "Missing key" in error_msg
                        or "Unexpected key" in error_msg
                        or "CheckpointException" in error_msg
                    ):
                        logger.warning(f"Optimizer checkpoint compatibility issue for {k}: {type(e).__name__}")
                        logger.warning(f"Initializing fresh optimizer state for {k} - training will continue")
                        # Reset to fresh optimizer state while preserving the optimizer's
                        # defaultdict-backed lazy state initialization.
                        v.state.clear()
                        logger.info(f"Reset optimizer state for {k} due to parameter mismatch")
                    else:
                        logger.error(f"Unexpected optimizer loading error for {k}: {e}")
                        raise e

        state = self._load_checkpoint(f"{path}.pth", device=device)

        if scheduler_dict is not None:
            logger.info("Loading the scheduler_dict...")
            for k, v in scheduler_dict.items():
                if k in state["scheduler"]:
                    v.load_state_dict(state["scheduler"][k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            # Check if saved grad_scaler state is non-empty (disabled scalers save empty state)
            if state.get("grad_scaler") and len(state["grad_scaler"]) > 0:
                grad_scaler.load_state_dict(state["grad_scaler"])
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            if "callbacks" in state:
                callbacks.load_state_dict(state["callbacks"])
            else:
                logger.warning("Callbacks not found in checkpoint.")

        return state["iteration"]
