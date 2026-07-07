# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple stdout loss logger callback for testing without wandb."""

from typing import Callable

import torch
from lipforcing.callbacks.callback import Callback
from lipforcing.methods.model import FastGenModel
import lipforcing.utils.logging_utils as logger


class StdoutLoggerCallback(Callback):
    """Prints loss values to stdout at every logging_iter."""

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        logging_iter = getattr(self.config.trainer, "logging_iter", 1) if self.config else 1
        if iteration % logging_iter == 0:
            parts = [f"iter {iteration:5d}"]
            for k, v in sorted(loss_dict.items()):
                if isinstance(v, torch.Tensor):
                    parts.append(f"{k}={v.item():.6f}")
                elif isinstance(v, (int, float)):
                    parts.append(f"{k}={v:.6f}")
            logger.info(" | ".join(parts))
