# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The training entry script for LipForcing. Works for both DDP and FSDP training.
"""

import argparse
import warnings

from lipforcing.configs.config import BaseConfig
from lipforcing.utils import instantiate
from lipforcing.trainer import Trainer
import lipforcing.utils.logging_utils as logger
from lipforcing.utils.distributed import synchronize, clean_up
from lipforcing.utils.scripts import parse_args, setup

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 2.6.


def main(config: BaseConfig):
    # initiate the model
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    synchronize()

    # initiate the trainer
    logger.info("Initializing trainer...")
    fastgen_trainer = Trainer(config)
    logger.success("Trainer initialized successfully")
    synchronize()

    # Start training
    fastgen_trainer.run(model)
    synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    args = parse_args(parser)
    config = setup(args)

    main(config)

    clean_up()
    logger.info("Training finished.")
