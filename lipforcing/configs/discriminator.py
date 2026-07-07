# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from lipforcing.utils import LazyCall as L
from lipforcing.networks.discriminators import Discriminator_VideoDiT

# 1.3B patchify: spatial-2, temporal-1; inner_dim=1536; layer=30
Discriminator_Wan_1_3B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=30,
    disc_type="dit_simple_conv3d",
    inner_dim=1536 // 4,
)

# 14B patchify: spatial-2, temporal-1; inner_dim=5120; layer=40
Discriminator_Wan_14B_Config: DictConfig = L(Discriminator_VideoDiT)(
    feature_indices=None,
    num_blocks=40,
    disc_type="dit_simple_conv3d",
    inner_dim=5120 // 4,
)
