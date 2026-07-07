# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from lipforcing.methods.model import FastGenModel as FastGenModel

from lipforcing.methods.distribution_matching.dmd2 import DMD2Model as DMD2Model
from lipforcing.methods.distribution_matching.causvid import CausVidModel as CausVidModel
from lipforcing.methods.distribution_matching.self_forcing import SelfForcingModel as SelfForcingModel

from lipforcing.methods.knowledge_distillation.KD import KDModel as KDModel
from lipforcing.methods.knowledge_distillation.KD import CausalKDModel as CausalKDModel

from lipforcing.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel as OmniAvatarSelfForcingModel
from lipforcing.methods.omniavatar_diffusion_forcing import OmniAvatarDiffusionForcingModel as OmniAvatarDiffusionForcingModel
