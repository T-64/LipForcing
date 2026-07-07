# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_namespace_packages
from pathlib import Path


def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="lipforcing",
    version="0.1.0",
    description="Lip Forcing: Few-Step Autoregressive Diffusion for Real-time Lip Synchronization.",
    license="Apache-2.0",
    packages=find_namespace_packages(include=["lipforcing*", "OmniAvatar*"]),
    python_requires=">=3.10",
    install_requires=read_requirements(),
)
