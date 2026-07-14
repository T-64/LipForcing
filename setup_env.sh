#!/usr/bin/env bash
# LipForcing environment setup via venv (conda channels are blocked in this env).
# Uses python 3.10.20 from the existing `synctalk` conda env -> clean isolated venv.
# pip goes through the working Tencent mirror (mirrors.cloud.tencent.com).
set -e

PYBIN=/root/miniconda3/envs/synctalk/bin/python
VENV=/root/code/LipForcing/.venv

echo "=== [1/4] create venv with python 3.10.20 ==="
$PYBIN --version
$PYBIN -m venv "$VENV"
source "$VENV/bin/activate"
python --version
which python pip

echo "=== [2/4] upgrade pip/wheel/setuptools ==="
pip install --upgrade pip wheel setuptools

echo "=== [3/4] pip install -e /root/code/LipForcing (Tencent mirror) ==="
cd /root/code/LipForcing
pip install -e .

echo "=== [4/4] sanity check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=== DONE ==="
