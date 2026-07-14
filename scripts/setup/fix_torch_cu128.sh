#!/usr/bin/env bash
# Fix torch: uninstall cu130 build, install torch 2.10.0+cu128 (matches README test config, driver 535 compatible).
source /root/code/LipForcing/proxy.env
source /root/code/LipForcing/.venv/bin/activate
set -e
echo "=== uninstall cu130 torch ==="
pip uninstall -y torch torchvision torchaudio
echo "=== install torch 2.10.0+cu128 ==="
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision torchaudio
echo "=== verify ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "TORCH_FIX_DONE"
