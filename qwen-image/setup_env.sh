#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Initializing Python 3.11 virtual environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

echo "=== Installing PyTorch with AMD ROCm 6.2 and Diffusers stack ==="
pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

echo "=== Installing Hugging Face and Diffusers dependencies ==="
pip install \
    "diffusers>=0.33.0" \
    "transformers>=4.49.0" \
    "accelerate>=1.4.0" \
    safetensors \
    pillow \
    "huggingface_hub[cli]" \
    sentencepiece \
    protobuf

echo ""
echo "=== Verifying ROCm GPU detection ==="
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python3 -c "
import torch
print('PyTorch Version:   ', torch.__version__)
print('CUDA / ROCm Built: ', torch.version.cuda or torch.version.hip)
print('GPU Available:     ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:       ', torch.cuda.get_device_name(0))
    print('VRAM Total (GB):   ', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
"

echo "=== Setup complete! ==="
