# Qwen-Image-2.1 Local Inference (PyTorch + AMD ROCm)

A pipeline and reproducible environment for running **Qwen-Image-2.1** (7B DiT + Qwen3-VL 9B text encoder) locally on AMD Radeon hardware (specifically validated on an AMD Radeon RX 7900 GRE with 16GB VRAM on NixOS).

## Key Features

- **Nix FHS Environment (`shell.nix`)**: Sets up POSIX driver paths and dependencies (`HSA_OVERRIDE_GFX_VERSION=11.0.0`, ROCm 6.2, Vulkan).
- **Official Hugging Face CLI Download (`download_model.sh`)**: Fast, resumable downloading directly using `hf download` without external curl calls.
- **Sequential CPU Offload**: Fits the full ~33 GB BF16 model within a 16 GB VRAM envelope without out-of-memory (OOM) errors.
- **Native Diffusers Pipeline (`generate.py`)**: Uses the `QwenImage21Pipeline` with full spatiotemporal VAE decoding.

---

## Quick Start

### 1. Enter the Nix Environment & Build Virtualenv

```bash
# Enter the Nix FHS shell
nix-shell shell.nix

# Initialize the venv and install PyTorch ROCm 6.2 stack
./setup_env.sh
```

### 2. Download Model Weights

Download the official `Qwen/Qwen-Image-2.1` weights:

```bash
./download_model.sh
```

The weights (~33 GB total) will be saved to `./models/Qwen-Image-2.1`.

### 3. Generate Images

```bash
python3 generate.py \
  --prompt "Retro 80s arcade spaceship, glowing cyan vector lines, black background" \
  --steps 20 \
  --width 1024 \
  --height 1024 \
  --output outputs/spaceship.png
```

---

## Sample Generation

**Prompt:** *"Retro 80s arcade spaceship, glowing cyan vector lines, black background"* (20 steps, 1024×1024)

![Retro 80s Arcade Spaceship](outputs/spaceship.png)

---

## Performance & Memory Notes

- **VRAM Required:** ~12-14 GB dynamic peak with sequential CPU offloading.
- **Inference Speed:** ~7.0s / step on AMD Radeon RX 7900 GRE (Navi 31 / gfx1100).
- **Architecture:** 7B Diffusion Transformer (DiT) conditioned on a 9B Qwen3-VL multimodal text encoder.
