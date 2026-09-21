#!/usr/bin/env python3
import os
import sys
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with Qwen-Image-2.1 on AMD Radeon RX 7900 GRE")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, default="outputs/generated.png", help="Output file path")
    parser.add_argument("--model-path", type=str, default="./models/Qwen-Image-2.1", help="Path to local model weights")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps (default: 28)")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="Guidance scale (default: 4.5)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

    print("==========================================================================")
    print("  🎨 Qwen-Image-2.1 Generator (PyTorch / ROCm)")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Model:    {args.model_path}")
    print(f"  Device:   AMD Radeon RX 7900 GRE (CUDA/ROCm: {torch.cuda.is_available()})")
    print("==========================================================================")

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path '{args.model_path}' not found.")
        print("Please run ./download_model.sh first to fetch the weights.")
        sys.exit(1)

    import diffusers
    from diffusers import DiffusionPipeline

    print(f"Diffusers version: {diffusers.__version__}")
    print("Loading pipeline with bfloat16...")

    try:
        if hasattr(diffusers, "QwenImage21Pipeline"):
            pipe = diffusers.QwenImage21Pipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
    except Exception as e:
        print(f"Direct pipeline load fallback: {e}")
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    # Enable sequential CPU offload to comfortably fit within 16GB VRAM
    if torch.cuda.is_available():
        print("Enabling sequential CPU offload for safe 16GB VRAM execution...")
        pipe.enable_sequential_cpu_offload()
    else:
        print("⚠️ Warning: Running on CPU.")

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"Generating image ({args.width}x{args.height}, {args.steps} steps)...")
    call_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "width": args.width,
        "height": args.height,
        "generator": generator,
        "output_type": "pil"
    }
    if args.negative_prompt:
        call_kwargs["negative_prompt"] = args.negative_prompt
    if pipe.__class__.__name__.startswith("QwenImage"):
        call_kwargs["true_cfg_scale"] = args.guidance_scale
    else:
        call_kwargs["guidance_scale"] = args.guidance_scale

    result = pipe(**call_kwargs)
    images = result.images
    import torchvision.transforms.functional as TF

    if isinstance(images, torch.Tensor):
        img_t = images[0] if images.ndim == 4 else images
        img_t = img_t.detach().cpu().float()
        if img_t.min() < 0:
            img_t = (img_t + 1.0) / 2.0
        image = TF.to_pil_image(img_t.clamp(0, 1))
    elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
        img_t = images[0].detach().cpu().float()
        if img_t.min() < 0:
            img_t = (img_t + 1.0) / 2.0
        image = TF.to_pil_image(img_t.clamp(0, 1))
    elif isinstance(images, list) and len(images) > 0:
        image = images[0]
    else:
        image = images

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    image.save(args.output)
    print(f"✅ Image successfully saved to: {args.output}")

if __name__ == "__main__":
    main()
