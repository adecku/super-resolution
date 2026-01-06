"""Visual sanity check for DIV2K dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work when running directly
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # scripts/../ = project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders


def tensor_to_image(tensor):
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to uint8 [0, 255]
    tensor = (tensor * 255).byte()
    # Convert to PIL Image
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Visual sanity check for DIV2K dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--out", type=str, required=True, help="Output directory for figures")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Create train loader
    train_loader, _ = make_div2k_loaders(cfg)
    
    # Get one batch
    lr, hr = next(iter(train_loader))
    
    # Get scale
    scale = cfg["data"]["scale"]
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process first 4 samples
    num_samples = min(4, lr.shape[0])
    
    for i in range(num_samples):
        sample_dir = out_dir / f"sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        
        # Get single sample
        lr_sample = lr[i]  # (C, H, W)
        hr_sample = hr[i]  # (C, H, W)
        
        # Upsample LR to HR size using bicubic interpolation
        lr_upsampled = F.interpolate(
            lr_sample.unsqueeze(0),  # Add batch dimension: (1, C, H, W)
            size=(hr_sample.shape[1], hr_sample.shape[2]),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)  # Remove batch dimension: (C, H, W)
        
        # Save images
        tensor_to_image(lr_sample).save(sample_dir / "lr.png")
        tensor_to_image(hr_sample).save(sample_dir / "hr.png")
        tensor_to_image(lr_upsampled).save(sample_dir / "bicubic.png")
    
    # Print summary
    print(f"Saved {num_samples} samples to: {out_dir}")
    print(f"LR shape: {lr.shape}")
    print(f"HR shape: {hr.shape}")


if __name__ == "__main__":
    main()

