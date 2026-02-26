"""Visual sanity check for DIV2K dataset."""

import argparse
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders


def tensor_to_image(tensor):
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Visual sanity check for DIV2K dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--out", type=str, required=True, help="Output directory for figures")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train_loader, _ = make_div2k_loaders(cfg)
    lr, hr = next(iter(train_loader))
    scale = cfg["data"]["scale"]
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(4, lr.shape[0])
    
    for i in range(num_samples):
        sample_dir = out_dir / f"sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        
        lr_sample = lr[i]
        hr_sample = hr[i]
        
        lr_upsampled = F.interpolate(
            lr_sample.unsqueeze(0),
            size=(hr_sample.shape[1], hr_sample.shape[2]),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)
        
        tensor_to_image(lr_sample).save(sample_dir / "lr.png")
        tensor_to_image(hr_sample).save(sample_dir / "hr.png")
        tensor_to_image(lr_upsampled).save(sample_dir / "bicubic.png")
    
    print(f"Saved {num_samples} samples to: {out_dir}")
    print(f"LR shape: {lr.shape}")
    print(f"HR shape: {hr.shape}")


if __name__ == "__main__":
    main()

