"""Evaluate bicubic upsampling baseline on DIV2K validation set."""

import argparse
import json
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders
from src.utils.metrics import psnr, ssim


def main():
    parser = argparse.ArgumentParser(description="Evaluate bicubic baseline on DIV2K validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    scale = cfg["data"]["scale"]
    _, val_loader = make_div2k_loaders(cfg)
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    for lr, hr in val_loader:
        lr_upsampled = F.interpolate(
            lr,
            size=(hr.shape[2], hr.shape[3]),
            mode="bicubic",
            align_corners=False
        )
        
        batch_psnr = psnr(lr_upsampled, hr)
        batch_ssim = ssim(lr_upsampled, hr)
        
        batch_size = lr.shape[0]
        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        num_samples += batch_size
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"Bicubic baseline (scale x{scale}):")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.6f}")
    print(f"  Number of samples: {num_samples}")
    
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"bicubic_baseline_x{scale}.json"
    
    results = {
        "scale": scale,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "num_samples": num_samples
    }
    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


# if __name__ == "__main__":
#     main()