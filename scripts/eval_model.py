"""Evaluate super-resolution models on DIV2K validation set."""

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
from PIL import Image

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders
from src.models.srcnn import SRCNN
from src.models.edsr import EDSR
from src.models.swinir import SwinIR
from src.utils.device import get_device
from src.utils.metrics import psnr, ssim


def create_model(model_name, cfg, device):
    """
    Create model based on config.
    
    Args:
        model_name: Name of the model ("srcnn", "edsr", or "swinir")
        cfg: Configuration dictionary
        device: Device to move model to
        
    Returns:
        Model instance
    """
    scale = cfg["data"]["scale"]
    params = cfg["model"].get("params", {})
    
    if model_name == "srcnn":
        channels = params.get("channels", 64)
        model = SRCNN(scale=scale, channels=channels)
    elif model_name == "edsr":
        num_feats = params.get("num_feats", 64)
        num_blocks = params.get("num_blocks", 16)
        res_scale = params.get("res_scale", 0.1)
        model = EDSR(scale=scale, num_feats=num_feats, num_blocks=num_blocks, res_scale=res_scale)
    elif model_name == "swinir":
        embed_dim = params.get("embed_dim", 96)
        depths = params.get("depths", [6, 6, 6, 6])
        num_heads = params.get("num_heads", [6, 6, 6, 6])
        window_size = params.get("window_size", 8)
        mlp_ratio = params.get("mlp_ratio", 4.0)
        qkv_bias = params.get("qkv_bias", True)
        drop_rate = params.get("drop_rate", 0.0)
        attn_drop_rate = params.get("attn_drop_rate", 0.0)
        drop_path_rate = params.get("drop_path_rate", 0.1)
        model = SwinIR(
            scale=scale,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
    else:
        raise ValueError(f"Unsupported model: '{model_name}'. Supported models: 'srcnn', 'edsr', 'swinir'")
    
    return model.to(device)


def tensor_to_image(tensor):
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Evaluate super-resolution model on DIV2K validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    device = get_device()
    print(f"Device: {device}")
    
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if "cfg" in checkpoint:
        ckpt_cfg = checkpoint["cfg"]
        print("Using config from checkpoint")
        scale = ckpt_cfg["data"]["scale"]
        model_name = ckpt_cfg["model"]["name"]
        cfg["data"]["scale"] = scale
        cfg["model"]["name"] = model_name
        if "params" in ckpt_cfg["model"]:
            cfg["model"]["params"] = ckpt_cfg["model"]["params"]
    else:
        print("Using config from --config argument")
        scale = cfg["data"]["scale"]
    
    _, val_loader = make_div2k_loaders(cfg)
    print(f"Validation batches: {len(val_loader)}")
    
    print(f"Creating {model_name} model: scale={scale}")
    model = create_model(model_name, cfg, device)
    
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        print("Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        raise RuntimeError(
            f"Model architecture mismatch. Please use the same config that was used for training this checkpoint."
        )
    
    model.eval()
    
    print(f"Loaded checkpoint: {ckpt_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    first_batch = None
    
    print("\nEvaluating on validation set...")
    with torch.no_grad():
        for batch_idx, (lr_batch, hr_batch) in enumerate(val_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            pred_batch = model(lr_batch)
            
            batch_psnr = psnr(pred_batch, hr_batch)
            batch_ssim = ssim(pred_batch, hr_batch)
            
            batch_size = lr_batch.shape[0]
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            num_samples += batch_size
            
            if first_batch is None:
                first_batch = (lr_batch, hr_batch, pred_batch)
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"\nResults:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.6f}")
    print(f"  Number of samples: {num_samples}")
    
    output_dir = Path("outputs/figures") / f"eval_{model_name}_x{scale}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "scale": scale,
        "psnr": float(avg_psnr),
        "ssim": float(avg_ssim),
        "num_samples": num_samples,
        "checkpoint": str(ckpt_path),
        "config": args.config
    }
    
    metrics_file = output_dir / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_file}")
    
    if first_batch is not None:
        lr_examples, hr_examples, pred_examples = first_batch
        
        num_examples = min(4, lr_examples.shape[0])
        print(f"\nSaving {num_examples} example images...")
        
        for i in range(num_examples):
            sample_dir = output_dir / f"sample_{i}"
            sample_dir.mkdir(exist_ok=True)
            
            lr_sample = lr_examples[i]
            hr_sample = hr_examples[i]
            pred_sample = pred_examples[i]
            
            lr_bicubic = F.interpolate(
                lr_sample.unsqueeze(0),
                size=(hr_sample.shape[1], hr_sample.shape[2]),
                mode="bicubic",
                align_corners=False
            ).squeeze(0)
            
            tensor_to_image(lr_bicubic).save(sample_dir / "bicubic.png")
            tensor_to_image(pred_sample).save(sample_dir / "pred.png")
            tensor_to_image(hr_sample).save(sample_dir / "hr.png")
            
            print(f"  Saved sample_{i} to: {sample_dir}")
        
        print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
