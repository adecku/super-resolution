"""Evaluate SRCNN model on DIV2K validation set."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so imports work when running directly
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
from src.utils.device import get_device
from src.utils.metrics import psnr, ssim


def tensor_to_image(tensor):
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRCNN model on DIV2K validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Validate model name
    model_name = cfg["model"]["name"]
    if model_name != "srcnn":
        raise ValueError(f"This script only supports SRCNN, got '{model_name}'")
    
    # Get device
    device = get_device()
    print(f"Device: {device}")
    
    # Load checkpoint first to check its config
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint with weights_only=False for compatibility with checkpoints
    # that may contain numpy scalars or other non-weight objects
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Try to use config from checkpoint if available, otherwise use provided config
    if "cfg" in checkpoint:
        ckpt_cfg = checkpoint["cfg"]
        print("Using config from checkpoint")
        # Use checkpoint config for model params, but keep provided config for data paths
        scale = ckpt_cfg["data"]["scale"]
        channels = ckpt_cfg["model"].get("params", {}).get("channels", 64)
        # Update cfg with checkpoint config for data loading
        cfg["data"]["scale"] = scale
    else:
        print("Using config from --config argument")
        scale = cfg["data"]["scale"]
        channels = cfg["model"].get("params", {}).get("channels", 64)
        
        # Try to infer channels from checkpoint weights if mismatch occurs
        state_dict = checkpoint["model"]
        if "conv1.weight" in state_dict:
            ckpt_conv1_out = state_dict["conv1.weight"].shape[0]
            if ckpt_conv1_out != channels:
                print(f"Warning: Config specifies channels={channels}, but checkpoint has conv1 output channels={ckpt_conv1_out}")
                print(f"Using channels={ckpt_conv1_out} from checkpoint")
                channels = ckpt_conv1_out
    
    # Create validation loader only
    _, val_loader = make_div2k_loaders(cfg)
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"Creating SRCNN model: scale={scale}, channels={channels}")
    model = SRCNN(scale=scale, channels=channels).to(device)
    
    # Load model weights with error handling
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        print("Model weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"\nWarning: Could not load weights with strict=True")
        print(f"Error: {e}")
        print("\nAttempting to infer model parameters from checkpoint weights...")
        
        # Try to infer parameters from checkpoint
        state_dict = checkpoint["model"]
        inferred_channels = None
        if "conv1.weight" in state_dict:
            inferred_channels = state_dict["conv1.weight"].shape[0]
            print(f"Inferred channels={inferred_channels} from conv1.weight shape")
        
        if inferred_channels and inferred_channels != channels:
            print(f"Recreating model with inferred channels={inferred_channels}")
            model = SRCNN(scale=scale, channels=inferred_channels).to(device)
            try:
                model.load_state_dict(checkpoint["model"], strict=True)
                print("Model weights loaded successfully with inferred parameters")
                channels = inferred_channels  # Update for later use
            except RuntimeError as e2:
                print(f"Still failed with inferred parameters: {e2}")
                raise RuntimeError(
                    f"Model architecture mismatch. Checkpoint was saved with different architecture.\n"
                    f"Current model: channels={channels}, scale={scale}\n"
                    f"Checkpoint suggests: channels={inferred_channels}\n"
                    f"Please use the same config that was used for training this checkpoint."
                )
        else:
            # Try strict=False as last resort
            print("Attempting to load with strict=False (partial loading)...")
            result = model.load_state_dict(checkpoint["model"], strict=False)
            if result.missing_keys:
                print(f"Missing keys (not loaded): {result.missing_keys}")
            if result.unexpected_keys:
                print(f"Unexpected keys (ignored): {result.unexpected_keys}")
            if result.missing_keys:
                raise RuntimeError(
                    f"Cannot load checkpoint: missing keys {result.missing_keys}. "
                    f"Checkpoint architecture doesn't match current model."
                )
            print("Model weights loaded with strict=False (some keys ignored)")
    
    model.eval()
    
    print(f"Loaded checkpoint: {ckpt_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Evaluate on validation set
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    # Store first batch for visualization
    first_batch = None
    
    print("\nEvaluating on validation set...")
    with torch.no_grad():
        for batch_idx, (lr_batch, hr_batch) in enumerate(val_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            # Forward pass
            pred_batch = model(lr_batch)
            
            # Calculate metrics for this batch
            batch_psnr = psnr(pred_batch, hr_batch)
            batch_ssim = ssim(pred_batch, hr_batch)
            
            batch_size = lr_batch.shape[0]
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            num_samples += batch_size
            
            # Store first batch for visualization
            if first_batch is None:
                first_batch = (lr_batch, hr_batch, pred_batch)
    
    # Calculate averages
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    # Print results
    print(f"\nResults:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.6f}")
    print(f"  Number of samples: {num_samples}")
    
    # Create output directory
    output_dir = Path("outputs/figures") / f"eval_srcnn_x{scale}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
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
    
    # Save example images (first 4 samples from first batch)
    if first_batch is not None:
        lr_examples, hr_examples, pred_examples = first_batch
        
        num_examples = min(4, lr_examples.shape[0])
        print(f"\nSaving {num_examples} example images...")
        
        for i in range(num_examples):
            sample_dir = output_dir / f"sample_{i}"
            sample_dir.mkdir(exist_ok=True)
            
            lr_sample = lr_examples[i]  # (C, H, W)
            hr_sample = hr_examples[i]  # (C, H, W)
            pred_sample = pred_examples[i]  # (C, H, W)
            
            # Bicubic upsampling for comparison
            lr_bicubic = F.interpolate(
                lr_sample.unsqueeze(0),  # Add batch dimension
                size=(hr_sample.shape[1], hr_sample.shape[2]),
                mode="bicubic",
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
            
            # Save images
            tensor_to_image(lr_bicubic).save(sample_dir / "bicubic.png")
            tensor_to_image(pred_sample).save(sample_dir / "pred.png")
            tensor_to_image(hr_sample).save(sample_dir / "hr.png")
            
            print(f"  Saved sample_{i} to: {sample_dir}")
        
        print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()