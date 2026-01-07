"""Training script for super-resolution models."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so imports work when running directly
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import yaml

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders
from src.models.srcnn import SRCNN
from src.utils.device import get_device
from src.utils.metrics import psnr, ssim
from src.utils.seed import set_seed


def tensor_to_image(tensor):
    """Convert tensor (C, H, W) in [0, 1] to PIL Image."""
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Train super-resolution model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Set seed
    set_seed(cfg["project"]["seed"])
    
    # Get device
    device = get_device()
    
    # Build data loaders
    train_loader, val_loader = make_div2k_loaders(cfg)
    
    # Extract config values
    model_name = cfg["model"]["name"]
    scale = cfg["data"]["scale"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    amp = cfg["runtime"]["amp"]
    save_every = cfg["train"]["save_every"]
    output_root = Path(cfg["paths"]["output_root"])
    
    # Create model (only SRCNN for now)
    if model_name != "srcnn":
        raise ValueError(f"Only 'srcnn' model is supported, got '{model_name}'")
    
    channels = cfg["model"].get("params", {}).get("channels", 64)
    model = SRCNN(scale=scale, channels=channels).to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # AMP setup
    use_amp = amp and device.type in ["cuda", "mps"]
    use_scaler = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step_offset = 0
    resume_checkpoint = None
    
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        
        print(f"Resuming from checkpoint: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(resume_checkpoint["model"])
        
        # Load optimizer state if available
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            print("Loaded optimizer state")
        
        # Load scaler state if available and using scaler
        if use_scaler and "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
            print("Loaded scaler state")
        
        # Get starting epoch
        if "epoch" in resume_checkpoint:
            start_epoch = resume_checkpoint["epoch"] + 1
            # Calculate global_step offset to continue TensorBoard logging
            global_step_offset = start_epoch * len(train_loader)
            print(f"Resuming from epoch {start_epoch} (checkpoint was at epoch {resume_checkpoint['epoch']})")
        else:
            print("Warning: No epoch found in checkpoint, starting from epoch 0")
        
        # Use checkpoint's run_dir if available, otherwise create new one
        if "run_dir" in resume_checkpoint:
            run_dir = Path(resume_checkpoint["run_dir"])
            print(f"Continuing in existing run directory: {run_dir}")
        else:
            # Create new run directory for resumed training
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = output_root / f"srcnn_x{scale}" / f"{timestamp}_resumed"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created new run directory for resumed training: {run_dir}")
    else:
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"srcnn_x{scale}" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save resolved config (only if not resuming to existing directory)
    if not args.resume or (args.resume and "run_dir" not in resume_checkpoint):
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    
    # TensorBoard writer (will append to existing logs if resuming)
    writer = SummaryWriter(log_dir=str(run_dir))
    
    # Print info
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_name}, Scale: x{scale}, Channels: {channels}")
    print(f"Epochs: {epochs}, Learning rate: {lr}, AMP: {use_amp}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if args.resume:
        print(f"Starting from epoch {start_epoch}/{epochs}, global_step offset: {global_step_offset}")
    print()
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            if use_amp:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    output = model(lr_batch)
                    loss = criterion(output, hr_batch)
                
                # Backward pass
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                output = model(lr_batch)
                loss = criterion(output, hr_batch)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log step loss (with offset if resuming)
            global_step = global_step_offset + epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/loss_step", loss.item(), global_step)
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        
        # Validation
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        val_num_samples = 0
        
        with torch.no_grad():
            for lr_val, hr_val in val_loader:
                lr_val = lr_val.to(device)
                hr_val = hr_val.to(device)
                
                if use_amp:
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        pred_val = model(lr_val)
                else:
                    pred_val = model(lr_val)
                
                # Calculate metrics
                batch_psnr = psnr(pred_val, hr_val)
                batch_ssim = ssim(pred_val, hr_val)
                
                batch_size = lr_val.shape[0]
                total_psnr += batch_psnr * batch_size
                total_ssim += batch_ssim * batch_size
                val_num_samples += batch_size
        
        avg_psnr = total_psnr / val_num_samples
        avg_ssim = total_ssim / val_num_samples
        
        writer.add_scalar("val/psnr", avg_psnr, epoch)
        writer.add_scalar("val/ssim", avg_ssim, epoch)
        
        # Save examples every save_every epochs
        if (epoch + 1) % save_every == 0:
            model.eval()
            with torch.no_grad():
                lr_example, hr_example = next(iter(val_loader))
                lr_example = lr_example.to(device)
                hr_example = hr_example.to(device)
                
                if use_amp:
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        pred_example = model(lr_example)
                else:
                    pred_example = model(lr_example)
                
                # Save first 4 samples
                examples_dir = run_dir / "examples" / f"ep{epoch}"
                examples_dir.mkdir(parents=True, exist_ok=True)
                
                num_examples = min(4, lr_example.shape[0])
                for i in range(num_examples):
                    sample_dir = examples_dir / f"sample_{i}"
                    sample_dir.mkdir(exist_ok=True)
                    
                    # Bicubic upsampling
                    lr_bicubic = F.interpolate(
                        lr_example[i:i+1],
                        size=(hr_example.shape[2], hr_example.shape[3]),
                        mode="bicubic",
                        align_corners=False
                    ).squeeze(0)
                    
                    # Save images
                    tensor_to_image(lr_bicubic).save(sample_dir / "bicubic.png")
                    tensor_to_image(pred_example[i]).save(sample_dir / "pred.png")
                    tensor_to_image(hr_example[i]).save(sample_dir / "hr.png")
        
        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "loss": avg_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "run_dir": str(run_dir),  # Save run_dir for resume
        }
        # Save scaler state if using scaler
        if use_scaler and scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(checkpoint, run_dir / f"ckpt_ep{epoch}.pth")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.6f}")
    
    writer.close()
    print(f"\nTraining completed. Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()

