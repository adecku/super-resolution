"""Training script for super-resolution models."""

import argparse
import sys
import time
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
from src.models.edsr import EDSR
from src.utils.device import get_device
from src.utils.metrics import psnr, ssim
from src.utils.seed import set_seed


def create_model(model_name, cfg, device):
    """
    Create model based on config.
    
    Args:
        model_name: Name of the model ("srcnn" or "edsr")
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
    else:
        raise ValueError(f"Unsupported model: '{model_name}'. Supported models: 'srcnn', 'edsr'")
    
    return model.to(device)


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
    save_every = cfg["train"].get("save_every", 5)
    val_every = cfg["train"].get("val_every", 1)
    grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)
    output_root = Path(cfg["paths"]["output_root"])
    
    # Create model
    model = create_model(model_name, cfg, device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # AMP setup
    use_amp = amp and device.type == "cuda"  # Only use AMP on CUDA
    use_scaler = use_amp
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step_offset = 0
    best_psnr = -float("inf")
    resume_checkpoint = None
    run_dir = None
    
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
        
        # Get starting epoch and best_psnr
        if "epoch" in resume_checkpoint:
            start_epoch = resume_checkpoint["epoch"] + 1
            global_step_offset = resume_checkpoint.get("global_step", start_epoch * len(train_loader))
            print(f"Resuming from epoch {start_epoch} (checkpoint was at epoch {resume_checkpoint['epoch']})")
        else:
            print("Warning: No epoch found in checkpoint, starting from epoch 0")
        
        if "best_psnr" in resume_checkpoint:
            best_psnr = resume_checkpoint["best_psnr"]
            print(f"Resuming best PSNR: {best_psnr:.4f} dB")
        
        # Use checkpoint's run_dir
        if "run_dir" in resume_checkpoint:
            run_dir = Path(resume_checkpoint["run_dir"])
            print(f"Continuing in existing run directory: {run_dir}")
        else:
            raise ValueError("Checkpoint missing 'run_dir'. Cannot resume.")
    else:
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"{model_name}_x{scale}" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save resolved config (only if not resuming)
    if not args.resume:
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    
    # TensorBoard writer (will append to existing logs if resuming)
    writer = SummaryWriter(log_dir=str(run_dir))
    
    # Print info
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_name}, Scale: x{scale}")
    print(f"Epochs: {epochs}, Learning rate: {lr}, AMP: {use_amp}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    lr_sample, hr_sample = next(iter(train_loader))
    print(f"Train batch shape - LR: {lr_sample.shape}, HR: {hr_sample.shape}")
    if args.resume:
        print(f"Starting from epoch {start_epoch}/{epochs}, global_step offset: {global_step_offset}")
    print()
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            # Forward pass with AMP if enabled
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    output = model(lr_batch)
                    loss = criterion(output, hr_batch)
                    loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
            else:
                output = model(lr_batch)
                loss = criterion(output, hr_batch)
                loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
            
            # Backward pass
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum_steps  # Unscale for logging
            num_batches += 1
            
            # Log step loss (with offset if resuming)
            global_step = global_step_offset + epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/loss_step", loss.item() * grad_accum_steps, global_step)
        
        # Handle remaining gradients if batch doesn't divide evenly
        if num_batches % grad_accum_steps != 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        
        # Validation (every val_every epochs)
        if (epoch + 1) % val_every == 0:
            model.eval()
            total_psnr = 0.0
            total_ssim = 0.0
            val_num_samples = 0
            
            with torch.no_grad():
                for lr_val, hr_val in val_loader:
                    lr_val = lr_val.to(device)
                    hr_val = hr_val.to(device)
                    
                    if use_amp:
                        with torch.autocast(device_type="cuda", enabled=use_amp):
                            pred_val = model(lr_val)
                    else:
                        pred_val = model(lr_val)
                    
                    # Clamp predictions to [0, 1] before metrics
                    pred_val = pred_val.clamp(0.0, 1.0)
                    
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
            
            # Save best checkpoint
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "psnr": avg_psnr,
                    "ssim": avg_ssim,
                    "best_psnr": best_psnr,
                    "run_dir": str(run_dir),
                    "global_step": global_step_offset + (epoch + 1) * len(train_loader),
                }
                if use_scaler and scaler is not None:
                    best_checkpoint["scaler"] = scaler.state_dict()
                
                torch.save(best_checkpoint, run_dir / "best.pth")
                print(f"  → New best PSNR: {best_psnr:.4f} dB (saved best.pth)")
        else:
            # If no validation this epoch, use previous values for checkpoint
            avg_psnr = resume_checkpoint.get("psnr", 0.0) if resume_checkpoint else 0.0
            avg_ssim = resume_checkpoint.get("ssim", 0.0) if resume_checkpoint else 0.0
        
        # Save examples every save_every epochs
        if (epoch + 1) % save_every == 0:
            model.eval()
            with torch.no_grad():
                lr_example, hr_example = next(iter(val_loader))
                lr_example = lr_example.to(device)
                hr_example = hr_example.to(device)
                
                if use_amp:
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        pred_example = model(lr_example)
                else:
                    pred_example = model(lr_example)
                
                # Clamp before saving images
                pred_example = pred_example.clamp(0.0, 1.0)
                
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
        
        # Save last checkpoint
        last_checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "loss": avg_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "best_psnr": best_psnr,
            "run_dir": str(run_dir),
            "global_step": global_step_offset + (epoch + 1) * len(train_loader),
        }
        if use_scaler and scaler is not None:
            last_checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(last_checkpoint, run_dir / "last.pth")
        
        # Also save epoch checkpoint
        torch.save(last_checkpoint, run_dir / f"ckpt_ep{epoch}.pth")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.6f}, Time: {epoch_time:.1f}s")
    
    writer.close()
    print(f"\nTraining completed. Checkpoints saved to: {run_dir}")
    print(f"Best PSNR: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
