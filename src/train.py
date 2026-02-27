"""Training script for super-resolution models."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

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
from src.models.factory import create_model
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
    
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device()
    train_loader, val_loader = make_div2k_loaders(cfg)
    model_name = cfg["model"]["name"]
    scale = cfg["data"]["scale"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    amp = cfg["runtime"]["amp"]
    save_every = cfg["train"].get("save_every", 5)
    val_every = cfg["train"].get("val_every", 1)
    grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)
    output_root = Path(cfg["paths"]["output_root"])
    
    model = create_model(model_name, cfg, device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    use_amp = amp and device.type == "cuda"
    use_scaler = use_amp
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
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
        
        model.load_state_dict(resume_checkpoint["model"])
        
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            print("Loaded optimizer state")
        
        if use_scaler and "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
            print("Loaded scaler state")
        
        if "epoch" in resume_checkpoint:
            start_epoch = resume_checkpoint["epoch"] + 1
            global_step_offset = resume_checkpoint.get("global_step", start_epoch * len(train_loader))
            print(f"Resuming from epoch {start_epoch} (checkpoint was at epoch {resume_checkpoint['epoch']})")
        else:
            print("Warning: No epoch found in checkpoint, starting from epoch 0")
        
        if "best_psnr" in resume_checkpoint:
            best_psnr = resume_checkpoint["best_psnr"]
            print(f"Resuming best PSNR: {best_psnr:.4f} dB")
        
        if "run_dir" in resume_checkpoint:
            run_dir = Path(resume_checkpoint["run_dir"])
            print(f"Continuing in existing run directory: {run_dir}")
        else:
            raise ValueError("Checkpoint missing 'run_dir'. Cannot resume.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"{model_name}_x{scale}" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.resume:
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    
    writer = SummaryWriter(log_dir=str(run_dir))
    
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
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_offset = epoch - start_epoch
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    output = model(lr_batch)
                    loss = criterion(output, hr_batch)
                    loss = loss / grad_accum_steps
            else:
                output = model(lr_batch)
                loss = criterion(output, hr_batch)
                loss = loss / grad_accum_steps
            
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum_steps
            num_batches += 1
            
            global_step = global_step_offset + epoch_offset * len(train_loader) + batch_idx
            writer.add_scalar("train/loss_step", loss.item() * grad_accum_steps, global_step)
        
        if num_batches % grad_accum_steps != 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        avg_psnr = None
        avg_ssim = None
        current_global_step = global_step_offset + (epoch_offset + 1) * len(train_loader)
        
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
                    
                    pred_val = pred_val.clamp(0.0, 1.0)
                    
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
                    "global_step": current_global_step,
                }
                if use_scaler and scaler is not None:
                    best_checkpoint["scaler"] = scaler.state_dict()
                
                torch.save(best_checkpoint, run_dir / "best.pth")
                print(f"  New best PSNR: {best_psnr:.4f} dB (saved best.pth)")
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
                
                pred_example = pred_example.clamp(0.0, 1.0)
                
                examples_dir = run_dir / "examples" / f"ep{epoch}"
                examples_dir.mkdir(parents=True, exist_ok=True)
                
                num_examples = min(4, lr_example.shape[0])
                for i in range(num_examples):
                    sample_dir = examples_dir / f"sample_{i}"
                    sample_dir.mkdir(exist_ok=True)
                    
                    lr_bicubic = F.interpolate(
                        lr_example[i:i+1],
                        size=(hr_example.shape[2], hr_example.shape[3]),
                        mode="bicubic",
                        align_corners=False
                    ).squeeze(0)
                    
                    tensor_to_image(lr_bicubic).save(sample_dir / "bicubic.png")
                    tensor_to_image(pred_example[i]).save(sample_dir / "pred.png")
                    tensor_to_image(hr_example[i]).save(sample_dir / "hr.png")
        
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
            "global_step": current_global_step,
        }
        if use_scaler and scaler is not None:
            last_checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(last_checkpoint, run_dir / "last.pth")
        torch.save(last_checkpoint, run_dir / f"ckpt_ep{epoch}.pth")
        
        epoch_time = time.time() - epoch_start_time
        
        psnr_str = f"{avg_psnr:.4f} dB" if avg_psnr is not None else "N/A"
        ssim_str = f"{avg_ssim:.6f}" if avg_ssim is not None else "N/A"
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, PSNR: {psnr_str}, SSIM: {ssim_str}, Time: {epoch_time:.1f}s")
    
    writer.close()
    print(f"\nTraining completed. Checkpoints saved to: {run_dir}")
    print(f"Best PSNR: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
