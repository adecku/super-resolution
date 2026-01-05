"""Training script for super-resolution models (smoke test)."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


class DummyDataset(Dataset):
    """Dummy dataset generating random LR/HR pairs."""
    
    def __init__(self, num_samples: int, patch_size: int, scale: int):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.scale = scale
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        lr = torch.randn(3, self.patch_size, self.patch_size)
        hr = torch.randn(3, self.patch_size * self.scale, self.patch_size * self.scale)
        return lr, hr


class DummyModel(nn.Module):
    """Simple dummy model for smoke test."""
    
    def __init__(self, scale: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.upsample(x)
        x = self.conv2(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Set seed
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    
    # Get device
    device = get_device()
    
    # Extract config values with fallbacks
    scale = cfg.get("data", {}).get("scale", 2)
    patch_size = cfg.get("data", {}).get("patch_size", 96)
    batch_size = cfg.get("data", {}).get("batch_size", 16)
    epochs = cfg.get("train", {}).get("epochs", 10)
    lr = cfg.get("train", {}).get("lr", 0.0002)
    amp = cfg.get("runtime", {}).get("amp", True)
    output_root = cfg.get("paths", {}).get("output_root", "outputs/runs")
    model_name = cfg.get("model", {}).get("name", "unknown")
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{model_name}_x{scale}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    import yaml
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(run_dir))
    
    # Print info
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_name}, Scale: x{scale}, Epochs: {epochs}, Batch size: {batch_size}")
    
    # Dataset and DataLoader
    dataset = DummyDataset(num_samples=1000, patch_size=patch_size, scale=scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = DummyModel(scale=scale).to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # AMP setup
    use_scaler = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (lr, hr) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            optimizer.zero_grad()
            
            if amp:
                with torch.autocast(device_type=device.type, enabled=amp):
                    output = model(lr)
                    loss = criterion(output, hr)
                
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                output = model(lr)
                loss = criterion(output, hr)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log step loss
            writer.add_scalar("train/loss_step", loss.item(), epoch * len(dataloader) + batch_idx)
        
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        
        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
        }
        torch.save(checkpoint, run_dir / f"ckpt_ep{epoch}.pth")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    writer.close()
    print(f"Training completed. Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()

# uruchomienie
# python -m src.train_smoke_test --config configs/train_srcnn_x2.yaml

