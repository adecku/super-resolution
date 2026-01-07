"""Benchmark inference time for super-resolution models."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so imports work when running directly
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

from src.config import load_config
from src.datasets.div2k import make_div2k_loaders
from src.models.srcnn import SRCNN
from src.models.edsr import EDSR
from src.utils.device import get_device


def create_model(model_name, cfg, device):
    """Create model based on config."""
    scale = cfg["data"]["scale"]
    
    if model_name == "srcnn":
        channels = cfg["model"].get("params", {}).get("channels", 64)
        model = SRCNN(scale=scale, channels=channels)
    elif model_name == "edsr":
        num_feats = cfg["model"].get("params", {}).get("num_feats", 64)
        num_blocks = cfg["model"].get("params", {}).get("num_blocks", 16)
        res_scale = cfg["model"].get("params", {}).get("res_scale", 0.1)
        model = EDSR(scale=scale, num_feats=num_feats, num_blocks=num_blocks, res_scale=res_scale)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to(device)


def benchmark_inference(model, val_loader, device, num_batches=10, warmup_batches=2):
    """
    Benchmark inference time.
    
    Args:
        model: Model to benchmark
        val_loader: Validation data loader
        device: Device to run on
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup batches
    
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    
    # Warmup
    print(f"Warming up with {warmup_batches} batches...")
    with torch.no_grad():
        for i, (lr_batch, _) in enumerate(val_loader):
            if i >= warmup_batches:
                break
            lr_batch = lr_batch.to(device)
            _ = model(lr_batch)
    
    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking with {num_batches} batches...")
    times = []
    
    with torch.no_grad():
        for i, (lr_batch, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            
            lr_batch = lr_batch.to(device)
            
            # Synchronize before timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Measure time
            start_time = time.perf_counter()
            _ = model(lr_batch)
            
            # Synchronize after forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
    
    avg_time_ms = sum(times) / len(times) if times else 0.0
    return avg_time_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference time for super-resolution models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to benchmark (default: 10)")
    parser.add_argument("--warmup-batches", type=int, default=2, help="Number of warmup batches (default: 2)")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Get device
    device = get_device()
    print(f"Device: {device}")
    
    # Extract model name
    model_name = cfg["model"]["name"]
    scale = cfg["data"]["scale"]
    
    # Create validation loader
    _, val_loader = make_div2k_loaders(cfg)
    batch_size = cfg["data"]["val_batch_size"]
    
    print(f"Model: {model_name}, Scale: x{scale}")
    print(f"Batch size: {batch_size}")
    print(f"Validation batches available: {len(val_loader)}")
    
    # Create model
    model = create_model(model_name, cfg, device)
    
    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Loaded checkpoint: {ckpt_path}")
    print()
    
    # Benchmark
    avg_time_ms = benchmark_inference(
        model, 
        val_loader, 
        device, 
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches
    )
    
    # Print results
    print()
    print("=" * 60)
    print("Benchmark Results:")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Scale: x{scale}")
    print(f"  Batch size: {batch_size}")
    print(f"  Average inference time: {avg_time_ms:.2f} ms")
    print(f"  Throughput: {1000.0 / avg_time_ms:.2f} batches/sec" if avg_time_ms > 0 else "  Throughput: N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()

