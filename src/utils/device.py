"""Utilities for device selection in PyTorch."""

import torch


def get_device() -> torch.device:
    """Automatically select the best available PyTorch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Test code
if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")


# PC 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# MAC
# pip install torch torchvision torchaudio